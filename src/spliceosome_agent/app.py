"""Minimal current-task TUI. Queries and errors survive background refreshes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from .commands import command_names, help_text
from .config import ModelProfile, PRESETS, config_ready, load_config, save_config
from .credentials import KeychainUnavailable, has_api_key, set_api_key
from .data import resource
from .io import object_hash
from .planning import resolve_study

CONFIG_HINT = "API keys are stored in the OS keychain, never in the config file."

COMMAND_LIST = command_names(tui=True)
TUI_HELP = help_text(tui=True)


STAGE_LABELS = {
    "ready": "Starting",
    "start_failed": "Startup failed",
    "running": "Running",
    "waiting_plan_revision": "Plan revision required",
    "complete": "Completed",
    "partial": "Partially completed",
    "failed": "Run failed",
    "stopped": "Stopped",
    "budget_exhausted": "Budget exhausted",
    "stalled": "Run interrupted",
}


# Fixed cell fonts keep the welcome mark independent of external font packages.
BLOCK_FONT_LARGE = {
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
}
BLOCK_FONT_COMPACT = {
    "S": ("111", "100", "111", "001", "111"),
    "P": ("110", "101", "110", "100", "100"),
    "L": ("100", "100", "100", "100", "111"),
    "I": ("111", "010", "010", "010", "111"),
    "C": ("111", "100", "100", "100", "111"),
    "E": ("111", "100", "110", "100", "111"),
    "O": ("111", "101", "101", "101", "111"),
    "M": ("101", "111", "101", "101", "101"),
    "A": ("010", "101", "111", "101", "101"),
    "G": ("111", "100", "101", "101", "111"),
    "N": ("101", "111", "111", "111", "101"),
    "T": ("111", "010", "010", "010", "010"),
}


def block_banner(compact=False):
    """Two centered words on a fixed cell canvas, separated by one blank row."""
    font = BLOCK_FONT_COMPACT if compact else BLOCK_FONT_LARGE
    words = ("SPLICEOSOME", "AGENT")
    width = len(words[0]) * (len(font["S"][0]) + 1) - 1
    pixels = str.maketrans("01", " █")
    lines = []
    for index, word in enumerate(words):
        if index:
            lines.append(" " * width)
        for row in range(len(font["S"])):
            line = " ".join(font[letter][row] for letter in word).translate(pixels)
            lines.append(line.center(width))
    return "\n".join(lines)


class WelcomeBanner(ScrollableContainer):
    """Center when it fits; keep the full mark scrollable in smaller viewports."""

    def compose(self) -> ComposeResult:
        yield Static("", id="welcome-banner", markup=False)

    def on_mount(self):
        self.call_after_refresh(self._fit_banner)

    def on_resize(self):
        self.call_after_refresh(self._fit_banner)

    def _fit_banner(self):
        size = self.content_size
        if not size.width or not size.height:
            self._last_size = None
            return
        if getattr(self, "_last_size", None) == size:
            return
        self._last_size = size
        self.compact = size.width < 65 or size.height < 15
        text = block_banner(self.compact)
        banner = self.query_one("#welcome-banner", Static)
        banner.update(Text(text, no_wrap=True, overflow="ignore"))
        banner.styles.width = len(text.splitlines()[0])
        banner.styles.height = len(text.splitlines())
        banner.styles.margin = 0
        # Scrollbars settle after sizing; use the real remaining viewport to center.
        self.call_after_refresh(self._center_banner)

    def _center_banner(self):
        banner = self.query_one("#welcome-banner", Static)
        viewport = self.scrollable_content_region
        left = max(0, (viewport.width - banner.size.width) // 2)
        top = max(0, (viewport.height - banner.size.height) // 2)
        banner.styles.margin = (top, 0, 0, left)


def preset_for(base_url):
    """Map a saved base URL back to a provider preset, or 'custom' when it matches none."""
    normalized = (base_url or "").strip().rstrip("/")
    for name, data in PRESETS.items():
        if data["base_url"].rstrip("/") == normalized:
            return name
    return "custom"


class _CommandSuggester(SuggestFromList):
    def __init__(self):
        super().__init__(COMMAND_LIST, case_sensitive=False)


class ModelScreen(ModalScreen[bool]):
    """Model configuration. In required mode the dialog cannot be dismissed without saving."""

    def __init__(self, config_path, required=False):
        super().__init__()
        self.config_path = config_path
        self.required = required
        self.config = load_config(config_path)
        name = self.config.default if self.config.default in self.config.profiles else next(iter(self.config.profiles), None)
        self.profile = self.config.profiles[name] if name else None
        # Resolve initial select values here: Select.__init__ sets the value directly, whereas
        # assigning .value after mount posts Select.Changed and would reset the other fields.
        self.initial_preset = preset_for(self.profile.base_url) if self.profile else "openai"
        self.initial_kind = self.profile.kind if self.profile else "remote"

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Static("Model setup", id="model-title")
            with VerticalScroll(id="model-form"):
                yield Label("Provider preset", classes="field-label")
                yield Select([(name, name) for name in PRESETS] + [("Custom endpoint", "custom")],
                             value=self.initial_preset, id="preset")
                yield Label("Profile name", classes="field-label")
                yield Input(placeholder="e.g. cloud", id="profile-name")
                yield Label("Base URL (include /v1 if required)", classes="field-label")
                yield Input(placeholder="https://api.openai.com/v1", id="base-url")
                yield Label("Model ID", classes="field-label")
                yield Input(placeholder="gpt-4o-mini", id="model-id")
                yield Label("Endpoint type", classes="field-label")
                yield Select([("Remote service", "remote"), ("Local loopback", "local")],
                             value=self.initial_kind, id="model-kind")
                yield Label("API key environment variable (optional for local)", classes="field-label")
                yield Input(placeholder="OPENAI_API_KEY", id="key-env")
                yield Label("API key (saved to the OS keychain)", classes="field-label")
                yield Input(password=True, placeholder="Leave blank to keep the saved key", id="key-value")
                yield Static(CONFIG_HINT, id="model-hint")
            yield Static("", id="model-error", markup=False)
            with Horizontal(id="model-buttons"):
                yield Button("Save", id="model-save", variant="primary")
                if not self.required:
                    yield Button("Cancel", id="model-back")

    def on_mount(self):
        self.prefill()

    def prefill(self):
        # Only text inputs are set here. Select values are fixed at construction time so that
        # programmatic assignment cannot post Select.Changed and overwrite the saved endpoint.
        profile = self.profile
        if profile:
            self.query_one("#profile-name", Input).value = profile.name
            self.query_one("#base-url", Input).value = profile.base_url
            self.query_one("#model-id", Input).value = profile.model
            self.query_one("#key-env", Input).value = profile.api_key_env or ""
            self.refresh_hint(profile)
        else:
            self.query_one("#profile-name", Input).value = "default"
            self.query_one("#base-url", Input).value = PRESETS["openai"]["base_url"]
            self.query_one("#key-env", Input).value = PRESETS["openai"]["api_key_env"] or ""

    def refresh_hint(self, profile):
        if has_api_key(profile):
            self.query_one("#model-hint", Static).update("A key is already saved. Leave blank to keep it.")
        else:
            self.query_one("#model-hint", Static).update(CONFIG_HINT)

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "preset" and event.value in PRESETS:
            data = PRESETS[event.value]
            self.query_one("#base-url", Input).value = data["base_url"]
            self.query_one("#key-env", Input).value = data["api_key_env"] or ""
            self.query_one("#model-kind", Select).value = data["kind"]

    def on_input_changed(self, event: Input.Changed):
        if event.input.id != "base-url":
            return
        preset = self.query_one("#preset", Select).value
        if preset in PRESETS:
            if PRESETS[preset]["base_url"].rstrip("/") != event.value.strip().rstrip("/"):
                # Keep the provider label honest after a hand-edited endpoint. Selecting a preset
                # never hits this branch because it writes the preset's own URL back.
                self.query_one("#preset", Select).value = "custom"

    def on_key(self, event):
        if self.required and event.key == "escape":
            event.stop()
            self.query_one("#model-error", Static).update("Complete model setup to continue.")

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        if event.button.id == "model-back":
            self.dismiss(False)
        elif event.button.id == "model-save":
            self.save()

    def save(self):
        error = self.query_one("#model-error", Static)
        try:
            field = lambda key: self.query_one(f"#{key}", Input).value.strip()
            name = field("profile-name")
            env = field("key-env") or None
            entered = field("key-value")
            if not name:
                raise ValueError("Profile name is required")
            if not field("model-id"):
                raise ValueError("Model ID is required")
            profile = ModelProfile(name=name, base_url=field("base-url"), model=field("model-id"),
                                   api_key_env=env, kind=self.query_one("#model-kind", Select).value)
            if profile.kind == "remote" and not (entered or env or has_api_key(profile)):
                raise ValueError("Remote models need an API key or its environment variable")
            config = load_config(self.config_path)
            config.profiles[name] = profile
            config.default = name
            save_config(config, self.config_path)
            if entered:
                try:
                    set_api_key(name, entered)
                except KeychainUnavailable as exc:
                    raise ValueError(f"Could not save the key to the OS keychain: {exc}") from exc
            self.dismiss(True)
        except Exception as exc:
            error.update(str(exc))


class StudyEditorScreen(ModalScreen):
    """Structured StudySpec editor; saving creates a new plan version and voids old approvals."""

    def __init__(self, session):
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        initial = self.session.study.model_dump_json(indent=2) if self.session.study else "{}"
        with Vertical(id="editor-dialog"):
            yield Static("Study editor · saving requires a new plan confirmation", id="editor-title")
            with VerticalScroll(id="editor-scroll"):
                yield TextArea(initial, id="study-editor", show_line_numbers=True)
            yield Static("", id="editor-error", markup=False)
            with Horizontal(id="editor-buttons"):
                yield Button("Save", id="save-study", variant="primary")
                yield Button("Cancel", id="editor-cancel")

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        if event.button.id == "editor-cancel":
            self.dismiss(None)
        elif event.button.id == "save-study":
            self.save()

    def save(self):
        error = self.query_one("#editor-error", Static)
        try:
            study = resolve_study(json.loads(self.query_one("#study-editor", TextArea).text), Path.cwd())
            self.dismiss(study)
        except Exception as exc:
            error.update(f"{type(exc).__name__}: {exc}")


class SpliceosomeApp(App):
    TITLE = "Spliceosome Agent"
    CSS_PATH = resource("styles.tcss")
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
        Binding("f1", "help", "Help", priority=True, show=False),
        Binding("escape", "close_help", "Back", priority=True, show=False),
        Binding("ctrl+r", "refresh", "Refresh", priority=True, show=False),
    ]

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._busy = False
        # Only task view follows polling. Replies, errors and explicit queries stay put.
        self._view = "home"
        self._return_view = "home"
        self._response = ""
        self._displayed_digest = None
        self._run_error = ""
        self._stop_requested = None

    def compose(self) -> ComposeResult:
        yield WelcomeBanner(id="welcome-scroll", can_focus=True)
        with VerticalScroll(id="workspace-scroll"):
            yield Static("", id="current-content", markup=False)
            with Vertical(id="dataset-controls"):
                yield Label("Choose a dataset")
                yield Select([], prompt="Matrix and metadata", id="dataset-choice")
            with Vertical(id="scale-controls"):
                yield Label("Confirm measurement scale")
                yield Select(self._scale_options(None), value="unknown", allow_blank=False,
                             id="scale-choice")
        yield Static("Working…", id="activity", markup=False)
        with Horizontal(id="primary-actions"):
            yield Button("Confirm and run", id="approve", variant="success")
            yield Button("Stop", id="stop", variant="error")
        yield Input(placeholder="Describe your goal and data location", suggester=_CommandSuggester(), id="chat-input")

    def on_mount(self):
        self._refresh_state()
        self.set_interval(1, self._tick)
        self._focus_input()
        if not config_ready(load_config(self.session.config_path)):
            self.call_after_refresh(self.require_model_setup)

    def _get(self, selector, expect_type=None):
        """Use the base screen even while a configuration modal is open."""
        if not self.is_running:
            return None
        try:
            screen = self.screen_stack[0]
        except ScreenStackError:
            return None
        for widget in screen.query(selector):
            if expect_type is None or isinstance(widget, expect_type):
                return widget
        return None

    def _show(self, selector, visible):
        widget = self._get(selector)
        if widget is not None:
            widget.display = visible

    def _update(self, selector, value):
        widget = self._get(selector, Static)
        if widget is not None and getattr(widget, "_current_text", None) != value:
            widget.update(value)
            widget._current_text = value

    def _focus_input(self):
        widget = self._get("#chat-input", Input)
        if widget is not None and widget.is_attached:
            widget.focus()

    def _message(self, value):
        if self._view != "response":
            self._return_view = self._view
        self._view = "response"
        self._response = value or ""
        self._refresh_state()
        if value == TUI_HELP:
            scroll = self._get("#workspace-scroll", VerticalScroll)
            if scroll is not None:
                self.call_after_refresh(scroll.scroll_home, animate=False)

    def _plan_and_approval(self):
        plan = None
        if self.session.store.get("current_plan"):
            try:
                plan = self.session.current_plan()
            except (ValueError, OSError) as exc:
                self._run_error = f"Could not read the plan: {type(exc).__name__}: {exc}"
        approved = False
        if plan is not None:
            try:
                self.session.store.require_approval(object_hash(plan))
                approved = True
            except (PermissionError, ValueError):
                pass
        return plan, approved

    def _run_state(self):
        run_id = self.session.store.get("latest_run")
        if not run_id:
            return None, None
        try:
            folder = self.session.store.run_dir(run_id)
            current = self.session.store.get("current_plan")
            if current and (folder / "plan.json").is_file():
                if object_hash(json.loads((folder / "plan.json").read_text(encoding="utf-8"))) != current:
                    return None, None
            path = folder / "run_state.json"
            if not path.is_file():
                self._run_error = "Run state is missing. Check the task with /runs."
                return run_id, None
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("status"), str):
                raise ValueError("Invalid run state format")
            # A killed worker may leave a running snapshot; never label it as active.
            job_path = folder / "job.json"
            if state["status"] in {"ready", "running"} and job_path.is_file():
                from .jobs import same_process
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("status") == "stop_requested":
                    state["status"] = "stop_requested"
                elif not same_process(job.get("identity")):
                    state["status"] = "stopped" if job.get("status") == "stopped" else "stalled"
            return run_id, state
        except (ValueError, OSError) as exc:
            self._run_error = f"Could not read run state: {type(exc).__name__}: {exc}"
            return run_id, None

    @staticmethod
    def _scale_options(domain):
        options = [("Unknown; do not run", "unknown"), ("Already log2 abundance", "log2"),
                   ("Positive abundance; apply log2 (no pseudocount)", "raw_log2")]
        if domain == "rna":
            options.append(("Raw integer counts", "counts"))
        return options

    def _sync_choices(self, card, visible):
        alternatives = [item for item in card.get("alternatives", []) if isinstance(item, dict)]
        dataset = self._get("#dataset-choice", Select)
        signature = json.dumps(alternatives, sort_keys=True)
        if dataset is not None and getattr(self, "_dataset_signature", None) != signature:
            registry = self.session.store.get("input_registry", {})
            dataset.set_options([
                (f"{i+1}. {Path(registry.get(item['matrix'], item['matrix'])).name} + "
                 f"{Path(registry.get(item['metadata'], item['metadata'])).name} · {item['group_column']}", str(i))
                for i, item in enumerate(alternatives)])
            self._dataset_signature = signature
        self._show("#dataset-controls", visible and len(alternatives) > 1)
        needs_scale = bool(visible and card.get("study") and card.get("status") == "needs_scale")
        choice = self._get("#scale-choice", Select)
        if choice is not None:
            domain = next(iter(card.get("study", {}).get("assays", {})), None)
            if getattr(self, "_scale_domain", "unset") != domain:
                choice.set_options(self._scale_options(domain))
                self._scale_domain = domain
            target = card.get("scale_choice", "unknown")
            if choice.value != target:
                choice.value = target
        self._show("#scale-controls", needs_scale)
        for widget in (dataset, choice):
            if widget is not None:
                widget.disabled = self._busy

    def _confirmation_text(self, plan, card):
        """Present verified essentials locally; full contracts remain in /plan."""
        pending = self.session.draft_pending()
        study = card.get("study") or None
        if plan is not None and not pending:
            study = plan.study.model_dump()
        if not study:
            draft = self.session.store.get("study_draft", {})
            questions = draft.get("questions", [])
            errors = self.session.store.get("latest_probe", {}).get("errors", []) if pending else []
            lines = list(dict.fromkeys(errors + questions))
            message = card.get("message")
            if message:
                lines.insert(0, message)
            return "\n".join(dict.fromkeys(lines)) or "Spliceosome Agent"
        lines = []
        if study.get("synthetic"):
            lines.append("Synthetic example · software validation")
        lines.append("Differential analysis" if study.get("analysis_mode") == "differential" else "Candidate screening")
        design = "paired" if study["design"] == "paired" else "unpaired"
        lines.append(f"{study['case']} − {study['reference']} · {design}")
        for domain, assay in study["assays"].items():
            info = plan.preflight.summary.get(domain, {}) if plan and plan.preflight and not pending else card
            label = {"rna": "RNA", "protein": "Protein"}.get(domain, domain)
            lines.append(f"{label}: {Path(assay['matrix']).name} + {Path(assay['metadata']).name}")
            unit = "pairs" if study["design"] == "paired" else "independent samples"
            if info:
                lines.append(f"{info.get('n_features', '?')} features · {info.get('n_samples', '?')} samples · {info.get('n_units', '?')} {unit}")
                counts = info.get("group_counts", {})
                if counts:
                    lines.append(", ".join(f"{group}={count}" for group, count in counts.items()))
            scale = assay.get("scale")
            method = "DESeq2" if scale == "counts" else "limma" if scale else "Scale confirmation required"
            scale_label = {"counts": "Raw counts", "log2_expression": "log2 expression", "log2_abundance": "log2 abundance",
                           "abundance": "Positive abundance → log2 (no pseudocount)"}.get(scale, "Not confirmed")
            lines.append(f"{scale_label} · {method}")
        lines.append(f"Full-feature BH q≤{study.get('alpha', .05):g} · {study.get('confidence', .95):.0%} CI")
        if study.get("min_effect"):
            lines.append(f"Additional effect threshold: {study['min_effect']:g}")
        if study.get("covariates"):
            lines.append("Covariates: " + ", ".join(study["covariates"]))
        if plan and not plan.preflight:
            lines.append("Legacy plan: preflight and a new confirmation are required before execution.")
        probe = self.session.store.get("latest_probe", {}) if pending else {}
        errors = probe.get("errors", [])
        warnings = probe.get("warnings", [])
        if plan and plan.preflight and not pending:
            errors = plan.preflight.errors
            warnings = plan.preflight.warnings
        lines.extend("Blocked: " + str(item) for item in dict.fromkeys(errors))
        lines.extend("Warning: " + str(item) for item in dict.fromkeys(warnings))
        if pending and card.get("status") != "needs_scale":
            lines.extend(str(item) for item in self.session.store.get("study_draft", {}).get("questions", []) if str(item) not in lines)
        return "\n".join(lines)

    def _run_text(self, run_id, state):
        status = state["status"]
        folder = self.session.store.run_dir(run_id)
        if status == "stop_requested" or (self._stop_requested == run_id and status in {"ready", "running"}):
            heading = "Stopping"
        else:
            heading = STAGE_LABELS.get(status, status)
        if status in {"ready", "running"} and self._stop_requested != run_id:
            # Show only the latest role event, not the event payload/history.
            events = self.session.store.events(run_id, 12)
            role = next((e.get("payload", {}).get("role") for e in reversed(events)
                         if e["kind"] in {"role_started", "role_finished"} and e.get("payload", {}).get("role")), None)
            if role:
                heading += " · " + role
        lines = [heading]
        warnings = list(state.get("warnings") or [])
        for result in (state.get("results") or {}).values():
            if isinstance(result, dict):
                warnings.extend(result.get("errors") or [])
                warnings.extend(result.get("warnings") or [])
        lines.extend("Warning: " + str(item) for item in dict.fromkeys(str(w) for w in warnings))
        if status in {"failed", "stopped", "stalled", "partial", "budget_exhausted"}:
            lines.append(f"Logs: /logs {run_id}")
            lines.append(f"Resume: /resume {run_id} (matching fingerprints required)")
        elif status == "waiting_plan_revision":
            lines.append("Revise the study configuration and confirm the new plan.")
        elif status == "start_failed":
            lines.append("Startup failed. Resolve the error, then retry confirmation.")
        if (folder / "report.md").is_file():
            lines.append(f"Report: {folder / 'report.md'}")
        lines.append(f"Output: {folder}")
        return "\n".join(lines)

    def _refresh_state(self):
        if not self.is_running:
            return
        self._show("#activity", self._busy)
        entry = self._get("#chat-input", Input)
        if entry is not None:
            entry.disabled = self._busy
        # Home and queries do not load or follow the retained task. In particular,
        # a broken state file cannot replace the welcome page during polling.
        if self._view != "task":
            welcome = self._view == "home" and not self._busy
            self._show("#welcome-scroll", welcome)
            self._show("#workspace-scroll", not welcome)
            self._update("#current-content", self._response if self._view == "response" else "")
            for selector in ("#dataset-controls", "#scale-controls", "#approve", "#stop", "#primary-actions"):
                self._show(selector, False)
            self._displayed_digest = None
            return
        self._run_error = ""
        plan, _ = self._plan_and_approval()
        run_id, state = self._run_state()
        status = state.get("status") if state else None
        pending = self.session.draft_pending()
        card = self.session.store.get("confirmation_card", {})
        self._show("#welcome-scroll", False)
        self._show("#workspace-scroll", True)
        has_run = state is not None and not pending
        if self._run_error:
            text = self._run_error
        elif has_run and status != "start_failed":
            text = self._run_text(run_id, state)
        else:
            text = self._confirmation_text(plan, card)
            if status == "start_failed":
                text = self._run_text(run_id, state) + "\n\n" + text
        self._update("#current-content", text)
        self._sync_choices(card, not has_run and not self._run_error)
        can_confirm = bool(plan and not pending and not self._run_error and
                           (run_id is None or status == "start_failed"))
        # Approval always refers to the card actually on screen, never a hidden query view.
        self._displayed_digest = object_hash(plan) if can_confirm else None
        self._show("#approve", can_confirm)
        stop = bool(state and status in {"ready", "running"} and self._stop_requested != run_id)
        self._show("#stop", stop)
        self._show("#primary-actions", can_confirm or stop)
        for selector in ("#approve", "#stop"):
            button = self._get(selector, Button)
            if button is not None:
                button.disabled = self._busy

    def refresh_controls(self):
        self._refresh_state()

    def _tick(self):
        try:
            self._refresh_state()
        except NoMatches:
            return

    def _set_busy(self, busy):
        self._busy = busy
        self._refresh_state()

    def _submit(self, value):
        text = (value or "").strip()
        if not text or self._busy or isinstance(self.screen, ModalScreen):
            return False
        # Reserve the operation before yielding to prevent queued double submissions.
        self._set_busy(True)
        if text == "/models":
            self._set_busy(False)
            self.run_worker(self.open_models(), exclusive=False, group="dialogs")
        elif text == "/edit":
            self._set_busy(False)
            self.run_worker(self.open_study_editor(), exclusive=False, group="dialogs")
        else:
            self.run_worker(self._handle_async(text), exclusive=False, group="conversation")
        return True

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "chat-input" and self._submit(event.value):
            event.input.value = ""

    async def _handle_async(self, text):
        self._set_busy(True)
        try:
            if text.strip() == "/home":
                self._view = self._return_view = "home"
                self._response = ""
                return
            if text.strip() == "/current":
                if any(self.session.store.get(key) for key in
                       ("current_plan", "latest_run", "study", "study_draft", "confirmation_card")):
                    self._view = "task"
                else:
                    self._message("No current task. Use /tasks to view saved runs.")
                return
            if text.strip() == "/help":
                self._message(TUI_HELP)
                return
            reply = await asyncio.to_thread(self.session.handle, text)
            if reply == "__quit__":
                self.exit()
                return
            command = text.split(maxsplit=1)[0]
            query = command in {"/inputs", "/plans", "/runs", "/tasks", "/status", "/report", "/logs", "/doctor"} or text.strip() == "/plan"
            # State-changing replies often contain full plans or worker JSON. Render the
            # current task once; explicit queries retain their complete original output.
            task_command = command in {"/demo", "/study", "/edit", "/scale", "/swap", "/dataset", "/stop", "/resume"}
            plan_created = command == "/plan" and reply and reply.startswith("Plan v")
            run_created = command in {"/confirm", "/run"} and self._run_state()[1] is not None
            card = self.session.store.get("confirmation_card", {})
            card_reply = card and reply == self.session.confirmation_text()
            if not query and (task_command or plan_created or card_reply or run_created):
                self._view = "task"
            else:
                self._message(reply)
        except Exception as exc:
            self._message(f"Action failed: {type(exc).__name__}: {exc}")
        finally:
            self._set_busy(False)
            self._focus_input()

    async def _run_session(self, command):
        await self._handle_async(command)

    def on_select_changed(self, event: Select.Changed):
        if isinstance(self.screen, ModalScreen) or event.value != event.select.value or self._busy or self._view != "task":
            return
        if event.select.id == "scale-choice":
            card = self.session.store.get("confirmation_card", {})
            if event.value is Select.BLANK or not card.get("study") or event.value == card.get("scale_choice", "unknown"):
                return
            self._submit(f"/scale {event.value}")
        elif event.select.id == "dataset-choice" and event.value is not Select.BLANK:
            self._submit(f"/dataset {int(event.value)+1}")

    async def confirm_approve(self, digest=None):
        digest = digest or self._displayed_digest
        plan, _ = self._plan_and_approval()
        if not digest or not plan or self.session.draft_pending():
            self._set_busy(False)
            return
        if digest != object_hash(plan):
            self._message("The plan changed. Press Esc to review the new plan before confirming.")
            self._set_busy(False)
            return
        await self._run_session(f"/confirm {digest}")

    async def confirm_stop(self, run_id=None):
        run_id = run_id or self._run_state()[0]
        if run_id:
            self._stop_requested = run_id
            await self._run_session(f"/stop {run_id}")
            self._stop_requested = None
            self._refresh_state()

    async def open_models(self):
        await self.push_screen_wait(ModelScreen(self.session.config_path, required=False))
        self.session.config = load_config(self.session.config_path)
        self._refresh_state()
        self._focus_input()

    async def open_study_editor(self):
        study = await self.push_screen_wait(StudyEditorScreen(self.session))
        if study is None:
            self._focus_input()
            return
        self._view = "task"
        self._set_busy(True)
        try:
            self.session.store.set("confirmation_card", {})
            previous, _ = self._plan_and_approval()
            goal = previous.goal if previous is not None else study.name
            await asyncio.to_thread(self.session.create_plan, study, goal)
        except Exception as exc:
            self._message(f"Action failed: {type(exc).__name__}: {exc}")
        finally:
            self._set_busy(False)
            self._focus_input()

    def on_button_pressed(self, event: Button.Pressed):
        if isinstance(self.screen, ModalScreen):
            return
        event.stop()
        if self._busy or self._view != "task":
            return
        if event.button.id == "approve" and self._displayed_digest:
            digest = self._displayed_digest
            self._set_busy(True)
            self.run_worker(self.confirm_approve(digest), exclusive=False, group="actions")
        elif event.button.id == "stop":
            run_id, state = self._run_state()
            if run_id and state and state["status"] in {"ready", "running"}:
                self._set_busy(True)
                self.run_worker(self.confirm_stop(run_id), exclusive=False, group="actions")

    def on_key(self, event):
        if (event.key not in {"pageup", "pagedown"} or self._busy
                or isinstance(self.screen, ModalScreen)
                or self._view != "response" or self._response != TUI_HELP):
            return
        scroll = self._get("#workspace-scroll", VerticalScroll)
        if scroll is not None:
            if event.key == "pagedown":
                scroll.scroll_page_down(animate=False)
            else:
                scroll.scroll_page_up(animate=False)
            event.stop()
            event.prevent_default()

    def action_help(self):
        if isinstance(self.screen, ModalScreen) or self._busy:
            return
        if self._view == "response" and self._response == TUI_HELP:
            self.action_close_help()
        else:
            self._message(TUI_HELP)

    def action_close_help(self):
        if isinstance(self.screen, ModalScreen):
            if isinstance(self.screen, ModelScreen) and self.screen.required:
                self.screen.query_one("#model-error", Static).update("Complete model setup to continue.")
                return
            self.screen.dismiss(None)
            return
        if not self._busy:
            if self._view == "response":
                self._view = self._return_view
            self._refresh_state()
            self._focus_input()

    def action_refresh(self):
        self._refresh_state()

    def require_model_setup(self):
        self.push_screen(ModelScreen(self.session.config_path, required=True), self.after_model_setup)

    def after_model_setup(self, saved):
        self.session.config = load_config(self.session.config_path)
        self._refresh_state()
        if not saved or not config_ready(self.session.config):
            self.call_after_refresh(self.require_model_setup)
