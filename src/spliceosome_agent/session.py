"""Shared interactive controller: approval is explicit and hash-bound in every UI."""
from __future__ import annotations

import json
import shlex
import re
import copy
import subprocess
from pathlib import Path

from . import jobs
from .commands import help_text
from .config import load_config
from .data import private_ids
from .engine import Engine
from .io import object_hash, read_json
from .llm import ModelClient
from .models import StudySpec, StudyDraft
from pydantic import ValidationError
from .preflight import discover, inspect_file, run_preflight, verify_environment, model_summary, inventory_identifiers
from .planning import describe, diff_plans, load_study, make_plan, resolve_study, understand
from .store import Store

HELP = help_text()


def plan_lines(store):
    """Human-readable inventory of every stored plan version, newest first."""
    records = store.list_plans()
    if not records:
        return "No stored plans yet. Build one with /plan <goal> or load /study first."
    lines = []
    for record in records:
        if record["current"]:
            scope = "current"
        elif record["superseded"]:
            scope = "superseded"
        else:
            scope = "not current"
        if record["approved"]:
            state = "approved"
        elif record["approved_ever"]:
            state = "approval expired"
        else:
            state = "not approved"
        lines.append(f"v{record['version']}  {record['plan_hash'][:12]}  {scope}, {state}  "
                     f"{record['created_at']}  {record['goal'][:80]}")
    lines.append("")
    lines.append("Show one version with /plans <hash prefix>.")
    return "\n".join(lines)


class Session:
    def __init__(self, workspace, config_path=None, client=None):
        self.store = Store(workspace)
        self.config_path = config_path
        self.config = load_config(config_path)
        self.client = client

    @property
    def study(self):
        payload = self.store.get("study")
        return StudySpec.model_validate(payload) if payload else None

    def invalidate_draft(self):
        self.store.set("draft_dirty", True)
        digest = self.store.get("current_plan")
        if digest and self.store.is_approved(digest):
            self.store.cancel(digest)

    def set_study(self, study):
        payload = study.model_dump()
        if self.store.get("study") != payload:
            self.invalidate_draft()
        self.store.set("study", payload)
        self.store.set("study_draft", StudyDraft(study=payload, goal=self.store.get("draft_goal", "")).model_dump())
        # A locally loaded config explicitly authorizes its declared input files.
        from .planning import input_paths
        self._register_files(input_paths(study))

    def _register_files(self, paths):
        registry = self.store.get("input_registry", {})
        for path in paths:
            resolved = str(Path(path).expanduser().resolve())
            if resolved not in registry.values():
                registry[f"file_{len(registry):04d}"] = resolved
        self.store.set("input_registry", registry)
        return registry

    def _authorize_text(self, text):
        # Paths must appear in the user's own message, never just in model output.
        paths = []
        pattern = r"[\"']([^\"']+)[\"']|(?<![A-Za-z0-9_])((?:/|~/|\./|\.\./)[^\s，。；：,;]+)"
        for match in re.finditer(pattern, text):
            value = (match.group(1) or match.group(2)).rstrip(').]）')
            if value.startswith(("/", "~/", "./", "../")):
                paths.append(value)
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()  # Ordinary prose may contain an unmatched apostrophe.
        for token in tokens:
            token = token.rstrip('，。；;')
            if not token.startswith(("/", "~/", "./", "../")) and Path(token).suffix.lower() in {".csv", ".tsv", ".rds", ".txt", ".yaml", ".yml"}:
                paths.append(token)
        if paths:
            found = discover(paths)
            previous = set(self.store.get("input_registry", {}).values())
            if set(found) - previous:
                self.invalidate_draft()
            registry = self._register_files(found)
            self.store.set("active_input_refs", [ref for ref,path in registry.items() if path in found])
        return self.store.get("input_registry", {})

    def _remote_value(self, value):
        registry = self.store.get("input_registry", {})
        reverse = {path: ref for ref, path in registry.items()}
        if isinstance(value, dict):
            return {k: self._remote_value(v) for k, v in value.items() if k not in {"r_library", "rscript"}}
        if isinstance(value, list):
            return [self._remote_value(v) for v in value]
        if isinstance(value, str):
            for path, ref in sorted(reverse.items(), key=lambda item: len(item[0]), reverse=True):
                value = value.replace(path, ref)
            # Resolve relative filenames to opaque references too, when unambiguous.
            names = {}
            for ref, path in registry.items():
                names.setdefault(Path(path).name, []).append(ref)
                try:
                    relative = str(Path(path).relative_to(Path.cwd()))
                except ValueError:
                    continue
                value = value.replace(relative, ref)
            for name, refs in names.items():
                if len(refs) == 1:
                    value = value.replace(name, refs[0])
            # Match path boundaries, preserving scientific prose such as RNA/protein.
            return re.sub(r"(?<![A-Za-z0-9_])(?:/|~/|\./|\.\./)[^\s，。；,;\"']+", "[LOCAL_PATH]", value)
        return value

    def _local_patch(self, patch):
        registry = self.store.get("input_registry", {})
        payload = copy.deepcopy(patch)
        assays = payload.get("assays") or {}
        locations = [(payload, "candidates")]
        for assay in assays.values():
            if isinstance(assay, dict):
                locations += [(assay, k) for k in ["matrix", "metadata", "mapping"]]
        for owner, key in locations:
            value = owner.get(key)
            if value:
                if value in registry:
                    owner[key] = registry[value]
                elif str(Path(value).expanduser().resolve()) in registry.values():
                    owner[key] = str(Path(value).expanduser().resolve())
                else:
                    raise ValueError("Model selected an undeclared input; specify the file or directory in your message")
        # Interpreter/library selection is local configuration, never a model-provided executable.
        payload.pop("rscript", None)
        payload.pop("r_library", None)
        return payload


    def resume_summary(self):
        """A short, factual status for the Continue button; never invents progress."""
        lines = []
        if self.study:
            lines.append(f"Loaded study: {self.study.name}")
        digest = self.store.get("current_plan")
        if digest:
            lines.append(f"Current plan: {digest}")
            try:
                self.store.require_approval(digest)
                lines.append("Approval: current plan is approved.")
            except PermissionError:
                lines.append("Approval: current plan is not approved.")
        run_id = self.store.get("latest_run")
        if run_id:
            lines.append(f"Latest run: {run_id}")
        return "\n".join(lines) if lines else "Nothing to continue yet. Use New analysis or load a config with /study."


    def get_client(self):
        if self.client is not None:
            return self.client
        self.config = load_config(self.config_path)
        # Before canonicalization, read local IDs without transmitting any table.
        try:
            identifiers = private_ids(self.study) if self.study else ()
        except (ValueError, OSError):
            identifiers = ()
        return ModelClient(self.config, self.store.state_dir / "planning_calls", identifiers=identifiers)

    def current_plan(self):
        return self.store.load_plan()

    def create_plan(self, study, goal, limits=None):
        previous_hash = self.store.get("current_plan")
        previous = self.store.load_plan(previous_hash) if previous_hash else None
        changed = previous is None or object_hash(previous.study) != object_hash(study) or previous.goal != goal
        if changed:
            self.invalidate_draft()
        self.set_study(study)
        self.store.set("draft_goal", goal)
        result = run_preflight(study, self.store.state_dir, cache_dir=self.store.cache_dir)
        card = self.store.get("confirmation_card", {})
        if study.schema_version == "2.0" and card.get("study"):
            result.summary["_field_sources"] = card.get("sources", {})
        self.store.set("latest_probe", result.model_dump())
        card = self.store.get("confirmation_card", {})
        draft = StudyDraft(goal=goal, study=study.model_dump(), status=result.status, questions=result.errors,
                           sources=card.get("sources", {}), confirmations={"scale": card.get("scale_choice")} if card.get("scale_choice") else {})
        self.store.set("study_draft", draft.model_dump())
        if result.status != "ready_for_confirmation":
            self.invalidate_draft()
            return "Data preflight blocked; nothing can run:\n" + "\n".join(result.errors)
        if (previous and not changed and previous.preflight == result and
                (not limits or all(getattr(previous, k) == v for k, v in limits.items()))):
            self.store.set("draft_dirty", False)
            return describe(previous) + f"\n\nConfirm with: /confirm {previous_hash}"
        plan = make_plan(study, goal, previous, preflight=result)
        if limits:
            from .models import AnalysisPlan
            plan = AnalysisPlan.model_validate({**plan.model_dump(), **limits})
        digest = self.store.save_plan(plan)
        diff = diff_plans(previous, plan) if previous else ""
        return describe(plan) + ("\n\nPlan diff:\n" + diff if diff else "") + f"\n\nConfirm with: /confirm {digest}"

    def draft_pending(self):
        """True when a loaded StudySpec differs from the confirmed plan (or no plan exists)."""
        if self.store.get("draft_dirty", False):
            return True
        study = self.study
        if study is None:
            return False
        digest = self.store.get("current_plan")
        if not digest:
            return True
        try:
            plan = self.store.load_plan(digest)
        except ValueError:
            return True
        return object_hash(plan.study) != object_hash(study)

    @staticmethod
    def _merge_draft(old, patch):
        merged = copy.deepcopy(old)
        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = Session._merge_draft(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _understand(self, text):
        registry = self._authorize_text(text)
        client = self.get_client()
        guard = getattr(client, "guard", None)
        if guard is not None:
            guard.identifiers = sorted(set(guard.identifiers) | inventory_identifiers(registry, **self.local_toolchain()) | set(self.store.get("local_identifiers", [])), key=len, reverse=True)
        history = self.store.get("planning_conversation", [])
        draft = StudyDraft.model_validate(self.store.get("study_draft", {}))
        inventory = [{"ref": ref, "format": Path(path).suffix.lower()} for ref, path in registry.items()]
        probes = {}
        active = self.store.get("active_input_refs", list(registry))
        for ref in active[:32]:
            if ref not in registry or Path(registry[ref]).suffix.lower() in {".yaml", ".yml"}:
                continue
            try:
                probes[ref] = inspect_file(registry[ref], self.study)
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                probes[ref] = {"status": "blocked", "error_type": type(exc).__name__}
        self.store.set("file_probes", probes)
        last_probe = self.store.get("latest_probe")
        if last_probe:
            from .models import ProbeReport
            probes["preflight"] = model_summary(ProbeReport.model_validate(last_probe))
        # Probe only explicit local references. At most three model/tool exchanges per message.
        for turn in range(3):
            reply, _ = understand(client, self._remote_value(text), None, history=history[-12:],
                                  draft=self._remote_value(draft.study), inventory=inventory, probes=probes)
            if reply.goal and reply.goal.strip() != draft.goal:
                draft.goal = reply.goal.strip()
                self.invalidate_draft()
            patch = reply.study if reply.study is not None else reply.draft
            if patch is not None:
                updated = self._merge_draft(draft.study, self._local_patch(patch))
                if updated != draft.study:
                    self.invalidate_draft()
                draft.study = updated
            if reply.questions and self.store.get("current_plan"):
                self.invalidate_draft()
            draft.questions = reply.questions
            draft.status = "needs_information"
            self.store.set("study_draft", draft.model_dump())
            self.store.set("draft_goal", draft.goal)
            history += [{"user": self._remote_value(text)}, {"assistant": self._remote_value(reply.model_dump())}]
            guard = getattr(client, "guard", None)
            if guard is not None:
                history = guard.clean_value(history)
            self.store.set("planning_conversation", history[-12:])
            if not reply.probes:
                return reply, draft, patch is not None or bool(reply.goal)
            for request in reply.probes:
                if request.ref not in registry:
                    raise ValueError("Probe requested an undeclared file reference")
                try:
                    probes[request.ref] = inspect_file(registry[request.ref], self.study)
                except (ValueError, OSError) as exc:
                    probes[request.ref] = {"status": "blocked", "error_type": type(exc).__name__}
            self.store.set("file_probes", probes)
        draft.questions = ["Local probes are saved. Continue the conversation to resolve input roles and missing declarations."]
        self.store.set("study_draft", draft.model_dump())
        return reply, draft, True

    def local_toolchain(self):
        current = self.study
        import os
        library = current.r_library if current else None
        library = library or os.environ.get("SPLICEOSOME_R_LIBRARY")
        if not library and (Path.cwd() / ".r-library").is_dir():
            library = str((Path.cwd() / ".r-library").resolve())
        return {"rscript": current.rscript if current else "Rscript", "r_library": library}

    def auto_prepare(self, text, selection=None):
        from .discovery import propose_inputs
        registry = self.store.get("input_registry", {})
        active = self.store.get("active_input_refs", list(registry))
        registry = {ref: registry[ref] for ref in active if ref in registry}
        previous = self.store.get("study_draft", {})
        if previous:
            self.store.event("planning", "draft_rebased_on_current_inputs", {"previous_draft_hash": object_hash(previous)})
        self.invalidate_draft()
        self.store.set("draft_goal", text)
        proposal = propose_inputs(registry, text, selection=selection, **self.local_toolchain())
        card, identifiers = proposal if isinstance(proposal, tuple) else (proposal, set())
        self.store.set("confirmation_card", card)
        self.store.set("planning_status", card["status"])
        self.store.set("local_identifiers", sorted(identifiers))
        if card.get("study"):
            card["study"].update(self.local_toolchain())
            self.store.set("confirmation_card", card)
            self.store.set("study_draft", StudyDraft(goal=text, study=card["study"], sources=card["sources"],
                                                     questions=[]).model_dump())
            self.store.set("draft_goal", text)
            if card["status"] == "ready":
                study = resolve_study(card["study"], Path.cwd())
                self.create_plan(study, text)
        else:
            self.store.set("study_draft", StudyDraft(goal=text, status="blocked" if card["status"] == "blocked" else "needs_information",
                                                     questions=[card["message"]]).model_dump())
        self.store.event("planning", "local_input_proposal", {"status": card["status"], "n_files": len(registry)})
        return self.confirmation_text()

    def confirmation_text(self):
        card = self.store.get("confirmation_card", {})
        if not card.get("study"):
            return card.get("message", "Provide the data directory for this analysis.")
        study = card["study"]
        domain = next(iter(study["assays"]))
        assay = study["assays"][domain]
        unit = "pairs" if study["design"] == "paired" else "independent samples"
        groups = ", ".join(f"{k}={v}" for k,v in card["group_counts"].items())
        lines = ["Inputs identified. Proposed configuration:",
                 f"Matrix: {card['matrix_name']}; metadata: {card['metadata_name']}",
                 f"{card['n_features']} features × {card['n_samples']} samples; {groups}",
                 f"Design: {study['design']}, {card['n_units']} {unit}; comparison: {study['case']} − {study['reference']}",
                 f"Method: {'DESeq2' if assay.get('scale') == 'counts' else 'limma'}; full-feature BH q≤0.05, 95% CI; no extra covariate selection.",
                 "Output: full differential results, PDF/SVG volcano plots, source data and report. No candidate list or species annotation required."]
        if card["status"] == "needs_scale":
            options = "already log2 abundance / positive abundance requiring log2 / unknown"
            if domain == "rna":
                options += " / raw integer counts"
            lines += ["Confirm measurement scale: " + options,
                      "Select the scale to update the plan, or use /scale log2 or /scale raw_log2."]
        elif self.store.get("draft_dirty", False):
            probe = self.store.get("latest_probe", {})
            lines += ["Preflight blocked: " + "; ".join(probe.get("errors", [])[:2])]
        else:
            label = {"log2": "Confirmed log2; analyze directly", "raw_log2": "Confirmed positive abundance; apply log2 without a pseudocount",
                     "counts": "Confirmed raw counts"}.get(card.get("scale_choice"), "Scale explicitly declared in the input")
            lines += [label, "Review the configuration, then select Confirm and run."]
        return "\n".join(lines)

    def select_dataset(self, index):
        card = self.store.get("confirmation_card", {})
        options = card.get("alternatives", [])
        if index < 0 or index >= len(options) or not isinstance(options[index], dict):
            raise ValueError("Choose a dataset from the inspection results")
        from .io import sha256
        if any(sha256(path) != digest for path,digest in card.get("hashes", {}).items()):
            return self.auto_prepare(self.store.get("draft_goal") or "Differential analysis")
        return self.auto_prepare(self.store.get("draft_goal") or "Differential analysis", selection=options[index])

    def choose_scale(self, choice):
        card = self.store.get("confirmation_card", {})
        if not card.get("study"):
            raise ValueError("Specify the data directory first so the matrix and metadata can be identified.")
        if choice not in {"log2", "raw_log2", "unknown", "counts"}:
            raise ValueError("Choose log2, raw_log2, counts or unknown")
        # An input replacement while the card was visible invalidates its suggestions too.
        from .io import sha256
        if any(sha256(path) != checksum for path,checksum in card["hashes"].items()):
            self.auto_prepare(self.store.get("draft_goal") or card["study"]["name"])
            raise ValueError("Input changed and was inspected again. Review the updated confirmation card.")
        self.invalidate_draft()
        domain = next(iter(card["study"]["assays"]))
        assay = card["study"]["assays"][domain]
        if choice == "counts" and domain != "rna":
            raise ValueError("Protein data cannot use an RNA counts model")
        if choice == "raw_log2" and not card["positive"]:
            raise ValueError("Not all values are positive, so log2 cannot be applied. No pseudocount will be added; check the data.")
        if choice == "unknown":
            assay.pop("scale", None)
            assay.pop("transform", None)
            card["status"] = "needs_scale"
            card["scale_choice"] = "unknown"
            card["sources"]["scale"]["status"] = "needs_confirmation"
            self.store.set("confirmation_card", card)
            self.store.set("study_draft", StudyDraft(goal=card["study"]["name"], study=card["study"], sources=card["sources"]).model_dump())
            return "The scale is not confirmed. Analysis will not run.\n" + self.confirmation_text()
        assay["scale"] = "counts" if choice == "counts" else "abundance" if choice == "raw_log2" else "log2_abundance" if domain == "protein" else "log2_expression"
        assay["transform"] = "log2" if choice == "raw_log2" else "none"
        assay["preprocessing"] = "Measurement scale confirmed by the user; upstream preprocessing history was not provided" + ("; explicit log2 transformation, no pseudocount" if choice == "raw_log2" else "; no further transformation")
        card["status"], card["scale_choice"] = "ready", choice
        card["sources"]["scale"] = {"status": "user_confirmed", "value": choice}
        self.store.set("confirmation_card", card)
        self.create_plan(resolve_study(card["study"], Path.cwd()), card.get("goal") or card["study"]["name"])
        return self.confirmation_text()

    def swap_direction(self):
        card = self.store.get("confirmation_card", {})
        if not card.get("study"):
            raise ValueError("No proposed comparison is available to swap")
        study = card["study"]
        study["case"], study["reference"] = study["reference"], study["case"]
        card["sources"]["contrast"] = {"status": "user_confirmed", "case": study["case"], "reference": study["reference"]}
        self.invalidate_draft()
        self.store.set("confirmation_card", card)
        if card["status"] == "ready":
            self.create_plan(resolve_study(study, Path.cwd()), card.get("goal") or study["name"])
        else:
            self.store.set("study_draft", StudyDraft(goal=study["name"], study=study, sources=card["sources"]).model_dump())
        return self.confirmation_text()

    def converse(self, text):
        stripped = text.strip()
        if stripped in {"进度", "查看状态", "当前状态", "运行状态", "progress", "status"}:
            return self.handle("/status")
        if stripped in {"运行历史", "查看运行", "runs", "history"}:
            return self.handle("/runs")
        # A free-text confirmation can never be mistaken for authorization.
        if stripped.lower() in {"确认执行", "确认并执行", "confirm run", "approve", "run"}:
            return "Review the displayed plan and use /confirm <hash> or Confirm and run; ordinary text cannot approve or start a run."
        if self.store.get("confirmation_card", {}).get("study"):
            short = stripped.lower()
            scale_phrases = {"已是log2": "log2", "已是 log2": "log2", "已经log2": "log2", "需要log2": "raw_log2", "需要 log2": "raw_log2", "不确定": "unknown"}
            if short in scale_phrases:
                return self.choose_scale(scale_phrases[short])
        registry = self._authorize_text(text)
        explicit_input = bool(re.search(r"(?:/|\.rds|\.csv|\.tsv)", text))
        ordinary = any(word in stripped.lower() for word in ["差异", "火山", "differential", "volcano"])
        screening = any(word in stripped.lower() for word in ["候选", "candidate", "筛选剪接体"])
        if registry and explicit_input and ordinary and not screening:
            return self.auto_prepare(text)
        reply, draft, changed = self._understand(text)
        if not changed and not reply.probes:
            return self._reply_text(reply)
        try:
            study = resolve_study(draft.study, Path.cwd())
        except ValidationError as exc:
            missing = [".".join(map(str, error["loc"])) + ": " + error["type"] for error in exc.errors()]
            draft.questions = list(dict.fromkeys(draft.questions + missing))
            self.store.set("study_draft", draft.model_dump())
            self.store.set("validation_details", draft.questions)
            return "One detail needs confirmation: " + (reply.questions[0] if reply.questions else "Review the study configuration and provide the information needed for this analysis.")
        if draft.questions:
            return "One detail needs confirmation: " + draft.questions[0]
        return self.create_plan(study, draft.goal or study.name)

    def _reply_text(self, reply):
        if reply.questions:
            return "More information is needed:\n" + reply.questions[0]
        return reply.explanation or "Describe the study and specify data files or a data directory."

    def generate_plan(self):
        study = self.study
        if study is None:
            raise ValueError("No complete StudySpec yet; describe the study or load /study first")
        return self.create_plan(study, self.store.get("draft_goal") or study.name)

    def natural(self, text):
        return self.converse(text)

    def confirm_and_run(self, plan_hash):
        import fcntl
        with (self.store.state_dir / ".confirmation.lock").open("a+") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            if plan_hash != self.store.get("current_plan") or self.draft_pending():
                raise PermissionError("The displayed plan is stale; inspect the revised draft and confirm again")
            plan = self.store.load_plan(plan_hash)
            if plan.preflight is None:
                self.create_plan(plan.study, plan.goal)
                raise PermissionError("Legacy plan re-probed; review and confirm the new plan hash")
            from .planning import verify_inputs
            try:
                verify_inputs(plan)
                verify_environment(plan.preflight, plan.study)
            except (ValueError, OSError) as exc:
                self.invalidate_draft()
                self.create_plan(plan.study, plan.goal)
                raise PermissionError("Inputs/environment changed; a refreshed plan needs confirmation") from exc
            key = "confirmed_run:" + plan_hash
            run_id = self.store.get(key)
            if run_id:
                engine = Engine(self.store, run_id, self.config)
                state = read_json(engine.run / "run_state.json")
                if state["status"] != "start_failed":
                    return f"Run already recorded: {run_id} ({state['status']}); use /resume explicitly if interrupted."
                self.store.require_approval(plan_hash)
            else:
                if not self.store.is_approved(plan_hash):
                    self.store.approve(plan_hash)
                engine = Engine.create(self.store, plan_hash, load_config(self.config_path))
                self.store.set(key, engine.run_id)
            try:
                engine.set_status("ready", warnings=[])
                job = jobs.start(self.store, engine.run_id, self.config_path)
            except Exception as exc:
                engine.set_status("start_failed", warnings=[f"Worker startup failed: {type(exc).__name__}: {exc}"])
                raise
            return f"Run started: {engine.run_id}\nOutput: {engine.run}\nBackground process: {job['identity']}"

    def handle(self, text):
        text = text.strip()
        if not text:
            return "Enter a goal or /help."
        if not text.startswith("/"):
            return self.converse(text)
        args = shlex.split(text)
        command, args = args[0], args[1:]
        if command == "/help":
            return HELP
        if command == "/dataset":
            if len(args) != 1:
                raise ValueError("Usage: /dataset <1-based number>")
            return self.select_dataset(int(args[0])-1)
        if command == "/scale":
            if len(args) != 1:
                raise ValueError("Usage: /scale log2|raw_log2|counts|unknown")
            return self.choose_scale(args[0])
        if command == "/swap":
            return self.swap_direction()
        if command == "/inputs":
            registry = self.store.get("input_registry", {})
            return "\n".join(f"{ref}: {path}" for ref, path in registry.items()) or "No declared inputs yet."
        if command == "/demo":
            import uuid
            from .examples import create_example
            target = self.store.state_dir / "examples" / ("synthetic_" + uuid.uuid4().hex[:8])
            path = create_example(target)
            self.set_study(load_study(path))
            return f"Synthetic example created at {target}\n" + self.create_plan(self.study, self.study.name)
        if command in {"/study", "/edit"}:
            self.store.set("confirmation_card", {})
            study = load_study(args[0])
            if command == "/edit":
                return self.create_plan(study, self.current_plan().goal)
            self.set_study(study)
            return self.create_plan(study, self.store.get("draft_goal") or study.name)
        if command == "/plan":
            if not args:
                digest = self.store.get("current_plan")
                if digest:
                    return describe(self.store.load_plan(digest))
                if self.study is not None:
                    return (f"No plan yet. Draft StudySpec loaded: {self.study.name}. "
                            "Continue describing the study or use /plan <goal>.")
                return "No plan yet. Describe the study, load /study, or create the synthetic demo with /demo."
            if self.study is None:
                raise ValueError("Load /study first, or complete the study description in plain language")
            return self.create_plan(self.study, " ".join(args))
        if command == "/plans":
            if not args:
                return plan_lines(self.store)
            digest = self.store.resolve_plan(args[0])
            plan = self.store.load_plan(digest)
            status = "approved" if self.store.is_approved(digest) else "not approved"
            if self.store.get("current_plan") == digest:
                status += " (current)"
            return f"Approval: {status}\n\n" + describe(plan)
        if command == "/confirm":
            if len(args) != 1:
                raise ValueError("Usage: /confirm <full plan hash>")
            return self.confirm_and_run(args[0])
        if command == "/approve":
            record = self.store.approve(args[0])
            return f"Approved {record.plan_hash}. Enter /run to execute."
        if command == "/cancel":
            digest = self.store.get("current_plan")
            if digest:
                self.store.cancel(digest)
            return "Approval revoked; no new job was started."
        if command == "/run":
            engine = Engine.create(self.store, self.store.get("current_plan"), self.config)
            job = jobs.start(self.store, engine.run_id, self.config_path)
            return f"Run started: {engine.run_id}\nOutput: {engine.run}\nBackground process: {job['identity']}"
        if command == "/resume":
            job = jobs.start(self.store, args[0], self.config_path, resume=True)
            return f"Resume started: {job['run_id']}"
        if command == "/stop":
            return json.dumps(jobs.stop(self.store, args[0]), ensure_ascii=False, indent=2)
        if command in {"/runs", "/tasks"}:
            rows = jobs.list_runs(self.store)
            if not rows:
                return "No runs yet. Approve a plan, then use /run."
            lines = []
            for row in rows:
                goal = ""
                plan_path = self.store.run_dir(row["run_id"]) / "plan.json"
                if plan_path.is_file():
                    goal = str(read_json(plan_path).get("goal", ""))[:70]
                lines.append(f"{row['run_id']}  {row['status']}  {row.get('updated_at', '')}  {goal}")
            return "\n".join(lines)
        if command in {"/report", "/logs"}:
            run_id = args[0] if args else self.store.get("latest_run")
            if not run_id:
                return "No runs yet. Approve a plan, then use /run."
            folder = self.store.run_dir(run_id)
            if command == "/report":
                path = folder / "report.md"
                if not path.is_file():
                    return f"No report for {run_id} yet; it is written when a run finishes."
                return path.read_text(encoding="utf-8", errors="replace")
            path = folder / "worker.log"
            if not path.is_file():
                return f"No worker log for {run_id} yet."
            count = 40
            if len(args) > 1:
                if not args[1].isdigit() or int(args[1]) < 1:
                    raise ValueError("/logs [run-id] [n]: n must be a positive integer")
                count = int(args[1])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-count:]) if lines else "The worker log is empty."
        if command == "/status":
            run_id = args[0] if args else self.store.get("latest_run")
            state = read_json(self.store.run_dir(run_id or "") / "run_state.json")
            return json.dumps(state, ensure_ascii=False, indent=2)
        if command == "/doctor":
            from .doctor import check
            return json.dumps(check(self.study), ensure_ascii=False, indent=2)
        if command == "/models":
            return load_config(self.config_path).model_dump_json(indent=2)
        if command == "/quit":
            return "__quit__"
        raise ValueError("Unknown command; use /help")
