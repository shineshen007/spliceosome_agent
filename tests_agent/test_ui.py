import json
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Select, Static, TextArea

from spliceosome_agent import credentials
from spliceosome_agent.app import (
    BLOCK_FONT_COMPACT, BLOCK_FONT_LARGE, ModelScreen, SpliceosomeApp,
    StudyEditorScreen, WelcomeBanner, block_banner,
)
from spliceosome_agent.config import ModelConfig, ModelProfile, load_config, save_config
from spliceosome_agent.session import Session


class FakeKeyring:
    """In-memory keyring so tests never touch the real OS keychain."""

    priority = 5

    def __init__(self):
        self.store = {}

    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, value):
        self.store[(service, account)] = value

    def delete_password(self, service, account):
        self.store.pop((service, account), None)


@pytest.fixture
def keyring():
    backend = FakeKeyring()
    credentials.set_backend(backend)
    yield backend
    credentials.set_backend(None)


def local_config(path, name="local"):
    save_config(ModelConfig(default=name, profiles={
        name: ModelProfile(name=name, base_url="http://127.0.0.1:11434/v1", model="fixture", kind="local", api_key_env=None)}), path)
    return path


def remote_config(path, name="cloud"):
    save_config(ModelConfig(default=name, profiles={
        name: ModelProfile(name=name, base_url="https://api.openai.com/v1", model="fixture", kind="remote", api_key_env="OPENAI_API_KEY")}), path)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (80, 24), (120, 40)])
async def test_minimal_home_and_query_at_all_sizes(tmp_path, keyring, size):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=size) as pilot:
        await pilot.pause(.1)
        assert app.query_one("#welcome-scroll").display
        assert not app.query_one("#workspace-scroll").display
        assert "█" in str(app.query_one("#welcome-banner", Static).render())
        assert app.query_one("#chat-input", Input).placeholder == "Describe your goal and data location"
        assert not app.query_one("#primary-actions").display
        for selector in ["#history-panel", "#job-log", "#chat-log", "#sources-panel", "#details-panel",
                         "#send-prompt", "#home-examples", "#phase-bar", "#status-bar", "#prompt-meta"]:
            assert not app.query(selector)
        app.query_one("#chat-input", Input).value = "/help"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert not app.query_one("#welcome-scroll").display
        assert "/study" in str(app.query_one("#current-content", Static).render())
        app._tick()
        assert "/study" in str(app.query_one("#current-content", Static).render())
        assert app.query_one("#chat-input", Input).region.bottom <= size[1]
        await pilot.press("escape")
        assert app.query_one("#welcome-scroll").display
        assert not app.query_one("#workspace-scroll").display


@pytest.mark.asyncio
async def test_first_run_requires_model_setup(tmp_path, keyring):
    config = tmp_path / "models.json"
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.6)
        assert isinstance(app.screen, ModelScreen)
        assert app.screen.required is True
        assert not app.screen.query("#model-back")  # no way to skip
        # Escape must not dismiss the mandatory dialog.
        await pilot.press("escape")
        await pilot.pause(.3)
        assert isinstance(app.screen, ModelScreen)
        # A remote profile without a key is rejected until a key is supplied.
        app.screen.query_one("#profile-name", Input).value = "cloud"
        app.screen.query_one("#model-id", Input).value = "fixture-remote"
        app.screen.query_one("#key-env", Input).value = ""
        app.screen.query_one("#model-save", Button).press()
        await pilot.pause(.3)
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#key-value", Input).value = "sk-fixture-secret"
        app.screen.query_one("#model-save", Button).press()
        await pilot.pause(.8)
        assert not isinstance(app.screen, ModelScreen)
        stored = load_config(config)
        assert stored.default == "cloud" and stored.profiles["cloud"].kind == "remote"
        assert "sk-fixture-secret" not in config.read_text()
        assert credentials.get_api_key("cloud") == "sk-fixture-secret"
    # A second launch must not prompt again.
    app2 = SpliceosomeApp(Session(tmp_path / "workspace2", config))
    async with app2.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.6)
        assert not isinstance(app2.screen, ModelScreen)


@pytest.mark.asyncio
async def test_local_model_needs_no_key_and_persists(tmp_path, keyring):
    config = tmp_path / "models.json"
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.6)
        assert isinstance(app.screen, ModelScreen)
        app.screen.query_one("#preset", Select).value = "ollama"
        await pilot.pause(.2)
        app.screen.query_one("#profile-name", Input).value = "ollama-local"
        app.screen.query_one("#model-id", Input).value = "fixture-local"
        app.screen.query_one("#model-save", Button).press()
        await pilot.pause(.8)
        assert not isinstance(app.screen, ModelScreen)
        stored = load_config(config)
        assert stored.profiles[stored.default].kind == "local"
        assert stored.profiles[stored.default].api_key_env is None
        assert "key-value" not in config.read_text()


@pytest.mark.asyncio
async def test_model_screen_preserves_saved_endpoint(tmp_path, keyring):
    """A saved custom base URL must not be overwritten by the preset defaults on open."""
    config = remote_config(tmp_path / "models.json")
    save_config(ModelConfig(default="default", profiles={"default": ModelProfile(
        name="default", base_url="http://127.0.0.1:8787", model="fixture", kind="remote", api_key_env="OPENAI_API_KEY")}), config)
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.7)
        assert isinstance(app.screen, ModelScreen)
        assert app.screen.query_one("#base-url", Input).value == "http://127.0.0.1:8787"
        assert app.screen.query_one("#model-kind", Select).value == "remote"
        assert app.screen.query_one("#key-env", Input).value == "OPENAI_API_KEY"
        assert app.screen.query_one("#preset", Select).value == "custom"
        # Only the key is supplied; the endpoint must stay untouched on save.
        app.screen.query_one("#key-value", Input).value = "sk-keep-endpoint"
        app.screen.query_one("#model-save", Button).press()
        await pilot.pause(.8)
        stored = load_config(config)
        assert stored.profiles["default"].base_url == "http://127.0.0.1:8787"


@pytest.mark.asyncio
async def test_model_screen_preserves_local_profile(tmp_path, keyring):
    config = tmp_path / "models.json"
    save_config(ModelConfig(default="ollama", profiles={"ollama": ModelProfile(
        name="ollama", base_url="http://127.0.0.1:11434/v1", model="fixture", kind="local", api_key_env=None)}), config)
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.5)
        assert not isinstance(app.screen, ModelScreen)  # a local profile is ready
        worker = app.run_worker(app.open_models(), exclusive=False)
        await pilot.pause(.5)
        assert isinstance(app.screen, ModelScreen)
        assert app.screen.query_one("#preset", Select).value == "ollama"
        assert app.screen.query_one("#base-url", Input).value == "http://127.0.0.1:11434/v1"
        assert app.screen.query_one("#model-kind", Select).value == "local"
        assert app.screen.query_one("#key-env", Input).value == ""
        app.screen.query_one("#model-back", Button).press()
        await worker.wait()


@pytest.mark.asyncio
async def test_preset_selection_updates_endpoint(tmp_path, keyring):
    config = tmp_path / "models.json"
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.6)
        screen = app.screen
        assert isinstance(screen, ModelScreen)
        screen.query_one("#preset", Select).value = "ollama"
        await pilot.pause(.3)
        assert screen.query_one("#base-url", Input).value == "http://127.0.0.1:11434/v1"
        assert screen.query_one("#model-kind", Select).value == "local"
        assert screen.query_one("#key-env", Input).value == ""
        # A hand-edited endpoint labels the preset as custom without losing the typed value.
        screen.query_one("#base-url", Input).value = "http://127.0.0.1:9999/v1"
        await pilot.pause(.3)
        assert screen.query_one("#preset", Select).value == "custom"
        assert screen.query_one("#base-url", Input).value == "http://127.0.0.1:9999/v1"


@pytest.mark.asyncio
async def test_models_command_opens_panel(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause(.3)
        assert not isinstance(app.screen, ModelScreen)
        app.query_one("#chat-input", Input).value = "/models"
        await pilot.press("enter")
        await pilot.pause(.5)
        assert isinstance(app.screen, ModelScreen)
        assert app.screen.required is False
        app.screen.query_one("#model-back", Button).press()
        await pilot.pause(.3)
        assert not isinstance(app.screen, ModelScreen)


@pytest.mark.asyncio
async def test_model_modal_keeps_key_out_of_config(tmp_path, keyring):
    config = local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", config))
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause(.4)
        assert not isinstance(app.screen, ModelScreen)  # already configured
        worker = app.run_worker(app.open_models(), exclusive=False)
        await pilot.pause(.3)
        assert isinstance(app.screen, ModelScreen)
        assert app.screen.required is False
        app.screen.query_one("#preset", Select).value = "ollama"
        await pilot.pause(.2)
        app.screen.query_one("#model-id", Input).value = "fixture-local"
        app.screen.query_one("#model-save", Button).press()
        await worker.wait()
        stored = load_config(config)
        assert stored.profiles[stored.default].kind == "local"
        assert stored.profiles[stored.default].api_key_env is None
        assert "key-value" not in config.read_text()


@pytest.mark.asyncio
async def test_study_editor_creates_new_version(tmp_path, keyring, study):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.create_plan(study, "SYNTHETIC UI test")
    old = session.store.get("current_plan")
    session.store.approve(old)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(140, 55)) as pilot:
        await app._handle_async("/current")
        await pilot.pause(.3)
        # Explicitly open the retained plan; startup itself remains on the welcome page.
        assert app.query_one("#workspace-scroll").display
        assert not app.query("#run")
        worker = app.run_worker(app.open_study_editor(), exclusive=False)
        await pilot.pause(.4)
        assert isinstance(app.screen, StudyEditorScreen)
        changed = study.model_dump()
        changed["alpha"] = .01
        app.screen.query_one("#study-editor", TextArea).load_text(json.dumps(changed))
        app.screen.query_one("#save-study", Button).press()
        await worker.wait()
        await pilot.pause(.2)
        digest = session.store.get("current_plan")
        assert digest != old
        with pytest.raises(PermissionError):
            session.store.require_approval(old)
        assert app.query_one("#approve", Button).disabled is False
        assert not app.query("#run")


@pytest.mark.asyncio
async def test_one_confirmation_starts_exact_displayed_plan(tmp_path, keyring, study, monkeypatch):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.create_plan(study, "SYNTHETIC UI test")
    digest = session.store.get("current_plan")
    calls = []
    def confirm(value):
        calls.append(value)
        session.store.approve(value)
        return "Run started: simulated UI worker"
    monkeypatch.setattr(session, "confirm_and_run", confirm)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(120, 44)) as pilot:
        await app._handle_async("/current")
        await pilot.pause(.3)
        assert not app.query("#run") and not app.query("#build-plan")
        assert str(app.query_one("#approve", Button).label) == "Confirm and run"
        assert not session.store.is_approved(digest)
        await pilot.click("#approve")
        await app.workers.wait_for_complete()
        assert calls == [digest]
        assert session.store.is_approved(digest)
        assert not app.query("#action-buttons")  # no second confirmation step


@pytest.mark.asyncio
async def test_controls_hidden_without_ready_plan(tmp_path, keyring, study):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    app = SpliceosomeApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        assert not app.query_one("#approve", Button).display
        assert not app.query_one("#stop", Button).display
        for selector in ["#cancel", "#revise", "#resume", "#swap-direction"]:
            assert not app.query(selector)
        session.create_plan(study, "SYNTHETIC UI test")
        app.refresh_controls()
        assert app.query_one("#welcome-scroll").display
        assert not app.query_one("#approve", Button).display
        await app._handle_async("/current")
        assert app.query_one("#approve", Button).display
        assert not app.query_one("#approve", Button).disabled
        session.invalidate_draft()
        app.refresh_controls()
        assert not app.query_one("#approve", Button).display


@pytest.mark.asyncio
async def test_study_load_automatically_probes_and_presents_plan(tmp_path, keyring, study):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    app = SpliceosomeApp(session)
    async with app.run_test(size=(120, 50)) as pilot:
        worker = app.run_worker(app._handle_async(f'/study "{Path(study.candidates).parent / "study.yaml"}"'))
        await worker.wait()
        await pilot.pause(.2)
        assert session.store.get("current_plan") is not None
        assert session.current_plan().preflight.status == "ready_for_confirmation"
        assert not app.query_one("#approve", Button).disabled
        assert not list(session.store.runs_dir.glob("*/run_state.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (80, 24), (120, 40)])
async def test_workspace_actions_present_at_small_terminal(tmp_path, keyring, study, monkeypatch, size):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.create_plan(study, "SYNTHETIC UI test")
    calls = []
    monkeypatch.setattr(session, "confirm_and_run", lambda digest: calls.append(digest) or "Simulated startup")
    app = SpliceosomeApp(session)
    async with app.run_test(size=size) as pilot:
        await app._handle_async("/current")
        await pilot.pause(.1)
        action = app.query_one("#approve", Button)
        assert action.region.height > 0
        assert action.region.bottom <= app.query_one("#chat-input").region.y
        assert app.query_one("#chat-input").region.bottom <= size[1]
        await pilot.click("#approve")
        await app.workers.wait_for_complete()
        assert calls == [session.store.get("current_plan")]


@pytest.mark.asyncio
async def test_help_toggle(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause(.3)
        await pilot.press("f1")
        await pilot.pause(.3)
        assert app._view == "response"
        assert "/study" in str(app.query_one("#current-content", Static).render())
        await pilot.press("escape")
        await pilot.pause(.3)
        assert app._view == "home"


def test_demo_command_loads_synthetic_study(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    reply = session.handle("/demo")
    assert "synthetic" in reply.lower()
    assert session.study is not None and session.study.synthetic is True

@pytest.mark.asyncio
async def test_compact_scale_card_updates_plan_without_a_questionnaire(tmp_path, keyring, study, monkeypatch):
    import subprocess
    from spliceosome_agent.examples import create_example
    local_config(tmp_path / "models.json")
    folder = tmp_path / "protein_rds"
    folder.mkdir()
    subprocess.run(["Rscript", "--vanilla", "-e",
        "args<-commandArgs(TRUE);x<-read.csv(args[1],row.names=1,check.names=FALSE);m<-read.csv(args[2]);saveRDS(x,args[3]);saveRDS(m,args[4])",
        study.assays["protein"].matrix,study.assays["protein"].metadata,
        str(folder/"protein_expr.rds"),str(folder/"protein_info.rds")],check=True)
    session=Session(tmp_path/"workspace",tmp_path/"models.json")
    session._register_files([str(folder/"protein_expr.rds"),str(folder/"protein_info.rds")])
    session.auto_prepare("蛋白差异分析并画火山图")
    calls=[]
    monkeypatch.setattr(session,"confirm_and_run",lambda digest:calls.append(digest) or "Simulated UI startup")
    app=SpliceosomeApp(session)
    async with app.run_test(size=(60,24)) as pilot:
        await app._handle_async("/current")
        await pilot.pause(.4)
        assert app.query_one("#scale-controls").display
        assert app.query_one("#scale-choice",Select).region.height>0
        assert not app.query_one("#approve",Button).display
        assert "missing" not in str(app.query_one("#current-content",Static).render())
        app.query_one("#scale-choice",Select).value="log2"
        await pilot.pause(.1)
        await app.workers.wait_for_complete()
        await pilot.pause(.3)
        assert session.current_plan().study.analysis_mode=="differential"
        assert not app.query_one("#approve",Button).disabled
        assert not app.query_one("#scale-controls").display
        assert app.query_one("#scale-choice",Select).value=="log2"
        await pilot.click("#approve")
        await app.workers.wait_for_complete()
        assert calls==[session.store.get("current_plan")]


@pytest.fixture
def ui_session(tmp_path, study, monkeypatch, keyring):
    # State tests need a real plan and structural R checks, not repeated dependency probes.
    monkeypatch.setattr("spliceosome_agent.preflight.check_environment", lambda study: ({"fixture": "stable"}, []))
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.create_plan(study, "合成界面状态验证")
    return session


def write_ui_run(session, status, **extra):
    from spliceosome_agent.io import write_json
    run_id = "20261020T100000_12345678"
    folder = session.store.run_dir(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "plan.json", session.current_plan())
    write_json(folder / "run_state.json", {"run_id": run_id, "status": status, "results": {}, "warnings": [], **extra})
    session.store.set("latest_run", run_id)
    return run_id, folder


@pytest.mark.asyncio
async def test_task_lifecycle_warnings_and_restart(ui_session):
    session = ui_session
    run_id, folder = write_ui_run(session, "running")
    session.store.event(run_id, "role_started", {"role": "RNA"})
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        await app._handle_async("/current")
        assert "Running · RNA" in str(app.query_one("#current-content", Static).render())
        assert app.query_one("#stop").display
        assert not app.query_one("#approve").display
        for status, label in [("complete", "Completed"), ("partial", "Partially completed"), ("failed", "Run failed"),
                              ("stopped", "Stopped"), ("budget_exhausted", "Budget exhausted"),
                              ("waiting_plan_revision", "Plan revision required")]:
            write_ui_run(session, status, warnings=["不可忽略的合成警告"],
                         results={"RNA": {"errors": ["具体失败原因"], "warnings": ["结果限制"]}})
            app._tick()
            body = str(app.query_one("#current-content", Static).render())
            assert label in body and "不可忽略的合成警告" in body and "具体失败原因" in body
            assert "结果限制" in body and str(folder) in body
            assert not app.query_one("#stop").display
            assert not app.query_one("#approve").display
        (folder / "report.md").write_text("合成结果")
        write_ui_run(session, "complete")
    restored = Session(session.store.workspace, session.config_path)
    app2 = SpliceosomeApp(restored)
    async with app2.run_test(size=(80, 24)):
        assert app2.query_one("#welcome-scroll").display
        assert not app2.query_one("#primary-actions").display
        await app2._handle_async("/current")
        assert "Completed" in str(app2.query_one("#current-content", Static).render())
        assert str(folder / "report.md") in str(app2.query_one("#current-content", Static).render())
        assert not app2.query_one("#approve").display


@pytest.mark.asyncio
async def test_explicit_queries_and_errors_survive_polling(ui_session, monkeypatch):
    session = ui_session
    run_id, folder = write_ui_run(session, "running")
    (folder / "worker.log").write_text("SYNTHETIC_LOG_LINE\n")
    app = SpliceosomeApp(session)
    async with app.run_test(size=(80, 24)) as pilot:
        for command, expected in [("/logs", "SYNTHETIC_LOG_LINE"), ("/plan", "Inputs:"),
                                  ("/runs", run_id), ("/help", "Help")]:
            await app._handle_async(command)
            body = str(app.query_one("#current-content", Static).render())
            assert expected in body
            app._tick()
            await pilot.pause(.05)
            assert str(app.query_one("#current-content", Static).render()) == body
            assert not app.query_one("#approve").display
        await app._handle_async("/unknown_command")
        error = str(app.query_one("#current-content", Static).render())
        assert "Action failed" in error
        write_ui_run(session, "complete")
        app._tick()
        assert str(app.query_one("#current-content", Static).render()) == error
        await pilot.press("escape")
        assert app.query_one("#welcome-scroll").display
        await app._handle_async("/current")
        assert "Completed" in str(app.query_one("#current-content", Static).render())


@pytest.mark.asyncio
async def test_stop_uses_current_run_once_without_another_confirmation(ui_session, monkeypatch):
    session = ui_session
    run_id, folder = write_ui_run(session, "running")
    calls = []
    def stop(store, value):
        calls.append(value)
        write_ui_run(session, "stopped")
        return {"status": "stopped", "run_id": value}
    monkeypatch.setattr("spliceosome_agent.jobs.stop", stop)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        await app._handle_async("/current")
        await pilot.pause(.1)
        await pilot.click("#stop")
        await app.workers.wait_for_complete()
        assert calls == [run_id]
        assert "Stopped" in str(app.query_one("#current-content", Static).render())
        assert not app.query_one("#stop").display
        assert not app.query("#action-buttons")


@pytest.mark.asyncio
async def test_hidden_history_cannot_replace_current_plan(ui_session):
    session = ui_session
    write_ui_run(session, "complete")
    changed = session.study.model_copy(deep=True)
    changed.alpha = .01
    session.create_plan(changed, "新的合成计划")
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)):
        await app._handle_async("/current")
        assert app.query_one("#approve").display
        assert "Completed" not in str(app.query_one("#current-content", Static).render())
        assert "q≤0.01" in str(app.query_one("#current-content", Static).render())
        assert session.store.list_plans()  # persisted history remains available


@pytest.mark.asyncio
async def test_queued_confirmation_rejects_replaced_plan(ui_session, monkeypatch):
    session = ui_session
    calls = []
    monkeypatch.setattr(session, "confirm_and_run", lambda digest: calls.append(digest))
    app = SpliceosomeApp(session)
    async with app.run_test(size=(80, 24)):
        await app._handle_async("/current")
        old = app._displayed_digest
        changed = session.study.model_copy(deep=True)
        changed.alpha = .01
        session.create_plan(changed, "新计划")
        await app.confirm_approve(old)
        assert calls == []
        assert not app._busy
        assert "The plan changed" in str(app.query_one("#current-content", Static).render())


@pytest.mark.asyncio
async def test_long_content_scrolls_without_moving_input(ui_session):
    session = ui_session
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        text = "\n".join(f"第{i}条：" + "/long_path/" * 18 for i in range(70))
        app._message(text)
        await pilot.pause(.1)
        scroll = app.query_one("#workspace-scroll")
        assert scroll.max_scroll_y > 0
        assert scroll.max_scroll_x == 0
        assert str(app.query_one("#current-content", Static).render()) == text
        entry = app.query_one("#chat-input", Input)
        position = entry.region
        scroll.scroll_end(animate=False)
        await pilot.pause(.1)
        app._tick()
        assert entry.region == position
        assert app.focused is entry


@pytest.mark.asyncio
async def test_retry_and_corrupt_state_do_not_silently_approve(ui_session):
    session = ui_session
    _, folder = write_ui_run(session, "start_failed", warnings=["启动进程失败"])
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)):
        await app._handle_async("/current")
        assert app.query_one("#approve").display
        assert "启动进程失败" in str(app.query_one("#current-content", Static).render())
        (folder / "run_state.json").write_text("{broken")
        app._tick()
        assert not app.query_one("#approve").display
        assert "Could not read run state" in str(app.query_one("#current-content", Static).render())


@pytest.mark.asyncio
async def test_dataset_choice_only_appears_for_multiple_candidates(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    option = {"matrix": "m1", "metadata": "md1", "group_column": "group", "sample_column": "sample_id"}
    session.store.set("confirmation_card", {"message": "选择本次输入", "alternatives": [option]})
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)):
        await app._handle_async("/current")
        assert not app.query_one("#dataset-controls").display
        session.store.set("confirmation_card", {"message": "选择本次输入", "alternatives": [option, {**option, "matrix": "m2"}]})
        app._tick()
        assert app.query_one("#dataset-controls").display
        assert "选择本次输入" in str(app.query_one("#current-content", Static).render())


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (80, 24), (120, 40)])
async def test_english_interface_and_modals_preserve_user_text(ui_session, size):
    import re
    from textual.widgets import Label

    def assert_english(screen):
        for widget in screen.query("Static, Label, Button"):
            text = str(widget.label) if isinstance(widget, Button) else str(widget.render())
            assert not re.search(r"[\u4e00-\u9fff]", text), text
        for entry in screen.query(Input):
            assert not re.search(r"[\u4e00-\u9fff]", entry.placeholder)

    app = SpliceosomeApp(ui_session)
    async with app.run_test(size=size) as pilot:
        assert_english(app.screen)
        await app._handle_async("/help")
        assert_english(app.screen)
        await pilot.press("escape")
        worker = app.run_worker(app.open_models())
        await pilot.pause(.1)
        assert isinstance(app.screen, ModelScreen)
        assert_english(app.screen)
        assert app.screen.query_one("#model-save").region.bottom <= size[1]
        app.screen.dismiss(False)
        await worker.wait()
        worker = app.run_worker(app.open_study_editor())
        await pilot.pause(.1)
        assert isinstance(app.screen, StudyEditorScreen)
        assert_english(app.screen)
        app.screen.dismiss(None)
        await worker.wait()
        original = "保留用户标签和路径：/data/患者样本.csv"
        app._message(original)
        app._tick()
        assert str(app.query_one("#current-content", Static).render()) == original


@pytest.mark.parametrize("compact,width,height", [(False, 65, 15), (True, 43, 11)])
def test_block_banner_spells_two_words_without_wrapping(compact, width, height):
    font = BLOCK_FONT_COMPACT if compact else BLOCK_FONT_LARGE
    rows = block_banner(compact).splitlines()
    assert len(rows) == height
    assert all(len(row) == width for row in rows)
    assert set("".join(rows)) == {"█", " "}
    glyph_width = 3 if compact else 5
    glyph_height = 5 if compact else 7
    assert not rows[glyph_height].strip()
    decode = {tuple(row.translate(str.maketrans("01", " █")) for row in glyph): letter
              for letter, glyph in font.items()}
    assert len(decode) == len(font)
    for start, word in [(0, "SPLICEOSOME"), (glyph_height + 1, "AGENT")]:
        word_width = len(word) * (glyph_width + 1) - 1
        offset = (width - word_width) // 2
        decoded = ""
        for column in range(offset, offset + word_width, glyph_width + 1):
            glyph = tuple(row[column:column + glyph_width] for row in rows[start:start + glyph_height])
            decoded += decode[glyph]
        assert decoded == word


@pytest.mark.asyncio
@pytest.mark.parametrize("size,compact", [((60, 24), True), ((80, 24), False), ((120, 40), False)])
async def test_welcome_banner_centers_and_keeps_input_fixed(tmp_path, keyring, size, compact):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=size) as pilot:
        await pilot.pause(.2)
        welcome = app.query_one("#welcome-scroll", WelcomeBanner)
        banner = app.query_one("#welcome-banner", Static)
        entry = app.query_one("#chat-input", Input)
        viewport = welcome.scrollable_content_region
        assert welcome.compact is compact
        assert banner.region.width == (43 if compact else 65)
        assert banner.region.height == (11 if compact else 15)
        assert abs((banner.region.x * 2 + banner.region.width) - (viewport.x * 2 + viewport.width)) <= 1
        assert abs((banner.region.y * 2 + banner.region.height) - (viewport.y * 2 + viewport.height)) <= 1
        assert welcome.max_scroll_x == 0 and welcome.max_scroll_y == 0
        assert banner.region.bottom <= entry.region.y
        assert entry.region.bottom <= size[1]
        assert app.focused is entry


@pytest.mark.asyncio
async def test_banner_resize_and_extreme_viewport_remain_scrollable(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.2)
        welcome = app.query_one("#welcome-scroll", WelcomeBanner)
        banner = app.query_one("#welcome-banner", Static)
        entry = app.query_one("#chat-input", Input)
        for size in [(60, 24), (80, 24), (80, 16), (32, 12), (120, 40)]:
            await pilot.resize_terminal(*size)
            await pilot.pause(.2)
            assert app.focused is entry
            assert entry.region.bottom <= size[1]
            assert banner.region.width in {43, 65}
            assert banner.region.height in {11, 15}
            if size == (32, 12):
                assert welcome.max_scroll_x > 0 and welcome.max_scroll_y > 0
                assert banner.styles.margin.left == 0 and banner.styles.margin.top == 0
                content = str(banner.render())
                welcome.scroll_to(welcome.max_scroll_x, welcome.max_scroll_y, animate=False)
                await pilot.pause(.1)
                assert welcome.scroll_x > 0 and welcome.scroll_y > 0
                assert str(banner.render()) == content
                assert app.focused is entry
                welcome.scroll_home(animate=False)
                welcome.focus()
                await pilot.press("right", "down")
                await pilot.pause(.3)
                assert welcome.scroll_x > 0 and welcome.scroll_y > 0
                entry.focus()
            else:
                assert welcome.max_scroll_x == 0 and welcome.max_scroll_y == 0


@pytest.mark.asyncio
async def test_banner_does_not_replace_task_reply_or_active_work(ui_session, monkeypatch):
    app = SpliceosomeApp(ui_session)
    async with app.run_test(size=(80, 24)):
        await app._handle_async("/current")
        assert not app.query_one("#welcome-scroll").display
        app._message("Spliceosome Agent")
        assert not app.query_one("#welcome-scroll").display
        assert str(app.query_one("#current-content", Static).render()) == "Spliceosome Agent"
        app.action_close_help()
        assert not app.query_one("#welcome-scroll").display
        write_ui_run(ui_session, "running")
        app._tick()
        assert not app.query_one("#welcome-scroll").display


@pytest.mark.asyncio
async def test_banner_hides_while_empty_session_is_busy(tmp_path, keyring):
    local_config(tmp_path / "models.json")
    app = SpliceosomeApp(Session(tmp_path / "workspace", tmp_path / "models.json"))
    async with app.run_test(size=(80, 24)):
        assert app.query_one("#welcome-scroll").display
        app._set_busy(True)
        assert not app.query_one("#welcome-scroll").display
        app._set_busy(False)
        assert app.query_one("#welcome-scroll").display
        app._message("A question from the user")
        app._tick()
        assert not app.query_one("#welcome-scroll").display


def store_snapshot(session):
    """Compare logical records, independent of SQLite WAL/checkpoint layout."""
    with session.store.connect() as db:
        return {table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("kv", "plans", "approvals", "events")}


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", ["empty", "draft", "plan", "approved", "running", "complete", "partial", "failed", "stopped", "corrupt"])
async def test_every_launch_and_poll_stays_home(ui_session, monkeypatch, saved):
    session = ui_session
    if saved in {"empty", "draft"}:
        with session.store.connect() as db:
            db.execute("DELETE FROM kv")
        if saved == "draft":
            session.store.set("study_draft", {"questions": ["Confirm the scale."]})
    elif saved not in {"plan", "approved"}:
        _, folder = write_ui_run(session, "running" if saved == "corrupt" else saved)
        if saved == "corrupt":
            (folder / "run_state.json").write_text("{broken")
    elif saved == "approved":
        session.store.approve(session.store.get("current_plan"))
    before = store_snapshot(session)
    def unexpected(*args, **kwargs):
        pytest.fail("Home must not load, approve, start or stop the retained task")
    monkeypatch.setattr(session, "current_plan", unexpected)
    monkeypatch.setattr(session, "draft_pending", unexpected)
    monkeypatch.setattr(session, "confirm_and_run", unexpected)
    monkeypatch.setattr("spliceosome_agent.jobs.start", unexpected)
    monkeypatch.setattr("spliceosome_agent.jobs.stop", unexpected)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause(1.1)  # Include an actual scheduled polling interval.
        app.action_refresh()
        await pilot.press("escape")
        assert app._view == "home"
        assert app.query_one("#welcome-scroll").display
        assert not app.query_one("#workspace-scroll").display
        assert app._displayed_digest is None
        for selector in ("#primary-actions", "#approve", "#stop", "#scale-controls", "#dataset-controls"):
            assert not app.query_one(selector).display
        assert store_snapshot(session) == before


@pytest.mark.asyncio
async def test_navigation_queries_preserve_origin_records_and_approval(ui_session, monkeypatch):
    from spliceosome_agent.app import COMMAND_LIST, TUI_HELP
    session = ui_session
    run_id, folder = write_ui_run(session, "running")
    (folder / "report.md").write_text("Synthetic report")
    (folder / "worker.log").write_text("Synthetic log")
    digest = session.store.get("current_plan")
    session.store.approve(digest)
    before = store_snapshot(session)
    def unexpected(*args, **kwargs):
        pytest.fail("Read-only navigation must not start or stop workers")
    monkeypatch.setattr("spliceosome_agent.jobs.start", unexpected)
    monkeypatch.setattr("spliceosome_agent.jobs.stop", unexpected)
    assert all(command in COMMAND_LIST and command in TUI_HELP for command in ("/home", "/current"))
    app = SpliceosomeApp(session)
    async with app.run_test(size=(80, 24)) as pilot:
        for origin in ("home", "task"):
            await app._handle_async("/home" if origin == "home" else "/current")
            assert app._view == origin
            for command in ("/tasks", "/runs", "/plans", f"/status {run_id}",
                            f"/report {run_id}", f"/logs {run_id}", "/plan", "/help"):
                await app._handle_async(command)
                assert app._view == "response"
                assert not app._response.startswith("Action failed"), app._response
                body = app._response
                app._tick()
                assert str(app.query_one("#current-content", Static).render()) == body
                assert not app.query_one("#primary-actions").display
                assert not app.query_one("#stop").display
                assert app._displayed_digest is None
            await pilot.press("escape")
            assert app._view == origin
            assert app.query_one("#welcome-scroll").display == (origin == "home")
            assert app.query_one("#stop").display == (origin == "task")
            await pilot.press("f1", "f1")
            assert app._view == origin
            assert store_snapshot(session) == before
        # Home also resets the return destination after a task-origin query.
        await app._handle_async("/help")
        await app._handle_async("/home")
        await app._handle_async("/tasks")
        await pilot.press("escape")
        assert app._view == "home"
        assert store_snapshot(session) == before


@pytest.mark.asyncio
async def test_current_empty_and_corrupt_are_read_only(ui_session):
    session = ui_session
    _, folder = write_ui_run(session, "running")
    (folder / "run_state.json").write_text("{broken")
    before = store_snapshot(session)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        await app._handle_async("/current")
        assert "Could not read run state" in str(app.query_one("#current-content", Static).render())
        assert not app.query_one("#primary-actions").display
        assert store_snapshot(session) == before
        await app._handle_async("/home")
        app._tick()
        assert app.query_one("#welcome-scroll").display
        with session.store.connect() as db:
            db.execute("DELETE FROM kv")
        before = store_snapshot(session)
        await app._handle_async("/current")
        assert app._response == "No current task. Use /tasks to view saved runs."
        await pilot.press("escape")
        assert app._view == "home"
        assert store_snapshot(session) == before


@pytest.mark.asyncio
async def test_model_setup_with_retained_task_returns_home(ui_session):
    session = ui_session
    session.config_path.write_text("{}")
    before = store_snapshot(session)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(.2)
        assert isinstance(app.screen, ModelScreen) and app.screen.required
        app.screen.query_one("#preset", Select).value = "ollama"
        await pilot.pause(.1)
        app.screen.query_one("#model-id", Input).value = "fixture-local"
        app.screen.query_one("#model-save", Button).press()
        await pilot.pause(.2)
        assert not isinstance(app.screen, ModelScreen)
        assert app._view == "home" and app.query_one("#welcome-scroll").display
        assert store_snapshot(session) == before
        worker = app.run_worker(app.open_models())
        await pilot.pause(.1)
        app.screen.query_one("#model-save", Button).press()
        await worker.wait()
        assert app._view == "home"
        assert store_snapshot(session) == before


@pytest.mark.asyncio
async def test_home_does_not_interrupt_a_live_background_process(ui_session):
    import subprocess
    import sys
    from spliceosome_agent.io import write_json
    from spliceosome_agent.jobs import identity, same_process
    session = ui_session
    run_id, folder = write_ui_run(session, "running")
    # Harmless isolated process, never a research worker or model request.
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                               start_new_session=True)
    job = {"run_id": run_id, "identity": identity(process.pid)}
    write_json(folder / "job.json", job)
    before = store_snapshot(session)
    try:
        app = SpliceosomeApp(session)
        async with app.run_test(size=(60, 24)) as pilot:
            await app._handle_async("/current")
            assert app.query_one("#stop").display
            await app._handle_async("/home")
            await pilot.pause(1.1)
            assert process.poll() is None and same_process(job["identity"])
            assert json.loads((folder / "job.json").read_text()) == job
            assert store_snapshot(session) == before
            assert app._view == "home" and not app.query_one("#stop").display
            write_ui_run(session, "complete")
            app._tick()
            assert app.query_one("#welcome-scroll").display
            await app._handle_async("/current")
            assert "Completed" in str(app.query_one("#current-content", Static).render())
    finally:
        process.stdin.close()
        process.wait(timeout=5)


@pytest.mark.asyncio
async def test_history_queries_do_not_select_an_old_task(ui_session):
    session = ui_session
    old_id, folder = write_ui_run(session, "complete")
    (folder / "report.md").write_text("Old synthetic report")
    (folder / "worker.log").write_text("Old synthetic log")
    revised = session.study.model_copy(deep=True)
    revised.alpha = .01
    session.create_plan(revised, "Revised synthetic task")
    before = store_snapshot(session)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(60, 24)) as pilot:
        await app._handle_async("/current")
        digest = app._displayed_digest
        assert digest == session.store.get("current_plan")
        for command in (f"/status {old_id}", f"/report {old_id}", f"/logs {old_id}", "/plans"):
            await app._handle_async(command)
            assert app._view == "response" and not app.query_one("#primary-actions").display
            await pilot.press("escape")
            assert app._view == "task" and app._displayed_digest == digest
            assert "q≤0.01" in str(app.query_one("#current-content", Static).render())
        assert store_snapshot(session) == before


@pytest.mark.asyncio
async def test_request_from_home_uses_retained_context(ui_session, monkeypatch):
    session = ui_session
    digest = session.store.get("current_plan")
    calls = []
    session.store.set("confirmation_card", {"message": "Review the retained task."})
    def handle(text):
        calls.append((text, session.store.get("current_plan"), session.study.name))
        return session.confirmation_text()
    monkeypatch.setattr(session, "handle", handle)
    app = SpliceosomeApp(session)
    async with app.run_test(size=(80, 24)):
        await app._handle_async("Continue the analysis")
        assert calls == [("Continue the analysis", digest, session.study.name)]
        assert app._view == "task" and not app.query_one("#welcome-scroll").display
        assert not session.store.is_approved(digest)


def test_home_current_are_tui_only(ui_session):
    for command in ("/home", "/current"):
        with pytest.raises(ValueError, match="Unknown command"):
            ui_session.handle(command)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (80, 24), (120, 40)])
async def test_complete_help_keyboard_scroll_and_origin(ui_session, size):
    from spliceosome_agent.app import TUI_HELP
    app = SpliceosomeApp(ui_session)
    async with app.run_test(size=size) as pilot:
        for origin in ("home", "task"):
            await app._handle_async("/home" if origin == "home" else "/current")
            await pilot.press("f1")
            await pilot.pause(.1)
            scroll = app.query_one("#workspace-scroll")
            entry = app.query_one("#chat-input", Input)
            assert app._response == TUI_HELP
            assert scroll.scroll_y == 0
            assert "/tasks" in TUI_HELP.splitlines()[2]
            # /tasks is actually within the first visible content rows.
            assert scroll.region.y <= app.query_one("#current-content").region.y + 2 < scroll.region.bottom
            assert not app.query_one("#primary-actions").display
            position = entry.region
            entry.value = "Keep this draft input"
            for _ in range(8):
                await pilot.press("pagedown")
            await pilot.pause(.1)
            assert scroll.scroll_y == scroll.max_scroll_y
            if size[1] == 24:
                assert scroll.scroll_y > 0
            offset = scroll.scroll_y
            app._tick()
            await pilot.pause(.1)
            assert app._response == TUI_HELP and scroll.scroll_y == offset
            assert entry.region == position and position.bottom <= size[1]
            assert app.focused is entry and entry.value == "Keep this draft input"
            for _ in range(8):
                await pilot.press("pageup")
            assert scroll.scroll_y == 0
            # Reopening help also starts at the command list after previous scrolling.
            await pilot.press("pagedown")
            await app._handle_async("/help")
            await pilot.pause(.1)
            assert scroll.scroll_y == 0
            await pilot.press("escape")
            assert app._view == origin
            assert app.focused is entry and entry.value == "Keep this draft input"
