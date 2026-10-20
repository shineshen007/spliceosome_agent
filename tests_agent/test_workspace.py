"""Visible output layout and legacy-workspace compatibility, with synthetic inputs."""
import concurrent.futures
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from spliceosome_agent.cli import main, parser
from spliceosome_agent.engine import Engine, verify_run
from spliceosome_agent.io import artifact, read_json, sha256, write_json
from spliceosome_agent.models import ToolResult
from spliceosome_agent.planning import make_plan
from spliceosome_agent.runner import execute_run, source_for, verify
from spliceosome_agent.session import Session
from spliceosome_agent.store import Store
from spliceosome_agent import jobs


@pytest.fixture(params=[False, True], ids=["flat", "legacy"])
def workspace(tmp_path, request):
    root = tmp_path / "custom-output"
    if request.param:
        (root / "runs").mkdir(parents=True)
    return Store(root)


def test_new_workspace_default_and_reopen(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert parser().parse_args([]).workspace == tmp_path / "result"
    assert main(["plans"]) == 0
    assert "No stored plans" in capsys.readouterr().out
    root = tmp_path / "result"
    assert (root / "state/workspace.json").is_file()
    assert (root / "state/sessions.sqlite3").is_file()
    assert (root / "state/cache").is_dir()
    assert not (root / "runs").exists()
    assert not (tmp_path / ".spliceosome-agent").exists()
    store = Store(root)
    assert store.flat and store.runs_dir == root
    store.set("sentinel", "retained")
    assert Store(root).get("sentinel") == "retained"
    explicit = tmp_path / "explicit"
    assert main(["-w", str(explicit), "plans"]) == 0
    assert (explicit / "state/workspace.json").is_file()
    assert not (explicit / "result").exists()


def test_layout_initialization_is_shared_across_concurrent_launches(tmp_path):
    root = tmp_path / "result"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        stores = list(pool.map(lambda _: Store(root), range(4)))
    assert all(s.flat and s.path == root / "state/sessions.sqlite3" for s in stores)
    assert not list(root.glob("state-init-*"))


@pytest.mark.parametrize("record", [{}, [], {"version": 2, "layout": "flat"},
                                    {"version": True, "layout": "flat"}, "{broken"])
def test_invalid_marker_is_not_repaired(tmp_path, record):
    marker = tmp_path / "result/state/workspace.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(record if isinstance(record, str) else json.dumps(record))
    before = marker.read_bytes()
    with pytest.raises(ValueError, match="workspace layout marker"):
        Store(marker.parent.parent)
    assert marker.read_bytes() == before
    assert not (marker.parent / "sessions.sqlite3").exists()


@pytest.mark.parametrize("legacy", ["runs", "plans", "sessions.sqlite3"])
def test_mixed_layout_does_not_overwrite_either_side(tmp_path, legacy):
    root = tmp_path / "result"
    store = Store(root)
    store.set("sentinel", "keep")
    target = root / legacy
    if legacy.endswith("sqlite3"):
        target.write_bytes(b"original database")
    else:
        target.mkdir()
    before = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Conflicting workspace layouts"):
        Store(root)
    assert before == {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}


def test_unknown_state_and_symlink_are_not_overwritten(tmp_path):
    root = tmp_path / "unknown"
    (root / "state").mkdir(parents=True)
    (root / "state/keep.txt").write_text("original")
    with pytest.raises(ValueError, match="Unrecognized state"):
        Store(root)
    assert (root / "state/keep.txt").read_text() == "original"
    other = tmp_path / "other"
    other.mkdir()
    (other / "state").symlink_to(root / "state", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        Store(other)


def test_paths_plans_and_standalone_runner(workspace, study, monkeypatch):
    plan = make_plan(study, "Synthetic layout acceptance")
    digest = workspace.save_plan(plan)
    workspace.approve(digest)
    assert (workspace.state_dir / "plans" / f"{digest}.json").is_file()
    engine = Engine.create(workspace, digest)
    assert engine.run.parent == workspace.runs_dir
    assert engine.run == workspace.run_dir(engine.run_id)
    verify(engine)
    if workspace.flat:
        assert engine.run.parent == workspace.workspace
        assert not (workspace.workspace / "runs").exists()
    calls = []
    monkeypatch.setattr("spliceosome_agent.cli.main", lambda args: calls.append(args) or 0)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(engine.run / "run_analysis.py"), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [["--workspace", str(workspace.workspace), "resume", engine.run_id, "--foreground"]]
    if not workspace.flat:
        assert engine.run.joinpath("run_analysis.py").read_text() == source_for(plan)
    with pytest.raises(ValueError):
        workspace.run_dir("../escape")
    if workspace.flat:
        for reserved in ("state", "runs", "plans", "sessions.sqlite3", "RUNS", "Plans"):
            with pytest.raises(ValueError, match="reserved"):
                workspace.run_dir(reserved)


def test_history_and_cli_queries_use_selected_layout(workspace, study, capsys):
    plan = make_plan(study, "Synthetic history acceptance")
    digest = workspace.save_plan(plan)
    workspace.approve(digest)
    engine = Engine.create(workspace, digest)
    engine.set_status("complete")
    (engine.run / "report.md").write_text("SYNTHETIC REPORT")
    (engine.run / "worker.log").write_text("SYNTHETIC LOG")
    frozen = {str(p): sha256(p) for p in engine.run.rglob("*") if p.is_file()}
    session = Session(workspace.workspace)
    for command, expected in [("/tasks", engine.run_id), ("/runs", engine.run_id),
                              ("/plans", digest[:12]), (f"/status {engine.run_id}", "complete"),
                              (f"/report {engine.run_id}", "SYNTHETIC REPORT"),
                              (f"/logs {engine.run_id}", "SYNTHETIC LOG")]:
        assert expected in session.handle(command)
    for command, extra, expected in [("runs", [], engine.run_id), ("status", [], "complete"),
                                      ("report", [engine.run_id], "SYNTHETIC REPORT"),
                                      ("logs", [engine.run_id], "SYNTHETIC LOG")]:
        assert main(["-w", str(workspace.workspace), command, *extra]) == 0
        assert expected in capsys.readouterr().out
    assert session.store.get("current_plan") == digest
    assert session.store.is_approved(digest)
    assert frozen == {str(p): sha256(p) for p in engine.run.rglob("*") if p.is_file()}
    if not workspace.flat:
        assert not (workspace.workspace / "state").exists()


def test_detached_start_stop_resume_preserves_workspace_root(workspace, study, monkeypatch):
    monkeypatch.setattr("spliceosome_agent.engine.r_fingerprint", lambda study: {"fixture": "stable"})
    plan = make_plan(study, "Synthetic process identity acceptance")
    digest = workspace.save_plan(plan)
    workspace.approve(digest)
    engine = Engine.create(workspace, digest)
    real_popen = subprocess.Popen
    processes, commands = [], []
    def harmless_process(argv, **kwargs):
        commands.append(argv)
        # Exercise actual detached PID/create-time identity and stop signaling,
        # substituting an idle stdin reader for research computation.
        kwargs["stdin"] = subprocess.PIPE
        process = real_popen([sys.executable, "-c", "import sys; sys.stdin.read()"], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(jobs.subprocess, "Popen", harmless_process)
    try:
        for resume in (False, True):
            record = jobs.start(Store(workspace.workspace), engine.run_id, resume=resume)
            assert jobs.same_process(record["identity"])
            assert commands[-1][commands[-1].index("--workspace") + 1] == str(workspace.workspace)
            assert ("--resume" in commands[-1]) == resume
            assert jobs.list_runs(workspace)[0]["worker_alive"]
            stopped = jobs.stop(workspace, engine.run_id)
            assert stopped["status"] == "stopped"
            processes[-1].wait(timeout=5)
            engine.set_status("stopped", warnings=["Synthetic idle process stopped; no analysis performed"])
        assert Store(workspace.workspace).get("latest_run") == engine.run_id
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdin.close()


def test_synthetic_execution_resume_and_manifest(workspace, study, monkeypatch):
    # Real orchestration/checkpoints/runner, deterministic text tools only.
    monkeypatch.setattr("spliceosome_agent.engine.r_fingerprint", lambda study: {"fixture": "stable"})
    def cache_only(plan, run):
        (run / "plot-cache").mkdir(exist_ok=True)
        (run / "plot-cache/fonts.json").write_text("disposable cache")
        return {"fixture": True}
    monkeypatch.setattr("spliceosome_agent.report.figures", cache_only)
    plan = make_plan(study, "Synthetic orchestration layout acceptance")
    digest = workspace.save_plan(plan)
    workspace.approve(digest)
    def tool(task, plan, run, results):
        path = run / task.role.lower() / "synthetic.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SOFTWARE FIXTURE ONLY")
        return ToolResult(role=task.role, status="complete", artifacts=[artifact(run, path)], summary={})
    engine = Engine.create(workspace, digest, tool_executor=tool)
    assert execute_run(engine)["status"] == "complete"
    assert (engine.run / "checkpoints.sqlite3").is_file()
    (engine.run / "plot-cache/fonts.json").write_text("cache may change")
    assert verify_run(workspace, engine.run_id)["status"] == "verified"
    assert execute_run(engine, resume=True)["status"] == "complete"
    assert verify_run(workspace, engine.run_id)["status"] == "verified"


def test_preflight_confirmation_and_audit_paths(workspace, study, monkeypatch, tmp_path):
    from spliceosome_agent.config import ModelConfig, save_config
    monkeypatch.setattr("spliceosome_agent.preflight.check_environment", lambda study: ({"fixture": "stable"}, []))
    monkeypatch.setattr("spliceosome_agent.engine.r_fingerprint", lambda study: {"fixture": "stable"})
    config = tmp_path / "models.json"
    save_config(ModelConfig(), config)
    session = Session(workspace.workspace, config)
    session.create_plan(study, "Synthetic confirmation layout acceptance")
    digest = session.store.get("current_plan")
    assert not session.store.is_approved(digest)
    assert (workspace.state_dir / "probes").is_dir()
    assert session.get_client().audit_dir == workspace.state_dir / "planning_calls"
    calls = []
    monkeypatch.setattr(jobs, "start", lambda store, run_id, config_path: calls.append(store.run_dir(run_id)) or {"identity": {"pid": 0}})
    assert "Run started" in session.confirm_and_run(digest)
    assert "already recorded" in session.confirm_and_run(digest)
    assert len(calls) == 1 and calls[0].parent == workspace.runs_dir
    assert workspace.state_dir.joinpath(".confirmation.lock").is_file()
    assert Session(workspace.workspace, config).store.get("current_plan") == digest
    if workspace.flat:
        assert all(not p.name.startswith(".") for p in workspace.workspace.rglob("*") if p.is_dir())


def test_cli_worker_and_foreground_resume_routing(workspace, study, monkeypatch, capsys):
    plan = make_plan(study, "Synthetic CLI dispatch acceptance")
    digest = workspace.save_plan(plan)
    workspace.approve(digest)
    engine = Engine.create(workspace, digest)
    calls = []
    def no_research_execute(current, resume=False):
        current._validate()
        calls.append((current.run, resume))
        return current.set_status("complete", warnings=["SYNTHETIC DISPATCH ONLY"])
    monkeypatch.setattr("spliceosome_agent.runner.execute_run", no_research_execute)
    monkeypatch.setattr("spliceosome_agent.cli.signal.signal", lambda *args: None)
    for command, extra in [("_worker", []), ("_worker", ["--resume"]), ("resume", ["--foreground"])]:
        assert main(["-w", str(workspace.workspace), command, engine.run_id, *extra]) == 0
        assert "SYNTHETIC DISPATCH ONLY" in capsys.readouterr().out
    assert calls == [(engine.run, False), (engine.run, True), (engine.run, True)]
    assert read_json(engine.run / "job.json")["exit_code"] == 0
