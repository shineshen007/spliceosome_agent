"""Behavioral checks for pre-approval probes, confirmation and frozen entrypoints."""
import json
from pathlib import Path

import pandas as pd
import pytest

from spliceosome_agent.engine import Engine
from spliceosome_agent.io import object_hash, read_json, sha256, write_json
from spliceosome_agent.models import AnalysisPlan, AgentDecision, PlanningReply, ApprovalRecord
from spliceosome_agent.planning import make_plan
from spliceosome_agent.preflight import discover, inspect_file, run_preflight
from spliceosome_agent.runner import execute_run
from spliceosome_agent.session import Session
from spliceosome_agent.store import Store


@pytest.fixture
def quick_environment(monkeypatch):
    # Only skip redundant package loading; actual R structural design validation still runs.
    monkeypatch.setattr("spliceosome_agent.preflight.check_environment", lambda study: ({"fixture": "stable"}, []))
    monkeypatch.setattr("spliceosome_agent.engine.r_fingerprint", lambda study: {"fixture": "stable"})


class Planner:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.contexts = []

    def json(self, role, system, payload, contract):
        self.contexts.append(payload)
        return PlanningReply.model_validate(next(self.replies))


def test_planning_uses_english_while_preserving_user_input():
    from spliceosome_agent.data import resource
    from spliceosome_agent.planning import understand

    class EnglishPlanner:
        def json(self, role, system, payload, contract):
            assert "explanations in English, regardless of input language" in system
            assert payload["request"] == "比较病例与对照，保留分组名"
            return contract(explanation="Provide the data directory for this comparison.")

    reply, _ = understand(EnglishPlanner(), "比较病例与对照，保留分组名")
    assert reply.explanation == "Provide the data directory for this comparison."
    assert "conclusions briefly in English" in resource("shared.md").read_text()


def loaded(tmp_path, study):
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.set_study(study)
    return session


def test_discover_is_bounded_to_explicit_roots(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "one.csv").write_text("id,A\ng,1\n")
    (root / "two.tsv").write_text("id\tB\ng\t2\n")
    (root / ".hidden").mkdir()
    (root / ".hidden/secret.csv").write_text("secret")
    external = tmp_path / "external.csv"
    external.write_text("secret")
    (root / "escape.csv").symlink_to(external)
    assert discover([root]) == [str(root / "one.csv"), str(root / "two.tsv")]
    assert discover([root / "escape.csv"]) == []
    summary = inspect_file(root / "one.csv")
    assert summary["n_rows"] == 1 and summary["n_columns"] == 2
    assert "A" not in summary["declared_columns"]


def test_partial_draft_persists_and_never_runs(tmp_path):
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=Planner([
        {"goal": "two-group comparison", "draft": {"species": "human"}, "questions": ["请提供文件与尺度"]}]))
    reply = session.converse("比较两组RNA")
    assert "One detail needs confirmation" in reply
    assert "missing" not in reply
    restored = Session(session.store.workspace, session.config_path)
    assert restored.store.get("study_draft")["study"] == {"species": "human"}
    assert restored.store.get("current_plan") is None
    assert not list(restored.store.workspace.rglob("run_analysis.py"))


def test_model_cannot_authorize_paths_or_interpreters(tmp_path, study):
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=Planner([
        {"study": study.model_dump()}]))
    with pytest.raises(ValueError, match="undeclared input"):
        session.converse("I have data")
    session.set_study(study)
    patch = session._local_patch({"rscript": "/untrusted/program", "r_library": "/untrusted/lib"})
    assert patch == {}


def test_directory_probes_have_only_local_refs_and_safe_summaries(tmp_path, study):
    client = Planner([{"probes": [{"ref": "file_0000"} ]}, {"questions": ["Which matrix is RNA and what is its scale?"]}])
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    reply = session.converse(f'Inspect "{Path(study.candidates).parent}"')
    assert "More information" in reply
    payload = json.dumps(client.contexts)
    assert str(tmp_path) not in payload
    assert "SYNTH_CONTROL_00" not in payload and "SYNTH_SUBJECT_00" not in payload
    assert len(client.contexts[0]["inventory"]) >= 4
    assert client.contexts[1]["probe_results"]
    assert session.store.get("current_plan") is None


def test_ready_plan_deduplicates_and_no_code_before_approval(tmp_path, study, quick_environment):
    session = loaded(tmp_path, study)
    before = {p: sha256(p) for p in make_plan(study, "x").input_hashes}
    session.create_plan(study, "synthetic test")
    digest = session.store.get("current_plan")
    session.create_plan(study, "synthetic test")
    assert session.store.get("current_plan") == digest
    assert len(session.store.list_plans()) == 1
    plan = session.current_plan()
    assert plan.schema_version == "1.1" and plan.preflight.summary["rna"]["design"]["residual_df"] > 0
    assert not list(session.store.workspace.rglob("run_analysis.py"))
    assert not list(session.store.runs_dir.glob("*/run_state.json"))
    assert before == {p: sha256(p) for p in before}


@pytest.mark.parametrize("problem", ["missing_file", "empty_file", "duplicate_id", "bad_pair", "sample_mismatch", "fractional_counts", "namespace", "species", "rank"])
def test_preflight_blocks_invalid_data(tmp_path, study, quick_environment, problem):
    path = Path(study.assays["rna"].metadata)
    md = pd.read_csv(path)
    if problem == "missing_file":
        study.assays["rna"].matrix = str(tmp_path / "absent.csv")
    elif problem == "empty_file":
        Path(study.assays["rna"].matrix).write_text("")
    elif problem == "duplicate_id":
        md.loc[1, "sample_id"] = md.loc[0, "sample_id"]
    elif problem == "bad_pair":
        md.loc[0, "subject_id"] = "not_a_pair"
    elif problem == "sample_mismatch":
        md.loc[0, "sample_id"] = "not_in_matrix"
    elif problem == "fractional_counts":
        matrix = pd.read_csv(study.assays["rna"].matrix)
        matrix[matrix.columns[1]] = matrix[matrix.columns[1]].astype(float)
        matrix.iloc[0, 1] = 0.5
        matrix.to_csv(study.assays["rna"].matrix, index=False)
    elif problem == "namespace":
        study.assays["rna"].id_type = "ENSEMBL"
    elif problem == "species":
        study.species = "different"
    elif problem == "rank":
        md["confounded"] = md.group
        study.covariates = {"confounded": "categorical"}
    md.to_csv(path, index=False)
    session = loaded(tmp_path, study)
    reply = session.create_plan(study, "blocked test")
    assert "blocked" in reply
    assert session.store.get("current_plan") is None
    assert session.store.get("study_draft")["status"] == "blocked"
    assert not list(session.store.workspace.rglob("run_analysis.py"))


def test_missing_scientific_declarations_cannot_become_a_plan(tmp_path):
    session = Session(tmp_path / "w", tmp_path / "m.json", client=Planner([
        {"draft": {"name": "test", "species": "human", "design": "paired"}, "goal": "test"}]))
    assert "One detail needs confirmation" in session.converse("test")
    assert session.store.get("current_plan") is None


def test_missing_dependencies_and_font_are_blockers(monkeypatch, study):
    from spliceosome_agent.preflight import check_environment
    monkeypatch.setattr("spliceosome_agent.doctor.check", lambda s: {"rscript": None, "r_packages": {}, "fonts": {}})
    study.font = "NotInstalledFixtureFont"
    study.allow_font_fallback = False
    env, errors = check_environment(study)
    assert any("Rscript" in error for error in errors)
    assert any("DESeq2" in error for error in errors)
    assert any("font" in error for error in errors)


def test_confirm_is_idempotent_start_failure_retries_same_run(tmp_path, study, quick_environment, monkeypatch):
    session = loaded(tmp_path, study)
    session.create_plan(study, "confirm test")
    digest = session.store.get("current_plan")
    started = []
    def start(store, run_id, config_path):
        started.append(run_id)
        if len(started) == 1:
            raise OSError("fixture startup failure")
        return {"identity": {"pid": 0}}
    monkeypatch.setattr("spliceosome_agent.jobs.start", start)
    with pytest.raises(OSError):
        session.confirm_and_run(digest)
    run_id = session.store.get("confirmed_run:" + digest)
    assert read_json(session.store.run_dir(run_id) / "run_state.json")["status"] == "start_failed"
    approval = session.store.require_approval(digest)
    assert approval.execution_scope == "registered_tools_with_automatic_code"
    assert (session.store.run_dir(run_id) / "run_analysis.py").is_file()
    assert "Run started" in session.confirm_and_run(digest)
    # Real jobs.start moves state to ready. Reflect that without launching a real worker here.
    Engine(session.store, run_id).set_status("ready")
    assert "already recorded" in session.confirm_and_run(digest)
    assert started == [run_id, run_id]
    restored = Session(session.store.workspace, session.config_path)
    assert len(started) == 2 and restored.store.get("confirmed_run:" + digest) == run_id


def test_confirm_reprobes_input_drift_and_needs_new_confirmation(tmp_path, study, quick_environment, monkeypatch):
    session = loaded(tmp_path, study)
    session.create_plan(study, "drift test")
    old = session.store.get("current_plan")
    path = Path(study.candidates)
    path.write_text(path.read_text() + "\n")
    monkeypatch.setattr("spliceosome_agent.jobs.start", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(PermissionError, match="refreshed plan"):
        session.confirm_and_run(old)
    assert session.store.get("current_plan") != old
    assert not session.store.is_approved(session.store.get("current_plan"))


def test_environment_drift_requires_new_plan(tmp_path, study, quick_environment, monkeypatch):
    session = loaded(tmp_path, study)
    session.create_plan(study, "environment test")
    old = session.store.get("current_plan")
    monkeypatch.setattr("spliceosome_agent.preflight.check_environment", lambda s: ({"fixture": "changed"}, []))
    with pytest.raises(PermissionError, match="refreshed plan"):
        session.confirm_and_run(old)
    assert session.store.get("current_plan") != old


def test_partial_or_goal_revision_revokes_approval_immediately(tmp_path, study, quick_environment):
    session = loaded(tmp_path, study)
    session.create_plan(study, "original")
    old = session.store.get("current_plan")
    session.store.approve(old)
    session.client = Planner([{"goal": "different purpose", "draft": {"species": None}, "questions": ["Specify species"]}])
    session.converse("Change purpose and species")
    assert session.draft_pending()
    with pytest.raises(PermissionError):
        session.store.require_approval(old)
    with pytest.raises(PermissionError):
        session.confirm_and_run(old)


def test_runner_is_the_executed_entry_and_tamper_blocks_resume(tmp_path, study, quick_environment, monkeypatch):
    from spliceosome_agent.runner import verify
    session = loaded(tmp_path, study)
    session.create_plan(study, "entrypoint test")
    digest = session.store.get("current_plan")
    session.store.approve(digest)
    engine = Engine.create(session.store, digest)
    called = []
    monkeypatch.setattr(engine, "execute", lambda resume=False: called.append(resume) or {"status": "complete"})
    assert execute_run(engine)["status"] == "complete"
    assert called == [False]
    path = engine.run / "run_analysis.py"
    path.write_text(path.read_text() + "\nraise RuntimeError('tampered')\n")
    # Even replacing the manifest SHA cannot authorize a different template.
    record = read_json(engine.run / "runner.json")
    record["source_sha256"] = sha256(path)
    write_json(engine.run / "runner.json", record)
    with pytest.raises(ValueError, match="runner changed"):
        execute_run(engine, resume=True)
    assert called == [False]


def test_legacy_plan_hash_and_approval_roundtrip_without_new_fields(tmp_path, study):
    plan = make_plan(study, "legacy")
    payload = plan.model_dump(mode="json")
    assert payload["schema_version"] == "1.0" and "preflight" not in payload
    # Simulate the exact old canonical representation, independently of the model serializer.
    digest = object_hash(payload)
    store = Store(tmp_path / "legacy")
    with store.connect() as db:
        db.execute("INSERT INTO plans VALUES(?,?,?)", (digest, json.dumps(payload), "2026-10-19"))
    store.set("current_plan", digest)
    assert object_hash(store.load_plan()) == digest
    old = ApprovalRecord(plan_hash=digest, plan_version=1, approved_at="2026-10-19")
    with store.connect() as db:
        db.execute("INSERT INTO approvals VALUES(?,?)", (digest, old.model_dump_json()))
    assert store.require_approval(digest) == old
    assert "preflight" not in json.loads(store.load_plan().model_dump_json())


def test_full_conversation_merges_partial_fields_and_resolves_file_refs(tmp_path, study, quick_environment):
    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    # Toolchain is local operator configuration, while the scientific draft is initially incomplete.
    session.store.set("study_draft", {"study": {"r_library": study.r_library}, "goal": "", "status": "needs_information", "questions": []})
    registry = session._register_files(make_plan(study, "x").input_hashes)
    reverse = {path: ref for ref, path in registry.items()}
    spec = study.model_dump()
    for assay in spec["assays"].values():
        for key in ["matrix", "metadata"]:
            assay[key] = reverse[assay[key]]
    spec["candidates"] = reverse[spec["candidates"]]
    spec.pop("r_library")
    species = spec.pop("species")
    session.client = Planner([
        {"goal": "synthetic comparison", "draft": {"species": species}, "questions": ["请声明设计和尺度"]},
        {"draft": spec},
    ])
    assert "One detail needs confirmation" in session.converse("These files are synthetic; compare two groups")
    assert session.store.get("current_plan") is None
    assert "Confirm with" in session.converse("Use the declared paired design and counts/log2 scales")
    plan = session.current_plan()
    assert plan.study.assays["rna"].matrix == study.assays["rna"].matrix
    assert plan.study.species == species
    assert plan.preflight.status == "ready_for_confirmation"
    assert not session.store.is_approved(object_hash(plan))


def test_confirmation_double_click_concurrently_creates_one_run(tmp_path, study, quick_environment, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    session = loaded(tmp_path, study)
    session.create_plan(study, "double click")
    digest = session.store.get("current_plan")
    calls = []
    monkeypatch.setattr("spliceosome_agent.jobs.start", lambda store, rid, config: calls.append(rid) or {"identity": None})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(session.confirm_and_run, [digest, digest]))
    assert len(calls) == 1
    assert len(list(session.store.runs_dir.glob("*/run_state.json"))) == 1
    assert sum("already recorded" in reply for reply in results) == 1


def test_planning_requests_and_persistent_history_redact_local_ids_and_secrets(tmp_path, study, monkeypatch):
    import httpx
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    secret = "fixture-secret-for-privacy"
    monkeypatch.setenv("FIXTURE_PLANNING_KEY", secret)
    payloads = []
    def handle(request):
        payloads.append(request.content.decode())
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"explanation": "What is your purpose?"})}}],
                                         "usage": {"total_tokens": 10}})
    config = ModelConfig(default="fixture", profiles={"fixture": ModelProfile(
        name="fixture", model="fixture", base_url="https://fixture.invalid/v1", api_key_env="FIXTURE_PLANNING_KEY")})
    client = ModelClient(config, tmp_path / "audit", transport=httpx.MockTransport(handle))
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    session._register_files([study.assays["rna"].metadata])
    session.converse(f"Question for SYNTH_SUBJECT_00. {secret}")
    serialized = "\n".join(payloads) + json.dumps(session.store.get("planning_conversation"))
    for path in (tmp_path / "audit").glob("*.json"):
        serialized += path.read_text()
    assert "SYNTH_SUBJECT_00" not in serialized
    assert secret not in serialized
    assert "[SUBJECT]" in serialized and "[SECRET]" in serialized


def test_prose_slashes_and_relative_filename_references_are_preserved_safely(tmp_path, study, monkeypatch):
    session = loaded(tmp_path, study)
    assert session._remote_value("RNA/protein comparison with counts/log2") == "RNA/protein comparison with counts/log2"
    assert "/private-data" not in session._remote_value("数据在/private-data/patients")
    assert "rna.csv" not in session._remote_value("Use rna.csv with protein.csv")
    monkeypatch.chdir(Path(study.candidates).parent)
    session._authorize_text("Use rna.csv")
    session.client = Planner([{"explanation": "Please describe the purpose."}])
    assert "purpose" in session.converse("Don't alter the existing thresholds")
