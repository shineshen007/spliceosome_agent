from pathlib import Path

import pytest

from spliceosome_agent.engine import Engine
from spliceosome_agent.io import artifact, read_json
from spliceosome_agent.llm import BudgetExceeded, PrivacyGuard
from spliceosome_agent.models import AgentDecision, ToolResult
from spliceosome_agent.generated_code import checked_proposal, execute_code, propose_code


class Client:
    def __init__(self, behavior=None):
        self.guard = PrivacyGuard()
        self.calls = []
        self.behavior = behavior

    def available(self, role="Orchestrator"):
        return True

    def json(self, role, system, payload, contract):
        self.calls.append(role)
        if self.behavior:
            return self.behavior(role, self.calls)
        return AgentDecision()


def fake_tool(task, plan, run, results):
    folder = {"QC": "inputs", "Knowledge": "knowledge", "RNA": "rna", "Protein": "protein", "Integration": "integration", "Critic": "critic"}[task.role]
    path = run / folder / "fixture.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("deterministic synthetic fixture")
    return ToolResult(role=task.role, status="complete", artifacts=[artifact(run, path)], summary={})


@pytest.fixture(autouse=True)
def no_expensive_probes(monkeypatch):
    monkeypatch.setattr("spliceosome_agent.engine.r_fingerprint", lambda study: {"fixture": "stable"})
    monkeypatch.setattr("spliceosome_agent.report.figures", lambda plan, run: {"fixture": True})


def test_full_model_loop_and_critic_feedback(approved):
    store, plan, digest = approved
    def behavior(role, calls):
        if role == "Critic" and calls.count("Critic") == 1:
            return AgentDecision(action="retry", target="RNA", explanation="technical fixture review")
        return AgentDecision()
    client = Client(behavior)
    engine = Engine.create(store, digest, client=client, tool_executor=fake_tool)
    state = engine.execute()
    assert state["status"] == "complete"
    assert client.calls.count("Critic") == 2
    assert set(client.calls) == {"QC", "Knowledge", "RNA", "Protein", "Integration", "Critic"}
    assert (engine.run / "attempts/round_00/rna/fixture.txt").exists()
    assert client.calls.count("QC") == 1 and client.calls.count("Protein") == 1
    assert (engine.run / "checkpoints.sqlite3").exists()
    assert read_json(engine.run / "loop_counters.json")["repairs"] == {"RNA": 1}


def test_repair_limit_preserves_partial_evidence(approved):
    store, plan, digest = approved
    client = Client(lambda role, calls: AgentDecision(action="retry", target="RNA") if role == "Critic" else AgentDecision())
    engine = Engine.create(store, digest, client=client, tool_executor=fake_tool)
    state = engine.execute()
    assert client.calls.count("Critic") == plan.max_repairs + 1
    assert any("Repair limit" in w for w in state["warnings"])


def test_design_revision_stops_for_new_plan(approved):
    store, plan, digest = approved
    client = Client(lambda role, calls: AgentDecision(action="revise", explanation="New covariate needed") if role == "Critic" else AgentDecision())
    engine = Engine.create(store, digest, client=client, tool_executor=fake_tool)
    assert engine.execute()["status"] == "waiting_plan_revision"
    with pytest.raises(PermissionError, match="new run"):
        engine.execute(resume=True)


def test_budget_stops_without_claiming_success(approved):
    store, plan, digest = approved
    def behavior(role, calls):
        raise BudgetExceeded("fixture")
    engine = Engine.create(store, digest, client=Client(behavior), tool_executor=fake_tool)
    with pytest.raises(BudgetExceeded):
        engine.execute()
    assert read_json(engine.run / "run_state.json")["status"] == "budget_exhausted"
    assert (engine.run / "report.md").exists()


def test_generated_code_runs_automatically_without_review(approved):
    store, plan, digest = approved
    source = ("import os\n"
              "open(os.path.join(os.environ['SPLICEOSOME_OUTPUT_DIR'], 'auto.txt'), 'w').write('auto')\n")
    def behavior(role, calls):
        return AgentDecision(action="code", language="python", source=source) if role == "RNA" else AgentDecision()
    engine = Engine.create(store, digest, client=Client(behavior), tool_executor=fake_tool)
    state = engine.execute()
    assert state["status"] == "complete"          # no interrupt, no review gate
    proposal = next((engine.run / "code").glob("*/proposal.json"))
    folder = proposal.parent
    assert (folder / "output" / "auto.txt").read_text() == "auto"
    assert not (folder / "approval.json").exists()  # execution no longer requires a review record
    assert any("supplemental and unvalidated" in warning for warning in state["warnings"])


def test_blocked_code_is_recorded_without_stopping_the_run(approved):
    store, plan, digest = approved
    def behavior(role, calls):
        return AgentDecision(action="code", language="python", source="import socket\n") if role == "RNA" else AgentDecision()
    engine = Engine.create(store, digest, client=Client(behavior), tool_executor=fake_tool)
    state = engine.execute()
    assert state["status"] == "complete"
    assert any("blocked by static safety checks" in warning for warning in state["warnings"])
    result = read_json(next((engine.run / "code").glob("*/result.json")))
    assert result["status"] == "blocked" and result["artifacts"] == []


def test_forbidden_code_is_blocked_before_any_execution(approved):
    store, plan, digest = approved
    run = store.workspace / "code-test"
    proposal = propose_code(run, plan, "RNA", "python", "import socket\nprint('fixture')")
    assert proposal["checks"]["passed"] is False
    assert any("Forbidden pattern" in finding for finding in proposal["checks"]["findings"])
    assert "image" not in proposal and proposal["toolchain"]["python_executable"]
    folder, proposal, _ = checked_proposal(run, proposal["proposal_id"])
    result = execute_code(run, plan, proposal["proposal_id"])
    assert result["status"] == "blocked"
    assert not (folder / "execution.log").exists()


def test_missing_python_dependency_and_changed_code(approved):
    store, plan, digest = approved
    run = store.workspace / "code-test"
    proposal = propose_code(run, plan, "RNA", "python", "import definitely_missing_spliceosome_pkg\n")
    assert proposal["checks"]["passed"] is False
    assert any("not importable" in finding for finding in proposal["checks"]["findings"])
    folder, _, source = checked_proposal(run, proposal["proposal_id"])
    source.write_text("print('changed')")
    with pytest.raises(PermissionError):
        checked_proposal(run, proposal["proposal_id"])


def test_reviewed_python_runs_as_restricted_subprocess(approved, monkeypatch):
    store, plan, digest = approved
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fixture-secret-must-not-leak")
    run = store.workspace / "code-run"
    source = (
        "import json, os\n"
        "audit = sorted(key for key in os.environ if 'KEY' in key.upper() or 'TOKEN' in key.upper())\n"
        "inputs = sorted(key for key in os.environ if key.startswith('SPLICEOSOME_INPUT_'))\n"
        "print(json.dumps({'keys': audit, 'inputs': inputs}))\n"
        "open(os.path.join(os.environ['SPLICEOSOME_OUTPUT_DIR'], 'out.txt'), 'w').write('supplemental ok')\n"
    )
    proposal = propose_code(run, plan, "RNA", "python", source)
    assert proposal["checks"]["passed"] is True
    pid = proposal["proposal_id"]
    folder, _, _ = checked_proposal(run, pid)
    result = execute_code(run, plan, pid)
    assert result["status"] == "complete", result
    assert (folder / "output" / "out.txt").read_text() == "supplemental ok"
    log = (folder / "execution.log").read_text()
    assert "sk-fixture-secret-must-not-leak" not in log
    assert '"keys": []' in log and "SPLICEOSOME_INPUT_000" in log
    assert read_json(folder / "result.json")["scientific_status"] == "supplemental_unvalidated"


def test_toolchain_change_invalidates_execution(approved, monkeypatch):
    store, plan, digest = approved
    run = store.workspace / "code-test"
    proposal = propose_code(run, plan, "RNA", "python", "print('fixture')")
    pid = proposal["proposal_id"]
    monkeypatch.setattr("spliceosome_agent.generated_code.toolchain",
                        lambda study: {"python": "changed", "python_executable": "x", "rscript": None, "r_version": None})
    with pytest.raises(PermissionError, match="toolchain"):
        execute_code(run, plan, pid)


def test_contract_hint_lists_exact_fields_and_enums():
    from spliceosome_agent.llm import contract_hint
    from spliceosome_agent.models import AgentDecision, PlanningReply
    decision = contract_hint(AgentDecision)
    assert "continue|retry|revise|code" in decision
    assert "QC|Knowledge|RNA|Protein|Integration" in decision
    assert "python|r" in decision
    planning = contract_hint(PlanningReply)
    for field in ["questions", "study", "explanation"]:
        assert field in planning


def test_model_prompt_carries_contract_and_accepts_message_alias(tmp_path):
    import httpx
    import json
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    from spliceosome_agent.models import PlanningReply
    captured = {}

    def handle(request):
        body = json.loads(request.content)
        captured["system"] = body["messages"][0]["content"]
        result = {"message": "alias fixture", "questions": ["synthetic question"], "study": None}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                       "finish_reason": "stop"}], "usage": {"total_tokens": 10}})

    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(
        name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    reply = client.json("Orchestrator", "Synthetic system prompt.", {}, PlanningReply)
    assert reply.explanation == "alias fixture" and reply.questions == ["synthetic question"]
    for field in ["questions", "study", "explanation"]:
        assert field in captured["system"]


def test_model_reports_invalid_field_location_without_raw_content(tmp_path):
    import httpx
    import json
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient, ModelUnavailable
    from spliceosome_agent.models import PlanningReply

    def handle(request):
        result = {"explanation": "fixture", "unexpected_private_field": "SYNTHETIC_VALUE"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                       "finish_reason": "stop"}], "usage": {"total_tokens": 10}})

    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(
        name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    with pytest.raises(ModelUnavailable, match="unexpected_private_field"):
        client.json("Orchestrator", "Synthetic system prompt.", {}, PlanningReply)
    record = read_json(tmp_path / "calls" / "00003_Orchestrator.json")
    assert record["error_type"] == "ValidationError" and "unexpected_private_field" in record["error"]
    assert "SYNTHETIC_VALUE" not in json.dumps(record)  # model content is never persisted


def test_greeting_reply_is_shown_without_creating_a_plan(tmp_path):
    import httpx
    import json
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    from spliceosome_agent.session import Session

    def handle(request):
        result = {"message": "Hello. Describe the study or load a configuration.", "questions": [], "study": None}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                       "finish_reason": "stop"}], "usage": {"total_tokens": 10}})

    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(
        name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    reply = session.natural("hello")
    assert reply == "Hello. Describe the study or load a configuration."
    assert session.store.get("current_plan") is None


def test_plan_and_run_history_commands(tmp_path, study):
    from spliceosome_agent.session import Session

    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    assert "No stored plans" in session.handle("/plans")
    assert "No runs yet" in session.handle("/runs")
    assert "No runs yet" in session.handle("/tasks")
    assert "No report" in session.handle("/report") or "No runs yet" in session.handle("/report")

    created = session.create_plan(study, "SYNTHETIC history fixture")
    digest = session.store.get("current_plan")
    listing = session.handle("/plans")
    assert digest[:12] in listing and "v1" in listing and "SYNTHETIC history fixture" in listing
    assert "not approved" in listing
    # A newer version supersedes the older one and expires its approval.
    session.store.approve(digest)
    assert "approved" in session.handle("/plans")
    session.create_plan(study, "SYNTHETIC history fixture v2")
    second = session.store.get("current_plan")
    listing = session.handle("/plans")
    assert second[:12] in listing and "current" in listing and "superseded" in listing
    assert "approval expired" in listing  # v1 was approved, then superseded by v2
    detail = session.handle(f"/plans {second[:8]}")
    assert "Approval:" in detail and "current" in detail and "SYNTHETIC history fixture v2" in detail
    with pytest.raises(ValueError, match="hexadecimal"):
        session.handle("/plans zzzz")
    with pytest.raises(ValueError, match="No stored plan"):
        session.handle("/plans 0000000000")


def test_logs_command_reads_worker_tail(tmp_path, study):
    from spliceosome_agent.session import Session

    session = Session(tmp_path / "workspace", tmp_path / "models.json")
    session.create_plan(study, "SYNTHETIC log fixture")
    run_id = "fixture-run-01"
    folder = session.store.run_dir(run_id)
    folder.mkdir(parents=True)
    (folder / "worker.log").write_text("\n".join(f"line {i}" for i in range(1, 61)) + "\n", encoding="utf-8")
    session.store.set("latest_run", run_id)
    tail = session.handle("/logs")
    assert "line 60" in tail and "line 21" in tail and "line 20" not in tail  # default 40 lines
    assert "line 55" in session.handle(f"/logs {run_id} 6")
    with pytest.raises(ValueError, match="positive integer"):
        session.handle(f"/logs {run_id} zero")
    assert "No report" in session.handle("/report")  # report is written only when a run finishes


def test_conversation_probes_and_creates_a_plan_without_executing(tmp_path, study):
    import httpx
    import json
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    from spliceosome_agent.session import Session

    def handle(request):
        result = {"goal": "synthetic comparison", "questions": [], "study": study.model_dump(),
                  "explanation": "draft ready"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                       "finish_reason": "stop"}], "usage": {"total_tokens": 10}})

    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(
        name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    session.set_study(study)  # explicit locally loaded inputs, never invented by the mock model
    reply = session.handle("我有一组合成RNA和蛋白数据想做两组比较")
    assert session.store.get("current_plan") is not None
    assert session.study is not None and session.study.name == study.name
    assert session.store.get("draft_goal") == "synthetic comparison"
    assert "Confirm with" in reply
    assert session.draft_pending() is False
    digest = session.store.get("current_plan")
    assert session.store.load_plan().preflight.status == "ready_for_confirmation"
    assert session.generate_plan()  # identical goal/config/input reuses the same version
    assert session.store.get("current_plan") == digest
    assert len(session.store.list_plans()) == 1
    assert not list(session.store.runs_dir.glob("*/run_state.json"))
    with pytest.raises(PermissionError):
        session.store.require_approval(digest)


def test_confirm_run_phrase_no_longer_approves_or_runs(tmp_path, study):
    import httpx
    import json
    from spliceosome_agent import jobs
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    from spliceosome_agent.session import Session

    def handle(request):
        result = {"questions": [], "study": None, "explanation": "Use /approve and /run explicitly."}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)},
                                                       "finish_reason": "stop"}], "usage": {"total_tokens": 10}})

    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(
        name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    session.create_plan(study, "SYNTHETIC software test")
    digest = session.store.get("current_plan")
    reply = session.handle("确认执行")
    with pytest.raises(PermissionError):
        session.store.require_approval(digest)
    assert jobs.list_runs(session.store) == []
    assert "approve" in reply.lower()


def test_natural_clarification_plan_approval_and_model_loop(tmp_path, study):
    import httpx
    import json
    from spliceosome_agent.config import ModelConfig, ModelProfile
    from spliceosome_agent.llm import ModelClient
    from spliceosome_agent.session import Session
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("response_format", {}).get("json_schema", {}).get("name") == "PlanningReply":
            if len(requests) == 1:
                result = {"questions": ["请确认输入配置和测量尺度"], "study": None}
            else:
                context = json.loads(body["messages"][1]["content"])
                assert len(context["conversation"]) == 2
                result = {"questions": [], "study": study.model_dump(), "explanation": "使用明确声明的合成输入"}
        else:
            result = {"action": "continue", "explanation": "模拟模型审查"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}], "usage": {"total_tokens": 20}})
    config = ModelConfig(default="mock", profiles={"mock": ModelProfile(name="mock", model="fixture", base_url="http://127.0.0.1:9999/v1", kind="local", api_key_env=None)})
    client = ModelClient(config, tmp_path / "calls", transport=httpx.MockTransport(handle))
    session = Session(tmp_path / "workspace", tmp_path / "models.json", client=client)
    session.set_study(study)
    assert "More information is needed" in session.natural("我想筛选剪接体候选")
    assert "Confirm with" in session.natural("我确认使用这些合成输入和已声明尺度")
    digest = session.store.get("current_plan")
    with pytest.raises(PermissionError):
        session.store.require_approval(digest)
    session.store.approve(digest)
    engine = Engine.create(session.store, digest, config, client=client, tool_executor=fake_tool)
    assert engine.execute()["status"] == "complete"
    assert len(requests) == 8  # Two planner turns plus six specialist reviews.


def test_worker_console_tail_does_not_invalidate_frozen_artifacts(approved):
    from spliceosome_agent.engine import verify_run
    store, plan, digest = approved
    engine = Engine.create(store, digest, client=Client(), tool_executor=fake_tool)
    log = engine.run / "worker.log"
    log.write_text("worker started\n")
    assert engine.execute()["status"] == "complete"
    log.write_text(log.read_text() + "final CLI response\n")
    assert verify_run(store, engine.run_id)["status"] == "verified"
