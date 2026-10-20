import json

import httpx
import pytest

from spliceosome_agent.config import ModelConfig, ModelProfile
from spliceosome_agent.llm import BudgetExceeded, ModelClient, ModelUnavailable, PrivacyGuard
from spliceosome_agent.models import AgentDecision, PlanningReply
from spliceosome_agent.planning import understand


def config():
    return ModelConfig(default="test", profiles={"test": ModelProfile(name="test", base_url="http://127.0.0.1:9999/v1", kind="local", model="fixture", api_key_env=None)})


def completion(value, tokens=20):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], "usage": {"total_tokens": tokens}})


def test_outbound_privacy_and_no_secret_in_audit(tmp_path):
    requests = []
    def handle(request):
        requests.append(request.content.decode())
        return completion({"action": "continue", "explanation": "safe", "evidence_ids": []})
    client = ModelClient(config(), tmp_path, identifiers=["PATIENT_123"], transport=httpx.MockTransport(handle))
    client.guard.secrets.append("secretvalue")
    client.json("RNA", "JSON only", {"summary": "PATIENT_123 secretvalue"}, AgentDecision)
    assert "PATIENT_123" not in requests[0] and "secretvalue" not in requests[0]
    assert "request_messages" not in (tmp_path / "00001_RNA.json").read_text()
    assert "PATIENT_123" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_incompatible_schema_falls_back_and_validates(tmp_path):
    modes = []
    def handle(request):
        mode = json.loads(request.content).get("response_format", {}).get("type", "plain")
        modes.append(mode)
        if mode != "plain":
            return httpx.Response(400, text="unsupported response_format")
        return completion({"action": "continue"})
    client = ModelClient(config(), tmp_path, transport=httpx.MockTransport(handle))
    assert client.json("RNA", "JSON only", {}, AgentDecision).action == "continue"
    assert modes == ["json_schema", "json_object", "plain"]


def test_invalid_json_is_not_an_execution_instruction(tmp_path):
    def handle(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "run rm arbitrary"}, "finish_reason": "stop"}], "usage": {"total_tokens": 20}})
    client = ModelClient(config(), tmp_path, transport=httpx.MockTransport(handle))
    with pytest.raises(ModelUnavailable):
        client.json("RNA", "JSON only", {}, AgentDecision)
    assert client.metrics["requests"] == 3 and client.metrics["tokens"] == 60


def test_budget_persists_across_new_client(tmp_path):
    transport = httpx.MockTransport(lambda request: completion({"action": "continue"}, tokens=190))
    client = ModelClient(config(), tmp_path, token_budget=300, transport=transport)
    client.json("RNA", "JSON", {}, AgentDecision)
    client = ModelClient(config(), tmp_path, token_budget=300, transport=transport)
    with pytest.raises(BudgetExceeded):
        client.json("RNA", "JSON", {"long": "x" * 100}, AgentDecision)


def test_401_does_not_fallback_or_leak_error_body(tmp_path):
    def handle(request):
        return httpx.Response(401, text="secret body")
    client = ModelClient(config(), tmp_path, transport=httpx.MockTransport(handle))
    with pytest.raises(ModelUnavailable, match="401"):
        client.json("RNA", "JSON", {}, AgentDecision)
    assert client.metrics["requests"] == 1


def test_planner_clarifies_missing_scales_instead_of_inventing(tmp_path):
    transport = httpx.MockTransport(lambda request: completion({"questions": ["RNA是什么测量尺度？"], "study": None, "explanation": "需要确认"}))
    client = ModelClient(config(), tmp_path, transport=transport)
    reply, study = understand(client, "请筛选剪接体")
    assert study is None and len(reply.questions) == 1


def test_model_rejects_unknown_actions(tmp_path):
    transport = httpx.MockTransport(lambda request: completion({"action": "shell", "argv": ["sh"]}))
    client = ModelClient(config(), tmp_path, transport=transport)
    with pytest.raises(ModelUnavailable):
        client.json("RNA", "JSON", {}, AgentDecision)


def test_numeric_subject_labels_do_not_corrupt_statistical_numbers(tmp_path):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content)["messages"][1]["content"])
        return completion({"action": "continue", "explanation": "subject 1"})
    client = ModelClient(config(), tmp_path, identifiers=["1"], transport=httpx.MockTransport(handle))
    result = client.json("RNA", "JSON", {"n_units": 12, "effect": 1, "subject": "1"}, AgentDecision)
    payload = json.loads(requests[0])
    assert payload["effect"] == 1 and payload["subject"] == "[SUBJECT]"
    assert result.explanation == "subject [SUBJECT]"
