"""Seven-role LangGraph loop with SQLite checkpoints and shared approval checks."""
from __future__ import annotations

import json
import operator
import os
import re
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

# Patient-level tools and graph state must not be forwarded to external tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import report
from .config import ModelConfig
from .data import child_environment, private_ids, resource
from .io import artifact, code_fingerprint, environment, now, object_hash, read_json, verify_artifacts, write_json
from .llm import BudgetExceeded, ModelClient, ModelUnavailable
from .models import AgentDecision, ToolResult
from .generated_code import checked_proposal, execute_code, propose_code
from .planning import input_paths, verify_inputs
from .store import Store
from .tools import execute_tool


def merge(left, right):
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    results: Annotated[dict, merge]
    decisions: Annotated[dict, merge]
    warnings: Annotated[list[str], operator.add]
    round: int
    repair_target: str | None
    route: str
    status: str


ROLE_INSTRUCTIONS = {
    "QC": "Audit identity, pairing, scales, missingness and design feasibility. Do not remove samples based on outcomes.",
    "Knowledge": "Audit source/version/species and core versus regulator membership; broad pathway members are not structural core.",
    "RNA": "Audit the RNA tool result, full-feature BH, scale and count suitability. Never treat expression as PSI.",
    "Protein": "Audit the protein result, declared scale, paired/unpaired model and uncertainty. Do not impute or renormalize implicitly.",
    "Integration": "Audit independent units, separate cohorts, coverage, discordance and stable-ID ordering. Never tune a weighted score.",
    "Critic": "Review scientific validity and provenance. Continue for valid negative results. Return retry with a target only for a concrete technical defect. Return revise for design changes.",
}

DEPENDENTS = {
    "QC": {"QC", "RNA", "Protein", "Integration", "Critic"},
    "Knowledge": {"Knowledge", "Integration", "Critic"},
    "RNA": {"RNA", "Integration", "Critic"},
    "Protein": {"Protein", "Integration", "Critic"},
    "Integration": {"Integration", "Critic"},
}


def valid_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("Invalid run ID")
    return value


def r_fingerprint(study):
    import subprocess
    script = ('cat(R.version.string,"\\n"); for(p in c("jsonlite","limma","DESeq2")) '
              'cat(p,if(requireNamespace(p,quietly=TRUE))as.character(packageVersion(p))else"NOT_INSTALLED","\\n")')
    try:
        result = subprocess.run([study.rscript, "--vanilla", "-e", script], env=child_environment(study),
                                capture_output=True, text=True, timeout=90)
        return {"exit_code": result.returncode, "versions": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": type(exc).__name__}


def provenance(plan):
    return {"plan_hash": object_hash(plan), "inputs": plan.input_hashes, "code": code_fingerprint(),
            "environment": environment(), "r": r_fingerprint(plan.study)}


class Engine:
    def __init__(self, store: Store, run_id, model_config=None, client=None, tool_executor=None):
        self.store = store
        self.run_id = valid_run_id(run_id)
        self.run = store.run_dir(self.run_id)
        self.config = model_config or ModelConfig()
        self.client = client
        self.tool_executor = tool_executor or execute_tool
        self.lock = threading.RLock()

    @classmethod
    def create(cls, store, digest, model_config=None, run_id=None, client=None, tool_executor=None):
        approval = store.require_approval(digest)
        plan = store.load_plan(digest)
        verify_inputs(plan)
        run_id = run_id or now().replace(":", "").replace("-", "").split(".")[0].replace("+", "_") + "_" + uuid.uuid4().hex[:8]
        engine = cls(store, run_id, model_config, client, tool_executor)
        # Refuse raw-input overlap before creating any execution directory.
        for path in input_paths(plan.study):
            if Path(path).resolve().is_relative_to(engine.run.resolve()):
                raise ValueError("An input may not live inside this new run")
        engine.run.mkdir(parents=True, exist_ok=False)
        write_json(engine.run / "plan.json", plan)
        write_json(engine.run / "approval.json", approval)
        write_json(engine.run / "provenance.json", provenance(plan))
        from .runner import generate
        generate(engine)
        # Persist only non-secret profile names/URLs/environment variable names.
        write_json(engine.run / "model_config.json", engine.config)
        write_json(engine.run / "loop_counters.json", {"role_calls": 0, "repairs": {}})
        engine.set_status("ready", results={}, warnings=[], agent_mode="pending")
        store.set("latest_run", run_id)
        return engine

    @property
    def plan(self):
        from .models import AnalysisPlan
        return AnalysisPlan.model_validate(read_json(self.run / "plan.json"))

    def event(self, kind, payload=None):
        with self.lock:
            record = self.store.event(self.run_id, kind, payload)
            with (self.run / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def set_status(self, status, **extra):
        path = self.run / "run_state.json"
        state = read_json(path) if path.exists() else {}
        state.update({"run_id": self.run_id, "status": status, "updated_at": now(), **extra})
        write_json(path, state)
        return state

    def _validate(self):
        plan = self.plan
        self.store.require_approval(object_hash(plan))
        verify_inputs(plan)
        if plan.preflight:
            from .preflight import verify_environment
            verify_environment(plan.preflight, plan.study)
        from .runner import verify
        verify(self)
        if read_json(self.run / "provenance.json") != provenance(plan):
            raise ValueError("Resume rejected: input, code or environment fingerprint changed")
        return plan

    def _count_role(self):
        with self.lock:
            counters = read_json(self.run / "loop_counters.json")
            if counters["role_calls"] >= self.plan.max_role_calls:
                raise BudgetExceeded("Approved role-call budget exhausted")
            counters["role_calls"] += 1
            write_json(self.run / "loop_counters.json", counters)

    def _summary_context(self, role, result, state):
        # Do not feed paths, raw metadata, arbitrary logs or exception messages to a model.
        context = {"role": role, "tool_status": result.status, "summary": result.summary,
                   "error_present": bool(result.errors), "round": state.get("round", 0),
                   "approved_methods": {"design": self.plan.study.design, "alpha": self.plan.study.alpha,
                                        "confidence": self.plan.study.confidence, "missing_policy": self.plan.study.missing_policy},
                   "evidence": []}
        registry = self.run / "integration/evidence_registry.csv"
        if registry.exists():
            import pandas as pd
            table = pd.read_csv(registry).head(100)
            context["evidence"] = json.loads(table.to_json(orient="records"))
        context["input_files"] = [{"env": f"SPLICEOSOME_INPUT_{i:03d}", "kind": Path(path).suffix}
                                  for i, path in enumerate(input_paths(self.plan.study))]
        return context

    def _step(self, role, state):
        task = next((t for t in self.plan.tasks if t.role == role), None)
        if task is None:
            return {}
        round_number = state.get("round", 0)
        if round_number and role not in DEPENDENTS[state["repair_target"]]:
            previous = state.get("results", {}).get(role)
            if previous:
                verify_artifacts(self.run, previous["artifacts"])
                self.event("role_reused", {"role": role, "round": round_number})
                return {"results": {role: previous}, "decisions": {role: AgentDecision().model_dump()}}
        cache = self.run / "steps" / f"{round_number:02d}_{role}.json"
        if cache.exists():
            saved = read_json(cache)
            result = ToolResult.model_validate(saved["result"])
            verify_artifacts(self.run, result.artifacts)
            decision = AgentDecision.model_validate(saved["decision"])
        else:
            self._count_role()
            self.event("role_started", {"role": role, "round": round_number})
            result = self.tool_executor(task, self.plan, self.run, state.get("results", {}))
            decision = AgentDecision()
            model_mode = "deterministic_tool_checks"
            if self.client.available(role):
                try:
                    # Fresh QC can reveal identifiers previously stored only in local RDS.
                    self.client.guard.identifiers = sorted(private_ids(self.plan.study, self.run), key=len, reverse=True)
                    context = self._summary_context(role, result, state)
                    decision = self.client.json(role, resource("shared.md").read_text() + "\n" + ROLE_INSTRUCTIONS[role], context, AgentDecision)
                    valid_ids = {item["evidence_id"] for item in context["evidence"]}
                    if not set(decision.evidence_ids).issubset(valid_ids):
                        raise ValueError("Model cited unknown evidence")
                    model_mode = "model_reviewed"
                except ModelUnavailable as exc:
                    self.event("model_degraded", {"role": role, "error_type": type(exc).__name__})
                    model_mode = "model_unavailable"
            if decision.action == "retry" and role != "Critic" and result.status == "complete":
                decision = AgentDecision(explanation="Successful deterministic results are not rerun to seek significance")
            write_json(cache, {"result": result.model_dump(), "decision": decision.model_dump(), "mode": model_mode})
            self.event("role_finished", {"role": role, "status": result.status, "mode": model_mode})
        update = {"results": {role: result.model_dump()}, "decisions": {role: decision.model_dump()}}
        if read_json(cache).get("mode") == "model_unavailable":
            update["warnings"] = [f"{role}: model review unavailable; only deterministic tool checks completed"]
        if decision.action == "code":
            previous = sorted((self.run / "code").glob("*/analysis.*"))
            proposal = propose_code(self.run, self.plan, role, decision.language, decision.source,
                                    previous[-1] if previous else None)
            proposal_id = proposal["proposal_id"]
            self.event("code_proposed", {"proposal_id": proposal_id, "file_sha256": proposal["file_sha256"],
                                        "checks_passed": proposal["checks"]["passed"]})
            folder, _, _ = checked_proposal(self.run, proposal_id)
            if not (folder / "result.json").exists():
                # Approval of the plan authorises execution; static checks and the restricted
                # local environment gate the code, with no separate per-file review step.
                code_result = execute_code(self.run, self.plan, proposal_id)
                self.event("supplemental_code_finished", {"proposal_id": proposal_id, "status": code_result["status"]})
            else:
                code_result = read_json(folder / "result.json")
            update.setdefault("warnings", []).append("Generated code output is supplemental and unvalidated; registered statistics retain authority")
            if code_result["status"] == "failed":
                update["warnings"].append("Supplemental code execution failed; inspect its result.json and execution.log")
            elif code_result["status"] == "blocked":
                update["warnings"].append("Supplemental code blocked by static safety checks; nothing was executed")
        return update

    def _route(self, state):
        decisions = state.get("decisions", {})
        for role, payload in decisions.items():
            decision = AgentDecision.model_validate(payload)
            if decision.action == "revise":
                return {"route": "revision", "status": "waiting_plan_revision"}
        for role, payload in decisions.items():
            decision = AgentDecision.model_validate(payload)
            if decision.action == "retry":
                target = decision.target if role == "Critic" else role
                if not target or target not in {task.role for task in self.plan.tasks} or target == "Critic":
                    return {"route": "finish", "status": "partial", "warnings": ["Invalid critic repair target; retained for review"]}
                with self.lock:
                    counters = read_json(self.run / "loop_counters.json")
                    count = counters["repairs"].get(target, 0)
                    if count >= self.plan.max_repairs:
                        return {"route": "finish", "status": "partial", "warnings": [f"Repair limit reached for {target}"]}
                    counters["repairs"][target] = count + 1
                    write_json(self.run / "loop_counters.json", counters)
                return {"route": "repair", "repair_target": target}
        return {"route": "finish"}

    def _repair(self, state):
        # Preserve exact previous outputs before rebuilding any descendant artifacts.
        number = state.get("round", 0)
        history = self.run / "attempts" / f"round_{number:02d}"
        history.mkdir(parents=True, exist_ok=False)
        mapping = {}
        folders = {"QC": "inputs", "Knowledge": "knowledge", "RNA": "rna", "Protein": "protein", "Integration": "integration", "Critic": "critic"}
        for role in DEPENDENTS[state["repair_target"]]:
            folder = folders[role]
            source = self.run / folder
            if source.exists():
                shutil.move(str(source), history / folder)
                mapping[folder] = str((history / folder).relative_to(self.run))
            result = self.run / "results" / f"{role}.json"
            if result.exists():
                (history / "results").mkdir(exist_ok=True)
                shutil.move(str(result), history / "results" / result.name)
        write_json(history / "path_map.json", mapping)
        self.event("repair_round", {"target": state["repair_target"], "round": number + 1})
        return {"round": number + 1, "decisions": {role: AgentDecision().model_dump() for role in ROLE_INSTRUCTIONS}}

    def _finish(self, state):
        review_failed = any("model review unavailable" in warning or "Supplemental code execution failed" in warning for warning in state.get("warnings", []))
        status = "complete" if not review_failed and state.get("status") != "partial" and all(result["status"] == "complete" for result in state.get("results", {}).values()) else "partial"
        return {"status": status}

    def _graph(self, saver):
        graph = StateGraph(GraphState)
        for role in ROLE_INSTRUCTIONS:
            graph.add_node(role, lambda state, role=role: self._step(role, state))
        graph.add_node("dispatch", lambda state: {})
        graph.add_node("route", self._route)
        graph.add_node("repair", self._repair)
        graph.add_node("finish", self._finish)
        graph.add_node("revision", lambda state: {"status": "waiting_plan_revision"})
        graph.add_edge(START, "QC")
        screening = self.plan.study.analysis_mode == "candidate_screening"
        if screening:
            graph.add_edge(START, "Knowledge")
            graph.add_edge(["QC", "Knowledge"], "dispatch")
        else:
            graph.add_edge("QC", "dispatch")
        domains = ["RNA" if domain == "rna" else "Protein" for domain in self.plan.study.assays]
        for role in domains:
            graph.add_edge("dispatch", role)
        if screening:
            graph.add_edge(domains, "Integration")
            graph.add_edge("Integration", "Critic")
        else:
            graph.add_edge(domains, "Critic")
        graph.add_edge("Critic", "route")
        graph.add_conditional_edges("route", lambda state: state["route"],
                                    {"repair": "repair", "finish": "finish", "revision": "revision"})
        graph.add_edge("repair", "QC")
        if screening:
            graph.add_edge("repair", "Knowledge")
        graph.add_edge("finish", END)
        graph.add_edge("revision", END)
        return graph.compile(checkpointer=saver)

    def execute(self, resume=False):
        import fcntl
        with (self.run / ".run.lock").open("a+") as lease:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another worker is executing this run") from exc
            try:
                return self._execute(resume)
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)

    def _execute(self, resume=False):
        plan = self._validate()
        if self.client is None:
            self.client = ModelClient(self.config, self.run / "llm", plan.token_budget, private_ids(plan.study, self.run))
        old = read_json(self.run / "run_state.json")
        if resume and old["status"] == "waiting_plan_revision":
            raise PermissionError("A revised plan must be approved and started in a new run")
        if old["status"] == "complete":
            verify_run(self.store, self.run_id)
            return old
        if resume:
            manifest_results = old.get("results", {})
            for result in manifest_results.values():
                verify_artifacts(self.run, result.get("artifacts", []))
        self.set_status("running", agent_mode="model_assisted" if self.client.available() else "deterministic_tools")
        connection = sqlite3.connect(self.run / "checkpoints.sqlite3", check_same_thread=False)
        saver = SqliteSaver(connection)
        graph = self._graph(saver)
        config = {"configurable": {"thread_id": self.run_id}, "recursion_limit": 200, "max_concurrency": 2}
        snapshot = graph.get_state(config) if resume else None
        if resume and old["status"] == "waiting_plan_revision":
            raise PermissionError("A revised plan must be approved and started in a new run")
        elif resume and snapshot and snapshot.values:
            initial = None
        else:
            initial = {"results": {}, "decisions": {}, "warnings": [], "round": 0}
        try:
            state = graph.invoke(initial, config)
            final = self.set_status(state.get("status", "partial"), results=state.get("results", {}), warnings=state.get("warnings", []))
            try:
                plots = report.figures(plan, self.run)
                write_json(self.run / "figure_validation.json", plots)
            except Exception as exc:
                final["warnings"].append("Figure export failed: " + str(exc))
                if final["status"] == "complete":
                    final["status"] = "partial"
            report.render(plan, self.run, final)
            final = self.set_status(final["status"], results=final["results"], warnings=final["warnings"])
            self.event("run_finished", {"status": final["status"]})
            self.finalize()
            return final
        except BaseException as exc:
            # Graph interrupts are returned above; budget/errors/stops remain resumable and visible.
            status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
            values = graph.get_state(config).values
            state = self.set_status(status, results=values.get("results", {}), warnings=[f"{type(exc).__name__}: {exc}"])
            self.event("run_stopped", {"status": status, "error_type": type(exc).__name__})
            report.render(plan, self.run, state)
            raise
        finally:
            connection.close()

    def finalize(self):
        audit = self.run / "audit"
        audit.mkdir(exist_ok=True)
        shutil.copy2(self.run / "events.jsonl", audit / "trace.jsonl")
        # The CLI appends its final response to worker.log after this method returns.
        # Numerical/model logs and the frozen event trace remain hash-verified artifacts.
        excluded = {"artifact_manifest.json", "run_state.json", "events.jsonl", "checkpoints.sqlite3", "checkpoints.sqlite3-wal", "checkpoints.sqlite3-shm", ".run.lock", "job.json", "worker.log"}
        records = [artifact(self.run, p).model_dump() for p in sorted(self.run.rglob("*"))
                   if p.is_file() and p.name not in excluded and not {".plot-cache", "plot-cache"}.intersection(p.parts)]
        write_json(self.run / "artifact_manifest.json", {"artifacts": records, "created_at": now()})


def verify_run(store, run_id):
    engine = Engine(store, run_id)
    plan = engine.plan
    from .runner import verify
    verify(engine)
    verify_inputs(plan)
    if read_json(engine.run / "provenance.json") != provenance(plan):
        raise ValueError("Code/environment/source drift; historical artifacts are retained")
    manifest = read_json(engine.run / "artifact_manifest.json")
    verify_artifacts(engine.run, manifest["artifacts"])
    return {"status": "verified", "run_id": run_id, "artifacts": len(manifest["artifacts"])}
