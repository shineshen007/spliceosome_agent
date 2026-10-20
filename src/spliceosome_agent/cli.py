"""Installable command surface. No execution command bypasses plan approval."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import ModelConfig, ModelProfile, PRESETS, load_config, readiness, save_config
from .io import read_json
from .store import Store


def parser():
    p = argparse.ArgumentParser(prog="spliceosome-agent", description="Spliceosome multi-agent screening; no command starts the welcome UI")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--workspace", "-w", type=Path, default=Path.cwd() / "result")
    p.add_argument("--config", type=Path, help="non-secret model configuration file")
    sub = p.add_subparsers(dest="command")
    tui = sub.add_parser("tui")
    tui.add_argument("--repl", action="store_true")
    init = sub.add_parser("init", help="load a study configuration")
    init.add_argument("study", type=Path)
    plan = sub.add_parser("plan", help="create/view a plan without running analysis")
    plan.add_argument("--study", type=Path)
    plan.add_argument("--goal")
    plan.add_argument("--natural", action="store_true", help="let the model interpret the goal and ask necessary questions")
    plan.add_argument("--max-repairs", type=int)
    plan.add_argument("--max-role-calls", type=int)
    plan.add_argument("--token-budget", type=int)
    plan.add_argument("--disable-generated-code", action="store_true")
    plans = sub.add_parser("plans", help="list stored plan versions, or show one by hash prefix")
    plans.add_argument("plan_hash", nargs="?", help="unique hexadecimal hash prefix")
    edit = sub.add_parser("edit", help="create a new plan version from an edited study")
    edit.add_argument("study", type=Path)
    approve = sub.add_parser("approve", help="explicitly approve the given plan hash")
    approve.add_argument("plan_hash")
    confirm = sub.add_parser("confirm", help="confirm an exact ready plan and start analysis")
    confirm.add_argument("plan_hash")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("plan_hash", nargs="?")
    run = sub.add_parser("run")
    run.add_argument("plan_hash", nargs="?")
    run.add_argument("--foreground", action="store_true")
    run.add_argument("--run-id")
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--foreground", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("run_id", nargs="?")
    sub.add_parser("runs")
    for name in ["verify", "stop", "report", "logs"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("run_id")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--study", type=Path)
    examples = sub.add_parser("examples")
    examples.add_argument("destination", type=Path)
    examples.add_argument("--design", choices=["paired", "unpaired"], default="paired")
    examples.add_argument("--rna-scale", choices=["counts", "log2_expression"], default="counts")
    examples.add_argument("--r-library", type=Path)
    models = sub.add_parser("models")
    msub = models.add_subparsers(dest="model_command", required=True)
    msub.add_parser("list")
    add = msub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--preset", choices=list(PRESETS), default="openai")
    add.add_argument("--model", required=True)
    add.add_argument("--base-url")
    add.add_argument("--key-env")
    add.add_argument("--key", help="API key value; saved to the OS keychain, never to the config file")
    add.add_argument("--kind", choices=["local", "remote"])
    add.add_argument("--format", choices=["auto", "schema", "json", "plain"], default="auto")
    add.add_argument("--default", action="store_true")
    role = msub.add_parser("role")
    role.add_argument("role", choices=["Orchestrator", "QC", "Knowledge", "RNA", "Protein", "Integration", "Critic"])
    role.add_argument("profile")
    test = msub.add_parser("test")
    test.add_argument("--role", default="Orchestrator")
    deps = sub.add_parser("deps")
    deps.add_argument("--install-r", action="store_true", required=True)
    deps.add_argument("--library", type=Path, required=True)
    knowledge = sub.add_parser("knowledge", help="import candidate tables from explicitly chosen frozen annotations")
    knowledge.add_argument("kind", choices=["go", "reactome"])
    knowledge.add_argument("input", type=Path)
    knowledge.add_argument("output", type=Path)
    knowledge.add_argument("--terms", nargs="+", required=True)
    knowledge.add_argument("--species", required=True)
    knowledge.add_argument("--source", required=True)
    knowledge.add_argument("--resource-version", required=True)
    knowledge.add_argument("--taxon", help="GO GAF taxon ID, e.g. 9606")
    knowledge.add_argument("--obo", type=Path)
    knowledge.add_argument("--member-type", choices=["core", "regulator", "pathway_member"], default="pathway_member")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("run_id")
    worker.add_argument("--resume", action="store_true")
    return p


def emit(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return dispatch(args) or 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"spliceosome-agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 2


def state_exit_code(state):
    return 0 if state["status"] in {"complete", "waiting_plan_revision"} else 1


def dispatch(args):
    from . import jobs
    from .data import resource
    from .engine import Engine, verify_run
    from .runner import execute_run
    from .planning import describe, load_study
    from .session import Session
    store = Store(args.workspace)
    cache = store.cache_dir
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    cmd = args.command
    if cmd is None or cmd == "tui":
        session = Session(args.workspace, args.config)
        if (cmd == "tui" and args.repl) or not sys.stdin.isatty():
            return repl(session)
        from .app import SpliceosomeApp
        SpliceosomeApp(session).run()
        return 0
    if cmd == "init":
        session = Session(args.workspace, args.config)
        session.set_study(load_study(args.study))
        emit("Study config loaded; next: plan --goal 'research goal'.")
    elif cmd in {"plan", "edit"}:
        session = Session(args.workspace, args.config)
        if cmd == "edit":
            emit(session.create_plan(load_study(args.study), session.current_plan().goal))
        elif args.study or args.goal:
            if args.study:
                session.set_study(load_study(args.study))
            if args.natural:
                emit(session.natural(args.goal or "Build a spliceosome screening plan"))
            else:
                if session.study is None:
                    raise ValueError("Provide --study or run init first")
                limits = {key: getattr(args, key) for key in ["max_repairs", "max_role_calls", "token_budget"] if getattr(args, key) is not None}
                if args.disable_generated_code:
                    limits["generated_code"] = False
                emit(session.create_plan(session.study, args.goal or "Spliceosome multi-omics screening", limits))
        else:
            emit(describe(store.load_plan()))
    elif cmd == "plans":
        from .session import plan_lines
        if not args.plan_hash:
            emit(plan_lines(store))
        else:
            digest = store.resolve_plan(args.plan_hash)
            plan = store.load_plan(digest)
            status = "approved" if store.is_approved(digest) else "not approved"
            if store.get("current_plan") == digest:
                status += " (current)"
            emit(f"Approval: {status}\n\n" + describe(plan))
    elif cmd == "confirm":
        from .session import Session
        emit(Session(args.workspace, args.config).confirm_and_run(args.plan_hash))
    elif cmd == "approve":
        emit(store.approve(args.plan_hash))
    elif cmd == "cancel":
        store.cancel(args.plan_hash or store.get("current_plan"))
        emit("Approval revoked.")
    elif cmd == "run":
        engine = Engine.create(store, args.plan_hash or store.get("current_plan"), load_config(args.config), args.run_id)
        if args.foreground:
            state = execute_run(engine)
            emit(state)
            return state_exit_code(state)
        emit(jobs.start(store, engine.run_id, args.config))
    elif cmd in {"resume", "_worker"}:
        folder = store.run_dir(args.run_id)
        # Resume the original recipient configuration; secrets remain environment-only.
        config = ModelConfig.model_validate(read_json(folder / "model_config.json"))
        engine = Engine(store, args.run_id, config)
        if cmd == "_worker":
            def interrupted(signum, frame):
                raise InterruptedError("Worker stopped by user")
            signal.signal(signal.SIGTERM, interrupted)
            try:
                state = execute_run(engine, resume=args.resume)
                emit(state)
                exit_code = state_exit_code(state)
                jobs.finish(store, args.run_id, exit_code, state["status"])
                return exit_code
            except BaseException as exc:
                state = read_json(engine.run / "run_state.json")
                if state["status"] not in {"failed", "stopped", "budget_exhausted"}:
                    engine.set_status("stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                                      warnings=[f"Worker failed: {type(exc).__name__}: {exc}"])
                jobs.finish(store, args.run_id, 130 if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 2,
                            "stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed")
                raise
        else:
            if args.foreground:
                state = execute_run(engine, resume=True)
                emit(state)
                return state_exit_code(state)
            emit(jobs.start(store, args.run_id, args.config, resume=True))
    elif cmd == "status":
        run_id = args.run_id or store.get("latest_run")
        emit(read_json(store.run_dir(run_id or "") / "run_state.json"))
    elif cmd == "runs":
        emit(jobs.list_runs(store))
    elif cmd == "verify":
        emit(verify_run(store, args.run_id))
    elif cmd == "stop":
        emit(jobs.stop(store, args.run_id))
    elif cmd in {"report", "logs"}:
        folder = store.run_dir(args.run_id)
        emit((folder / ("report.md" if cmd == "report" else "worker.log")).read_text(errors="replace"))
    elif cmd == "doctor":
        from .doctor import check
        emit(check(load_study(args.study) if args.study else None))
    elif cmd == "examples":
        from .examples import create_example
        emit(str(create_example(args.destination, args.design, args.rna_scale, args.r_library)))
    elif cmd == "models":
        config = load_config(args.config)
        if args.model_command == "list":
            emit(config)
        elif args.model_command == "add":
            data = {**PRESETS[args.preset], "name": args.name, "model": args.model, "format_mode": args.format}
            for key in ["base_url", "api_key_env", "kind"]:
                value = getattr(args, "key_env" if key == "api_key_env" else key, None)
                if value:
                    data[key] = value
            config.profiles[args.name] = ModelProfile(**data)
            if args.default or not config.default:
                config.default = args.name
            emit(str(save_config(config, args.config)))
            if args.key:
                from .credentials import set_api_key
                set_api_key(args.name, args.key)
                emit("API key saved to the OS keychain.")
        elif args.model_command == "role":
            if args.profile not in config.profiles:
                raise ValueError("Unknown profile")
            config.roles[args.role] = args.profile
            emit(str(save_config(config, args.config)))
        else:
            from .llm import ModelClient
            from .models import AgentDecision
            ready, profile, reason = readiness(config)
            if not ready:
                raise ValueError(reason)
            client = ModelClient(config, store.state_dir / "model_checks")
            emit(client.json(args.role, "Return JSON: action=continue, explanation=connection test; no data.", {}, AgentDecision))
    elif cmd == "deps":
        return subprocess.call(["Rscript", "--vanilla", str(resource("r/00_install.R")), str(args.library.resolve())])
    elif cmd == "knowledge":
        from .knowledge import import_go, import_reactome
        if args.output.exists():
            raise FileExistsError("Knowledge import refuses to overwrite an existing file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.kind == "go":
            if not args.taxon:
                raise ValueError("GO import requires explicit --taxon")
            import_go(args.input, args.terms, args.species, args.resource_version, args.output,
                      args.source, args.taxon, args.member_type, args.obo)
        else:
            import_reactome(args.input, args.terms, args.species, args.resource_version, args.output, args.source)
        emit(str(args.output.resolve()))


def repl(session):
    print("spliceosome_agent · welcome\nNew study /study; configure a model with models add; help /help.")
    for line in sys.stdin:
        try:
            response = session.handle(line)
            if response == "__quit__":
                return 0
            print(response)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")
    return 0
