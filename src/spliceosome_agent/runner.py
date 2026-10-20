"""A frozen, inspectable entrypoint for the registered scientific workflow."""
from __future__ import annotations

import runpy
from pathlib import Path

from .io import object_hash, sha256, write_json, read_json


def source_for(plan, *, flat=False):
    parent = "folder.parent" if flat else "folder.parent.parent"
    return ("# Generated from an explicitly approved plan. No free-form statistical logic.\n"
            "from spliceosome_agent.io import object_hash\n\n"
            "def run(engine, resume=False):\n"
            f"    if object_hash(engine.plan) != {object_hash(plan)!r}:\n"
            "        raise ValueError('Runner does not match the approved plan')\n"
            "    return engine.execute(resume=resume)\n\n"
            "if __name__ == '__main__':\n"
            "    import sys\n"
            "    from pathlib import Path\n"
            "    from spliceosome_agent.cli import main\n"
            "    folder = Path(__file__).resolve().parent\n"
            f"    sys.exit(main(['--workspace', str({parent}), 'resume', folder.name, '--foreground']))\n")


def generate(engine):
    # Engine.create has already required approval and checked inputs.
    plan = engine.plan
    path = engine.run / "run_analysis.py"
    path.write_text(source_for(plan, flat=engine.store.flat), encoding="utf-8")
    write_json(engine.run / "runner.json", {"plan_hash": object_hash(plan), "source_sha256": sha256(path),
                                           "entrypoint": "run_analysis.py:run", "kind": "registered_workflow"})
    if plan.preflight:
        write_json(engine.run / "preflight.json", plan.preflight)


def verify(engine):
    if engine.plan.schema_version == "1.0" and not (engine.run / "runner.json").exists():
        return  # Legacy read path; no new files are written into historical runs.
    source = engine.run / "run_analysis.py"
    record = read_json(engine.run / "runner.json")
    if (record["plan_hash"] != object_hash(engine.plan) or sha256(source) != record["source_sha256"]
            or source.read_text(encoding="utf-8") != source_for(engine.plan, flat=engine.store.flat)):
        raise ValueError("Generated runner changed; execution/resume rejected")


def execute_run(engine, resume=False):
    engine._validate()
    verify(engine)
    if not (engine.run / "runner.json").exists():
        return engine.execute(resume=resume)
    entry = runpy.run_path(str(engine.run / "run_analysis.py"))
    return entry["run"](engine, resume=resume)
