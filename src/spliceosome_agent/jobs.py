"""Detached POSIX supervision with cross-platform PID/create-time identity checks."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .io import now, read_json, write_json


def identity(pid):
    try:
        process = psutil.Process(int(pid))
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {"pid": process.pid, "created": process.create_time()}
    except (psutil.Error, ValueError, TypeError):
        return None


def same_process(record):
    return bool(record and identity(record.get("pid")) == record)


def start(store, run_id, config_path=None, resume=False):
    from .engine import Engine, valid_run_id
    valid_run_id(run_id)
    engine = Engine(store, run_id)
    engine._validate()  # Same gate as foreground execution, before starting a child.
    folder = store.run_dir(run_id)
    record_path = folder / "job.json"
    if record_path.exists() and same_process(read_json(record_path).get("identity")):
        raise ValueError("This run already has a live worker")
    argv = [sys.executable, "-m", "spliceosome_agent", "--workspace", str(store.workspace)]
    if config_path:
        argv += ["--config", str(Path(config_path).resolve())]
    argv += ["_worker", run_id]
    if resume:
        argv += ["--resume"]
    with (folder / "worker.log").open("ab") as log:
        worker = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, cwd=folder)
    record = {"run_id": run_id, "identity": identity(worker.pid), "started_at": now(), "argv": argv}
    write_json(record_path, record)
    return record


def stop(store, run_id):
    folder = store.run_dir(run_id)
    record = read_json(folder / "job.json")
    if not same_process(record.get("identity")):
        raise ValueError("Worker identity changed or worker already exited; no signal sent")
    pid = record["identity"]["pid"]
    if os.getpgid(pid) != pid:
        raise ValueError("Worker process group identity does not match")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while same_process(record["identity"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    if (folder / "job.json").exists():
        record = read_json(folder / "job.json")
    record["stop_requested_at"] = now()
    record["status"] = "stop_requested" if same_process(record["identity"]) else "stopped"
    write_json(folder / "job.json", record)
    return record


def finish(store, run_id, exit_code, status):
    folder = store.run_dir(run_id)
    path = folder / "job.json"
    record = read_json(path) if path.exists() else {"run_id": run_id, "identity": identity(os.getpid())}
    record.update({"exit_code": exit_code, "ended_at": now(), "status": status})
    write_json(path, record)


def list_runs(store):
    rows = []
    for path in sorted(store.runs_dir.glob("*/run_state.json"), reverse=True):
        record = read_json(path)
        job = path.parent / "job.json"
        if job.exists():
            record["worker_alive"] = same_process(read_json(job).get("identity"))
            if record["status"] == "running" and not record["worker_alive"]:
                record["status"] = "stalled"
        rows.append(record)
    return rows
