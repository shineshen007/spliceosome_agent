"""SQLite session history, approvals and process-independent event records."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

from .io import now, object_hash, write_json
from .models import AnalysisPlan, ApprovalRecord


class Store:
    def __init__(self, workspace):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        marker = self.workspace / "state" / "workspace.json"
        if (self.workspace / "state").is_symlink() or marker.is_symlink():
            raise ValueError("Workspace state must not be a symbolic link")
        legacy = any((self.workspace / name).exists() for name in ("sessions.sqlite3", "runs", "plans"))
        if marker.exists() and legacy:
            raise ValueError("Conflicting workspace layouts; retain both locations and choose a separate workspace")
        if marker.exists():
            self._check_layout(marker)
            self.flat = True
        elif legacy:
            if (self.workspace / "state" / "sessions.sqlite3").exists():
                raise ValueError("Conflicting workspace databases; no files were moved")
            self.flat = False
        else:
            state = self.workspace / "state"
            if state.exists():
                # Another initializer may have atomically published the directory.
                if not marker.exists():
                    raise ValueError("Unrecognized state directory; workspace.json is required")
            else:
                # Publish a complete marker atomically, without overwriting an existing
                # state directory. Concurrent launches then read the same layout.
                with tempfile.TemporaryDirectory(prefix="state-init-", dir=self.workspace) as scratch:
                    write_json(Path(scratch) / "workspace.json", {"version": 1, "layout": "flat"})
                    try:
                        os.rename(scratch, state)
                    except OSError:
                        if not marker.exists():
                            raise
            self._check_layout(marker)
            self.flat = True
        self.state_dir = self.workspace / "state" if self.flat else self.workspace
        self.runs_dir = self.workspace if self.flat else self.workspace / "runs"
        self.cache_dir = self.state_dir / ("cache" if self.flat else ".cache")
        self.path = self.state_dir / "sessions.sqlite3"
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS plans(hash TEXT PRIMARY KEY, payload TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approvals(hash TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)

    @staticmethod
    def _check_layout(marker):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Invalid workspace layout marker; no automatic repair performed") from exc
        if (record != {"version": 1, "layout": "flat"}
                or type(record.get("version")) is not int):
            raise ValueError("Unsupported workspace layout marker; no automatic repair performed")

    def run_dir(self, run_id):
        from .engine import valid_run_id
        run_id = valid_run_id(run_id)
        if self.flat and run_id.casefold() in {"state", "runs", "plans", "sessions.sqlite3"}:
            raise ValueError("Run ID is reserved for workspace management")
        return self.runs_dir / run_id

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def save_plan(self, plan: AnalysisPlan):
        digest = object_hash(plan)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO plans VALUES(?,?,?)", (digest, plan.model_dump_json(), now()))
            if plan.parent_hash:
                db.execute("DELETE FROM approvals WHERE hash=?", (plan.parent_hash,))
        write_json(self.state_dir / "plans" / f"{digest}.json", plan)
        self.set("draft_dirty", False)
        self.set("current_plan", digest)
        self.event("planning", "plan_created", {"plan_hash": digest, "version": plan.version})
        return digest

    def load_plan(self, digest=None):
        digest = digest or self.get("current_plan")
        with self.connect() as db:
            row = db.execute("SELECT payload FROM plans WHERE hash=?", (digest,)).fetchone()
        if not row:
            raise ValueError("No such plan; create or import a plan first")
        plan = AnalysisPlan.model_validate_json(row[0])
        if object_hash(plan) != digest:
            raise ValueError("Stored plan integrity failed")
        return plan

    def list_plans(self):
        """Every stored plan version, newest first, with its live approval state."""
        current = self.get("current_plan")
        with self.connect() as db:
            rows = db.execute("SELECT hash, payload, created FROM plans ORDER BY created DESC").fetchall()
            approved = {row[0] for row in db.execute("SELECT hash FROM approvals").fetchall()}
            event_rows = db.execute("SELECT payload FROM events WHERE payload LIKE '%\"approved\"%'").fetchall()
        approved_ever = set()
        for (payload,) in event_rows:
            record = json.loads(payload)
            if record.get("kind") == "approved":
                digest = (record.get("payload") or {}).get("plan_hash")
                if digest:
                    approved_ever.add(digest)
        records = []
        parents = set()
        for digest, payload, created in rows:
            plan = AnalysisPlan.model_validate_json(payload)
            if plan.parent_hash:
                parents.add(plan.parent_hash)
            records.append({"plan_hash": digest, "version": plan.version, "goal": plan.goal,
                            "created_at": created, "current": digest == current,
                            "approved": digest in approved, "superseded": digest in parents,
                            "approved_ever": digest in approved_ever})
        return records

    def resolve_plan(self, prefix):
        """Match a plan hash by unique hexadecimal prefix; ambiguity fails closed."""
        prefix = (prefix or "").strip().lower()
        if not prefix or any(c not in "0123456789abcdef" for c in prefix):
            raise ValueError("Provide a hexadecimal plan-hash prefix")
        with self.connect() as db:
            rows = [row[0] for row in db.execute("SELECT hash FROM plans WHERE hash LIKE ?", (prefix + "%",)).fetchall()]
        if not rows:
            raise ValueError(f"No stored plan matches prefix {prefix!r}")
        if len(rows) > 1:
            raise ValueError(f"Prefix {prefix!r} matches {len(rows)} plans; provide more characters")
        return rows[0]

    def is_approved(self, digest):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM approvals WHERE hash=?", (digest,)).fetchone() is not None

    def approve(self, digest):
        plan = self.load_plan(digest)
        if self.get("current_plan") != digest:
            raise ValueError("Only the current plan can be approved; old confirmation expired")
        if self.get("draft_dirty", False):
            raise PermissionError("Draft changed; inspect and confirm the revised plan")
        from .planning import verify_inputs
        verify_inputs(plan)
        if plan.preflight:
            from .preflight import verify_environment
            verify_environment(plan.preflight, plan.study)
        record = ApprovalRecord(plan_hash=digest, plan_version=plan.version, approved_at=now(),
                                **({"schema_version": "1.1", "execution_scope": "registered_tools_with_automatic_code"}
                                   if plan.schema_version == "1.1" else {}))
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO approvals VALUES(?,?)", (digest, record.model_dump_json()))
        self.event("planning", "approved", record.model_dump())
        return record

    def require_approval(self, digest):
        plan = self.load_plan(digest)
        with self.connect() as db:
            row = db.execute("SELECT payload FROM approvals WHERE hash=?", (digest,)).fetchone()
        if not row:
            raise PermissionError("Plan has not been approved, or approval expired")
        approval = ApprovalRecord.model_validate_json(row[0])
        if approval.plan_hash != object_hash(plan) or approval.plan_version != plan.version:
            raise PermissionError("Approval does not match this plan")
        return approval

    def cancel(self, digest):
        with self.connect() as db:
            db.execute("DELETE FROM approvals WHERE hash=?", (digest,))
        self.event("planning", "cancelled", {"plan_hash": digest})

    def set(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def event(self, session, kind, payload=None):
        record = {"at": now(), "kind": kind, "payload": payload or {}}
        with self.connect() as db:
            db.execute("INSERT INTO events(session,payload) VALUES(?,?)", (session, json.dumps(record, ensure_ascii=False)))
        return record

    def events(self, session, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE session=? ORDER BY seq DESC LIMIT ?", (session, limit)).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

