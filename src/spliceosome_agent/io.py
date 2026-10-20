"""Atomic writes and immutable provenance, adapted from the previous io module."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_document(path):
    path = Path(path).expanduser().resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes declared directory: {relative}")
    return path


def artifact(root, path):
    from .models import Artifact
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("Artifact must be a regular file inside this run")
    return Artifact(path=str(path.relative_to(root)), sha256=sha256(path), bytes=path.stat().st_size)


def verify_artifacts(root, records):
    for item in records:
        item = item.model_dump() if hasattr(item, "model_dump") else item
        path = inside(root, item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact missing or changed: {item['path']}")


def environment():
    packages = {}
    for package in ["spliceosome-agent", "numpy", "pandas", "langgraph", "pydantic", "textual", "matplotlib"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def code_fingerprint():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".R", ".md", ".json", ".tcss"}}
