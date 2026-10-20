"""Reviewed supplemental code, executed as a restricted local subprocess.

No container runtime is involved. Proposals are hash-bound to their source, the
execution toolchain and the frozen plan inputs; each file needs an explicit
approval before it runs. The child process starts in the proposal work folder
with a credentials-free allowlisted environment, writes only inside its own
output tree, and is bounded by a wall-clock timeout and POSIX resource limits.
Static checks reject obvious network, subprocess, deletion and installation
patterns and verify Python/R imports; they filter unintended code but are not a
security boundary and cannot stop deliberately obfuscated code.
"""
from __future__ import annotations

import ast
import difflib
import importlib.util
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no POSIX resource limits
    resource = None

from .io import artifact, now, object_hash, read_json, sha256, write_json
from .planning import input_paths

TIMEOUT_SECONDS = 600
ADDRESS_SPACE_BYTES = 4 * 1024 ** 3

FORBIDDEN_CODE = re.compile(
    r"\b(?:subprocess|os\.system|os\.popen|os\.spawn\w*|shutil\.rmtree|socket|requests|"
    r"urllib\.request|httpx|pip\s+install|install\.packages|file\.remove|unlink\s*\(|"
    r"system2?\s*\(|rm\s+-|curl\s+|wget\s+)\b",
    re.I,
)


def toolchain(study):
    """Exact interpreters that would run reviewed code; proposals bind to this."""
    rscript = shutil.which(study.rscript)
    version = None
    if rscript:
        try:
            probe = subprocess.run([rscript, "--version"], capture_output=True, text=True, timeout=30)
            version = (probe.stdout.strip().splitlines() or [None])[0] if probe.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            version = None
    return {"python": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
            "rscript": str(Path(rscript).resolve()) if rscript else None, "r_version": version}


def _python_findings(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Python syntax error: {exc}"]
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    missing = []
    for name in sorted(modules):
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(name)
    return [f"Python packages not importable in this environment: {', '.join(missing)}"] if missing else []


def _r_packages(source):
    names = re.findall(r"\b(?:library|require|requireNamespace)\s*\(\s*['\"]?([A-Za-z][A-Za-z0-9.]*)", source)
    names += re.findall(r"\b([A-Za-z][A-Za-z0-9.]*):::{0,1}", source)
    return sorted(set(names))


def _r_findings(source, study):
    rscript = shutil.which(study.rscript)
    if not rscript:
        return [f"Rscript not found: {study.rscript}"]
    findings = []
    try:
        parsed = subprocess.run([rscript, "--vanilla", "-e", "invisible(parse(text = readLines('stdin')))"],
                                input=source, capture_output=True, text=True, timeout=60)
        if parsed.returncode:
            detail = (parsed.stderr.strip().splitlines() or ["unknown error"])[-1][:300]
            findings.append("R syntax error: " + detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        findings.append(f"R syntax check failed: {type(exc).__name__}")
    packages = _r_packages(source)
    if packages:
        expression = ("pkgs <- commandArgs(TRUE); missing <- pkgs[!vapply(pkgs, requireNamespace, "
                      "logical(1), quietly=TRUE)]; if (length(missing)) cat(paste(missing, collapse=','))")
        try:
            checked = subprocess.run([rscript, "--vanilla", "-e", expression, *packages],
                                     env=_environment(study), capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            findings.append(f"R package check failed: {type(exc).__name__}")
        else:
            if checked.stdout.strip():
                findings.append("R packages not installed: " + checked.stdout.strip())
    return findings


def check_code(language, source, study):
    """Static preflight recorded on the proposal and re-run before execution."""
    findings = []
    match = FORBIDDEN_CODE.search(source)
    if match:
        findings.append(f"Forbidden pattern: {match.group(0)!r}")
    findings += _python_findings(source) if language == "python" else _r_findings(source, study)
    return {"passed": not findings, "findings": findings}


def _environment(study, **extra):
    """Allowlisted child environment: no credential or token variables ever pass through."""
    env = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "TZ", "SystemRoot"] if key in os.environ}
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
    if study.r_library:
        env["R_LIBS_USER"] = study.r_library
    env.update({key: str(value) for key, value in extra.items()})
    return env


def _limits():
    if resource is None:
        return
    for kind, value in [(resource.RLIMIT_CPU, TIMEOUT_SECONDS), (resource.RLIMIT_AS, ADDRESS_SPACE_BYTES)]:
        try:
            _, hard = resource.getrlimit(kind)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(kind, (value, hard))
        except (OSError, ValueError):
            continue


def propose_code(run, plan, role, language, source, previous=None):
    if not plan.generated_code:
        raise PermissionError("Generated code is disabled in this approved plan")
    if language not in {"python", "r"}:
        raise ValueError("Only Python/R supplemental code is supported")
    payload = {"role": role, "language": language, "source_hash": object_hash(source),
               "plan_hash": object_hash(plan), "inputs": plan.input_hashes,
               "toolchain": toolchain(plan.study),
               "limits": {"timeout_seconds": TIMEOUT_SECONDS, "address_space_mb": ADDRESS_SPACE_BYTES // (1024 ** 2)},
               "checks": check_code(language, source, plan.study),
               "purpose": "supplemental_unvalidated_output_not_a_replacement_for_registered_statistics"}
    proposal_id = object_hash(payload)
    folder = Path(run) / "code" / proposal_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("analysis.py" if language == "python" else "analysis.R")
    if path.exists() and path.read_text() != source:
        raise ValueError("Code hash collision or changed code proposal")
    path.write_text(source, encoding="utf-8")
    payload["file_sha256"] = sha256(path)
    payload["proposal_id"] = proposal_id
    write_json(folder / "proposal.json", payload)
    before = Path(previous).read_text() if previous else ""
    diff = "\n".join(difflib.unified_diff(before.splitlines(), source.splitlines(), fromfile="previous", tofile=path.name, lineterm=""))
    (folder / "review.diff").write_text(diff + "\n", encoding="utf-8")
    return payload


def checked_proposal(run, proposal_id):
    if len(proposal_id) != 64 or any(c not in "0123456789abcdef" for c in proposal_id):
        raise ValueError("Invalid code proposal ID")
    folder = Path(run) / "code" / proposal_id
    proposal = read_json(folder / "proposal.json")
    path = folder / ("analysis.py" if proposal["language"] == "python" else "analysis.R")
    if sha256(path) != proposal["file_sha256"] or object_hash(path.read_text()) != proposal["source_hash"]:
        raise PermissionError("Code changed since proposal; review a new proposal")
    canonical = {key: value for key, value in proposal.items() if key not in {"proposal_id", "file_sha256"}}
    if object_hash(canonical) != proposal_id:
        raise PermissionError("Code proposal was modified")
    return folder, proposal, path


def execute_code(run, plan, proposal_id):
    folder, proposal, source = checked_proposal(run, proposal_id)
    if proposal["plan_hash"] != object_hash(plan):
        raise PermissionError("Code belongs to a different plan")
    if proposal["toolchain"] != toolchain(plan.study):
        raise PermissionError("Execution toolchain changed since the proposal; create a new plan")
    for path, digest in plan.input_hashes.items():
        if sha256(path) != digest:
            raise PermissionError("Input changed before code execution")
    checks = check_code(proposal["language"], source.read_text(), plan.study)
    if not checks["passed"]:
        result = {"status": "blocked", "proposal_id": proposal_id, "artifacts": [],
                  "scientific_status": "supplemental_unvalidated",
                  "reason": "Static safety checks failed; nothing was executed",
                  "findings": checks["findings"]}
        write_json(folder / "result.json", result)
        return result
    # Each proposal runs inside its own trees: inputs are read-only by contract, writes belong to output/.
    output = folder / "output"
    work = folder / "work"
    home = folder / "home"
    for path in [output, work, home]:
        path.mkdir(exist_ok=True)
    extra = {"HOME": home, "TMPDIR": home, "SPLICEOSOME_OUTPUT_DIR": output, "SPLICEOSOME_WORK_DIR": work}
    for index, path in enumerate(input_paths(plan.study)):
        extra[f"SPLICEOSOME_INPUT_{index:03d}"] = Path(path).resolve()
    env = _environment(plan.study, **extra)
    if proposal["language"] == "python":
        argv = [sys.executable, str(source)]
    else:
        rscript = shutil.which(plan.study.rscript)
        if not rscript:
            raise RuntimeError("Rscript is unavailable; cannot execute reviewed R code")
        argv = [rscript, "--vanilla", str(source)]
    write_json(folder / "command.json", {"argv": argv, "cwd": str(work), "environment": sorted(env),
                                         "credential_env": "none"})
    result = {}
    try:
        with (folder / "execution.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=_limits if os.name == "posix" else None)
            try:
                exit_code = process.wait(timeout=TIMEOUT_SECONDS)
                result = {"status": "complete" if exit_code == 0 else "failed", "exit_code": exit_code}
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
                result = {"status": "failed", "reason": "timeout"}
    except OSError as exc:
        result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    result.update({"proposal_id": proposal_id, "scientific_status": "supplemental_unvalidated",
                   "artifacts": [artifact(run, p).model_dump() for p in sorted((folder / "output").rglob("*")) if p.is_file()]})
    write_json(folder / "result.json", result)
    return result
