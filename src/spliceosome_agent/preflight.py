"""Local, non-inferential probes. Raw data never enter the planning model context."""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from . import doctor
from .data import child_environment, inspect_assay, read_table, resource
from .io import environment, object_hash, read_json, sha256, write_json
from .knowledge import mapping_for, prepare_candidates
from .models import ProbeReport

SUPPORTED = {".csv", ".tsv", ".txt", ".rds", ".yaml", ".yml"}
SAFE_COLUMNS = {"sample_id", "subject_id", "group", "ID", "Type", "Pair", "batch", "age", "sex",
                "gene_id", "member_type", "source", "version", "species", "feature_id"}


def discover(paths):
    """Only walk explicitly supplied roots; never follow symlinks or hidden subtrees."""
    found = set()
    for supplied in paths:
        root = Path(supplied).expanduser().absolute()
        if root.is_symlink():
            continue
        if root.is_file():
            if root.suffix.lower() in SUPPORTED:
                found.add(str(root.resolve()))
        elif root.is_dir():
            for base, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(base) / d).is_symlink())
                for name in sorted(files):
                    p = Path(base) / name
                    if not name.startswith(".") and not p.is_symlink() and p.suffix.lower() in SUPPORTED:
                        found.add(str(p.resolve()))
                    if len(found) >= 2000:
                        raise ValueError("Too many candidate files; specify a smaller data directory")
        else:
            raise ValueError(f"Input path does not exist: {supplied}")
    return sorted(found)


def inspect_file(path, study=None):
    """Only allowlisted schema information, never sample columns, row values or names."""
    path = Path(path)
    summary = {"format": path.suffix.lower(), "bytes": path.stat().st_size}
    if path.suffix.lower() in {".yaml", ".yml"}:
        summary["kind"] = "configuration; load locally with /study to declare its inputs"
        return summary
    if path.suffix.lower() == ".rds":
        from .discovery import inspect_local
        table, _, descriptor = inspect_local(path, study.rscript if study else "Rscript", study.r_library if study else None)
        summary.update(descriptor)
    else:
        table = read_table(path, study)
    columns = set(table.columns)
    summary.update(n_rows=len(table), n_columns=len(table.columns),
                   declared_columns=sorted(columns & SAFE_COLUMNS))
    if {"gene_id", "member_type", "source", "version", "species"}.issubset(columns):
        summary["kind"] = "candidate_table"
    elif columns & {"sample_id", "subject_id", "ID", "Pair"}:
        summary["kind"] = "possible_metadata"
    else:
        summary["kind"] = "possible_matrix_or_mapping"
    return summary


def check_environment(study):
    checked = doctor.check(study)
    required = {"jsonlite"}
    required.update("DESeq2" if a.scale == "counts" else "limma" for a in study.assays.values())
    errors = []
    if not checked["rscript"]:
        errors.append("Rscript is unavailable")
    for package in sorted(required):
        if checked["r_packages"].get(package) in {None, "NOT_INSTALLED"}:
            errors.append(f"Required R package unavailable: {package}")
    from matplotlib import font_manager
    for path in ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
    try:
        font = font_manager.findfont(study.font, fallback_to_default=False)
    except ValueError:
        font = font_manager.findfont("DejaVu Sans", fallback_to_default=False) if study.allow_font_fallback else None
    if font is None:
        errors.append("Requested font unavailable; explicitly allow a fallback or install the font")
    return {"python": environment(), "rscript": checked["rscript"],
            "r_packages": {p: checked["r_packages"].get(p) for p in sorted(required)}, "font": font}, errors


def run_preflight(study, workspace, *, cache_dir=None):
    from .planning import input_paths
    from .tools import validate_alignment
    folder = Path(workspace) / "probes" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    hashes, summary, errors, warnings = {}, {}, [], []
    for path in input_paths(study):
        try:
            hashes[path] = sha256(path)
        except OSError as exc:
            errors.append(f"Input unavailable: {path}: {type(exc).__name__}")
    os.environ.setdefault("MPLCONFIGDIR", str((Path(cache_dir) if cache_dir is not None else Path(workspace) / "cache") / "matplotlib"))
    env, env_errors = check_environment(study)
    errors.extend(env_errors)
    # Canonical metadata/RDS exports are disposable local cache; only summary snapshots persist.
    with tempfile.TemporaryDirectory(prefix="validation-", dir=folder) as scratch:
        for domain in study.assays:
            try:
                x, md, info = inspect_assay(study, domain, scratch)
                out = Path(scratch) / "inputs" / domain
                md.to_csv(out / "metadata.csv", index=False)
                if study.analysis_mode == "candidate_screening":
                    mapping = mapping_for(study, domain, list(x.index))
                    info["mapping_counts"] = {str(k): int(v) for k, v in mapping.mapping_status.value_counts().items()}
                    if any(k != "mapped" for k in info["mapping_counts"]):
                        warnings.append(f"{domain}: ambiguous or unmapped features retained in coverage")
                summary[domain] = info
                if not env_errors:
                    job = {"metadata": str(out / "metadata.csv"), "output": str(out / "design.json"),
                           "design": study.design, "reference": study.reference, "case": study.case,
                           "covariates": study.covariates}
                    write_json(out / "job.json", job)
                    response = subprocess.run([study.rscript, "--vanilla", str(resource("r/04_preflight.R")), str(out / "job.json")],
                                              env=child_environment(study), capture_output=True, text=True, timeout=90)
                    if response.returncode:
                        # Local-only error; never forward stderr or raw values to a model.
                        raise ValueError("Design validation failed (rank deficiency, constant/confounded covariates or no residual degrees of freedom)")
                    info["design"] = read_json(out / "design.json")
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{domain}: {exc}")
        validate_alignment(study, scratch, summary, errors)
        if study.analysis_mode == "candidate_screening":
            try:
                summary["candidates"] = prepare_candidates(study, scratch)
            except (ValueError, OSError) as exc:
                errors.append(f"candidates: {exc}")
    for path, checksum in hashes.items():
        try:
            if sha256(path) != checksum:
                errors.append("Input changed during preflight; inspect again")
        except OSError:
            errors.append("Input disappeared during preflight")
    result = ProbeReport(status="blocked" if errors else "ready_for_confirmation", input_hashes=hashes,
                         summary=summary, environment=env, errors=errors, warnings=warnings)
    write_json(folder / "preflight.json", result)
    return result


def verify_environment(report, study):
    current, errors = check_environment(study)
    if errors or object_hash(current) != object_hash(report.environment):
        raise ValueError("Environment changed since preflight; create a new plan and confirm again")


def model_summary(report):
    """No paths, labels, exception text or feature identifiers leave local preflight."""
    return {"status": report.status, "error_count": len(report.errors), "warning_count": len(report.warnings),
            "assays": {domain: {key: info[key] for key in ["n_samples", "n_units", "n_features", "n_features_with_missing"] if key in info}
                       for domain, info in report.summary.items() if domain in {"rna", "protein"}}}


def inventory_identifiers(registry, rscript="Rscript", r_library=None):
    """Recognize only standard metadata columns locally; never return their values to a model."""
    identifiers = set()
    for path in registry.values():
        path = Path(path)
        if path.suffix.lower() == ".rds":
            from .discovery import inspect_local
            try:
                table, _, _ = inspect_local(path, rscript, r_library)
                if {"group", "Type"} & set(table.columns):
                    for column in set(table.columns) & {"sample_id", "subject_id", "ID", "Pair"}:
                        identifiers.update(v for v in table[column] if v)
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass  # The bootstrap file probe records the error separately.
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            header = pd.read_csv(path, sep=separator, nrows=0, encoding="utf-8-sig").columns
            if not ({"group", "Type"} & set(header)):
                continue
            columns = sorted(set(header) & {"sample_id", "subject_id", "ID", "Pair"})
            if columns:
                table = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False,
                                    usecols=columns, encoding="utf-8-sig")
                identifiers.update(v for col in columns for v in table[col] if v)
        except (ValueError, OSError):
            continue  # The explicit file probe/preflight will report unreadable/malformed data.
    return identifiers
