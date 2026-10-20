"""Explicit sample identities, scale declarations and RDS table ingestion."""
from __future__ import annotations

import csv
import os
import subprocess
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd


def resource(name):
    return Path(str(files("spliceosome_agent").joinpath("resources", name)))


def child_environment(study):
    # Trusted numerical tools still do not need cloud credentials or tracing.
    env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SystemRoot"] if key in os.environ}
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
    if study.r_library:
        env["R_LIBS_USER"] = study.r_library
    return env


def read_table(path, study, converted=None):
    path = Path(path)
    if path.suffix.lower() == ".rds":
        if converted is None:
            raise ValueError("RDS ingestion requires an isolated conversion output")
        converted = Path(converted)
        converted.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([study.rscript, "--vanilla", str(resource("r/01_export.R")), str(path), str(converted)],
                                env=child_environment(study), capture_output=True, text=True)
        if result.returncode:
            raise ValueError("RDS conversion failed; only matrix/data.frame objects are supported: " + result.stderr[-1000:])
        path = converted
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt", ".gaf"} else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=separator), None)
    if header is None:
        raise ValueError("Input table is empty")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate input columns")
    return pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def inspect_assay(study, domain, run):
    assay = study.assays[domain]
    out = Path(run) / "inputs" / domain
    out.mkdir(parents=True, exist_ok=True)
    raw = read_table(assay.matrix, study, out / "matrix_export.csv")
    meta = read_table(assay.metadata, study, out / "metadata_export.csv")
    if raw.empty or meta.empty:
        raise ValueError("Matrix features and metadata samples must not be empty")
    if len(raw.columns) < 2:
        raise ValueError("Matrix must contain a feature ID column and sample columns")
    identifiers = raw.iloc[:, 0]
    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Feature IDs must be nonblank and unique; declare an upstream aggregation policy")
    x = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    # Missing is explicit, but malformed values are never silently converted to missing.
    missing_tokens = {"", "NA", "NaN", "nan"}
    malformed = x.isna() & ~raw.iloc[:, 1:].isin(missing_tokens)
    if malformed.any().any() or np.isinf(x.to_numpy()).any():
        raise ValueError("Matrix contains invalid numeric or infinite measurements")
    x.index = identifiers
    x.index.name = "feature_id"
    c = assay.columns
    required = [c.sample_id, c.group, *study.covariates]
    if c.subject_id:
        required.append(c.subject_id)
    if not set(required).issubset(meta.columns):
        raise ValueError("Metadata lacks declared mapping or covariate columns")
    if meta[c.sample_id].eq("").any() or meta[c.sample_id].duplicated().any():
        raise ValueError("Sample IDs must be nonblank and unique")
    if set(meta[c.sample_id]) != set(x.columns):
        raise ValueError("Matrix and metadata sample IDs must match exactly")
    if set(meta[c.group]) != {study.reference, study.case}:
        raise ValueError("Metadata groups do not match the two approved labels")
    md = pd.DataFrame({"sample_id": meta[c.sample_id], "group": meta[c.group],
                       "subject_id": meta[c.subject_id] if c.subject_id else meta[c.sample_id]})
    if md.subject_id.eq("").any():
        raise ValueError("Subject IDs may not be missing")
    if study.design == "paired":
        for _, group in md.groupby("subject_id", sort=False):
            if len(group) != 2 or set(group.group) != {study.reference, study.case}:
                raise ValueError("Each paired subject must have exactly one sample per group")
        n_units = md.subject_id.nunique()
        if n_units < study.min_units:
            raise ValueError("Insufficient biological pairs")
    else:
        if md.subject_id.duplicated().any():
            raise ValueError("Repeated subjects in an unpaired design are unsupported")
        if md.groupby("group").size().min() < study.min_units:
            raise ValueError("Insufficient independent subjects per group")
        n_units = len(md)
    if set(study.covariates) & {"sample_id", "group", "subject_id", "condition", "subject"}:
        raise ValueError("Covariate names conflict with canonical metadata fields")
    for key, kind in study.covariates.items():
        md[key] = meta[key]
        if md[key].eq("").any() or md[key].isin(["NA", "NaN", "nan"]).any():
            raise ValueError("Missing covariate values require a new explicit plan")
        if kind == "continuous":
            md[key] = pd.to_numeric(md[key], errors="raise")
            if not np.isfinite(md[key]).all():
                raise ValueError("Non-finite covariate")
    if assay.scale == "counts":
        values = x.to_numpy()
        finite = values[np.isfinite(values)]
        if (finite < 0).any() or (finite != np.floor(finite)).any() or (finite > np.iinfo(np.int32).max).any():
            raise ValueError("DESeq2 requires nonnegative integer counts within R integer range")
    if assay.transform == "log2":
        finite = x.to_numpy()[np.isfinite(x.to_numpy())]
        if (finite <= 0).any():
            raise ValueError("Log2 conversion requires strictly positive abundance; no pseudocount is added")
        x = np.log2(x)
    if study.missing_policy == "error" and x.isna().any().any():
        raise ValueError("Missing measurements; approved policy is error")
    md = md.set_index("sample_id").loc[list(x.columns)].rename_axis("sample_id").reset_index()
    summary = {"n_samples": len(md), "n_units": int(n_units), "n_features": len(x),
            "n_features_with_missing": int(x.isna().any(axis=1).sum()), "scale": assay.scale,
            "preprocessing": assay.preprocessing, "cohort_id": assay.cohort_id,
            **({"input_transform": "log2", "analysis_scale": "log2_abundance" if domain == "protein" else "log2_expression"} if assay.transform == "log2" else {}),
            "group_counts": {str(k): int(v) for k, v in md.groupby("group").size().items()}}

    return x, md, summary

def private_ids(study, run=None):
    identifiers = set()
    for domain, assay in study.assays.items():
        if run and (Path(run) / "inputs" / domain / "metadata.csv").exists():
            md = pd.read_csv(Path(run) / "inputs" / domain / "metadata.csv", dtype=str, keep_default_na=False)
            identifiers.update(md.sample_id)
            identifiers.update(md.subject_id)
        else:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="spliceosome-private-") as temporary:
                md = read_table(assay.metadata, study, Path(temporary) / "metadata.csv")
            for column in [assay.columns.sample_id, assay.columns.subject_id]:
                if column and column in md:
                    identifiers.update(md[column])
    return identifiers


def prepare_assay(study, domain, run):
    x, md, summary = inspect_assay(study, domain, run)
    out = Path(run) / "inputs" / domain
    x.to_csv(out / "matrix.csv", index=True)
    md.to_csv(out / "metadata.csv", index=False)
    return summary
