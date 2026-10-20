"""Evidence-led local draft assembly; no patient rows are sent to a planning model."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .data import child_environment, read_table, resource
from .io import sha256


def inspect_local(path, rscript="Rscript", r_library=None):
    path = Path(path)
    toolchain = SimpleNamespace(rscript=rscript, r_library=r_library)
    attributes = {}
    with tempfile.TemporaryDirectory(prefix="spliceosome-discovery-") as scratch:
        if path.suffix.lower() == ".rds":
            exported, props = Path(scratch) / "table.csv", Path(scratch) / "attributes.csv"
            response = subprocess.run([rscript, "--vanilla", str(resource("r/05_inspect.R")), str(path), str(exported), str(props)],
                                      env=child_environment(toolchain), capture_output=True, text=True, timeout=90)
            if response.returncode:
                raise ValueError("RDS could not be read or contains duplicate IDs. Check the local input; no object contents were sent to the model.")
            table = read_table(exported, toolchain)
            attributes = dict(pd.read_csv(props, keep_default_na=False).itertuples(index=False, name=None))
            attributes = {k: v for k, v in attributes.items() if v}
        else:
            table = read_table(path, toolchain)
    measurements = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    raw = table.iloc[:, 1:]
    invalid = measurements.isna() & ~raw.isin(["", "NA", "NaN", "nan"])
    numeric = len(table.columns) >= 2 and not invalid.any().any()
    descriptor = {"format": path.suffix.lower(), "n_rows": len(table), "n_columns": len(table.columns),
                  "kind": "numeric_table" if numeric else "possible_metadata"}
    if numeric:
        values = measurements.to_numpy()
        finite = values[np.isfinite(values)]
        descriptor.update(n_features=len(table), n_samples=len(measurements.columns),
                          strictly_positive=bool(len(finite) and (finite > 0).all()),
                          nonfinite=int((~np.isfinite(values)).sum()))
    return table, attributes, descriptor


def propose_inputs(registry, goal, rscript="Rscript", r_library=None, selection=None):
    """Return a local draft and evidence card, never infer log2 from distributions."""
    tables, attributes, summaries, errors, hashes = {}, {}, {}, [], {}
    selected = {ref: path for ref, path in registry.items() if Path(path).suffix.lower() in {".rds", ".csv", ".tsv", ".txt"}}
    if len(selected) > 32:
        return {"status": "needs_information", "message": "Too many files. Specify the matrix and metadata file paths for this analysis.", "alternatives": list(selected)}
    identifiers = set()
    for ref, path in selected.items():
        try:
            hashes[path] = sha256(path)
            tables[ref], attributes[ref], summaries[ref] = inspect_local(path, rscript, r_library)
            if sha256(path) != hashes[path]:
                raise ValueError("Input changed during inspection. Inspect it again.")
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{Path(path).name}: {exc}")
    matches = []
    for matrix_ref, matrix in tables.items():
        if summaries[matrix_ref]["kind"] != "numeric_table":
            continue
        samples = set(matrix.columns[1:])
        for metadata_ref, metadata in tables.items():
            if matrix_ref == metadata_ref:
                continue
            for column in metadata:
                if len(metadata) == len(samples) and not metadata[column].duplicated().any() and set(metadata[column]) == samples:
                    groups = [c for c in metadata if c != column and c.lower() in {"group", "type", "condition"}
                              and metadata[c].nunique() == 2 and not metadata[c].isin(["", "NA", "NaN", "nan"]).any()]
                    for group_column in groups:
                        match = (matrix_ref, metadata_ref, column, group_column)
                        # Prefer named sample IDs over a redundant RDS row-name export.
                        if column == "feature_id" and any(a==matrix_ref and b==metadata_ref and d==group_column for a,b,_,d in matches):
                            continue
                        if column.lower() in {"sample_id", "id"}:
                            matches = [item for item in matches if not (item[0]==matrix_ref and item[1]==metadata_ref and item[2]=="feature_id" and item[3]==group_column)]
                        matches.append(match)
    if selection:
        chosen = (selection["matrix"], selection["metadata"], selection["sample_column"], selection["group_column"])
        if chosen not in matches:
            return {"status": "blocked", "message": "The selected input relationship changed. Inspect it again.", "hashes": hashes}
        matches = [chosen]
    if len(matches) != 1:
        message = ("Multiple datasets found. Choose the matrix and metadata for this analysis." if matches else
                   "No metadata table matches the matrix samples one-to-one. Specify the matrix and matching metadata.")
        return {"status": "blocked" if errors else "needs_information", "message": "\n".join(errors) if errors else message,
                "alternatives": [{"matrix": a, "metadata": b, "sample_column": c, "group_column": d} for a,b,c,d in matches],
                "summaries": summaries, "hashes": hashes}
    matrix_ref, metadata_ref, sample_col, group_col = matches[0]
    metadata = tables[metadata_ref]
    identifiers.update(metadata[sample_col])
    levels = sorted(metadata[group_col].unique())
    subject_col = next((c for c in metadata if c.lower() in {"subject_id", "pair", "patient_id", "donor_id"}), None)
    design, n_units = "unpaired", len(metadata)
    if subject_col:
        identifiers.update(metadata[subject_col])
        blocks = list(metadata.groupby(subject_col, sort=False))
        repeated = metadata[subject_col].duplicated().any()
        if repeated:
            if metadata[subject_col].eq("").any() or not all(len(g) == 2 and set(g[group_col]) == set(levels) for _, g in blocks):
                return {"status": "blocked", "message": "Metadata contains repeated subjects that cannot form one-to-one pairs across the two groups. Check the sample relationships.", "summaries": summaries, "hashes": hashes}
            design, n_units = "paired", len(blocks)
    lower = goal.lower()
    if any(word in lower for word in ["癌栓", "thrombus"]) and not any("thrombus" in c.lower() or "癌栓" in c for c in metadata):
        return {"status": "needs_information", "message": "No thrombus field was found in the metadata. Confirm use of the observed groups or provide the corresponding metadata.", "hashes": hashes, "summaries": summaries}
    domain = "protein" if any(s in lower for s in ["蛋白", "protein", "proteomic"]) else "rna" if any(s in lower for s in ["rna", "转录"]) else None
    name = Path(registry[matrix_ref]).name.lower()
    if domain is None:
        domain = "protein" if "pro" in name else "rna" if "rna" in name else None
    if domain is None:
        return {"status": "needs_information", "message": "Matrix and metadata identified. Specify whether the data are RNA or protein.", "summaries": summaries, "hashes": hashes}
    facts = summaries[matrix_ref]
    assay = {"matrix": registry[matrix_ref], "metadata": registry[metadata_ref], "id_type": "UNKNOWN",
             "columns": {"sample_id": sample_col, "group": group_col, "subject_id": subject_col},
             "preprocessing": "Upstream preprocessing was not provided; normalization history is not inferred"}
    declared = attributes[matrix_ref]
    if declared.get("scale") in ({"log2_abundance"} if domain == "protein" else {"counts", "log2_expression"}):
        assay["scale"] = declared["scale"]
    elif declared.get("scale") == "abundance":
        # Even an explicit raw scale still needs authorization to transform it.
        pass
    if declared.get("preprocessing"):
        assay["preprocessing"] = declared["preprocessing"]
    study = {"schema_version": "2.0", "analysis_mode": "differential", "name": "Protein differential analysis" if domain == "protein" else "RNA differential analysis",
             "design": design, "reference": levels[0], "case": levels[1], "assays": {domain: assay}}
    sources = {"files": {"status": "observed", "evidence": "matrix/metadata sample ID sets match exactly"},
               "columns": {"status": "observed", "value": assay["columns"]},
               "design": {"status": "proposed", "value": design, "n_units": n_units},
               "contrast": {"status": "proposed", "reference": levels[0], "case": levels[1]},
               "scale": {"status": "observed" if assay.get("scale") else "needs_confirmation", "evidence": "input attribute" if assay.get("scale") else "not inferable from numeric range"}}
    card = {"status": "ready" if assay.get("scale") else "needs_scale", "study": study, "goal": goal, "sources": sources,
            "matrix_name": Path(registry[matrix_ref]).name, "metadata_name": Path(registry[metadata_ref]).name,
            "n_features": facts["n_features"], "n_samples": facts["n_samples"], "n_units": n_units,
            "group_counts": {str(k): int(v) for k,v in metadata[group_col].value_counts().sort_index().items()},
            "positive": facts["strictly_positive"], "hashes": hashes, "summaries": summaries}
    if assay.get("scale"):
        card["scale_choice"] = "counts" if assay["scale"] == "counts" else "log2"
    return card, identifiers
