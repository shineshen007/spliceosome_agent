"""Registered numerical tools and full-evidence integration; no model shell dispatch."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data import child_environment, prepare_assay, resource
from .io import artifact, read_json, write_json
from .knowledge import mapping_for, prepare_candidates
from .models import EvidenceRecord, ToolResult


TOOL_PERMISSIONS = {
    "qc": {"reads": "declared inputs", "writes": "run/inputs", "subprocess": "RDS export only"},
    "knowledge": {"reads": "declared candidate/mapping snapshots", "writes": "run/knowledge", "network": False},
    "differential": {"reads": "canonical run inputs", "writes": "run/<domain>", "subprocess": "packaged Rscript"},
    "integration": {"reads": "validated feature results", "writes": "run/integration", "network": False},
    "critic": {"reads": "registered results", "writes": "run/critic", "network": False},
}


def bh(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(ranked, 1)
    return adjusted


def validate_differential(table, study):
    required = {"feature_id", "effect", "se", "ci_low", "ci_high", "pvalue", "padj", "status", "selected", "direction", "n_units"}
    if not required.issubset(table) or table.feature_id.duplicated().any():
        raise ValueError("Invalid differential table contract")
    tested = table.pvalue.notna()
    if not table.loc[tested, "pvalue"].between(0, 1).all():
        raise ValueError("p-values outside [0,1]")
    if not np.allclose(table.loc[tested, "padj"], bh(table.loc[tested, "pvalue"]), atol=1e-12, rtol=1e-10):
        raise ValueError("BH differs from full-feature family")
    estimated = table.effect.notna()
    if not ((table.loc[estimated, "ci_low"] <= table.loc[estimated, "effect"]) &
            (table.loc[estimated, "ci_high"] >= table.loc[estimated, "effect"])).all():
        raise ValueError("Confidence intervals do not contain effects")
    expected = table.padj.notna() & (table.padj <= study.alpha) & (table.effect.abs() >= study.min_effect)
    actual = table.selected.astype(str).str.lower().eq("true")
    if not actual.equals(expected):
        raise ValueError("Selection inconsistent with approved thresholds")
    expected_direction = np.where(table.effect.isna(), "not_estimated", np.where(table.effect > 0, "up", np.where(table.effect < 0, "down", "zero")))
    if not (table.direction == expected_direction).all():
        raise ValueError("Effect direction inconsistent")


def qc(study, run):
    results, errors = {}, []
    for domain in study.assays:
        try:
            results[domain] = prepare_assay(study, domain, run)
        except Exception as exc:
            errors.append(f"{domain}: {type(exc).__name__}: {exc}")
    validate_alignment(study, run, results, errors)
    write_json(Path(run) / "inputs/qc.json", {"assays": results, "errors": errors})
    return results, errors


def validate_alignment(study, run, results, errors):
    if len(results) > 1:
        alignment = []
        domains = list(results)
        for i, left in enumerate(domains):
            for right in domains[i + 1:]:
                if study.assays[left].cohort_id != study.assays[right].cohort_id:
                    continue
                a = pd.read_csv(Path(run) / "inputs" / left / "metadata.csv", dtype=str)
                b = pd.read_csv(Path(run) / "inputs" / right / "metadata.csv", dtype=str)
                shared_subjects = set(a.subject_id) & set(b.subject_id)
                shared_samples = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
                conflicts = (shared_samples.group_a != shared_samples.group_b) | (shared_samples.subject_id_a != shared_samples.subject_id_b)
                if study.design == "unpaired":
                    common = a.merge(b, on="subject_id", suffixes=("_a", "_b"))
                    subject_conflict = bool((common.group_a != common.group_b).any())
                else:
                    subject_conflict = False
                blocked = not shared_subjects or conflicts.any() or subject_conflict
                alignment.append({"assays": [left, right], "cohort_id": study.assays[left].cohort_id,
                                  "n_shared_subjects": len(shared_subjects),
                                  "n_left_only": len(set(a.subject_id) - set(b.subject_id)),
                                  "n_right_only": len(set(b.subject_id) - set(a.subject_id)),
                                  "status": "blocked" if blocked else "aligned"})
                if blocked:
                    errors.append("Cross-omics identity/group conflict or no shared subjects under one cohort_id; revise explicit mappings/cohort declarations")
        results["_cohort_alignment"] = alignment


def differential(study, run, domain):
    run = Path(run)
    out = run / domain
    out.mkdir(parents=True, exist_ok=True)
    job = {"matrix": str(run / "inputs" / domain / "matrix.csv"),
           "metadata": str(run / "inputs" / domain / "metadata.csv"), "output": str(out),
           "scale": ("log2_abundance" if domain == "protein" else "log2_expression") if study.assays[domain].transform == "log2" else study.assays[domain].scale,
           "input_scale": study.assays[domain].scale, "input_transform": study.assays[domain].transform,
           **{key: getattr(study, key) for key in ["design", "reference", "case", "covariates", "alpha", "confidence",
                                                 "min_effect", "counts_min_total", "seed", "missing_policy", "r_library"]}}
    job_path = out / "job.json"
    write_json(job_path, job)
    argv = [study.rscript, "--vanilla", str(resource("r/02_differential.R")), str(job_path)]
    write_json(out / "command.json", {"argv": argv})
    with (out / "tool.log").open("a") as log:
        result = subprocess.run(argv, env=child_environment(study), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"R tool exited {result.returncode}; inspect {domain}/tool.log")
    table = pd.read_csv(out / "differential.csv", dtype={"feature_id": str})
    validate_differential(table, study)
    return read_json(out / "statistics.json")


def evidence_id(domain, cohort, feature):
    return "ev_" + hashlib.sha256(f"{domain}\0{cohort}\0{feature}".encode()).hexdigest()[:24]


def integration(study, run, results):
    run = Path(run)
    out = run / "integration"
    out.mkdir(parents=True, exist_ok=True)
    if any(item["status"] == "blocked" for item in results.get("QC", {}).get("summary", {}).get("_cohort_alignment", [])):
        raise ValueError("Cross-omics identity alignment blocked; do not merge these branches")
    candidates = pd.read_csv(run / "knowledge/candidates.csv", dtype=str, keep_default_na=False)
    genes = sorted(set(candidates.gene_id))
    registry = []
    per_cohort = {}
    coverage = []
    for domain, assay in study.assays.items():
        per_cohort.setdefault(assay.cohort_id, {})
        role = "RNA" if domain == "rna" else "Protein"
        if results.get(role, {}).get("status") != "complete":
            coverage.append({"omics": domain, "cohort_id": assay.cohort_id, "status": "failed_or_blocked", "n_mapped": 0})
            continue
        table = pd.read_csv(run / domain / "differential.csv", dtype={"feature_id": str})
        mapping = mapping_for(study, domain, table.feature_id)
        mapping.to_csv(out / f"{domain}_mapping_audit.csv", index=False)
        table = table.merge(mapping, how="left", on="feature_id")
        coverage.append({"omics": domain, "cohort_id": assay.cohort_id, "status": "available",
                         "n_samples": results.get("QC", {}).get("summary", {}).get(domain, {}).get("n_samples"),
                         "n_units": results.get("QC", {}).get("summary", {}).get(domain, {}).get("n_units"),
                         "n_mapped": int(table.loc[table.mapping_status.eq("mapped"), "gene_id"].nunique())})
        for row in table.itertuples(index=False):
            def number(value):
                return float(value) if pd.notna(value) else None
            rec = EvidenceRecord(evidence_id=evidence_id(domain, assay.cohort_id, row.feature_id),
                                 gene_id=row.gene_id or "UNMAPPED", omics=domain, cohort_id=assay.cohort_id,
                                 source=f"{domain}/differential.csv", effect=number(row.effect), ci_low=number(row.ci_low),
                                 ci_high=number(row.ci_high), pvalue=number(row.pvalue), padj=number(row.padj),
                                 status=row.status if row.mapping_status == "mapped" else row.mapping_status)
            registry.append(rec.model_dump())
        table = table.loc[table.mapping_status.eq("mapped")].copy()
        per_cohort[assay.cohort_id][domain] = table.set_index("gene_id").to_dict("index")
    registry_columns = list(EvidenceRecord.model_fields)
    pd.DataFrame(registry, columns=registry_columns).to_csv(out / "evidence_registry.csv", index=False)
    rows = []
    for cohort, domains in per_cohort.items():
        for gene in genes:
            row = {"gene_id": gene, "cohort_id": cohort,
                   "member_type": "|".join(sorted(set(candidates.loc[candidates.gene_id.eq(gene), "member_type"])))}
            supported = []
            tested = []
            for domain in ["rna", "protein"]:
                item = domains.get(domain, {}).get(gene)
                available = item is not None and pd.notna(item["padj"])
                is_selected = available and str(item["selected"]).lower() == "true"
                row.update({f"{domain}_status": item["status"] if item else "not_measured_or_unmapped",
                            f"{domain}_effect": item["effect"] if item else None,
                            f"{domain}_q": item["padj"] if item else None,
                            f"{domain}_evidence_id": evidence_id(domain, cohort, item["feature_id"]) if item else ""})
                if available:
                    tested.append(domain)
                if is_selected:
                    supported.append(domain)
            if len(supported) == 2:
                row["support"] = "dual_concordant" if np.sign(row["rna_effect"]) == np.sign(row["protein_effect"]) else "dual_discordant"
            elif supported:
                row["support"] = "single_omics"
            else:
                row["support"] = "unsupported" if tested else "not_evaluable"
            rows.append(row)
    columns = ["gene_id", "cohort_id", "member_type", "rna_status", "rna_effect", "rna_q", "rna_evidence_id",
               "protein_status", "protein_effect", "protein_q", "protein_evidence_id", "support"]
    evidence = pd.DataFrame(rows, columns=columns)
    if len(evidence):
        tiers = {name: i for i, name in enumerate(["dual_concordant", "dual_discordant", "single_omics", "unsupported", "not_evaluable"])}
        evidence["_order"] = evidence.support.map(tiers)
        evidence = evidence.sort_values(["cohort_id", "_order", "gene_id"]).drop(columns="_order")
    evidence.to_csv(out / "candidate_evidence.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    return {"n_candidate_rows": len(evidence), "n_evidence": len(registry),
            "support_counts": {str(k): int(v) for k, v in evidence.support.value_counts().items()},
            "cohorts_kept_separate": True}


def critic(study, run, results):
    run = Path(run)
    issues = []
    for domain in study.assays:
        role = "RNA" if domain == "rna" else "Protein"
        if results.get(role, {}).get("status") != "complete":
            issues.append({"role": role, "kind": "missing_branch", "message": "Branch failed or blocked; evidence is incomplete"})
            continue
        try:
            table = pd.read_csv(run / domain / "differential.csv", dtype={"feature_id": str})
            validate_differential(table, study)
            stats = read_json(run / domain / "statistics.json")
            if stats["bh_scope"] != "all_finite_pvalues_before_candidate_filter" or stats["independent_filtering"]:
                raise ValueError("Incorrect multiplicity policy")
            if stats["n_tested"] != int(table.pvalue.notna().sum()):
                raise ValueError("Reported test count differs from the full table")
        except Exception as exc:
            issues.append({"role": role, "kind": "statistical_validation", "message": str(exc)})
    if study.analysis_mode == "candidate_screening":
        registry = pd.read_csv(run / "integration/evidence_registry.csv", dtype=str, keep_default_na=False)
        known = set(registry.evidence_id)
        candidates = pd.read_csv(run / "integration/candidate_evidence.csv", dtype=str, keep_default_na=False)
        for key in ["rna_evidence_id", "protein_evidence_id"]:
            if not set(candidates[key]).difference({""}).issubset(known):
                issues.append({"role": "Integration", "kind": "unknown_evidence", "message": "Candidate cites unknown evidence"})
    report = {"status": "pass" if not issues else "partial", "issues": issues,
              "scientific_status": "synthetic_demonstration" if study.synthetic else "observational_screening_no_independent_validation",
              "limits": ["Expression/abundance is not PSI, spliceosome activity, assembly or causality",
                         "Same-subject multi-omics is complementary evidence, not independent validation",
                         "No discoveries is a valid result; non-significance does not establish equivalence"]}
    write_json(run / "critic/review.json", report)
    return report


def execute_tool(task, plan, run, results):
    run = Path(run)
    role = task.role
    errors = []
    summary = {}
    status = "complete"
    try:
        if task.tool == "qc":
            summary, errors = qc(plan.study, run)
            status = "partial" if errors and summary else "failed" if errors else "complete"
        elif task.tool == "knowledge":
            summary = prepare_candidates(plan.study, run)
        elif task.tool == "differential":
            domain = "rna" if role == "RNA" else "protein"
            if domain not in results.get("QC", {}).get("summary", {}):
                raise ValueError("Branch QC failed; numerical execution blocked")
            summary = differential(plan.study, run, domain)
        elif task.tool == "integration":
            if results.get("Knowledge", {}).get("status") != "complete":
                raise ValueError("Knowledge membership unavailable; integration blocked")
            summary = integration(plan.study, run, results)
        elif task.tool == "critic":
            if plan.study.analysis_mode == "candidate_screening" and results.get("Integration", {}).get("status") != "complete":
                raise ValueError("Integration unavailable; critic blocked")
            summary = critic(plan.study, run, results)
            status = "partial" if summary["issues"] else "complete"
        else:
            raise ValueError("Unregistered tool")
    except Exception as exc:
        status = "failed"
        errors = [f"{type(exc).__name__}: {exc}"]
    # Parallel branch isolation: collect only the tool's own output subtree.
    folder = {"QC": "inputs", "Knowledge": "knowledge", "RNA": "rna", "Protein": "protein",
              "Integration": "integration", "Critic": "critic"}[role]
    artifacts = [artifact(run, p) for p in sorted((run / folder).rglob("*")) if p.is_file()]
    result = ToolResult(role=role, status=status, artifacts=artifacts, summary=summary, errors=errors)
    write_json(run / "results" / f"{role}.json", result)
    return result
