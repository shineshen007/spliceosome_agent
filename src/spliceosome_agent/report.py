"""Evidence-derived reporting and vector figures; source data remain inspectable."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json


def figures(plan, run):
    import os
    cache = Path(run) / "plot-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    run = Path(run)
    folder = run / "figures"
    source = folder / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    for p in [Path("/Library/Fonts/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial.ttf")]:
        if p.is_file():
            font_manager.fontManager.addfont(str(p))
    requested = plan.study.font
    try:
        resolved = font_manager.findfont(requested, fallback_to_default=False)
        actual = font_manager.FontProperties(fname=resolved).get_name()
    except ValueError:
        if not plan.study.allow_font_fallback:
            raise ValueError("Requested font unavailable; install Arial or approve an explicit fallback in a new plan")
        actual = "DejaVu Sans"
    plt.rcParams.update({"font.family": actual, "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    write_json(folder / "font.json", {"requested": requested, "actual": actual, "fallback": actual != requested})
    captions = []
    colors = {plan.study.reference: "#3C5488", plan.study.case: "#E64B35"}
    for domain in plan.study.assays:
        de = run / domain / "differential.csv"
        qc = run / domain / "sample_qc.csv"
        if not de.exists() or not qc.exists():
            continue
        table = pd.read_csv(de)
        md = pd.read_csv(qc, dtype={"sample_id": str, "subject_id": str})
        shutil.copy2(de, source / f"{domain}_differential.csv")
        shutil.copy2(qc, source / f"{domain}_sample_qc.csv")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        xpositions = {plan.study.reference: 0, plan.study.case: 1}
        if plan.study.design == "paired":
            for _, group in md.groupby("subject_id"):
                group = group.assign(_position=group.group.map(xpositions)).sort_values("_position")
                axes[0].plot(group._position, group["median"], color="0.75", linewidth=0.4, zorder=1)
        for label, group in md.groupby("group"):
            offsets = np.linspace(-0.08, 0.08, len(group))
            axes[0].scatter(xpositions[label] + offsets, group["median"], s=8, color=colors[label], label=f"{label} (n={len(group)})", zorder=2)
        axes[0].set(xticks=[0, 1], xticklabels=[plan.study.reference, plan.study.case], ylabel="Median on display scale", title=f"A  {domain}: sample QC")
        axes[0].legend(frameon=False, fontsize=8)
        finite = table.effect.notna() & table.padj.notna()
        plotted = table.loc[finite]
        selected = plotted.selected.astype(str).str.lower().eq("true")
        y = -np.log10(np.maximum(plotted.padj.astype(float), np.finfo(float).tiny))
        axes[1].scatter(plotted.effect, y, s=5, c=np.where(selected, "#E64B35", "#9FA3A8"), rasterized=len(plotted) > 10_000)
        axes[1].axhline(-np.log10(plan.study.alpha), color="0.5", ls="--", lw=0.5)
        axes[1].set(xlabel=f"{plan.study.case} − {plan.study.reference}: effect", ylabel="−log10(BH q)", title=f"B  {domain}: all tested features")
        for extension in ["pdf", "svg"]:
            fig.savefig(folder / f"{domain}_qc_differential.{extension}", metadata={"Creator": "spliceosome_agent"})
        plt.close(fig)
        if plan.study.analysis_mode == "differential":
            fig, axis = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
            axis.scatter(plotted.effect, y, s=5, c=np.where(selected, "#E64B35", "#9FA3A8"), rasterized=len(plotted)>10_000)
            axis.axhline(-np.log10(plan.study.alpha), color="0.5", ls="--", lw=.5)
            axis.set(xlabel=f"{plan.study.case} − {plan.study.reference}: effect on log2 scale",
                     ylabel="−log10(BH q)", title=f"{domain}: all tested features")
            for extension in ["pdf", "svg"]:
                fig.savefig(folder / f"{domain}_volcano.{extension}")
            plt.close(fig)
        captions.append(f"{domain}: A displays every biological sample; paired subjects are connected when applicable. "
                        "Sample QC is descriptive. B displays all tested features; red marks the approved q/effect selection. "
                        f"Confidence level={plan.study.confidence}; numerical effect/CI/sample counts are in source_data. "
                        "No clinical or splicing-activity conclusion is implied.")
    evidence = run / "integration/candidate_evidence.csv"
    if evidence.exists():
        data = pd.read_csv(evidence)
        shutil.copy2(evidence, source / "candidate_evidence.csv")
        summary = data.groupby(["cohort_id", "support"]).size().rename("n").reset_index()
        summary.to_csv(source / "support_counts.csv", index=False)
        if len(summary):
            fig, axis = plt.subplots(figsize=(7.2, max(2.0, len(summary) * 0.22)), layout="constrained")
            labels = summary.cohort_id + ": " + summary.support
            axis.barh(labels, summary.n, color="#625D9E")
            axis.set(xlabel="Number of candidate genes", title="Cross-omics support (descriptive)")
            for extension in ["pdf", "svg"]:
                fig.savefig(folder / f"cross_omics_support.{extension}")
            plt.close(fig)
        captions.append("Cross-omics support counts include unsupported and non-evaluable candidates; cohorts remain separate. "
                        "Layers from the same subjects are not independent validation. No weighted score or significance optimization is used.")
    (folder / "captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    # The exact installed plotting source is retained alongside source data.
    shutil.copy2(Path(__file__), folder / "plotting_code.py")
    return {"font": actual, "formats": ["pdf", "svg"], "n_pdf": len(list(folder.glob("*.pdf")))}


def render(plan, run, state):
    run = Path(run)
    lines = ["# Differential analysis report" if plan.study.analysis_mode == "differential" else "# Spliceosome screening report", "", f"Goal: {plan.goal}", "",
             f"Technical status: **{state['status']}**", f"Agent mode: {state.get('agent_mode', 'deterministic_tools')}",
             f"Scientific status: {'SYNTHETIC_DEMONSTRATION' if plan.study.synthetic else 'OBSERVATIONAL_SCREENING_NO_INDEPENDENT_VALIDATION'}", "",
             f"Analysis mode: {plan.study.analysis_mode}; no species/ID annotation is implied when unknown.",
             f"Input transformations: { {d: a.transform for d,a in plan.study.assays.items()} }.",
             f"Design: {plan.study.design}; contrast: {plan.study.case} − {plan.study.reference}; species: {plan.study.species}.",
             f"BH q ≤ {plan.study.alpha}; confidence level {plan.study.confidence}; seed {plan.study.seed}.", "",
             "## Methods and results", ""]
    for role, result in state.get("results", {}).items():
        lines += [f"### {role}: {result['status']}", "", f"See [structured result](results/{role}.json).", ""]
        for error in result.get("errors", []):
            lines.append(f"- {error}")
    if (run / "integration/candidate_evidence.csv").exists():
        lines += ["", "[Complete candidate evidence](integration/candidate_evidence.csv) · "
                  "[Full evidence registry](integration/evidence_registry.csv) · [Coverage](integration/coverage.csv)", ""]
    if (run / "critic/review.json").exists():
        review = read_json(run / "critic/review.json")
        lines += ["## Evidence limits", "", *[f"- {x}" for x in review["limits"]]]
    else:
        lines += ["## Evidence limits", "", "Critic review was not completed; this run cannot establish accepted screening results."]
    lines += ["", "Expression and abundance do not establish PSI, splicing activity, complex assembly or causality. "
              "Negative, missing, ambiguous and failed results remain in the full tables. "
              "BH is calculated before membership filtering. Cohorts are not pooled.", "",
              "## Provenance", "", "See plan.json, approval.json, provenance.json, events.jsonl and artifact_manifest.json. "
              "The approved workflow entrypoint is run_analysis.py (hash in runner.json); "
              "optional model-generated code/* outputs remain supplemental and unvalidated.", ""]
    if state.get("warnings"):
        lines += ["## Warnings", "", *[f"- {warning}" for warning in state["warnings"]], ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
