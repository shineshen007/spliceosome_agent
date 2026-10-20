"""Clearly synthetic fixtures; never drawn from the protected local cohort."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def create_example(destination, design="paired", rna_scale="counts", r_library=None):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2024)
    genes = [f"SYNTH_GENE_{i:03d}" for i in range(160)]
    n = 12
    samples = [f"SYNTH_CONTROL_{i:02d}" for i in range(n)] + [f"SYNTH_CASE_{i:02d}" for i in range(n)]
    subjects = [f"SYNTH_SUBJECT_{i:02d}" for i in range(n)] * 2 if design == "paired" else samples
    md = pd.DataFrame({"sample_id": samples, "group": ["control"] * n + ["case"] * n,
                       "subject_id": subjects, "batch": ["b1", "b2"] * n})
    md.to_csv(destination / "metadata.csv", index=False)
    baseline = rng.uniform(5, 9, (len(genes), 1))
    subject_effect = rng.normal(0, .25, (len(genes), n))
    signal = np.zeros((len(genes), 2 * n))
    signal[:10, n:] = 1.5
    signal[10:20, n:] = -1.5
    expr = baseline + np.tile(subject_effect, (1, 2)) + signal + rng.normal(0, .2, (len(genes), 2 * n))
    mu = 2 ** expr
    counts = rng.negative_binomial(20, 20 / (20 + mu))
    rna = counts if rna_scale == "counts" else expr
    protein = baseline + np.tile(subject_effect, (1, 2)) + signal + rng.normal(0, .2, (len(genes), 2 * n))
    protein[5:8, n:] -= 3  # Preserve genuine discordance in this synthetic example.
    pd.DataFrame(rna, index=pd.Index(genes, name="feature_id"), columns=samples).to_csv(destination / "rna.csv")
    pd.DataFrame(protein, index=pd.Index(genes, name="feature_id"), columns=samples).to_csv(destination / "protein.csv")
    pd.DataFrame({"gene_id": genes[:30] + ["SYNTH_UNMEASURED"], "member_type": ["core"] * 15 + ["regulator"] * 16,
                  "source": "synthetic_fixture_not_biological_annotation", "version": "1", "species": "synthetic"}).to_csv(destination / "candidates.csv", index=False)
    study = {"name": "Synthetic spliceosome screening demonstration", "species": "synthetic", "design": design,
             "reference": "control", "case": "case", "candidates": "candidates.csv", "synthetic": True,
             "allow_font_fallback": True, "r_library": str(Path(r_library).resolve()) if r_library else None,
             "assays": {"rna": {"matrix": "rna.csv", "metadata": "metadata.csv", "scale": rna_scale,
                                 "preprocessing": "synthetic generated matrix", "id_type": "SYMBOL"},
                        "protein": {"matrix": "protein.csv", "metadata": "metadata.csv", "scale": "log2_abundance",
                                    "preprocessing": "synthetic log2 abundance; no normalization", "id_type": "SYMBOL"}}}
    (destination / "study.yaml").write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")
    (destination / "README.md").write_text("# SYNTHETIC DEMONSTRATION\n\nNo real patients, biological memberships or scientific findings.\n", encoding="utf-8")
    return destination / "study.yaml"

