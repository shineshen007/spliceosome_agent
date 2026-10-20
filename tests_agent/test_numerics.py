import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spliceosome_agent.data import child_environment
from spliceosome_agent.engine import Engine, verify_run
from spliceosome_agent.examples import create_example
from spliceosome_agent.io import sha256
from spliceosome_agent.io import read_json
from spliceosome_agent.planning import load_study, make_plan
from spliceosome_agent.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("design", ["paired", "unpaired"])
@pytest.mark.parametrize("rna_scale", ["counts", "log2_expression"])
def test_real_r_models_against_independent_reference(tmp_path, design, rna_scale):
    library = Path(os.environ.get("SPLICEOSOME_R_LIBRARY") or ROOT / ".r-library")
    if not (library / "DESeq2").is_dir():
        pytest.skip("Install the documented R environment before numerical acceptance")
    study_file = create_example(tmp_path / "fixture", design, rna_scale, library)
    study = load_study(study_file)
    store = Store(tmp_path / "workspace")
    plan = make_plan(study, "SYNTHETIC NUMERICAL ACCEPTANCE")
    hashes = {p: sha256(p) for p in plan.input_hashes}
    digest = store.save_plan(plan)
    store.approve(digest)
    engine = Engine.create(store, digest)
    state = engine.execute()
    assert state["status"] == "complete", state
    assert read_json(engine.run / "rna/statistics.json")["absorbed_subject_covariates"] == []
    for domain in ["rna", "protein"]:
        output = tmp_path / f"{domain}_reference.csv"
        response = subprocess.run([study.rscript, "--vanilla", str(ROOT / "tests_agent/reference_models.R"),
                                   str(engine.run / domain / "job.json"), str(output)],
                                  env=child_environment(study), capture_output=True, text=True)
        assert response.returncode == 0, response.stderr
        observed = pd.read_csv(engine.run / domain / "differential.csv").set_index("feature_id")
        reference = pd.read_csv(output).set_index("feature_id")
        for column in ["effect", "se", "pvalue", "padj", "ci_low", "ci_high"]:
            assert np.allclose(observed.loc[reference.index, column], reference[column], rtol=2e-7, atol=1e-9, equal_nan=True), (domain, column)
    assert hashes == {p: sha256(p) for p in hashes}
    assert verify_run(store, engine.run_id)["status"] == "verified"
    evidence = pd.read_csv(engine.run / "integration/candidate_evidence.csv")
    assert {"dual_concordant", "dual_discordant", "not_evaluable"}.issubset(set(evidence.support))
    assert not list(engine.run.rglob("*.png"))
    assert len(list((engine.run / "figures").glob("*.pdf"))) == 3
