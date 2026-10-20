import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spliceosome_agent.config import ModelConfig, ModelProfile
from spliceosome_agent.data import prepare_assay
from spliceosome_agent.engine import Engine
from spliceosome_agent.io import artifact, object_hash, verify_artifacts
from spliceosome_agent.knowledge import mapping_for, prepare_candidates
from spliceosome_agent.models import StudySpec
from spliceosome_agent.planning import make_plan, verify_inputs
from spliceosome_agent.tools import bh


def test_approval_required_before_run(approved):
    store, plan, digest = approved
    store.cancel(digest)
    with pytest.raises(PermissionError):
        Engine.create(store, digest, run_id="must_not_exist")
    assert not (store.workspace / "runs/must_not_exist").exists()


def test_revision_expires_old_approval(approved):
    store, plan, digest = approved
    updated = make_plan(plan.study.model_copy(update={"alpha": .01}), "revised", plan)
    new_hash = store.save_plan(updated)
    assert new_hash != digest and updated.version == 2
    with pytest.raises(PermissionError):
        store.require_approval(digest)
    with pytest.raises(ValueError, match="current"):
        store.approve(digest)


def test_changed_input_cannot_be_approved(approved):
    store, plan, digest = approved
    path = Path(plan.study.candidates)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Input changed"):
        store.approve(digest)


def test_unknown_or_inconsistent_model_profiles():
    with pytest.raises(ValueError):
        ModelConfig(default="missing")
    for url, kind in [("http://remote.example/v1", "remote"), ("https://remote.example/v1", "local"), ("https://key:secret@host/v1", "remote")]:
        with pytest.raises(ValueError):
            ModelProfile(name="bad", base_url=url, kind=kind, model="model")
    assert ModelProfile(name="local", base_url="http://127.0.0.1:11434/v1", kind="local", model="model", api_key_env=None)


def test_scale_is_never_guessed(study):
    data = study.model_dump()
    data["assays"]["rna"]["scale"] = "TPM"
    with pytest.raises(ValueError):
        StudySpec.model_validate(data)


def test_metadata_reordering_preserves_explicit_identity(study, tmp_path):
    original = prepare_assay(study, "rna", tmp_path / "a")
    path = Path(study.assays["rna"].metadata)
    pd.read_csv(path).sample(frac=1, random_state=31).to_csv(path, index=False)
    shuffled = prepare_assay(study, "rna", tmp_path / "b")
    assert original == shuffled
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a/inputs/rna/metadata.csv"), pd.read_csv(tmp_path / "b/inputs/rna/metadata.csv"))
    assert pd.read_csv(tmp_path / "a/inputs/rna/metadata.csv").columns[0] == "sample_id"


@pytest.mark.parametrize("problem", ["duplicate_sample", "broken_pair", "unknown_group", "missing_subject", "fractional_counts"])
def test_bad_inputs_fail_closed(study, tmp_path, problem):
    metadata = Path(study.assays["rna"].metadata)
    table = pd.read_csv(metadata)
    if problem == "duplicate_sample":
        table.loc[1, "sample_id"] = table.loc[0, "sample_id"]
    elif problem == "broken_pair":
        table.loc[0, "subject_id"] = "NEW_SUBJECT"
    elif problem == "unknown_group":
        table.loc[0, "group"] = "unknown"
    elif problem == "missing_subject":
        table.loc[0, "subject_id"] = ""
    else:
        matrix = Path(study.assays["rna"].matrix)
        x = pd.read_csv(matrix)
        x[x.columns[1]] = x[x.columns[1]].astype(float)
        x.iloc[0, 1] = 1.5
        x.to_csv(matrix, index=False)
    table.to_csv(metadata, index=False)
    with pytest.raises(ValueError):
        prepare_assay(study, "rna", tmp_path / "run")


def test_unpaired_repeated_subject_is_not_independent(study, tmp_path):
    study.design = "unpaired"
    with pytest.raises(ValueError, match="Repeated"):
        prepare_assay(study, "rna", tmp_path)


def test_mapping_ambiguity_is_retained(study, tmp_path):
    mapping = tmp_path / "map.csv"
    pd.DataFrame({"feature_id": ["f1", "f1", "f2", "f3"], "gene_id": ["A", "B", "C", "C"]}).to_csv(mapping, index=False)
    study.assays["rna"].mapping = str(mapping)
    mapped = mapping_for(study, "rna", ["f1", "f2", "f3", "missing"])
    assert set(mapped.mapping_status) == {"ambiguous", "multiple_features_requires_policy", "unmapped"}


def test_candidate_species_mismatch_rejected(study, tmp_path):
    table = pd.read_csv(study.candidates)
    table.loc[0, "species"] = "other species"
    table.to_csv(study.candidates, index=False)
    with pytest.raises(ValueError, match="species"):
        prepare_candidates(study, tmp_path)


def test_artifact_tamper_and_escape(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("original")
    record = artifact(tmp_path, file)
    verify_artifacts(tmp_path, [record])
    file.write_text("tampered")
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, [record])
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, [{"path": "../outside", "sha256": "x", "bytes": 0}])


def test_full_family_bh_is_not_candidate_only():
    p = [.001, .02, .04, .9]
    assert bh(p) == pytest.approx([.004, .04, .0533333333333, .9])
    assert bh(p[:2])[1] != bh(p)[1]


def test_empty_matrix_is_qc_failure(study, tmp_path):
    path = Path(study.assays["rna"].matrix)
    pd.read_csv(path).iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        prepare_assay(study, "rna", tmp_path / "run")


def test_cross_omics_shared_sample_conflicts_are_not_merged(study, tmp_path):
    from spliceosome_agent.tools import qc, integration
    source = pd.read_csv(study.assays["protein"].metadata)
    source["group"] = source.group.map({study.reference: study.case, study.case: study.reference})
    different = tmp_path / "conflicting_protein_metadata.csv"
    source.to_csv(different, index=False)
    study.assays["protein"].metadata = str(different)
    summary, errors = qc(study, tmp_path / "run")
    assert errors and summary["_cohort_alignment"][0]["status"] == "blocked"
    with pytest.raises(ValueError, match="identity alignment"):
        integration(study, tmp_path / "run", {"QC": {"summary": summary}})
