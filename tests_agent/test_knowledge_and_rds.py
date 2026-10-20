from pathlib import Path
import subprocess

import pandas as pd
import pytest

from spliceosome_agent.data import prepare_assay
from spliceosome_agent.knowledge import import_go, import_reactome


def test_go_not_evidence_taxon_and_descendants(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text("[Term]\nid: GO:ROOT\n\n[Term]\nid: GO:CHILD\nis_a: GO:ROOT ! parent\n")
    gaf = tmp_path / "annotations.gaf"
    def row(symbol, qualifier="", taxon="9606"):
        return "\t".join(["DB", "id", symbol, qualifier, "GO:CHILD", "PMID:1", "IDA", "", "C", "", "", "protein", "taxon:" + taxon, "20261019", "DB"])
    gaf.write_text("!gaf-version: 2.2\n" + "\n".join([row("KEEP"), row("NOT", "NOT"), row("WRONG_SPECIES", taxon="10090")]) + "\n")
    out = tmp_path / "candidates.csv"
    import_go(gaf, ["GO:ROOT"], "Homo sapiens", "frozen_test", out, "synthetic_go_fixture", "9606", obo=obo)
    assert pd.read_csv(out).gene_id.tolist() == ["KEEP"]


def test_reactome_members_stay_pathway_members(tmp_path):
    raw = tmp_path / "reactome.tsv"
    raw.write_text("123\tR-HSA-TEST\thttps://reactome.org/test\tSynthetic\tIEA\tHomo sapiens\n")
    out = tmp_path / "candidates.csv"
    import_reactome(raw, ["R-HSA-TEST"], "Homo sapiens", "frozen_test", out, "synthetic_reactome_fixture")
    table = pd.read_csv(out)
    assert table.member_type.tolist() == ["pathway_member"]
    assert table.id_type.tolist() == ["ENTREZ"]


def test_rds_id_type_pair_adapter_without_row_order(study, tmp_path):
    matrix = tmp_path / "matrix.rds"
    metadata = tmp_path / "metadata.rds"
    assay = study.assays["rna"]
    script = ('a<-commandArgs(TRUE);x<-read.csv(a[1],row.names=1,check.names=FALSE);'
              'm<-read.csv(a[2],stringsAsFactors=FALSE);names(m)<-c("ID","Type","Pair","batch");'
              'm<-m[rev(seq_len(nrow(m))),];saveRDS(x,a[3]);saveRDS(m,a[4])')
    response = subprocess.run(["Rscript", "--vanilla", "-e", script, assay.matrix, assay.metadata, str(matrix), str(metadata)],capture_output=True,text=True)
    assert response.returncode == 0, response.stderr
    assay.matrix = str(matrix)
    assay.metadata = str(metadata)
    assay.columns.sample_id = "ID"
    assay.columns.group = "Type"
    assay.columns.subject_id = "Pair"
    summary = prepare_assay(study, "rna", tmp_path / "run")
    assert summary["n_units"] == 12 and summary["n_samples"] == 24

