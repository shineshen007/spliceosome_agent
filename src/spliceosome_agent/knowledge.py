"""Frozen membership provenance; no invented annotation or silent ID aggregation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import read_table

MEMBER_TYPES = {"core", "regulator", "pathway_member"}


def prepare_candidates(study, run):
    table = read_table(study.candidates, study)
    required = {"gene_id", "member_type", "source", "version", "species"}
    if not required.issubset(table):
        raise ValueError("Candidate table requires gene_id, member_type, source, version, species")
    if table[list(required)].eq("").any().any():
        raise ValueError("Candidate provenance fields may not be empty")
    if not set(table.species).issubset({study.species}):
        raise ValueError("Candidate species differs from the approved study")
    if not set(table.member_type).issubset(MEMBER_TYPES):
        raise ValueError("Unknown candidate membership class")
    if "id_type" in table and not set(table.id_type).issubset({study.candidate_id_type}):
        raise ValueError("Candidate ID namespace differs from the approved study")
    table = table.drop_duplicates().sort_values(["gene_id", "member_type", "source"])
    out = Path(run) / "knowledge"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "candidates.csv", index=False)
    return {"n_genes": int(table.gene_id.nunique()), "sources": sorted(table.source.unique()),
            "versions": sorted(table.version.unique()), "species": study.species}


def mapping_for(study, domain, features):
    assay = study.assays[domain]
    if assay.mapping:
        mapping = read_table(assay.mapping, study)
        if not {"feature_id", "gene_id"}.issubset(mapping):
            raise ValueError("Mapping table requires feature_id and gene_id")
        mapping = mapping[["feature_id", "gene_id"]].drop_duplicates()
        if mapping.eq("").any().any():
            raise ValueError("Mapping IDs must be nonblank")
        counts = mapping.groupby("feature_id").gene_id.nunique()
        mapping["mapping_status"] = mapping.feature_id.map(counts).map(lambda n: "mapped" if n == 1 else "ambiguous")
        # Multiple measured features for one gene are not collapsed after inspecting outcomes.
        unique_features = mapping.groupby("gene_id").feature_id.nunique()
        duplicate_genes = set(unique_features[unique_features > 1].index)
        mapping.loc[mapping.gene_id.isin(duplicate_genes), "mapping_status"] = "multiple_features_requires_policy"
        mapping = pd.DataFrame({"feature_id": features}).merge(mapping, how="left", on="feature_id")
        mapping["mapping_status"] = mapping.mapping_status.fillna("unmapped")
        mapping["gene_id"] = mapping.gene_id.fillna("")
        return mapping
    if assay.id_type != study.candidate_id_type:
        raise ValueError("Different feature/candidate namespaces require a frozen explicit mapping table")
    return pd.DataFrame({"feature_id": features, "gene_id": features, "mapping_status": "mapped"})


def import_go(gaf, terms, species, version, output, source, taxon, member_type="pathway_member", obo=None):
    """Import selected GO terms plus optional is_a/part_of descendants; exclude NOT."""
    if member_type not in MEMBER_TYPES:
        raise ValueError("Unknown membership class")
    selected = set(terms)
    if obo:
        parents = {}
        current = None
        for line in Path(obo).read_text().splitlines():
            if line == "[Term]":
                current = None
            elif line.startswith("id: GO:"):
                current = line.split()[1]
            elif current and line.startswith("is_a: GO:"):
                parents.setdefault(current, set()).add(line.split()[1])
            elif current and line.startswith("relationship: part_of GO:"):
                parents.setdefault(current, set()).add(line.split()[2])
        while True:
            expanded = selected | {term for term, ps in parents.items() if ps & selected}
            if expanded == selected:
                break
            selected = expanded
    rows = []
    for line in Path(gaf).read_text().splitlines():
        if line.startswith("!") or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 15:
            raise ValueError("Malformed GAF row")
        if cols[4] in selected and "NOT" not in cols[3].split("|") and cols[12].split("|")[0] == f"taxon:{taxon}":
            rows.append({"gene_id": cols[2], "member_type": member_type, "source": source,
                         "version": version, "species": species, "term_id": cols[4],
                         "evidence_code": cols[6], "annotation_taxon": cols[12]})
    pd.DataFrame(rows, columns=["gene_id", "member_type", "source", "version", "species", "term_id", "evidence_code", "annotation_taxon"]).drop_duplicates().to_csv(output, index=False)


def import_reactome(path, pathway_ids, species, version, output, source):
    """Import official NCBI2Reactome_All_Levels TSV; class stays pathway_member."""
    table = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if table.shape[1] != 6:
        raise ValueError("Expected six-column NCBI2Reactome_All_Levels file")
    table.columns = ["gene_id", "term_id", "url", "pathway_name", "evidence_code", "species"]
    table = table.loc[table.term_id.isin(pathway_ids) & table.species.eq(species)].copy()
    table["member_type"] = "pathway_member"
    table["source"] = source
    table["version"] = version
    table["id_type"] = "ENTREZ"
    table.to_csv(output, index=False)
