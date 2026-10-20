"""Versioned public contracts. Extra fields fail closed at every model boundary."""
from __future__ import annotations

from typing import Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetadataColumns(Contract):
    sample_id: str = "sample_id"
    group: str = "group"
    subject_id: str | None = "subject_id"


class AssaySpec(Contract):
    matrix: str
    metadata: str
    scale: Literal["counts", "log2_expression", "log2_abundance", "abundance"]
    preprocessing: str = Field(min_length=1)
    id_type: Literal["SYMBOL", "ENSEMBL", "ENTREZ", "UNIPROT", "UNKNOWN"] = "UNKNOWN"
    cohort_id: str = "study"
    columns: MetadataColumns = Field(default_factory=MetadataColumns)
    mapping: str | None = None
    transform: Literal["none", "log2"] = "none"

    @model_serializer(mode="wrap")
    def preserve_legacy_fields(self, handler):
        payload = handler(self)
        if self.transform == "none":
            payload.pop("transform", None)
        return payload



class StudySpec(Contract):
    schema_version: Literal["1.0", "2.0"] = "1.0"
    analysis_mode: Literal["candidate_screening", "differential"] = "candidate_screening"
    name: str = Field(min_length=1)
    species: str = Field(default="unknown", min_length=1)
    design: Literal["paired", "unpaired"]
    reference: str
    case: str
    assays: dict[Literal["rna", "protein"], AssaySpec]
    candidates: str | None = None
    candidate_id_type: Literal["SYMBOL", "ENSEMBL", "ENTREZ", "UNIPROT"] = "SYMBOL"
    covariates: dict[str, Literal["continuous", "categorical"]] = Field(default_factory=dict)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    min_effect: float = Field(default=0, ge=0)
    counts_min_total: int = Field(default=10, ge=0)
    min_units: int = Field(default=3, ge=3)
    missing_policy: Literal["exclude_feature", "error"] = "exclude_feature"
    seed: int = Field(default=2024, ge=0)
    rscript: str = "Rscript"
    r_library: str | None = None
    font: str = "Arial"
    allow_font_fallback: bool = False
    synthetic: bool = False

    @model_serializer(mode="wrap")
    def preserve_legacy_fields(self, handler):
        payload = handler(self)
        if self.schema_version == "1.0":
            payload.pop("analysis_mode", None)
        return payload

    @model_validator(mode="after")
    def valid_design(self):
        if self.schema_version == "1.0" and self.analysis_mode != "candidate_screening":
            raise ValueError("Plain differential analysis requires StudySpec 2.0")
        if self.analysis_mode == "candidate_screening" and (not self.candidates or self.species == "unknown"):
            raise ValueError("Candidate screening requires a candidate table and declared species")
        if self.reference == self.case:
            raise ValueError("reference and case must differ")
        if not self.assays:
            raise ValueError("At least one assay is required")
        for domain, assay in self.assays.items():
            if self.analysis_mode == "candidate_screening" and assay.id_type == "UNKNOWN":
                raise ValueError("Candidate screening requires declared feature ID types")
            if assay.transform == "log2" and assay.scale != "abundance":
                raise ValueError("Only explicitly confirmed raw abundance can be log2 transformed")
            if assay.scale == "abundance" and assay.transform != "log2":
                raise ValueError("Raw abundance requires an explicitly confirmed log2 transformation")
            if domain == "protein" and assay.scale not in {"log2_abundance", "abundance"}:
                raise ValueError("Protein input must be explicitly log2_abundance")
            if domain == "rna" and assay.scale not in {"counts", "log2_expression", "abundance"}:
                raise ValueError("RNA requires counts or log2_expression")
            if self.design == "paired" and not assay.columns.subject_id:
                raise ValueError("Paired design requires an explicit subject_id column")
        return self


class TaskSpec(Contract):
    id: str
    role: Literal["QC", "Knowledge", "RNA", "Protein", "Integration", "Critic"]
    tool: Literal["qc", "knowledge", "differential", "integration", "critic"]
    depends_on: list[str] = Field(default_factory=list)


class ProbeRequest(Contract):
    ref: str


class ProbeReport(Contract):
    status: Literal["needs_information", "blocked", "ready_for_confirmation"]
    input_hashes: dict[str, str] = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StudyDraft(Contract):
    goal: str = ""
    study: dict = Field(default_factory=dict)
    status: Literal["needs_information", "blocked", "ready_for_confirmation"] = "needs_information"
    questions: list[str] = Field(default_factory=list)
    sources: dict = Field(default_factory=dict)
    confirmations: dict = Field(default_factory=dict)


class AnalysisPlan(Contract):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    version: int = Field(default=1, ge=1)
    parent_hash: str | None = None
    goal: str
    study: StudySpec
    tasks: list[TaskSpec]
    max_repairs: int = Field(default=3, ge=0, le=10)
    max_role_calls: int = Field(default=30, ge=1, le=300)
    token_budget: int = Field(default=100_000, ge=100)
    generated_code: bool = True
    privacy: Literal["deidentified_summary"] = "deidentified_summary"
    input_hashes: dict[str, str]
    preflight: ProbeReport | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_hash(self, handler):
        payload = handler(self)
        if self.schema_version == "1.0":
            payload.pop("preflight", None)
        return payload

    @model_validator(mode="after")
    def preflight_binding(self):
        if self.schema_version == "1.1":
            if self.preflight is None or self.preflight.status != "ready_for_confirmation":
                raise ValueError("New plans require a successful preflight")
            if self.preflight.input_hashes != self.input_hashes:
                raise ValueError("Preflight input hashes do not match the plan")
        elif self.preflight is not None:
            raise ValueError("Legacy plans cannot carry new preflight fields")
        return self


class ApprovalRecord(Contract):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    plan_hash: str
    plan_version: int
    approved_at: str
    approved_by: str = "user"
    execution_scope: Literal["registered_tools_with_code_review", "registered_tools_with_automatic_code"] = "registered_tools_with_code_review"


class Artifact(Contract):
    path: str
    sha256: str
    bytes: int


class ToolResult(Contract):
    role: str
    status: Literal["complete", "failed", "blocked", "partial"]
    artifacts: list[Artifact] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class EvidenceRecord(Contract):
    evidence_id: str
    gene_id: str
    omics: str
    cohort_id: str
    source: str
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    pvalue: float | None = None
    padj: float | None = None
    status: str


class AgentDecision(Contract):
    action: Literal["continue", "retry", "revise", "code"] = "continue"
    # Some OpenAI-compatible gateways name the free-text field ``message``; only this
    # prose field is tolerated under both names, scientific fields stay strict.
    explanation: str = Field(default="", validation_alias=AliasChoices("explanation", "message"))
    evidence_ids: list[str] = Field(default_factory=list)
    target: Literal["QC", "Knowledge", "RNA", "Protein", "Integration"] | None = None
    language: Literal["python", "r"] | None = None
    source: str | None = None

    @model_validator(mode="after")
    def code_required(self):
        if self.action == "code" and (not self.language or not self.source):
            raise ValueError("Code proposals require language and source")
        return self


class PlanningReply(Contract):
    draft: dict | None = None
    probes: list[ProbeRequest] = Field(default_factory=list)
    goal: str = ""
    questions: list[str] = Field(default_factory=list)
    study: dict | None = None
    explanation: str = Field(default="", validation_alias=AliasChoices("explanation", "message"))
