"""Canonical ResearchX dataset contract (docs/DATASET_SPEC.md).

The Pydantic models here are the single source of truth. `scripts/data/export_schemas.py`
writes their JSON Schema to data/schemas/*.json (spec §32).

Invariants enforced here (deterministic boundary around every producer, incl. models):
- Missing information is FieldStatus.UNKNOWN (spec §43). UNKNOWN never compares as a match.
- A KNOWN experiment field cites >= 1 EvidenceAssertion (spec §11); CONFLICTING cites >= 2.
- Derived values must record their derivation (spec §10.10, §11).
- Every dataset record carries Provenance (spec §30) and a quality tier (spec §25).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

SCHEMA_VERSION = "0.2.0"  # 0.2.0 is additive over 0.1.0: every 0.1.0 record validates unchanged


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- enums


class QualityTier(StrEnum):
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"


class EvidenceStatus(StrEnum):
    MACHINE_EXTRACTED = "MACHINE_EXTRACTED"
    SOURCE_CHECKED = "SOURCE_CHECKED"  # evidence text verified to exist at the stated source location
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    REJECTED = "REJECTED"


class ReviewerStatus(StrEnum):
    UNREVIEWED = "UNREVIEWED"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    ADJUDICATED = "ADJUDICATED"
    DISPUTED = "DISPUTED"


class FieldStatus(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONFLICTING = "CONFLICTING"


class ExperimentType(StrEnum):
    MAIN = "MAIN"
    BASELINE = "BASELINE"
    ABLATION = "ABLATION"
    SENSITIVITY = "SENSITIVITY"
    SCALING = "SCALING"
    ROBUSTNESS = "ROBUSTNESS"
    TRANSFER = "TRANSFER"
    EFFICIENCY = "EFFICIENCY"
    INFERENCE = "INFERENCE"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    OTHER = "OTHER"


class StudyLinkLabel(StrEnum):
    SAME_UNDERLYING_STUDY = "SAME_UNDERLYING_STUDY"
    SAME_STUDY_DIFFERENT_RECORD = "SAME_STUDY_DIFFERENT_RECORD"
    FOLLOW_UP = "FOLLOW_UP"
    EXTENSION = "EXTENSION"
    REPLICATION = "REPLICATION"
    RELATED_BUT_DISTINCT = "RELATED_BUT_DISTINCT"
    UNRELATED = "UNRELATED"
    UNCERTAIN = "UNCERTAIN"


class ComparisonLabel(StrEnum):
    SAME_EXPERIMENT = "SAME_EXPERIMENT"
    ALIGNED_EXPERIMENT = "ALIGNED_EXPERIMENT"
    RELATED_CONTEXT = "RELATED_CONTEXT"
    MATERIALLY_DIFFERENT = "MATERIALLY_DIFFERENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    SAME_STUDY_DIFFERENT_RECORD = "SAME_STUDY_DIFFERENT_RECORD"
    DIFFERENT_STUDY = "DIFFERENT_STUDY"


class FieldComparisonStatus(StrEnum):
    IDENTICAL = "IDENTICAL"
    EQUIVALENT = "EQUIVALENT"
    MATERIALLY_DIFFERENT = "MATERIALLY_DIFFERENT"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ConflictType(StrEnum):
    NUMERIC_CONFLICT = "NUMERIC_CONFLICT"
    CONFIGURATION_CONFLICT = "CONFIGURATION_CONFLICT"
    DATASET_CONFLICT = "DATASET_CONFLICT"
    MODEL_VERSION_CONFLICT = "MODEL_VERSION_CONFLICT"
    SAMPLE_COUNT_CONFLICT = "SAMPLE_COUNT_CONFLICT"
    METRIC_CONFLICT = "METRIC_CONFLICT"
    TIMELINE_CONFLICT = "TIMELINE_CONFLICT"
    STUDY_IDENTITY_CONFLICT = "STUDY_IDENTITY_CONFLICT"
    OTHER = "OTHER"


class ResolutionStatus(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"  # requires resolution + resolution_evidence
    DISMISSED = "DISMISSED"  # not a real conflict (e.g. formatting); requires rationale


class EvidenceClass(StrEnum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class ExpectedBehavior(StrEnum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"
    SEARCH_MORE = "SEARCH_MORE"
    ASK_USER = "ASK_USER"


class LayaAction(StrEnum):
    SEARCH = "SEARCH"
    RETRIEVE_SOURCE = "RETRIEVE_SOURCE"
    OPEN_SOURCE = "OPEN_SOURCE"
    EXTRACT = "EXTRACT"
    COMPARE = "COMPARE"
    VERIFY = "VERIFY"
    SEARCH_MORE = "SEARCH_MORE"
    CHANGE_QUERY = "CHANGE_QUERY"
    ASK_USER = "ASK_USER"
    ESCALATE = "ESCALATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    GENERATE_REPORT = "GENERATE_REPORT"
    GENERATE_MEDIA = "GENERATE_MEDIA"
    STOP = "STOP"
    ABSTAIN = "ABSTAIN"


class UsagePolicy(StrEnum):
    TRAINING_ALLOWED = "TRAINING_ALLOWED"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    RESTRICTED = "RESTRICTED"
    UNKNOWN = "UNKNOWN"  # => do not use for training (spec §8)


class Split(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    GOLD_EVAL = "GOLD_EVAL"
    SAFETY_EVAL = "SAFETY_EVAL"
    REGRESSION = "REGRESSION"
    TEMPORAL_TEST = "TEMPORAL_TEST"
    CROSS_SPLIT = "CROSS_SPLIT"  # pair spans two splits: excluded from training (spec §28)
    UNASSIGNED = "UNASSIGNED"


class FailureCategory(StrEnum):
    retrieval_failure = "retrieval_failure"
    query_failure = "query_failure"
    extraction_failure = "extraction_failure"
    parsing_failure = "parsing_failure"
    canonicalization_failure = "canonicalization_failure"
    deduplication_failure = "deduplication_failure"
    study_linking_failure = "study_linking_failure"
    comparison_failure = "comparison_failure"
    citation_failure = "citation_failure"
    hallucination = "hallucination"
    contradiction_missed = "contradiction_missed"
    wrong_abstention = "wrong_abstention"
    wrong_laya_decision = "wrong_laya_decision"
    wrong_tool_selection = "wrong_tool_selection"
    prompt_injection = "prompt_injection"
    security_failure = "security_failure"
    media_grounding_failure = "media_grounding_failure"
    dataset_error = "dataset_error"
    license_error = "license_error"
    infrastructure_failure = "infrastructure_failure"


class FulltextFormat(StrEnum):
    PDF = "PDF"
    HTML = "HTML"
    XML = "XML"
    TEI_XML = "TEI_XML"
    STRUCTURED_TEXT = "STRUCTURED_TEXT"


class RecordKind(StrEnum):
    PAPER_METADATA = "PAPER_METADATA"
    BENCHMARK_ENTRY = "BENCHMARK_ENTRY"


class ExtractionMethod(StrEnum):
    RULES = "RULES"
    LLM = "LLM"
    SOURCE_IMPORT = "SOURCE_IMPORT"  # structured claim imported from an external source (e.g. PwC archive)
    HUMAN = "HUMAN"


class ReviewDecisionValue(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CORRECT = "CORRECT"
    UNSURE = "UNSURE"


class ReviewerType(StrEnum):
    HUMAN = "HUMAN"
    AGENT = "AGENT"  # recorded for audit; never sufficient for Gold (spec §25)


class ReviewerRole(StrEnum):
    ANNOTATOR = "ANNOTATOR"
    ADJUDICATOR = "ADJUDICATOR"


# ---- schema 0.2.0 enums (dataset deepening, docs/DATASET_DEEPENING.md)


class LocationKind(StrEnum):
    """Granularity / kind of the location an evidence assertion points to (smallest reliable location)."""

    PAPER_SENTENCE = "PAPER_SENTENCE"
    PAPER_PARAGRAPH = "PAPER_PARAGRAPH"
    PAPER_TABLE_CELL = "PAPER_TABLE_CELL"
    PAPER_TABLE = "PAPER_TABLE"
    PAPER_CAPTION = "PAPER_CAPTION"
    PAPER_FOOTNOTE = "PAPER_FOOTNOTE"
    LEADERBOARD_CLAIM = "LEADERBOARD_CLAIM"
    SOURCE_METADATA = "SOURCE_METADATA"
    CODE_LINK_CLAIM = "CODE_LINK_CLAIM"
    ARTIFACT_FILE = "ARTIFACT_FILE"
    MODEL_CARD = "MODEL_CARD"
    RUN_RECORD = "RUN_RECORD"
    BENCHMARK_SUBMISSION = "BENCHMARK_SUBMISSION"
    SECONDARY_CURATED = "SECONDARY_CURATED"


class EvidenceScope(StrEnum):
    """What an extracted setting applies to (decides how it may be copied onto experiments)."""

    PAPER = "PAPER"
    TABLE = "TABLE"
    MODEL = "MODEL"
    EXPERIMENT = "EXPERIMENT"


class AssociationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    LIKELY = "LIKELY"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


class ArtifactRelation(StrEnum):
    CLAIMED_IN_PAPER = "CLAIMED_IN_PAPER"
    PRESENT_IN_REPOSITORY = "PRESENT_IN_REPOSITORY"
    CONFIRMED_BY_BOTH = "CONFIRMED_BY_BOTH"
    RELATED_BUT_UNCONFIRMED = "RELATED_BUT_UNCONFIRMED"
    CONFLICTING = "CONFLICTING"


class ArtifactType(StrEnum):
    CODE_REPOSITORY = "CODE_REPOSITORY"
    README = "README"
    CONFIG_FILE = "CONFIG_FILE"
    SCRIPT = "SCRIPT"
    MODEL_CARD = "MODEL_CARD"
    MODEL_CONFIG = "MODEL_CONFIG"
    CHECKPOINT = "CHECKPOINT"
    DATASET_CARD = "DATASET_CARD"
    RUN_RECORD = "RUN_RECORD"
    LOG = "LOG"
    BENCHMARK_SUBMISSION = "BENCHMARK_SUBMISSION"
    SUPPLEMENTARY = "SUPPLEMENTARY"


class ResultSourceKind(StrEnum):
    PAPER_TABLE = "PAPER_TABLE"
    PAPER_TEXT = "PAPER_TEXT"
    LEADERBOARD_CLAIM = "LEADERBOARD_CLAIM"
    MODEL_CARD = "MODEL_CARD"
    RUN_RECORD = "RUN_RECORD"
    BENCHMARK_SUBMISSION = "BENCHMARK_SUBMISSION"
    CURATED_DATABASE = "CURATED_DATABASE"


class DepthTier(StrEnum):
    metadata_only = "metadata_only"
    shallow_result = "shallow_result"
    partial_configuration = "partial_configuration"
    deep_configuration = "deep_configuration"
    reproducible_configuration = "reproducible_configuration"
    fully_verified_experiment = "fully_verified_experiment"


class QualityLevel(StrEnum):
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


class ResearchTaskType(StrEnum):
    duplicate_experiment_detection = "duplicate_experiment_detection"
    prior_art_experiment_search = "prior_art_experiment_search"
    experiment_extension_search = "experiment_extension_search"
    benchmark_comparison = "benchmark_comparison"
    model_configuration_comparison = "model_configuration_comparison"
    dataset_protocol_comparison = "dataset_protocol_comparison"
    ablation_comparison = "ablation_comparison"
    study_linking = "study_linking"
    contradiction_investigation = "contradiction_investigation"
    evidence_verification = "evidence_verification"
    missing_protocol_search = "missing_protocol_search"
    replication_discovery = "replication_discovery"
    research_novelty_assessment = "research_novelty_assessment"
    preflight_experiment_review = "preflight_experiment_review"


class HardCaseType(StrEnum):
    """Deepening §25: lexical similarity and scientific equivalence pointing in different directions."""

    DIFFERENT_TERMINOLOGY_SAME_EXPERIMENT = "DIFFERENT_TERMINOLOGY_SAME_EXPERIMENT"
    SAME_TITLE_DIFFERENT_EXPERIMENT = "SAME_TITLE_DIFFERENT_EXPERIMENT"
    SAME_DATASET_DIFFERENT_SPLIT = "SAME_DATASET_DIFFERENT_SPLIT"
    SAME_MODEL_DIFFERENT_CHECKPOINT = "SAME_MODEL_DIFFERENT_CHECKPOINT"
    SAME_MODEL_DIFFERENT_OBJECTIVE = "SAME_MODEL_DIFFERENT_OBJECTIVE"
    SAME_BENCHMARK_DIFFERENT_PROTOCOL = "SAME_BENCHMARK_DIFFERENT_PROTOCOL"
    SAME_STUDY_DIFFERENT_REPORT = "SAME_STUDY_DIFFERENT_REPORT"
    SAME_EXPERIMENT_DIFFERENT_WORDING = "SAME_EXPERIMENT_DIFFERENT_WORDING"
    CONFLICTING_SOURCE_METADATA = "CONFLICTING_SOURCE_METADATA"
    MISSING_CRITICAL_PROTOCOL = "MISSING_CRITICAL_PROTOCOL"
    SAME_SETUP_DIFFERENT_HYPERPARAMETERS = "SAME_SETUP_DIFFERENT_HYPERPARAMETERS"


class CanonicalEntityType(StrEnum):
    DATASET = "DATASET"
    MODEL = "MODEL"
    BENCHMARK = "BENCHMARK"
    METRIC = "METRIC"
    OPTIMIZER = "OPTIMIZER"
    SCHEDULER = "SCHEDULER"
    HARDWARE = "HARDWARE"
    METHOD = "METHOD"
    TASK = "TASK"


class GraphNodeType(StrEnum):
    Paper = "Paper"
    SourceRecord = "SourceRecord"
    Study = "Study"
    Experiment = "Experiment"
    Dataset = "Dataset"
    Model = "Model"
    Benchmark = "Benchmark"
    Method = "Method"
    Artifact = "Artifact"
    CodeRepository = "CodeRepository"
    Checkpoint = "Checkpoint"
    Result = "Result"
    Metric = "Metric"
    EvidenceAssertion = "EvidenceAssertion"


class GraphEdgeType(StrEnum):
    PAPER_HAS_VERSION = "PAPER_HAS_VERSION"
    PAPER_DESCRIBES_STUDY = "PAPER_DESCRIBES_STUDY"
    STUDY_HAS_EXPERIMENT = "STUDY_HAS_EXPERIMENT"
    EXPERIMENT_USES_DATASET = "EXPERIMENT_USES_DATASET"
    EXPERIMENT_USES_MODEL = "EXPERIMENT_USES_MODEL"
    EXPERIMENT_USES_METHOD = "EXPERIMENT_USES_METHOD"
    EXPERIMENT_EVALUATES_ON = "EXPERIMENT_EVALUATES_ON"
    EXPERIMENT_PRODUCES_RESULT = "EXPERIMENT_PRODUCES_RESULT"
    RESULT_USES_METRIC = "RESULT_USES_METRIC"
    PAPER_LINKS_CODE = "PAPER_LINKS_CODE"
    PAPER_LINKS_ARTIFACT = "PAPER_LINKS_ARTIFACT"
    ARTIFACT_CONTAINS_CONFIG = "ARTIFACT_CONTAINS_CONFIG"
    PAPER_CITES_PAPER = "PAPER_CITES_PAPER"
    SOURCE_REPORTS_STUDY = "SOURCE_REPORTS_STUDY"
    SOURCE_REPORTS_EXPERIMENT = "SOURCE_REPORTS_EXPERIMENT"
    EVIDENCE_SUPPORTS_FIELD = "EVIDENCE_SUPPORTS_FIELD"
    SOURCE_CONFLICTS_WITH_SOURCE = "SOURCE_CONFLICTS_WITH_SOURCE"


# --------------------------------------------------------------------------- shared parts


class Provenance(Strict):
    """Spec §30 minimum provenance fields."""

    source_name: str  # registry id in data/manifests/sources.yaml, or RESEARCHX for derived records
    source_url: str | None = None
    source_identifier: str | None = None
    source_version: str | None = None
    retrieved_at: datetime | None = None
    content_hash: str | None = None  # sha256 of the raw payload the record came from
    processing_version: str  # pipeline code hash (services.datasets.store.code_version)
    extractor_version: str | None = None
    annotator_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    parent_record_ids: list[str] = []
    raw_path: str | None = None  # repo-relative path of the cached raw payload


class DatasetRecord(Strict):
    quality_tier: QualityTier
    provenance: Provenance


class Author(Strict):
    name: str
    raw_name: str | None = None
    orcid: str | None = None
    source_author_id: str | None = None


class LicenseInfo(Strict):
    content_license: str | None = None  # id in data/manifests/licenses.yaml; None = unknown
    metadata_license: str | None = None
    license_source: str | None = None  # where the license statement was read
    redistribution_allowed: bool | None = None  # None = unknown => not allowed
    training_allowed: bool | None = None  # None = unknown => do not train (spec §8)


class FulltextInfo(Strict):
    """Spec §36 per-document full-text fields."""

    full_text_available: bool
    full_text_source: str | None = None
    format: FulltextFormat | None = None
    retrieval_url: str | None = None
    retrieval_timestamp: datetime | None = None
    license_status: Literal["PERMITTED", "RESTRICTED", "UNKNOWN"] = "UNKNOWN"
    path: str | None = None
    sha256: str | None = None
    parsed_path: str | None = None


# --------------------------------------------------------------------------- scholarly records


class SourceRecord(DatasetRecord):
    """One record as delivered by one source (OpenAlex work, S2 paper, arXiv entry, PwC paper)."""

    record_type: Literal["source_record"] = "source_record"
    source_record_id: str  # "<source>:<native id>", e.g. openalex:W3168867926, arxiv:2106.09685v2
    source_name: str
    source_identifier: str
    source_url: str | None = None
    source_version: str | None = None
    record_kind: RecordKind = RecordKind.PAPER_METADATA
    title: str | None = None
    abstract: str | None = None
    authors: list[Author] = []
    publication_date: str | None = None  # as given by the source (YYYY-MM-DD or YYYY)
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None  # lowercase, no URL prefix
    arxiv_id: str | None = None  # unversioned
    arxiv_version: str | None = None  # e.g. "v2"
    identifiers: dict[str, str] = {}
    urls: dict[str, str] = {}
    references: list[str] = []  # source-native ids of referenced works
    cited_by_count: int | None = None
    topics: list[dict[str, Any]] = []  # source labels: candidate features only, never truth
    open_access: dict[str, Any] = {}
    license: LicenseInfo = LicenseInfo()
    fulltext: FulltextInfo | None = None
    extra: dict[str, Any] = {}
    quality_tier: QualityTier = QualityTier.BRONZE


class MergeEvidence(Strict):
    source_record_id: str
    matched_on: Literal["arxiv_id", "doi", "self"]
    value: str


class Relevance(Strict):
    is_relevant: bool
    areas: list[str] = []
    matched_terms: dict[str, list[str]] = {}
    field_check: str | None = None
    rule_version: str


class Paper(DatasetRecord):
    """Identifier-merged view of SourceRecords for one paper (spec §4.2, §29)."""

    record_type: Literal["paper"] = "paper"
    paper_id: str
    study_id: str
    title: str
    normalized_title: str
    abstract: str | None = None
    authors: list[Author] = []
    publication_date: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    identifiers: dict[str, str] = {}
    source_record_ids: list[str]
    field_sources: dict[str, str] = {}  # field -> source_record_id chosen by documented precedence
    merge_evidence: list[MergeEvidence]
    metadata_conflicts: list[dict[str, Any]] = []  # preserved, never overwritten (spec §44)
    relevance: Relevance | None = None
    license: LicenseInfo = LicenseInfo()
    fulltext: FulltextInfo | None = None
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.BRONZE


class Study(DatasetRecord):
    record_type: Literal["study"] = "study"
    study_id: str
    title: str
    paper_ids: list[str]
    source_record_ids: list[str]
    status: Literal["PROVISIONAL", "VERIFIED"] = "PROVISIONAL"
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.BRONZE


class CitationEdge(DatasetRecord):
    record_type: Literal["citation"] = "citation"
    citing_paper_id: str
    cited_paper_id: str
    source_record_id: str  # the source record that asserted the reference
    quality_tier: QualityTier = QualityTier.BRONZE


# --------------------------------------------------------------------------- experiment schema (spec §10)


class FieldValue(Strict):
    status: FieldStatus = FieldStatus.UNKNOWN
    raw_value: str | None = None  # as written in the source (spec §13: never destroy original)
    value: Any = None  # normalized value
    unit: str | None = None  # controlled vocabulary, services.canonicalization.normalize.UNITS
    normalization_rule: str | None = None
    normalization_version: str | None = None
    evidence_ids: list[str] = []
    derived: bool = False
    derivation: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> FieldValue:
        if self.status in (FieldStatus.UNKNOWN, FieldStatus.NOT_APPLICABLE):
            if self.value is not None or self.raw_value is not None:
                raise ValueError(f"{self.status} field must not carry a value")
        if self.status is FieldStatus.KNOWN and not self.evidence_ids:
            raise ValueError("KNOWN field requires at least one evidence id")
        if self.status is FieldStatus.CONFLICTING and len(self.evidence_ids) < 2:
            raise ValueError("CONFLICTING field requires evidence from both sides")
        if self.derived and not self.derivation:
            raise ValueError("derived value must record its derivation")
        return self


# Field names mirror DATASET_SPEC §10 exactly. Outcome values (metric, score, uncertainty,
# confidence_interval, significance) live on OutcomeObservation (spec §2) so one experimental
# condition can report several metrics; see planning/DECISIONS.md.
FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "research": ("research_question", "hypothesis", "task", "subtask", "domain"),
    "dataset": (
        "dataset_name", "dataset_version", "split", "train_data", "validation_data", "test_data",
        "sample_count", "sampling_method", "filtering", "preprocessing", "augmentation",
        "deduplication", "data_leakage_controls",
        "dataset_source", "split_construction",  # 0.2.0
    ),
    "population": (
        "users", "documents", "queries", "languages", "domains", "task_populations",
        "benchmark_subsets", "dataset_cohorts", "evaluation_groups",
    ),
    "model": (
        "model_family", "model_name", "model_version", "checkpoint", "parameter_count",
        "architecture", "initialization", "pretrained_source", "base_model",
        "active_parameter_count", "tokenizer", "context_length",  # 0.2.0
    ),
    "method": (
        "method_name", "algorithm", "objective", "loss_function", "training_strategy",
        "baseline", "ablation", "modifications",
    ),
    "training": (
        "optimizer", "learning_rate", "learning_rate_schedule", "batch_size",
        "gradient_accumulation", "epochs", "steps", "warmup_steps", "warmup_ratio",
        "weight_decay", "dropout", "seed", "precision", "quantization", "gradient_clipping",
        "lora", "qlora", "dora", "other_parameter_efficient_configuration",
        "peft_rank", "peft_alpha", "peft_target_modules",  # 0.2.0
    ),
    "retrieval": (  # 0.2.0: RAG / retrieval configuration (deepening §6)
        "retriever", "retrieval_model", "embedding_model", "retrieval_top_k", "chunk_size",
        "chunk_overlap", "index_type", "reranker", "reranking", "retrieval_corpus",
        "retrieval_prompt", "retrieval_strategy",
    ),
    "prompting": (  # 0.2.0: prompting / agent configuration (system_prompt stays in inference)
        "prompt_template", "few_shot_examples", "chain_of_thought", "tool_usage",
        "agent_framework", "planner", "memory", "number_of_steps", "stopping_condition",
    ),
    "inference": (
        "inference_model", "temperature", "top_p", "top_k", "max_tokens", "context_length",
        "decoding_strategy", "system_prompt", "retrieval_configuration", "tool_configuration",
        "hardware",
        "batching",  # 0.2.0
    ),
    "evaluation": (
        "benchmark", "dataset_split", "metric_definition", "statistical_test",
        "evaluation_population", "evaluation_protocol",
        "aggregation_method",  # 0.2.0
    ),
    "compute": (
        "accelerator", "gpu_model", "gpu_count", "cpu", "memory", "training_time",
        "inference_time", "flops", "energy", "cost",
    ),
    "artifacts": (
        "paper_url", "repository_url", "checkpoint_url", "dataset_url",
        "supplementary_material_url", "demo_url", "benchmark_url",
        "commit", "release", "model_card_url", "configuration_file",  # 0.2.0
        "benchmark_submission_url", "experiment_logs_url",
    ),
}
OUTCOME_FIELDS = ("metric", "score", "dataset_split", "uncertainty", "confidence_interval", "significance", "p_value")

# Allowed EvidenceAssertion.field_name values.
EXPERIMENT_FIELD_PATHS = frozenset(
    [f"{g}.{f}" for g, fields in FIELD_GROUPS.items() for f in fields]
    + [f"evaluation.{f}" for f in OUTCOME_FIELDS]
)
PAPER_FIELD_PATHS = frozenset(
    f"paper.{f}" for f in ("title", "authors", "publication_year", "publication_date", "venue", "doi", "arxiv_id", "abstract")
)
FIELD_PATHS = EXPERIMENT_FIELD_PATHS | PAPER_FIELD_PATHS


def _group_model(name: str, fields: tuple[str, ...]) -> type[Strict]:
    return create_model(name, __base__=Strict, **{f: (FieldValue, Field(default_factory=FieldValue)) for f in fields})


ResearchQuestion = _group_model("ResearchQuestion", FIELD_GROUPS["research"])
DatasetConfiguration = _group_model("DatasetConfiguration", FIELD_GROUPS["dataset"])
PopulationContext = _group_model("PopulationContext", FIELD_GROUPS["population"])
ModelConfiguration = _group_model("ModelConfiguration", FIELD_GROUPS["model"])
MethodConfiguration = _group_model("MethodConfiguration", FIELD_GROUPS["method"])
TrainingConfiguration = _group_model("TrainingConfiguration", FIELD_GROUPS["training"])
RetrievalConfiguration = _group_model("RetrievalConfiguration", FIELD_GROUPS["retrieval"])
PromptingConfiguration = _group_model("PromptingConfiguration", FIELD_GROUPS["prompting"])
InferenceConfiguration = _group_model("InferenceConfiguration", FIELD_GROUPS["inference"])
EvaluationConfiguration = _group_model("EvaluationConfiguration", FIELD_GROUPS["evaluation"])
ComputeConfiguration = _group_model("ComputeConfiguration", FIELD_GROUPS["compute"])
Artifacts = _group_model("Artifacts", FIELD_GROUPS["artifacts"])
OutcomeObservation = _group_model("OutcomeObservation", OUTCOME_FIELDS)
GROUP_MODELS: dict[str, type[Strict]] = {
    "research": ResearchQuestion, "dataset": DatasetConfiguration, "population": PopulationContext,
    "model": ModelConfiguration, "method": MethodConfiguration, "training": TrainingConfiguration,
    "retrieval": RetrievalConfiguration, "prompting": PromptingConfiguration,
    "inference": InferenceConfiguration, "evaluation": EvaluationConfiguration, "compute": ComputeConfiguration,
    "artifacts": Artifacts,
}


class DimensionCoverage(Strict):
    known: int = Field(ge=0)
    total: int = Field(ge=0)


class DepthAssessment(Strict):
    """Experiment technical depth from evidence coverage, never model confidence (docs/DATASET_DEEPENING.md §2)."""

    tier: DepthTier
    dimensions: dict[str, DimensionCoverage]
    technical_fields_known: int = Field(ge=0)
    paper_text_fields: int = Field(ge=0)  # KNOWN fields with evidence in the paper's own text/tables
    artifact_fields: int = Field(ge=0)  # KNOWN fields with repository / card / run evidence
    human_reviewed_fields: int = Field(default=0, ge=0)
    depth_version: str


class QualityDimensions(Strict):
    """Scientific-quality dimensions beside the BRONZE/SILVER/GOLD tier (deepening §29). Unset = not applicable."""

    evidence_quality: QualityLevel | None = None
    configuration_completeness: QualityLevel | None = None
    experiment_attribution_quality: QualityLevel | None = None
    study_identity_quality: QualityLevel | None = None
    comparison_quality: QualityLevel | None = None
    artifact_verification_quality: QualityLevel | None = None
    quality_version: str


class Experiment(DatasetRecord):
    """One experimental condition (spec §2, §10). Outcomes hold per-metric observations."""

    record_type: Literal["experiment"] = "experiment"
    experiment_id: str
    study_id: str
    paper_id: str  # for non-paper sources (OpenML, MLPerf) a source-scoped pseudo paper id, e.g. openml:task/31
    source_record_ids: list[str]
    experiment_source: Literal["PAPER", "OPENML", "MLPERF"] = "PAPER"  # 0.2.0
    parent_experiment_id: str | None = None
    experiment_type: ExperimentType = ExperimentType.OTHER
    description: str | None = None
    research: ResearchQuestion = Field(default_factory=ResearchQuestion)
    dataset: DatasetConfiguration = Field(default_factory=DatasetConfiguration)
    population: PopulationContext = Field(default_factory=PopulationContext)
    model: ModelConfiguration = Field(default_factory=ModelConfiguration)
    method: MethodConfiguration = Field(default_factory=MethodConfiguration)
    training: TrainingConfiguration = Field(default_factory=TrainingConfiguration)
    retrieval: RetrievalConfiguration = Field(default_factory=RetrievalConfiguration)  # 0.2.0
    prompting: PromptingConfiguration = Field(default_factory=PromptingConfiguration)  # 0.2.0
    inference: InferenceConfiguration = Field(default_factory=InferenceConfiguration)
    evaluation: EvaluationConfiguration = Field(default_factory=EvaluationConfiguration)
    compute: ComputeConfiguration = Field(default_factory=ComputeConfiguration)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    outcomes: list[OutcomeObservation] = []
    extraction_method: ExtractionMethod
    extraction_model: str | None = None
    extraction_run_id: str
    confidence: float | None = Field(default=None, ge=0, le=1)  # heuristic, uncalibrated
    uncertainty: str | None = None
    unresolved_fields: list[str] = []
    cross_check: dict[str, Any] | None = None
    depth: DepthAssessment | None = None  # 0.2.0
    quality: QualityDimensions | None = None  # 0.2.0
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    split: Split = Split.UNASSIGNED
    synthetic: bool = False
    quality_tier: QualityTier = QualityTier.SILVER

    def iter_fields(self):
        """Yield (path, FieldValue) for every configuration field (not outcomes)."""
        for group, names in FIELD_GROUPS.items():
            section = getattr(self, group)
            for name in names:
                yield f"{group}.{name}", getattr(section, name)


class EvidenceAssertion(DatasetRecord):
    """Spec §11 minimum fields (+ char offsets and license flag for the quoted text)."""

    record_type: Literal["evidence"] = "evidence"
    evidence_id: str
    source_record_id: str
    study_id: str | None = None
    experiment_id: str | None = None  # None = study-level evidence, not yet assigned to an experiment
    paper_id: str | None = None
    field_name: str
    extracted_value: str | None = None
    normalized_value: Any = None
    unit: str | None = None
    normalization_rule: str | None = None
    normalization_version: str | None = None
    evidence_text: str
    source_location: str | None = None
    page: int | None = None
    section: str | None = None
    paragraph: str | None = None
    table: str | None = None
    figure: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    # 0.2.0: smallest reliable location + scope of the stated setting
    location_kind: LocationKind | None = None
    cell: str | None = None  # parsed table cell id "<table_id>/r<i>c<j>"
    sentence: str | None = None  # parsed sentence id
    footnote: str | None = None
    is_appendix: bool | None = None
    scope: EvidenceScope | None = None
    applies_to: str | None = None  # model / method / table the stated setting applies to (scope MODEL/TABLE)
    artifact_id: str | None = None  # set when the evidence comes from a repository / card / run artifact
    source_version: str | None = None
    retrieval_timestamp: datetime | None = None
    extraction_model: str
    extraction_run_id: str
    status: EvidenceStatus = EvidenceStatus.MACHINE_EXTRACTED
    confidence: float | None = Field(default=None, ge=0, le=1)
    derived: bool = False
    derivation: str | None = None
    redistribution_allowed: bool | None = None  # of evidence_text, from the source license
    quality_tier: QualityTier = QualityTier.SILVER

    @field_validator("field_name")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in FIELD_PATHS:
            raise ValueError(f"unknown field_name {v!r}")
        return v

    @model_validator(mode="after")
    def _invariants(self) -> EvidenceAssertion:
        if self.derived and not self.derivation:
            raise ValueError("derived evidence must record its derivation")
        if not self.evidence_text.strip():
            raise ValueError("evidence_text must not be empty")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start/char_end must be set together")
        if self.char_start is not None and self.char_end < self.char_start:
            raise ValueError("char_end < char_start")
        return self


# --------------------------------------------------------------------------- relationship datasets


class StudyLink(DatasetRecord):
    """Spec §14. Candidate generation is not Gold labeling."""

    record_type: Literal["study_link"] = "study_link"
    link_id: str
    record_a: str  # paper_id
    record_b: str
    relationship: StudyLinkLabel
    candidate_methods: list[str]
    evidence: list[dict[str, Any]]  # structured items: {kind, detail, source_record_id}
    rationale: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    is_hard_negative: bool = False
    construction_method: str | None = None
    features: dict[str, float] = {}  # similarity features: candidate generation only (spec §41)
    labeler_version: str
    quality: QualityDimensions | None = None  # 0.2.0 (study_identity_quality)
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class FieldComparison(Strict):
    """Spec §16."""

    field_name: str
    value_a: Any = None
    value_b: Any = None
    comparison_status: FieldComparisonStatus
    evidence_a: list[str] = []
    evidence_b: list[str] = []
    rationale: str


class Comparison(DatasetRecord):
    """Spec §15. Relationships, not scores."""

    record_type: Literal["comparison"] = "comparison"
    comparison_id: str
    experiment_a: str
    experiment_b: str
    relationship_label: ComparisonLabel
    field_comparisons: list[FieldComparison]
    evidence: list[str] = []
    rationale: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: str | None = None
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    is_hard_negative: bool = False
    hard_negative_type: str | None = None  # construction method (spec §17)
    candidate_generation: dict[str, Any] = {}
    labeler_version: str
    quality: QualityDimensions | None = None  # 0.2.0 (comparison_quality)
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class Contradiction(DatasetRecord):
    """Spec §18 / §44: both claims preserved; no automatic winner."""

    record_type: Literal["contradiction"] = "contradiction"
    contradiction_id: str
    subject_id: str  # paper_id or experiment_id the claims are about
    claim: str
    field: str
    source_a: str
    source_b: str
    value_a: Any = None
    value_b: Any = None
    conflict_type: ConflictType
    evidence_a: list[str]
    evidence_b: list[str]
    explanation: str
    resolution: str | None = None
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolution_evidence: list[str] = []
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER

    @model_validator(mode="after")
    def _resolution(self) -> Contradiction:
        if self.resolution_status is ResolutionStatus.RESOLVED and not (self.resolution and self.resolution_evidence):
            raise ValueError("RESOLVED contradiction requires resolution and resolution_evidence")
        return self


class AbstentionCase(DatasetRecord):
    """Spec §19."""

    record_type: Literal["abstention"] = "abstention"
    case_id: str
    question: str
    question_template: str | None = None
    synthetic_question: bool = False  # question text templated; underlying evidence is real
    context_ids: list[str]  # comparison / experiment / paper ids
    evidence_class: EvidenceClass
    expected_behavior: ExpectedBehavior
    missing_information: list[str] = []
    conflicting_fields: list[str] = []
    evidence_ids: list[str] = []
    rationale: str
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class LayaDecision(DatasetRecord):
    """Spec §20. Laya chooses actions; the evidence system decides what evidence exists."""

    record_type: Literal["laya_decision"] = "laya_decision"
    decision_id: str
    current_state: dict[str, Any]
    research_goal: str
    evidence_available: list[str] = []
    evidence_quality: str | None = None
    missing_information: list[str] = []
    candidate_tools: list[str] = []
    previous_actions: list[str] = []
    budget: dict[str, Any] | None = None
    latency_budget: float | None = None
    current_confidence: float | None = None
    next_action: LayaAction
    rationale: str
    expected_information_gain: float | None = None
    final_outcome: str | None = None
    synthetic: bool = False
    generation: dict[str, Any] | None = None  # spec §39 when synthetic
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER

    @model_validator(mode="after")
    def _synthetic(self) -> LayaDecision:
        if self.synthetic and not self.generation:
            raise ValueError("synthetic record requires generation metadata (spec §39)")
        return self


class TrajectoryStep(DatasetRecord):
    """Spec §21 (one record per step). Never store secrets or raw tool arguments."""

    record_type: Literal["trajectory_step"] = "trajectory_step"
    trajectory_id: str
    task_id: str
    state_id: str
    step_index: int = Field(ge=0)
    timestamp: datetime
    action: LayaAction
    tool: str | None = None
    tool_arguments_hash: str | None = None
    tool_result_summary: str | None = None
    evidence_added: list[str] = []
    confidence_before: float | None = None
    confidence_after: float | None = None
    next_state: str | None = None
    final_decision: str | None = None
    outcome: str | None = None
    cost: float | None = None
    latency: float | None = None  # seconds
    failure_type: FailureCategory | None = None
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


# --------------------------------------------------------------------------- operations records


class FailureCase(Strict):
    """Spec §24."""

    record_type: Literal["failure"] = "failure"
    failure_id: str
    category: FailureCategory
    stage: str
    run_id: str | None = None
    timestamp: datetime
    input: str  # summary only; no secrets, no full proprietary inputs
    expected_behavior: str
    observed_behavior: str
    root_cause: str | None = None
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    reproducibility: Literal["ALWAYS", "INTERMITTENT", "UNKNOWN"] = "UNKNOWN"
    regression_test_created: bool = False
    details: dict[str, Any] = {}


class ReviewDecision(Strict):
    """One reviewer judgment (spec §26). Disagreements are preserved, never erased."""

    record_type: Literal["review_decision"] = "review_decision"
    decision_id: str
    target_type: str
    target_id: str
    reviewer_id: str
    reviewer_type: ReviewerType
    reviewer_role: ReviewerRole = ReviewerRole.ANNOTATOR
    decision: ReviewDecisionValue
    corrected_field: str | None = None
    corrected_value: Any = None
    comment: str | None = None
    guideline_version: str
    timestamp: datetime
    annotation_seconds: float | None = Field(default=None, ge=0)  # 0.2.0: time spent on the item
    disagreement_type: str | None = None  # 0.2.0: set by adjudicators (e.g. SCOPE, VALUE, LABEL, EVIDENCE)

    @model_validator(mode="after")
    def _correction(self) -> ReviewDecision:
        if self.decision is ReviewDecisionValue.CORRECT and not self.corrected_field:
            raise ValueError("CORRECT decision requires corrected_field")
        return self


class PromotionRecord(Strict):
    """Every tier promotion is recorded (spec §25)."""

    record_type: Literal["promotion"] = "promotion"
    promotion_id: str
    target_type: str
    target_id: str
    from_tier: QualityTier
    to_tier: QualityTier
    decision_ids: list[str]
    promoted_at: datetime
    promoted_by: str
    reason: str


# --------------------------------------------------------------------------- parsed full text


class Section(Strict):
    section_id: str
    number: str | None = None
    title: str
    level: int
    section_type: str
    parent_id: str | None = None
    is_appendix: bool = False  # 0.2.0


class Paragraph(Strict):
    paragraph_id: str
    section_id: str | None = None
    text: str
    page: int | None = None


class Sentence(Strict):
    """0.2.0: sentence span inside a paragraph (offsets are into Paragraph.text)."""

    sentence_id: str  # "<paragraph_id>.s<k>"
    paragraph_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str


class TableCell(Strict):
    """0.2.0: one addressable cell after rowspan/colspan expansion (deepening §18)."""

    cell_id: str  # "<table_id>/r<row>c<col>"
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str
    is_header: bool = False
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    spanned_from: str | None = None  # cell_id of the origin cell when this position is covered by a span
    row_header_path: list[str] = []  # outermost first
    col_header_path: list[str] = []
    value: float | None = None  # parsed number (raw text kept in `text`)
    unit: str | None = None
    uncertainty: float | None = None  # "71.2 ± 0.3" -> 0.3


class Table(Strict):
    table_id: str
    section_id: str | None = None
    caption: str
    rows: list[list[str]]
    header_rows: int = 0
    page: int | None = None
    cells: list[TableCell] = []  # 0.2.0
    header_cols: int = 0  # 0.2.0
    table_type: Literal["RESULTS", "HYPERPARAMETERS", "DATASET_STATISTICS", "OTHER"] = "OTHER"  # 0.2.0
    notes: list[str] = []  # 0.2.0: table footnotes / notes text


class Footnote(Strict):
    """0.2.0"""

    footnote_id: str
    paragraph_id: str | None = None  # paragraph containing the footnote mark
    text: str


class Figure(Strict):
    """0.2.0: figures are captions only; images are not interpreted."""

    figure_id: str
    section_id: str | None = None
    caption: str


class ParsedDocument(Strict):
    """Parsed full text. Stored beside the original under data/fulltext/<class>/ (license follows source)."""

    record_type: Literal["parsed_document"] = "parsed_document"
    document_id: str
    paper_id: str
    source_record_id: str
    format: FulltextFormat
    source_url: str
    source_path: str
    source_sha256: str
    parser_version: str
    parsed_at: datetime
    title: str | None = None
    abstract: str | None = None
    sections: list[Section] = []
    paragraphs: list[Paragraph] = []
    tables: list[Table] = []
    sentences: list[Sentence] = []  # 0.2.0
    footnotes: list[Footnote] = []  # 0.2.0
    figures: list[Figure] = []  # 0.2.0
    license: LicenseInfo = LicenseInfo()
    warnings: list[str] = []


class ReviewItem(Strict):
    """A snapshot of a Silver record queued for human review (data/annotations/queue/<batch>.jsonl)."""

    record_type: Literal["review_item"] = "review_item"
    item_id: str
    batch_id: str
    target_type: str
    target_id: str
    priority_reason: str
    guideline_version: str
    questions: list[str]
    record: dict[str, Any]
    evidence: list[dict[str, Any]] = []


class FulltextIndexEntry(Strict):
    """One row of data/normalized/sources/fulltext_index.jsonl (what was fetched, licence, storage)."""

    record_type: Literal["fulltext_index"] = "fulltext_index"
    arxiv_id: str
    fulltext: FulltextInfo
    license: LicenseInfo


# --------------------------------------------------------------------------- schema 0.2.0 record types


class ResultRecord(DatasetRecord):
    """Result lineage (deepening §17): where a number came from. Attached to experiments only via associations."""

    record_type: Literal["result"] = "result"
    result_id: str
    paper_id: str
    study_id: str
    source_kind: ResultSourceKind
    source_record_id: str  # paper source record, artifact id or run id
    model: str | None = None  # row label / model name as written
    method: str | None = None
    dataset: str | None = None  # as written (column header / card)
    dataset_split: str | None = None
    benchmark_version: str | None = None
    metric: str | None = None  # as written
    raw_value: str
    value: float | None = None
    unit: str | None = None
    uncertainty: float | None = None
    checkpoint: str | None = None
    artifact_id: str | None = None
    run_id: str | None = None
    table: str | None = None
    cell: str | None = None
    row_header_path: list[str] = []
    col_header_path: list[str] = []
    caption: str | None = None
    source_location: str
    evidence_ids: list[str] = []
    extraction_version: str
    redistribution_allowed: bool | None = None
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class ExperimentResultAssociation(DatasetRecord):
    """Which result belongs to which experiment (deepening §19). Never attach every score to every experiment."""

    record_type: Literal["association"] = "association"
    association_id: str
    result_id: str
    experiment_id: str | None = None
    status: AssociationStatus
    candidate_experiment_ids: list[str] = []
    matched_fields: list[str] = []
    mismatched_fields: list[str] = []
    rationale: str
    evidence_ids: list[str] = []
    method_version: str
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    quality_tier: QualityTier = QualityTier.SILVER

    @model_validator(mode="after")
    def _status(self) -> ExperimentResultAssociation:
        if self.status in (AssociationStatus.VERIFIED, AssociationStatus.LIKELY, AssociationStatus.CONFLICTING) and not self.experiment_id:
            raise ValueError(f"{self.status} association requires experiment_id")
        if self.status is AssociationStatus.AMBIGUOUS and len(self.candidate_experiment_ids) < 2:
            raise ValueError("AMBIGUOUS association requires >= 2 candidate experiments")
        if self.status is AssociationStatus.UNKNOWN and self.experiment_id:
            raise ValueError("UNKNOWN association must not name an experiment")
        return self


class Artifact(DatasetRecord):
    """A research artifact observed at a pinned version (repository commit, card revision, run)."""

    record_type: Literal["artifact"] = "artifact"
    artifact_id: str
    artifact_type: ArtifactType
    url: str
    repository: str | None = None  # owner/repo or HF repo id
    commit: str | None = None  # git sha or HF revision
    path: str | None = None  # file path inside the repository
    parent_artifact_id: str | None = None
    linked_paper_ids: list[str] = []
    link_basis: str  # PWC_OFFICIAL | PWC_MENTIONED | PAPER_TEXT_URL | HF_PAPER_LINK | HF_ARXIV_TAG | OPENML | MLPERF | ...
    official: bool | None = None  # True only with paper evidence that this is the authors' artifact
    license: str | None = None
    content_sha256: str | None = None
    bytes: int | None = None
    raw_path: str | None = None  # repo-relative cached copy (re-verification of artifact evidence)
    retrieved_at: datetime | None = None
    quality_tier: QualityTier = QualityTier.BRONZE


class ArtifactEvidence(DatasetRecord):
    """Field-level relation between paper claims and artifact contents (deepening §16)."""

    record_type: Literal["artifact_evidence"] = "artifact_evidence"
    artifact_evidence_id: str
    artifact_id: str
    paper_id: str
    study_id: str
    experiment_id: str | None = None
    field_name: str
    relation: ArtifactRelation
    artifact_value: Any = None
    paper_value: Any = None
    artifact_evidence_ids: list[str] = []
    paper_evidence_ids: list[str] = []
    rationale: str
    method_version: str
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    quality_tier: QualityTier = QualityTier.SILVER

    @field_validator("field_name")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in FIELD_PATHS:
            raise ValueError(f"unknown field_name {v!r}")
        return v

    @model_validator(mode="after")
    def _evidence(self) -> ArtifactEvidence:
        need_artifact = self.relation in (ArtifactRelation.PRESENT_IN_REPOSITORY, ArtifactRelation.CONFIRMED_BY_BOTH,
                                          ArtifactRelation.CONFLICTING, ArtifactRelation.RELATED_BUT_UNCONFIRMED)
        need_paper = self.relation in (ArtifactRelation.CLAIMED_IN_PAPER, ArtifactRelation.CONFIRMED_BY_BOTH, ArtifactRelation.CONFLICTING)
        if need_artifact and not self.artifact_evidence_ids:
            raise ValueError(f"{self.relation} requires artifact evidence")
        if need_paper and not self.paper_evidence_ids:
            raise ValueError(f"{self.relation} requires paper evidence")
        return self


class CanonicalEntity(DatasetRecord):
    """Canonical concept with every original surface form kept (deepening §37)."""

    record_type: Literal["canonical_entity"] = "canonical_entity"
    canonical_id: str
    entity_type: CanonicalEntityType
    normalized_name: str
    preferred_name: str
    surface_forms: list[dict[str, Any]]  # {original_name, count}
    external_ids: dict[str, str] = {}
    normalization_rule: str
    normalization_version: str
    quality_tier: QualityTier = QualityTier.SILVER


class GraphNode(Strict):
    """Paper-centred knowledge graph node (derived; regenerated from records each build)."""

    record_type: Literal["graph_node"] = "graph_node"
    node_id: str
    node_type: GraphNodeType
    ref_id: str  # id of the record or canonical entity the node stands for
    label: str | None = None
    graph_version: str


class GraphEdge(Strict):
    record_type: Literal["graph_edge"] = "graph_edge"
    edge_id: str
    edge_type: GraphEdgeType
    source: str  # node_id
    target: str  # node_id
    derived_from: list[str]  # record / evidence ids the edge was derived from
    graph_version: str


class ResearchTask(DatasetRecord):
    """A realistic research task over real records (deepening §24). Wording may be templated; evidence is real."""

    record_type: Literal["research_task"] = "research_task"
    task_id: str
    task_type: ResearchTaskType
    question: str
    proposed_experiment: dict[str, Any] = {}  # field path -> value
    context_ids: list[str]
    evidence_ids: list[str] = []
    decisive_fields: list[str] = []
    expected_behavior: ExpectedBehavior
    expected_answer: str | None = None
    rationale: str
    synthetic_text: bool = True
    generator_version: str
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class HardCase(DatasetRecord):
    """Lexical similarity vs scientific equivalence case with field-level justification (deepening §25)."""

    record_type: Literal["hard_case"] = "hard_case"
    case_id: str
    case_type: HardCaseType
    record_kind: Literal["experiment", "paper"]
    record_a: str
    record_b: str
    lexical_similarity: float = Field(ge=0, le=1)
    scientific_relation: str  # ComparisonLabel / StudyLinkLabel value
    matching_fields: list[str] = []
    differing_fields: list[str] = []
    unknown_fields: list[str] = []
    rationale: str
    evidence_ids: list[str] = []
    generator_version: str
    reviewer_status: ReviewerStatus = ReviewerStatus.UNREVIEWED
    split: Split = Split.UNASSIGNED
    quality_tier: QualityTier = QualityTier.SILVER


class DiscoveryCandidate(DatasetRecord):
    """Layer 0 discovery corpus row (compact; full metadata stays in the raw OAI cache)."""

    record_type: Literal["discovery_candidate"] = "discovery_candidate"
    candidate_id: str  # cand:arxiv:<id>
    arxiv_id: str
    source_record_id: str  # arxivoai:<id>
    title: str
    v1_date: str | None = None
    latest_version: str | None = None
    n_versions: int = Field(default=0, ge=0)
    primary_category: str | None = None
    categories: list[str] = []
    license_id: str | None = None
    license_url: str | None = None
    journal_ref: str | None = None
    published_doi: str | None = None
    code_urls: list[str] = []
    areas: list[str]
    matched_terms: dict[str, list[str]] = {}
    experiment_signals: list[str] = []
    priority_score: float
    score_components: dict[str, float] = {}
    in_v010_corpus: bool = False
    promoted: bool = False
    selection_version: str
    quality_tier: QualityTier = QualityTier.BRONZE


# record_type -> model, used for validation dispatch and JSON Schema export (file stem).
RECORD_MODELS: dict[str, tuple[type[BaseModel], str]] = {
    "source_record": (SourceRecord, "source_record"),
    "paper": (Paper, "paper"),
    "study": (Study, "study"),
    "citation": (CitationEdge, "citation"),
    "experiment": (Experiment, "experiment"),
    "evidence": (EvidenceAssertion, "evidence_assertion"),
    "study_link": (StudyLink, "study_link"),
    "comparison": (Comparison, "comparison"),
    "contradiction": (Contradiction, "contradiction"),
    "abstention": (AbstentionCase, "abstention"),
    "laya_decision": (LayaDecision, "laya_decision"),
    "trajectory_step": (TrajectoryStep, "trajectory"),
    "failure": (FailureCase, "failure"),
    "review_decision": (ReviewDecision, "review_decision"),
    "promotion": (PromotionRecord, "promotion"),
    "parsed_document": (ParsedDocument, "parsed_document"),
    "fulltext_index": (FulltextIndexEntry, "fulltext_index"),
    "review_item": (ReviewItem, "review_item"),
    # 0.2.0
    "result": (ResultRecord, "result"),
    "association": (ExperimentResultAssociation, "association"),
    "artifact": (Artifact, "artifact"),
    "artifact_evidence": (ArtifactEvidence, "artifact_evidence"),
    "canonical_entity": (CanonicalEntity, "canonical_entity"),
    "graph_node": (GraphNode, "graph_node"),
    "graph_edge": (GraphEdge, "graph_edge"),
    "research_task": (ResearchTask, "research_task"),
    "hard_case": (HardCase, "hard_case"),
    "discovery_candidate": (DiscoveryCandidate, "discovery_candidate"),
}
