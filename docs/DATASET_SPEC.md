# ResearchX Dataset Specification

Version: 1.0
Status: Active
Owner: ResearchX ML/Data Pipeline
Primary domain: ML/CS scientific research
Primary objective: Build a reproducible, evidence-grounded dataset that teaches and evaluates ResearchX at the experiment-comparison level.

---

# 1. PURPOSE

ResearchX is not a generic scientific QA system and is not primarily a document-similarity system.

The dataset must support the following core question:

> Has this exact experiment — including task, population/context, intervention/method, conditions, dataset, evaluation protocol, and outcome — already been tested, and if something close exists, exactly how does it differ?

The dataset must therefore teach and evaluate:

1. Scientific paper and source discovery.
2. Structured extraction of experimental configurations.
3. Evidence-grounded extraction.
4. Linking multiple records to the same underlying study.
5. Comparing experiments field-by-field.
6. Distinguishing similarity from scientific equivalence.
7. Detecting materially different experimental conditions.
8. Detecting contradictions between sources.
9. Handling missing information explicitly.
10. Abstaining when evidence is insufficient.
11. Choosing the next research/tool action.
12. Producing evidence-backed research outputs.
13. Preserving provenance and auditability.

The dataset is a core product asset and must be treated as a versioned data product, not as a temporary collection of downloaded papers.

---

# 2. CORE PRINCIPLE

ResearchX must not be trained or evaluated as:

    document A is similar to document B

ResearchX must instead operate on:

    SourceRecord
        -> Study
            -> Experiment
                -> ExperimentalCondition
                    -> DatasetConfiguration
                    -> ModelConfiguration
                    -> TrainingConfiguration
                    -> InferenceConfiguration
                    -> EvaluationConfiguration
                    -> OutcomeObservation
                        -> EvidenceAssertion

Comparison must happen at field level.

Similarity is not equivalence.

A high textual or embedding similarity score must never by itself create a SAME_EXPERIMENT or SAME_STUDY label.

Missing information is UNKNOWN.

UNKNOWN must never be silently converted to MATCH.

---

# 3. DATASET ARCHITECTURE

ResearchX will use a layered dataset architecture rather than one monolithic dataset.

    External Scholarly Sources
            |
            v
    Raw Acquisition Layer
            |
            v
    Normalized Research Corpus
            |
            +----------------------------+
            |                            |
            v                            v
    Scientific IE Data             External Benchmarks
            |                            |
            v                            |
    Experiment Extraction               |
            |                            |
            v                            |
    Evidence Assertions                 |
            |                            |
            +-------------+--------------+
                          |
                          v
                  Study Linking
                          |
                          v
                Experiment Comparison
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
      Contradiction   Abstention    Hard Negatives
            |             |             |
            +-------------+-------------+
                          |
                          v
                  Laya Decision Data
                          |
                          v
                  Agent Trajectories
                          |
                          v
                   ResearchX Gold
                     Evaluation

The final ResearchX dataset consists of multiple coordinated datasets sharing common identifiers and provenance.

---

# 4. DATASET LAYERS

The repository must contain the following logical layers.

## 4.1 Raw external data

Purpose:
- Preserve downloaded source material.
- Preserve original source structure whenever legally permissible.
- Make transformations reproducible.

Examples:
- OpenAlex records
- Semantic Scholar records
- arXiv metadata/source/PDF where permitted
- Papers With Code metadata
- Existing benchmark datasets

Raw data must never be directly treated as Gold labels.

---

## 4.2 Normalized scholarly corpus

Purpose:
- Create a unified ResearchX representation of papers and scholarly records.

This layer must normalize:
- identifiers
- titles
- abstracts
- authors
- publication dates
- venue
- DOI
- arXiv ID
- citation relationships
- references
- URLs
- source provenance
- availability/licensing metadata

---

## 4.3 Scientific information extraction data

Purpose:
- Bootstrap scientific terminology and information extraction.

Use existing scientific IE datasets as auxiliary resources.

Candidate sources include:
- SciREX
- ScienceIE
- SciRIFF
- SciRepEval
- QASPER
- SciFact
- SciCite

These datasets do not define the ResearchX task.

They are supporting datasets for:
- entity extraction
- relation extraction
- scientific QA
- evidence retrieval
- claim verification
- scientific language understanding

---

## 4.4 ResearchX experiment dataset

This is the central original dataset.

Each verified experiment record must represent an actual experimental configuration rather than a paper-level summary.

An experiment may be a:
- main experiment
- baseline
- ablation
- sensitivity study
- scaling experiment
- benchmark evaluation
- transfer experiment
- robustness experiment
- efficiency experiment
- inference experiment
- fine-tuning experiment
- retrieval experiment
- agent evaluation
- multimodal evaluation

---

## 4.5 ResearchX comparison dataset

This dataset contains explicit relationships between pairs of experiments.

It is one of the most important ResearchX datasets.

Each comparison must include:
- experiment A
- experiment B
- relationship label
- field-level comparison
- evidence
- rationale
- confidence
- uncertainty
- reviewer status

---

## 4.6 Study-linking dataset

This dataset links multiple SourceRecords to the same underlying investigation.

Examples:
- preprint -> conference paper
- workshop paper -> journal extension
- registry -> publication
- paper -> follow-up report
- technical report -> benchmark paper
- multiple reports from one study

---

## 4.7 Contradiction dataset

This dataset captures cases where two sources make incompatible claims.

---

## 4.8 Abstention dataset

This dataset deliberately contains cases where the correct behavior is:

    INSUFFICIENT_EVIDENCE
    UNCERTAIN
    ABSTAIN
    ESCALATE

ResearchX must not be forced to produce a false binary decision when the sources do not support one.

---

## 4.9 Laya decision dataset

This dataset is specifically for the decision/orchestration layer.

It must represent:

    current state
    available evidence
    missing information
    previous actions
    available tools
    budget/cost
    confidence
        ->
    next action

Laya is a decision/routing layer, not the scientific source of truth.

---

## 4.10 Agent trajectory dataset

Store complete or selectively summarized ResearchX trajectories:

    State
      ->
    Action
      ->
    Tool
      ->
    Tool result
      ->
    Evidence added
      ->
    State update
      ->
    Next action
      ->
    Decision
      ->
    Outcome
      ->
    Cost / latency

This dataset can later support:
- imitation learning
- routing optimization
- Laya fine-tuning
- policy evaluation
- failure analysis

---

# 5. PRIMARY RESEARCH DOMAIN

The initial ResearchX domain is:

    MACHINE LEARNING / COMPUTER SCIENCE

Priority areas:

1. Large language models.
2. Retrieval-augmented generation.
3. Information retrieval.
4. AI agents.
5. Fine-tuning.
6. Parameter-efficient fine-tuning.
7. LoRA / QLoRA / DoRA.
8. Model evaluation.
9. Multimodal models.
10. Quantization.
11. Efficient inference.
12. Model compression.
13. AI safety/evaluation.
14. Reasoning models.
15. Benchmark methodology.
16. Scientific machine learning where relevant.

The domain must remain configurable in the pipeline.

Do not hard-code ML/CS assumptions into the core data model where a more general scientific representation is possible.

---

# 6. EXTERNAL DATA SOURCES

The acquisition pipeline should support the following sources.

## 6.1 OpenAlex

Purpose:
- scholarly discovery
- metadata
- citation relationships
- authors
- institutions
- topics
- publication metadata
- open-access metadata

Use:
- candidate corpus generation
- metadata enrichment
- scholarly graph construction

Acquisition modes:
- API
- snapshot when justified by scale

Do NOT download the complete OpenAlex snapshot for the initial ResearchX prototype unless there is a demonstrated need.

The initial target is a selective ML/CS candidate corpus.

Full text must be treated separately from metadata and must respect source-level copyright/licensing.

Reference:
https://openalex.org/
https://docs.openalex.org/

---

## 6.2 Semantic Scholar

Purpose:
- scholarly metadata
- citation graph
- paper relationships
- author/entity enrichment
- downloadable scholarly datasets

Use:
- metadata enrichment
- candidate linking
- citation relationships
- duplicate resolution
- study relationship candidates

Reference:
https://www.semanticscholar.org/
https://www.semanticscholar.org/product/api
https://www.semanticscholar.org/product/api/datasets

---

## 6.3 arXiv

Purpose:
- ML/CS paper discovery
- preprints
- metadata
- full text where permitted

Use:
- candidate corpus
- full-text scientific extraction
- version tracking

The system must distinguish:
- arXiv identifier
- version
- source date
- retrieved date

Do not blindly ingest source archives without inspection.

Reference:
https://arxiv.org/
https://info.arxiv.org/help/api/index.html

---

## 6.4 Papers With Code

Purpose:
- paper/task relationships
- datasets
- benchmarks
- evaluations
- metrics

Use:
- experiment enrichment
- benchmark normalization
- task/evaluation mapping
- candidate experiment generation

Reference:
https://paperswithcode.com/

The acquisition process must respect the current availability, API behavior, licensing, and terms of the source.

---

## 6.5 S2ORC / Semantic Scholar scientific corpora

Purpose:
- structured scientific full text
- scientific document processing
- section-level extraction
- auxiliary scientific NLP

Use:
- extraction bootstrapping
- scientific document parsing
- auxiliary training/evaluation where licensing permits

Reference:
https://www.semanticscholar.org/

---

# 7. EXISTING RESEARCH DATASETS

Existing datasets must be treated as supporting datasets, not as the ResearchX Gold dataset.

## 7.1 Scientific extraction

Candidate sources:
- SciREX
- ScienceIE
- SciRIFF
- SciRepEval

Use cases:
- scientific entity extraction
- scientific relation extraction
- scientific document understanding
- retrieval
- instruction-following

---

## 7.2 Scientific QA and evidence

Candidate sources:
- QASPER
- SciFact
- ResearchQA

Use cases:
- evidence retrieval
- scientific question answering
- claim verification
- research information needs
- evidence-backed generation

---

## 7.3 Retrieval

Candidate sources:
- BRIGHT
- SciRepEval retrieval tasks

Use cases:
- difficult scientific/reasoning retrieval
- retrieval evaluator development
- retrieval regression testing

---

## 7.4 General evidence/citation evaluation

Candidate sources:
- FEVER
- FActScore
- ALCE
- ASQA

Use cases:
- evidence grounding
- claim verification
- citation correctness
- answer support
- citation completeness

These datasets are auxiliary capabilities and must not be confused with the experiment-comparison task.

---

## 7.5 Research/agent evaluation

Candidate sources:
- BrowseComp
- BrowseComp-Plus
- DeepResearchGym
- DeepResearch Bench
- AssistantBench
- GAIA
- WebArena-Verified

These should normally be maintained under evaluation/restricted storage when their terms or benchmark design prohibit training use.

Do not place evaluation-only benchmark data into model-training corpora.

In particular, follow the current benchmark instructions for:
- BrowseComp
- BrowseComp-Plus
- GAIA
- Mind2Web
- WebArena-Verified
and any future benchmark added to the project.

---

# 8. BENCHMARK CONTAMINATION POLICY

Every external dataset must be classified as one of:

    TRAINING_ALLOWED
    EVALUATION_ONLY
    RESTRICTED
    UNKNOWN

If the license or benchmark policy is unknown:

    DO NOT USE FOR TRAINING.

Evaluation-only datasets must be isolated from:
- training data
- synthetic data generation
- weak supervision
- fine-tuning corpora
- retrieval indexes used during training

Do not expose benchmark examples to generative dataset creation pipelines unless the benchmark explicitly permits such use.

The system must maintain a contamination manifest.

Required metadata:

    dataset_name
    dataset_version
    source_url
    license
    training_allowed
    evaluation_allowed
    redistribution_allowed
    contamination_status
    retrieval_date

---

# 9. ORIGINAL RESEARCHX GOLD DATASET

The ResearchX Gold dataset is the core proprietary/original dataset.

Target initial scale:

    1,000–2,000 human-verified papers
    3,000–8,000 experiment records
    10,000+ evidence assertions
    10,000+ field-level annotations
    10,000+ comparison pairs
    2,000+ study-linking pairs
    1,000+ contradiction cases
    1,000+ abstention/insufficient-evidence cases

These are engineering targets, not claims about existing data availability.

Quality is more important than raw count.

A smaller expert-verified Gold dataset is preferable to a huge noisy dataset.

---

# 10. EXPERIMENT RECORD SCHEMA

Each experiment must support the following fields.

## 10.1 Identity

    experiment_id
    study_id
    source_record_ids
    parent_experiment_id
    experiment_type

Possible experiment_type:

    MAIN
    BASELINE
    ABLATION
    SENSITIVITY
    SCALING
    ROBUSTNESS
    TRANSFER
    EFFICIENCY
    INFERENCE
    TRAINING
    EVALUATION
    OTHER

---

## 10.2 Research question

    research_question
    hypothesis
    task
    subtask
    domain

---

## 10.3 Dataset configuration

    dataset_name
    dataset_version
    split
    train_data
    validation_data
    test_data
    sample_count
    sampling_method
    filtering
    preprocessing
    augmentation
    deduplication
    data_leakage_controls

---

## 10.4 Population/context

For ML/CS, population/context may include:

    users
    documents
    queries
    languages
    domains
    task populations
    benchmark subsets
    dataset cohorts
    evaluation groups

Capture the concept even when the term "population" is not used explicitly by the paper.

---

## 10.5 Model configuration

    model_family
    model_name
    model_version
    checkpoint
    parameter_count
    architecture
    initialization
    pretrained_source
    base_model

---

## 10.6 Method configuration

    method_name
    algorithm
    objective
    loss_function
    training_strategy
    baseline
    ablation
    modifications

---

## 10.7 Training configuration

    optimizer
    learning_rate
    learning_rate_schedule
    batch_size
    gradient_accumulation
    epochs
    steps
    warmup_steps
    warmup_ratio
    weight_decay
    dropout
    seed
    precision
    quantization
    gradient_clipping
    LoRA
    QLoRA
    DoRA
    other_parameter_efficient_configuration

---

## 10.8 Inference configuration

    inference_model
    temperature
    top_p
    top_k
    max_tokens
    context_length
    decoding_strategy
    system_prompt
    retrieval_configuration
    tool_configuration
    hardware

Only extract these fields when supported by evidence.

---

## 10.9 Evaluation configuration

    benchmark
    dataset_split
    metric
    metric_definition
    score
    uncertainty
    confidence_interval
    statistical_test
    significance
    evaluation_population
    evaluation_protocol

---

## 10.10 Compute

When reported:

    accelerator
    gpu_model
    gpu_count
    cpu
    memory
    training_time
    inference_time
    flops
    energy
    cost

Missing values remain UNKNOWN.

Do not estimate them unless a derived field is explicitly marked:

    derived = true

and the derivation is recorded.

---

## 10.11 Artifacts

    paper_url
    repository_url
    checkpoint_url
    dataset_url
    supplementary_material_url
    demo_url
    benchmark_url

---

# 11. EVIDENCE ASSERTION SCHEMA

Every important extracted field must be associated with an EvidenceAssertion.

Minimum fields:

    evidence_id
    source_record_id
    study_id
    experiment_id
    field_name
    extracted_value
    normalized_value
    evidence_text
    source_location
    page
    section
    paragraph
    table
    figure
    source_version
    retrieval_timestamp
    extraction_model
    extraction_run_id
    status
    confidence
    derived

Allowed status:

    MACHINE_EXTRACTED
    SOURCE_CHECKED
    HUMAN_REVIEWED
    REJECTED

Derived conclusions must never masquerade as direct evidence.

---

# 12. SOURCE LOCATION REQUIREMENTS

Whenever possible, evidence must identify the smallest useful source location.

Preferred order:

    paragraph
    section
    page
    table
    figure
    appendix
    supplementary material

A source URL alone is insufficient for high-value claims when a more precise location can be stored.

---

# 13. NORMALIZATION

Normalization is required for comparison.

However:

    normalization must not destroy original evidence.

For every normalized value retain:

    raw_value
    normalized_value
    normalization_rule
    normalization_version

Examples:

    0.00002
    2e-5

may normalize to the same numeric representation.

But:

    "10 epochs"
    "10 steps"

must never normalize to the same semantic unit.

Unit conversion must be explicit and deterministic.

Controlled vocabularies should preserve hierarchical information.

Example:

    Transformer
        -> Decoder-only Transformer
            -> Causal Language Model

Do not flatten hierarchical concepts unnecessarily.

---

# 14. STUDY-LINKING DATASET

Each candidate relationship should include:

    record_a
    record_b
    relationship
    evidence
    rationale
    confidence
    reviewer_status

Allowed labels:

    SAME_UNDERLYING_STUDY
    SAME_STUDY_DIFFERENT_RECORD
    FOLLOW_UP
    EXTENSION
    REPLICATION
    RELATED_BUT_DISTINCT
    UNRELATED
    UNCERTAIN

Candidate positives may be generated from:
- DOI
- arXiv version relationships
- citation relationships
- repository relationships
- author/project metadata
- explicit "extended version" statements
- benchmark/project links

Candidate generation is not Gold labeling.

Final Gold labels require verification.

---

# 15. EXPERIMENT COMPARISON DATASET

Each pair must contain:

    comparison_id
    experiment_a
    experiment_b
    relationship_label
    field_comparisons
    evidence
    rationale
    confidence
    reviewer_status

Core labels:

    SAME_EXPERIMENT
    ALIGNED_EXPERIMENT
    RELATED_CONTEXT
    MATERIALLY_DIFFERENT
    INSUFFICIENT_EVIDENCE
    CONFLICTING_EVIDENCE

Where useful, also record:

    SAME_STUDY_DIFFERENT_RECORD
    DIFFERENT_STUDY

These are relationships, not scores.

Do not collapse all comparison information into a single similarity number.

---

# 16. FIELD-LEVEL COMPARISON

Each comparison must support per-field labels.

Possible values:

    IDENTICAL
    EQUIVALENT
    MATERIALLY_DIFFERENT
    DIFFERENT
    UNKNOWN
    CONFLICTING
    NOT_APPLICABLE

Each field-level judgment should have:

    field_name
    value_a
    value_b
    comparison_status
    evidence_a
    evidence_b
    rationale

Example:

    dataset = IDENTICAL
    model = EQUIVALENT
    learning_rate = MATERIALLY_DIFFERENT
    training_steps = UNKNOWN
    evaluation_metric = CONFLICTING

Unknown must never be interpreted as identical.

---

# 17. HARD-NEGATIVE DATASET

Hard negatives are mandatory.

Examples:

1. Same dataset, different training objective.
2. Same model family, different checkpoint.
3. Same paper title, different underlying study.
4. Same paper, different experiment.
5. Same benchmark, materially different evaluation protocol.
6. Same method name, different implementation.
7. Same model, different fine-tuning setup.
8. Same dataset, different split.
9. Same benchmark, different metric definition.
10. Similar abstract wording, different scientific configuration.
11. Same intervention but different duration.
12. Same model and dataset but different inference configuration.
13. Same experiment reported in multiple sources.
14. Similar experiment with insufficient evidence to establish equivalence.

Hard negatives should be explicitly tagged with construction method.

---

# 18. CONTRADICTION DATASET

Each contradiction record should contain:

    contradiction_id
    claim
    field
    source_a
    source_b
    value_a
    value_b
    conflict_type
    evidence_a
    evidence_b
    explanation
    resolution
    resolution_status

Conflict types may include:

    NUMERIC_CONFLICT
    CONFIGURATION_CONFLICT
    DATASET_CONFLICT
    MODEL_VERSION_CONFLICT
    SAMPLE_COUNT_CONFLICT
    METRIC_CONFLICT
    TIMELINE_CONFLICT
    STUDY_IDENTITY_CONFLICT
    OTHER

Do not automatically decide which source is correct unless evidence supports the resolution.

---

# 19. ABSTENTION DATASET

Examples should be constructed in at least three classes:

    SUFFICIENT_EVIDENCE
    INSUFFICIENT_EVIDENCE
    CONFLICTING_EVIDENCE

Expected system behavior may be:

    ANSWER
    ABSTAIN
    ESCALATE
    SEARCH_MORE
    ASK_USER

The correct answer is sometimes to refuse a definitive comparison because the evidence is inadequate.

---

# 20. LAYA DECISION DATASET

Laya-related data must be separate from scientific facts.

Each example should contain:

    decision_id
    current_state
    research_goal
    evidence_available
    evidence_quality
    missing_information
    candidate_tools
    previous_actions
    budget
    latency_budget
    current_confidence
    next_action
    rationale
    expected_information_gain
    final_outcome

Candidate actions:

    SEARCH
    RETRIEVE_SOURCE
    OPEN_SOURCE
    EXTRACT
    COMPARE
    VERIFY
    SEARCH_MORE
    CHANGE_QUERY
    ASK_USER
    ESCALATE
    HUMAN_REVIEW
    GENERATE_REPORT
    GENERATE_MEDIA
    STOP
    ABSTAIN

Do not let Laya invent evidence.

Laya chooses actions.

The evidence system determines what evidence actually exists.

---

# 21. AGENT TRAJECTORY DATASET

Store structured trajectories:

    trajectory_id
    task_id
    state_id
    timestamp
    action
    tool
    tool_arguments_hash
    tool_result_summary
    evidence_added
    confidence_before
    confidence_after
    next_state
    final_decision
    outcome
    cost
    latency
    failure_type

Never store secrets in trajectories.

Never store unrestricted full proprietary inputs unless explicitly authorized.

---

# 22. RESEARCH REPORT DATASET

A research report example should contain:

    research_question
    evidence_set
    selected_sources
    structured_findings
    comparisons
    contradictions
    uncertainty
    citations
    final_report

Training examples must preserve the distinction between:

    directly supported claim
    derived conclusion
    unresolved claim
    speculation

Unsupported statements must not be used as positive Gold examples.

---

# 23. PREFERENCE DATASET

Preference examples may contain:

    question
    evidence
    answer_a
    answer_b
    preferred_answer
    preference_reason

Preference dimensions:

    factuality
    evidence_support
    citation_correctness
    citation_completeness
    uncertainty_handling
    comparison_quality
    completeness
    conciseness
    redundancy
    abstention_behavior

Preference labels must not replace factual Gold labels.

---

# 24. FAILURE DATASET

Every significant failure should be captured.

Failure categories:

    retrieval_failure
    query_failure
    extraction_failure
    parsing_failure
    canonicalization_failure
    deduplication_failure
    study_linking_failure
    comparison_failure
    citation_failure
    hallucination
    contradiction_missed
    wrong_abstention
    wrong_laya_decision
    wrong_tool_selection
    prompt_injection
    security_failure
    media_grounding_failure
    dataset_error
    license_error
    infrastructure_failure

Each failure should include:

    input
    expected_behavior
    observed_behavior
    root_cause
    severity
    reproducibility
    regression_test_created

---

# 25. DATA QUALITY TIERS

All datasets must be explicitly classified.

## GOLD

Human verified.

Allowed for:
- final evaluation
- regression evaluation
- critical fine-tuning
- benchmark reporting

## SILVER

Model-assisted or weakly supervised.

Allowed for:
- candidate generation
- preliminary training
- experimentation

Not suitable as final ground truth.

## BRONZE

Raw, noisy, harvested, or weakly processed.

Allowed for:
- discovery
- retrieval
- weak supervision
- candidate generation

Never silently promote Bronze to Gold.

Every promotion must be recorded.

---

# 26. ANNOTATION PROCESS

The annotation pipeline should be:

    Candidate Generation
          ->
    Automatic Extraction
          ->
    Evidence Attachment
          ->
    Human Verification
          ->
    Adjudication
          ->
    Gold Label
          ->
    Quality Check
          ->
    Dataset Version

For difficult/high-impact examples:

    Annotator A
        +
    Annotator B
        ->
    Adjudicator

Disagreement should be preserved.

Do not erase disagreement merely to make the dataset appear cleaner.

---

# 27. ACTIVE LEARNING

The dataset pipeline should support active learning.

Suggested process:

    Model prediction
        ->
    uncertainty/disagreement
        ->
    hard-case selection
        ->
    human review
        ->
    Gold label
        ->
    dataset update
        ->
    evaluation
        ->
    optional fine-tuning

Prioritize:
- low-confidence cases
- high-value comparison cases
- contradictory cases
- hard negatives
- model disagreements
- cases likely to expose systematic errors

---

# 28. DATA SPLITTING

Dataset splits must prevent leakage.

Default splits:

    TRAIN
    VALIDATION
    TEST
    GOLD_EVAL
    SAFETY_EVAL
    REGRESSION

Splitting should occur at the relevant entity level.

Never randomly split passages from the same paper across train and test.

Avoid leakage across:

    same paper
    same study
    same experiment
    same dataset-specific generated pair
    same source-version family
    same synthetic template family

When relevant, also maintain:

    TEMPORAL_TEST

The temporal split should evaluate generalization to later research.

---

# 29. DEDUPLICATION

Deduplication must operate at multiple levels.

## Source level

Possible duplicate indicators:

    DOI
    arXiv ID
    normalized title
    repository identifier
    exact source hash

## Study level

Possible duplicate indicators:

    explicit study references
    project/repository relation
    trial/study identifiers
    publication chains
    metadata relationships

## Experiment level

Use:
- normalized configurations
- experiment descriptions
- dataset/model/configuration evidence

Never merge records solely because titles are similar.

All merges must be auditable.

---

# 30. PROVENANCE

Every data record must retain provenance.

Minimum fields:

    source_name
    source_url
    source_identifier
    source_version
    retrieved_at
    content_hash
    processing_version
    extractor_version
    annotator_status

For transformed records:

    parent_record_ids

must be preserved.

---

# 31. LICENSING AND ACCESS CONTROL

Every source must have a license/access manifest.

Required fields:

    source_name
    source_url
    dataset_version
    content_license
    metadata_license
    training_allowed
    evaluation_allowed
    redistribution_allowed
    internal_use_allowed
    attribution_required
    restrictions
    verification_date

Do not assume:
- publicly accessible means freely redistributable
- downloadable means training-permitted
- open access means unrestricted reuse

For public full-text data, use content that is legally compatible with the intended distribution.

Copyrighted source material may be processed internally where permitted, but must not automatically be redistributed as part of the ResearchX dataset.

---

# 32. DATA STORAGE LAYOUT

Recommended structure:

    data/
    ├── README.md
    │
    ├── raw/
    │   ├── openalex/
    │   ├── semantic_scholar/
    │   ├── arxiv/
    │   ├── papers_with_code/
    │   └── external/
    │
    ├── normalized/
    │   ├── papers/
    │   ├── sources/
    │   ├── citations/
    │   └── entities/
    │
    ├── fulltext/
    │   ├── permitted/
    │   └── restricted/
    │
    ├── silver/
    │   ├── experiments/
    │   ├── evidence/
    │   ├── comparisons/
    │   └── trajectories/
    │
    ├── gold/
    │   ├── experiments/
    │   ├── evidence/
    │   ├── study_links/
    │   ├── comparisons/
    │   ├── contradictions/
    │   ├── abstention/
    │   ├── laya/
    │   └── trajectories/
    │
    ├── eval/
    │   ├── internal/
    │   └── external/
    │
    ├── restricted/
    │
    ├── manifests/
    │   ├── sources.yaml
    │   ├── licenses.yaml
    │   ├── datasets.yaml
    │   ├── contamination.yaml
    │   └── versions.yaml
    │
    └── schemas/
        ├── source_record.json
        ├── study.json
        ├── experiment.json
        ├── evidence_assertion.json
        ├── comparison.json
        ├── contradiction.json
        ├── abstention.json
        ├── laya_decision.json
        └── trajectory.json

Large files must not be committed to Git.

---

# 33. DOWNLOAD / ACQUISITION POLICY

Claude Code must implement the acquisition pipeline.

Claude Code must NOT blindly download every listed dataset.

The acquisition system must:

1. Inspect source availability.
2. Check current licensing/usage requirements.
3. Record the source version.
4. Record the retrieval timestamp.
5. Download only the required subset where practical.
6. Verify downloads.
7. Compute checksums.
8. Store provenance.
9. Normalize data.
10. Validate schema.
11. Generate statistics.
12. Record failures.
13. Make the process reproducible.

The pipeline must be resumable.

Partial downloads must not corrupt the canonical dataset.

Use retries and backoff where appropriate.

Avoid unnecessary repeated downloads.

---

# 34. INITIAL ACQUISITION STRATEGY

Do NOT begin by downloading massive complete scholarly snapshots.

Initial target:

    20,000–50,000 relevant ML/CS candidate papers

Priority:

    LLMs
    RAG
    retrieval
    agents
    fine-tuning
    PEFT
    multimodal
    evaluation
    quantization
    efficient inference
    safety

Initial acquisition stages:

    Stage 1:
        metadata discovery

    Stage 2:
        deduplication

    Stage 3:
        ML/CS relevance filtering

    Stage 4:
        licensing/full-text availability filtering

    Stage 5:
        full-text acquisition for selected papers

    Stage 6:
        experiment extraction

    Stage 7:
        evidence verification

    Stage 8:
        experiment pairing

    Stage 9:
        human annotation

    Stage 10:
        Gold dataset freeze

---

# 35. INITIAL CORPUS PIPELINE

Expected pipeline:

    OpenAlex
        +
    Semantic Scholar
        +
    arXiv
        +
    Papers With Code
        |
        v
    Candidate Paper Corpus
        |
        v
    Deduplication
        |
        v
    ML/CS Relevance Filter
        |
        v
    License / Access Filter
        |
        v
    Full-text Acquisition
        |
        v
    Scientific Document Parsing
        |
        v
    Candidate Experiment Extraction
        |
        v
    Evidence Assertions
        |
        v
    Human Verification
        |
        v
    ResearchX Gold

---

# 36. FULL-TEXT ACQUISITION POLICY

Full text must be downloaded selectively.

Prefer:
- legally available
- machine-readable
- stable
- versioned
- reproducible sources

For every document record:

    full_text_available
    full_text_source
    format
    retrieval_url
    retrieval_timestamp
    license_status

Possible formats:

    PDF
    HTML
    XML
    TEI/XML
    source-derived structured text

Do not assume all PDFs are equally parseable.

Preserve the original where permitted and maintain a parsed representation separately.

---

# 37. EXPERIMENT EXTRACTION PIPELINE

The initial extraction pipeline should be:

    Document
        ->
    Structure detection
        ->
    Section segmentation
        ->
    Candidate experiment identification
        ->
    Field extraction
        ->
    Evidence span extraction
        ->
    Canonicalization
        ->
    Validation
        ->
    Human verification

Extraction models may use NVIDIA/Nemotron models through Nebius as specified by the architecture.

Model output must never be treated as Gold automatically.

---

# 38. MODEL USAGE FOR DATA CONSTRUCTION

The ResearchX dataset-building pipeline should support model-assisted extraction.

Preferred process:

    NVIDIA/Nemotron model
        ->
    structured candidate output
        ->
    deterministic schema validation
        ->
    evidence verification
        ->
    human review

Use smaller/efficient models for:
- high-volume extraction
- classification
- candidate generation
- normalization proposals

Use larger reasoning models for:
- difficult scientific interpretation
- complex comparison candidates
- contradiction analysis
- high-impact verification

Never couple dataset correctness to a single model.

The pipeline must support model replacement.

---

# 39. DATASET GENERATION RULES

Synthetic examples may be generated for:
- hard negatives
- missing-field cases
- controlled perturbations
- comparison stress tests
- abstention cases
- tool-selection examples

Synthetic data must be marked:

    synthetic = true

and must retain:

    generation_model
    generation_prompt_version
    generation_timestamp
    parent_record_ids
    validation_status

Synthetic data cannot automatically become Gold.

Human verification is required for Gold synthetic examples.

---

# 40. COMPARISON DATA GENERATION

Comparison candidates may be produced using:

    metadata similarity
    citation relationships
    embeddings
    lexical similarity
    task overlap
    dataset overlap
    model overlap
    benchmark overlap
    explicit paper references

These are candidate-generation mechanisms only.

Final comparison labels must use explicit field-level evidence.

Study design/configuration is compared before outcome interpretation.

Do not use performance score similarity as evidence that two experiments are equivalent.

---

# 41. EMBEDDINGS

Embeddings may be used for:

- candidate retrieval
- nearest-neighbor search
- clustering
- hard-negative mining

Embeddings must NOT independently decide:

    SAME_STUDY
    SAME_EXPERIMENT
    MATERIALLY_DIFFERENT

Embedding similarity must remain a candidate-generation feature.

---

# 42. NUMERIC AND UNIT HANDLING

Numeric comparison must be deterministic where possible.

Examples:

    0.00002
    2e-5

must normalize consistently.

Units must be represented explicitly.

Examples:

    seconds
    minutes
    hours
    epochs
    steps
    tokens
    samples
    parameters

Never compare incompatible units without conversion.

Never infer an unstated unit.

---

# 43. MISSING DATA POLICY

Allowed state:

    UNKNOWN

Forbidden behavior:

    silently filling missing values
    guessing omitted configuration
    converting missing values into defaults
    assuming common research practice
    assuming information from another experiment applies

The dataset should contain explicit examples of missing information.

---

# 44. CONFLICT POLICY

If two sources disagree:

    preserve both claims

Do not silently overwrite one with another.

Store:

    source_a
    source_b
    value_a
    value_b
    evidence_a
    evidence_b
    conflict_type
    resolution_status

A resolved conflict must include evidence supporting the resolution.

---

# 45. ANNOTATION GUIDELINES

Annotators must answer:

1. What exactly does the source say?
2. What experiment does it belong to?
3. What evidence supports this field?
4. Is the field directly stated or derived?
5. Is the field missing?
6. Is there conflicting evidence?
7. Is this the same underlying study?
8. Is this the same experiment?
9. If not, which fields differ materially?
10. Is there enough evidence to make the decision?

Annotators must not infer information merely because it is statistically common or scientifically plausible.

---

# 46. DATASET VALIDATION

Every dataset build must run automated validation.

Checks include:

    schema validity
    required identifiers
    duplicate identifiers
    orphan references
    invalid enums
    invalid numerical values
    invalid units
    missing provenance
    missing source locations
    conflicting labels
    duplicate examples
    train/test leakage
    benchmark contamination
    license metadata completeness

The pipeline must fail loudly on critical violations.

---

# 47. DATASET STATISTICS

Every build must generate a statistics report.

At minimum:

    number_of_source_records
    number_of_papers
    number_of_studies
    number_of_experiments
    number_of_evidence_assertions
    number_of_comparisons
    number_of_same_study_pairs
    number_of_hard_negatives
    number_of_contradictions
    number_of_abstention_cases
    number_of_laya_decisions
    number_of_trajectories

Also report:

    label_distribution
    missing_field_distribution
    source_distribution
    domain_distribution
    temporal_distribution
    evidence_status_distribution
    Gold/Silver/Bronze distribution
    license distribution
    split distribution

---

# 48. QUALITY TARGETS

Target properties for the Gold dataset:

1. Every high-value claim has evidence.
2. Every Gold label is human verified or explicitly adjudicated.
3. Every comparison has field-level justification.
4. Every UNKNOWN remains UNKNOWN.
5. Every source relationship has provenance.
6. No benchmark contamination.
7. No silent data merges.
8. No unsupported inferred values.
9. No unexplained label imbalance.
10. No train/test entity leakage.

---

# 49. EVALUATION DATASET DESIGN

The final evaluation suite must independently test:

    paper retrieval
    scientific extraction
    study linking
    experiment comparison
    hard-negative handling
    contradiction detection
    abstention
    citation correctness
    evidence completeness
    Laya decision quality
    agent trajectory quality
    report quality

Do not rely on a single aggregate metric.

---

# 50. EVALUATION METRICS

Potential metrics include:

## Retrieval

    Recall@K
    Precision@K
    MRR
    nDCG

## Extraction

    field-level precision
    field-level recall
    F1
    exact match where appropriate

## Study linking

    precision
    recall
    F1
    pairwise accuracy

## Experiment comparison

    macro F1
    per-class F1
    field-level agreement
    false-match rate
    missed-match rate

## Evidence

    evidence precision
    evidence recall
    citation support rate
    citation correctness

## Abstention

    appropriate abstention rate
    false-certainty rate
    selective accuracy

## Decision/agent

    task success
    action accuracy
    unnecessary-action rate
    tool efficiency
    cost
    latency

Metrics must be reported per task and per dataset slice.

---

# 51. RESEARCHX DATASET SLICES

Evaluation should include slices such as:

    easy
    medium
    hard

and domain slices:

    LLM
    RAG
    Retrieval
    Agents
    Fine-tuning
    PEFT
    Multimodal
    Quantization
    Evaluation
    Safety

Also include:

    high_similarity_hard_negative
    low_similarity_same_experiment
    same_study_multi_record
    missing_evidence
    conflicting_evidence
    long_document
    table-heavy
    supplementary-material-dependent

---

# 52. TEMPORAL DATASET

Where possible, maintain a temporal benchmark.

Training data must only use information available before the evaluation period.

Purpose:

- measure temporal generalization
- reduce accidental information leakage
- evaluate performance on newer research

---

# 53. DATASET VERSIONING

Dataset versions must be immutable.

Use semantic versioning:

    MAJOR.MINOR.PATCH

Example:

    0.1.0
    0.2.0
    1.0.0

Maintain:

    dataset version
    schema version
    source versions
    processing version
    extraction model version
    annotation guideline version

A dataset version must be reproducible from its acquisition manifests and processing configuration where legally and technically possible.

---

# 54. DATASET MANIFEST

Every dataset release must contain a manifest:

    dataset_version
    created_at
    source_versions
    source_urls
    processing_commit
    schema_version
    extractor_versions
    annotation_versions
    record_counts
    checksums
    licensing_status
    contamination_status

---

# 55. REPRODUCIBILITY

The complete build should be executable through scripts.

Recommended layout:

    scripts/data/
        acquire_openalex.py
        acquire_semantic_scholar.py
        acquire_arxiv.py
        acquire_papers_with_code.py

        normalize_metadata.py
        deduplicate_papers.py
        filter_mlcs.py

        acquire_fulltext.py
        parse_fulltext.py

        extract_experiments.py
        extract_evidence.py

        build_study_links.py
        build_comparison_candidates.py
        build_hard_negatives.py
        build_contradictions.py
        build_abstention_cases.py

        build_laya_dataset.py
        build_trajectories.py

        validate_dataset.py
        detect_leakage.py
        check_licenses.py
        generate_statistics.py
        build_manifest.py

Do not create one giant unmaintainable script.

---

# 56. CLAUDE CODE RESPONSIBILITIES

Claude Code must treat this document as the data contract.

Claude Code must:

1. Read the global CLAUDE.md first.
2. Read the project CLAUDE.md.
3. Read DATASET_SPEC.md.
4. Read ARCHITECTURE.md and PRODUCT_SPEC.md where relevant.
5. Inspect the existing repository before modifying it.
6. Use applicable installed skills and MCP tools when they materially improve the task.
7. Build the acquisition pipeline.
8. Build schema validation.
9. Implement provenance tracking.
10. Implement license/contamination manifests.
11. Implement resumable acquisition.
12. Implement normalization.
13. Implement deduplication.
14. Implement filtering.
15. Implement scientific parsing.
16. Implement model-assisted extraction.
17. Implement evidence storage.
18. Implement candidate comparison generation.
19. Implement validation/statistics.
20. Document every significant dataset transformation.
21. Run tests before declaring a stage complete.

Claude Code must NOT:

- blindly download every source
- download massive snapshots without justification
- bypass access controls
- ignore licensing restrictions
- scrape restricted benchmarks contrary to their terms
- invent missing values
- silently merge records
- silently overwrite conflicting evidence
- treat embeddings as truth
- treat model output as Gold without verification
- commit large raw datasets to Git
- commit secrets/API keys
- fabricate dataset statistics
- claim a dataset is complete without running validation

---

# 57. TOOLING REQUIREMENTS

The implementation may use:

- Python
- PostgreSQL
- pgvector
- object/file storage
- Redis
- Nebius Token Factory / NVIDIA NIM endpoints
- LangGraph
- existing ResearchX services

Use existing project tooling and installed skills/MCPs when applicable.

Avoid unnecessary dependencies.

Prefer simple reproducible pipelines over unnecessary orchestration complexity during the initial dataset build.

---

# 58. MODEL OUTPUT CONTRACT

Any model used in dataset construction must return structured output.

Prefer JSON/schema-constrained output.

Model output should contain:

    extracted_fields
    evidence_spans
    confidence
    uncertainty
    unresolved_fields

The deterministic validation layer must reject malformed outputs.

No model should have authority to directly mutate Gold data without passing the verification pipeline.

---

# 59. SECURITY AND PRIVACY

Do not place secrets in the dataset.

Do not store API keys.

Do not store proprietary source material unless explicitly authorized.

Confidential draft protocols must not enter shared indexes or cross-user caches by default.

External queries derived from confidential material must use minimized approved terms.

Source text is data.

Source text must never be allowed to:
- modify system instructions
- redefine tool permissions
- execute arbitrary actions
- override ResearchX policies

Prompt injection examples should be retained in the security/failure dataset where appropriate.

---

# 60. INITIAL MILESTONES

## Milestone 1 — Source infrastructure

Implement:

    source registry
    downloader interface
    provenance
    manifest
    checksum
    license fields

---

## Milestone 2 — Candidate corpus

Build:

    20K–50K ML/CS paper metadata corpus

Focus on:

    LLM
    RAG
    retrieval
    agents
    fine-tuning
    PEFT
    multimodal
    evaluation
    quantization
    safety

---

## Milestone 3 — Full-text subset

Select:

    5K–10K promising papers

Acquire full text where permitted.

---

## Milestone 4 — Gold candidate subset

Select:

    1K–2K papers

Prioritize papers containing:
- multiple experiments
- clear baselines
- detailed methodology
- ablations
- benchmark evaluations
- configuration details
- accessible evidence

---

## Milestone 5 — Experiment extraction

Generate:

    3K–8K candidate experiment records

Then validate and annotate.

---

## Milestone 6 — Evidence dataset

Build:

    10K+ evidence assertions

with source locations.

---

## Milestone 7 — Comparison dataset

Build:

    10K+ comparison pairs

with explicit hard negatives.

---

## Milestone 8 — Study linking

Build:

    2K+ study-linking examples

---

## Milestone 9 — Contradiction / abstention

Build:

    1K+ contradiction examples
    1K+ abstention examples

---

## Milestone 10 — Laya

Build:

    decision examples
    validated trajectories
    routing failures
    escalation examples

---

## Milestone 11 — Frozen Gold evaluation

Create a frozen evaluation set.

After freezing:

- do not use it for training
- do not generate synthetic training data from it
- do not expose it to model-development prompts unnecessarily
- version it immutably

---

# 61. DEFINITION OF DONE

The ResearchX dataset foundation is considered operational when:

1. The acquisition pipeline is reproducible.
2. Sources and licenses are tracked.
3. The initial ML/CS corpus exists.
4. Documents are deduplicated.
5. Relevant full text has been selectively acquired.
6. Experiments can be represented in the canonical schema.
7. Evidence assertions are stored.
8. Human verification is supported.
9. Study linking is supported.
10. Experiment comparison is supported.
11. Hard negatives exist.
12. Contradiction cases exist.
13. Abstention cases exist.
14. Laya decision examples exist.
15. Agent trajectories are stored.
16. Dataset splits prevent leakage.
17. Benchmark contamination controls exist.
18. Dataset statistics are generated automatically.
19. Dataset versions are reproducible.
20. Gold evaluation data is isolated.

---

# 62. FIRST IMPLEMENTATION TASK FOR CLAUDE CODE

Do not immediately start downloading every dataset.

First implement:

    1. data directory structure
    2. canonical schemas
    3. source manifest
    4. license manifest
    5. contamination manifest
    6. downloader abstraction
    7. provenance model
    8. checksum/versioning
    9. validation framework
    10. statistics framework

Then implement acquisition in this order:

    OpenAlex metadata
        ->
    Semantic Scholar metadata/enrichment
        ->
    arXiv metadata/full text where permitted
        ->
    Papers With Code metadata
        ->
    candidate corpus
        ->
    filtering
        ->
    selective full text
        ->
    experiment extraction

Do not download the entire OpenAlex snapshot initially.

Do not download every existing benchmark immediately.

Download benchmark datasets only after classification into:

    training
    evaluation
    restricted

and after verifying their current usage terms.

---

# 63. RECOMMENDED SOURCE REGISTRY

Maintain a machine-readable registry similar to:

    source_id:
        name:
        purpose:
        url:
        acquisition_method:
        data_types:
        version:
        license:
        training_allowed:
        evaluation_allowed:
        redistribution_allowed:
        restricted:
        notes:

Initial source IDs:

    OPENALEX
    SEMANTIC_SCHOLAR
    ARXIV
    PAPERS_WITH_CODE
    S2ORC
    SCIREX
    SCIENCEIE
    SCIRIFF
    SCIREPEVAL
    QASPER
    SCIFACT
    RESEARCHQA
    FEVER
    FACTSCORE
    ALCE
    ASQA
    BRIGHT
    BROWSECOMP
    BROWSECOMP_PLUS
    DEEPRESEARCHGYM
    DEEPRESEARCH_BENCH
    ASSISTANTBENCH
    GAIA
    WEBARENA_VERIFIED
    MIND2WEB

The registry must be checked and updated when source versions or policies change.

---

# 64. RESEARCHX DATASET PHILOSOPHY

The raw paper corpus is not the main moat.

The valuable ResearchX dataset is:

    expert-reviewed experiment representations
        +
    study linking
        +
    field-level comparisons
        +
    hard negatives
        +
    evidence assertions
        +
    contradiction cases
        +
    abstention judgments
        +
    decision trajectories
        +
    citation verification
        +
    failure cases
        +
    uncertainty judgments

The dataset should therefore optimize for:

    scientific correctness
    provenance
    difficult cases
    expert disagreement
    realistic research workflows
    reproducibility

rather than merely maximizing document count.

---

# 65. REFERENCE SOURCES

Primary scholarly/data sources:

OpenAlex:
https://openalex.org/
https://docs.openalex.org/

Semantic Scholar:
https://www.semanticscholar.org/
https://www.semanticscholar.org/product/api

arXiv:
https://arxiv.org/
https://info.arxiv.org/help/api/index.html

Papers With Code:
https://paperswithcode.com/

Scientific benchmark/dataset references:

SciREX:
https://github.com/allenai/SciREX

ScienceIE:
https://scienceie.github.io/

SciRIFF:
https://github.com/allenai/SciRIFF

QASPER:
https://allenai.org/data/qasper

SciFact:
https://github.com/allenai/scifact

BRIGHT:
https://brightbenchmark.github.io/

BrowseComp:
https://openai.com/index/browsecomp/

BrowseComp-Plus:
https://github.com/texttron/BrowseComp-Plus

GAIA:
https://huggingface.co/gaia-benchmark

WebArena:
https://webarena.dev/

Mind2Web:
https://osu-nlp-group.github.io/Mind2Web/

These links are references for acquisition/evaluation planning. Current versions, terms, licensing, and benchmark restrictions must be re-verified at implementation time.

---

# 66. FINAL RULE

ResearchX must never optimize for the appearance of having a large dataset.

The objective is to build a dataset that can answer:

    What experiment was actually performed?
    What evidence supports that representation?
    Which other experiments are genuinely comparable?
    Which fields differ?
    Are two records from the same underlying study?
    What evidence conflicts?
    What is still unknown?
    When should ResearchX abstain?
    What should the research agent do next?

Every important answer must be traceable back to evidence.

Every uncertainty must remain visible.

Every dataset version must be reproducible.

Every benchmark restriction must be respected.

The ResearchX Gold dataset is the final source of truth for ResearchX model development and evaluation.

# 67. ADDENDUM — DATASET v0.2.0 AND DATA-CLASS SEPARATION (2026-10-02)

## 67.1 Versions

- v0.1.0 is the frozen bootstrap (Silver) release: manifest `data/manifests/releases/v0.1.0.json`, processing
  version `code-f948ff800013`, snapshot archives in `data/releases/v0.1.0/`. It is preserved and reproducible and
  is never modified.
- v0.2.0 (schema 0.2.0) deepens technical content per experiment instead of record counts. The design is in
  `docs/DATASET_DEEPENING.md`, the source map in `docs/DATASET_SOURCE_MAP.md`, the gap analysis in
  `data/reports/dataset_gap_analysis.md` and the staged plan in `planning/DATASET_ROADMAP.md`.

## 67.2 Record types added in schema 0.2.0

| Record | Path | Rule |
|---|---|---|
| DiscoveryCandidate | `normalized/discovery/candidates.jsonl` | Layer 0 candidates from arXiv OAI-PMH; selection version recorded |
| ResultRecord | `silver/results/results.jsonl` | where a number came from (table cell, text, claim); attached to experiments only through associations |
| ExperimentResultAssociation | `silver/results/associations.jsonl` | VERIFIED / LIKELY / AMBIGUOUS / UNKNOWN / CONFLICTING; never "every score to every experiment" |
| Artifact | `silver/artifacts/artifacts.jsonl` | repository / file pinned to a commit; official only with paper or PwC evidence |
| ArtifactEvidence | `silver/artifacts/artifact_evidence.jsonl` | paper-vs-artifact field relation; repository values never overwrite paper values |
| CanonicalEntity | `silver/entities/canonical_entities.jsonl` | case / punctuation-level normalization only; every surface form kept |
| GraphNode, GraphEdge | `silver/graph/` | derived each build; every edge names the records it came from |
| ResearchTask, HardCase | `silver/tasks/` | templated wording over real records; `synthetic_text` flag; never Gold without human review |

Experiments gained `depth` (technical depth tier) and `quality` (scientific-quality dimensions) and the retrieval
and prompting field groups. Fields are populated only from evidence; otherwise they stay UNKNOWN.

## 67.3 Data classes

ResearchX keeps three classes of data apart (brief §44):

1. **Training data**: dataset records whose sources allow training (`training_allowed: true` in
   `data/manifests/sources.yaml`) and that are outside every evaluation split.
2. **Evaluation data**: frozen evaluation splits, the internal Gold set and public benchmarks under
   `data/eval/`. Benchmarks are evaluation-only, never used for training, synthetic generation or training-time
   indexes, and every evaluation records the dataset version.
3. **Runtime research sources**: documents retrieved at answer time (internal corpus, scholarly APIs, Tavily web
   results). They are session data with provenance. They enter the dataset only through the acquisition policy
   (§33, licence check, versioned release) and never become Gold without human verification. Runtime answers are
   never training or evaluation labels by default.

Tavily is not used to build datasets when equivalent public or static data can be acquired directly (credit
budget and licence clarity).

## 67.4 Retrieval units

Parsed documents keep stable ids for paragraphs, sentences, table cells, captions and footnotes. These ids are
the retrieval units of the runtime index (ARCHITECTURE §51), so a retrieved unit maps directly to an
`EvidenceAssertion` location. Embeddings and indexes are derived artifacts, rebuilt from a release; they are
never part of the dataset system of record.