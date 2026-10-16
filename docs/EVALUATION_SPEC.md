# EVALUATION_SPEC.md

# ResearchX — Evaluation Specification

## 1. Purpose

The ResearchX evaluation system measures whether the complete research system produces reliable, evidence-grounded, reproducible, secure, and useful results.

Evaluation must cover:

- Models
- Prompts
- Retrieval
- Extraction
- Canonicalization
- Evidence linking
- Comparison
- Decision/orchestration
- Final synthesis
- Citations
- Tool usage
- Guardrails
- Media generation
- Fine-tuned models
- End-to-end research workflows

The evaluation system is a first-class component of ResearchX.

No model, prompt, retrieval strategy, fine-tuning method, or major architectural change should be accepted solely because its output appears better qualitatively.

---

# 2. Evaluation Principles

ResearchX follows these principles:

1. Measure before optimizing.
2. Establish a baseline before fine-tuning.
3. Evaluate individual components and the complete pipeline.
4. Prefer ground-truth evidence over LLM judgment.
5. Use deterministic metrics where possible.
6. Use LLM-as-a-judge only where deterministic evaluation is insufficient.
7. Human evaluation is required for high-impact subjective dimensions.
8. Never treat model confidence as factual correctness.
9. Every evaluation result must identify the model/version/configuration used.
10. Evaluation datasets must be versioned.
11. Evaluation must be reproducible.
12. Failed safety evaluations block release where applicable.

---

# 3. Evaluation Layers

ResearchX uses multiple evaluation layers.

## Layer 1 — Unit Evaluation

Tests individual functions.

Examples:

- Parser
- Entity normalizer
- Citation parser
- Schema validator
- Evidence linker

## Layer 2 — Component Evaluation

Tests complete services.

Examples:

- Retrieval service
- Extraction service
- Canonicalization service
- Comparison service
- Laya service
- Media service

## Layer 3 — Model Evaluation

Tests individual models on controlled tasks.

## Layer 4 — Pipeline Evaluation

Tests:

Research Question
→ Retrieval
→ Extraction
→ Evidence
→ Reasoning
→ Synthesis

## Layer 5 — End-to-End Evaluation

Tests the complete ResearchJob.

## Layer 6 — Adversarial Evaluation

Tests:

- Prompt injection
- Malicious documents
- Contradictory sources
- Missing information
- Misleading evidence
- Tool abuse
- Data leakage
- Unsupported claims

## Layer 7 — Human Evaluation

Used for dimensions where automated metrics are insufficient.

---

# 4. Evaluation Dataset

Every evaluation dataset must contain:

- dataset_id
- version
- task
- input
- expected_output where applicable
- ground_truth
- evidence
- source references
- difficulty
- category
- metadata
- created_at
- validation_status

Dataset examples:

research_eval_v1
research_retrieval_v1
citation_eval_v1
extraction_eval_v1
security_eval_v1
comparison_eval_v1
media_grounding_eval_v1

---

# 5. Evaluation Dataset Categories

The dataset should contain examples covering:

## Basic

Straightforward factual research.

## Multi-hop

Questions requiring information from multiple sources.

## Comparative

Questions requiring comparison across entities.

## Temporal

Questions where information changes over time.

## Contradictory

Questions where sources disagree.

## Incomplete

Questions where required information is unavailable.

## Ambiguous

Questions requiring clarification or careful interpretation.

## Multimodal

Questions involving:

- Images
- Charts
- PDFs
- Audio
- Video

## Adversarial

Inputs specifically designed to expose system weaknesses.

## Enterprise

Examples involving:

- Large document sets
- Long context
- Access restrictions
- Sensitive information
- Auditability
- Reproducibility

---

# 6. Baseline

Every major feature must have a baseline.

Baseline candidates:

- Base NVIDIA model
- Base model + prompt
- Base model + retrieval
- Base model + retrieval + structured output

For fine-tuning:

Baseline:

Base Model + Existing Pipeline

Comparison:

Fine-Tuned Model + Existing Pipeline

Fine-tuning must not be considered successful without measurable improvement on the target evaluation set.

---

# 7. Retrieval Evaluation

Retrieval must be evaluated independently from generation.

Metrics:

- Recall@K
- Precision@K
- MRR
- NDCG
- Evidence recall
- Source coverage
- Query coverage

Important metric:

Evidence Recall

Definition:

Percentage of required ground-truth evidence items that appear in the retrieved evidence set.

Example:

Required evidence:
A, B, C

Retrieved:
A, B, D

Evidence Recall = 2 / 3

---

# 8. Extraction Evaluation

Evaluate structured extraction against validated ground truth.

Metrics:

- Exact Match
- Precision
- Recall
- F1
- Field-level accuracy
- Schema validity
- Missing-field rate
- Unsupported-field rate

For numerical values:

- Exact numeric match where appropriate
- Normalized numeric match
- Unit correctness

For dates:

- Normalized date accuracy

For entities:

- Entity resolution accuracy

---

# 9. Canonicalization Evaluation

Measure whether equivalent entities are correctly resolved.

Metrics:

- Entity precision
- Entity recall
- Entity F1
- Duplicate resolution rate
- False merge rate
- False split rate

Critical failure:

Two unrelated entities must not be merged into one canonical entity.

---

# 10. Evidence Linking Evaluation

Every claim should ideally link to supporting evidence.

Metrics:

- Evidence precision
- Evidence recall
- Claim-evidence correctness
- Evidence span accuracy
- Source attribution accuracy
- Unsupported claim rate

A citation that exists but does not support the claim must be counted as incorrect.

---

# 11. Citation Evaluation

Citation evaluation has two primary dimensions.

## Citation Correctness

Does the cited source actually support the claim?

## Citation Completeness

Are important externally verifiable claims supported by citations?

Additional metrics:

- Citation precision
- Citation recall
- Citation completeness
- Citation placement accuracy
- Source provenance accuracy
- Broken citation rate
- Fabricated citation rate

Fabricated citations are a critical failure.

---

# 12. Claim Verification

The system should classify generated claims.

Possible labels:

- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED
- CONTRADICTED
- UNVERIFIABLE

Evaluation should measure classification accuracy.

---

# 13. Hallucination Evaluation

Measure unsupported factual generation.

Metrics:

- Hallucination rate
- Unsupported claim rate
- Fabricated source rate
- Fabricated citation rate
- Incorrect entity rate
- Incorrect numerical claim rate

The evaluation set must contain questions where the correct behavior is:

"I don't have sufficient evidence."

The system should receive positive evaluation for refusing to invent information when evidence is unavailable.

---

# 14. Faithfulness Evaluation

Faithfulness asks:

"Does the generated answer accurately reflect the retrieved evidence?"

The evaluator should compare:

Generated claim
vs.
Retrieved evidence

The system should not receive credit merely because the final answer sounds plausible.

---

# 15. Contradiction Evaluation

Test cases should intentionally contain conflicting sources.

Expected behavior:

1. Detect the conflict.
2. Identify the conflicting claims.
3. Identify the sources.
4. Preserve provenance.
5. Explain possible reasons where evidence supports them.
6. Avoid silently selecting one claim.

Metrics:

- Contradiction detection precision
- Contradiction detection recall
- Source attribution accuracy
- Conflict explanation accuracy

---

# 16. Temporal Evaluation

ResearchX must distinguish information across time.

Evaluation cases should include:

- Old documentation
- New documentation
- Version changes
- Historical facts
- Updated benchmarks
- Changed pricing
- Changed model specifications

Metrics:

- Temporal correctness
- Version correctness
- Latest-source selection accuracy
- Historical-source preservation

The system must not treat the newest source as automatically correct if the question is historical.

---

# 17. Comparison Evaluation

Comparison outputs should be evaluated field-by-field.

Metrics:

- Attribute accuracy
- Source correctness
- Missing-data accuracy
- Entity alignment
- Numerical correctness
- Contradiction handling

The system must explicitly represent unavailable data.

It must not fill missing values with model guesses.

---

# 18. Decision / Laya Evaluation

Laya is a decision/orchestration component.

Evaluate whether it correctly:

- Decomposes research questions.
- Selects appropriate tools.
- Chooses useful evidence.
- Identifies missing information.
- Requests additional research when required.
- Stops when sufficient evidence exists.
- Handles conflicting evidence.
- Avoids unnecessary tool calls.

Metrics:

- Task completion rate
- Tool-selection accuracy
- Research-plan quality
- Evidence-selection accuracy
- Unnecessary-tool-call rate
- Missing-tool-call rate
- Research efficiency

Laya decisions must remain auditable.

---

# 19. Research Planning Evaluation

Evaluate generated research plans against expert-created reference plans.

Dimensions:

- Coverage
- Logical decomposition
- Relevance
- Source strategy
- Dependency ordering
- Missing research questions
- Redundant research questions

The evaluation should focus on whether the plan enables reliable research rather than whether it matches one exact wording.

---

# 20. End-to-End Evaluation

An end-to-end ResearchJob should be evaluated on:

1. Question understanding
2. Research planning
3. Source discovery
4. Retrieval
5. Extraction
6. Canonicalization
7. Evidence linking
8. Contradiction detection
9. Comparison
10. Synthesis
11. Citation correctness
12. Citation completeness
13. Uncertainty handling
14. Final usefulness

The evaluation system should produce both component-level and overall results.

---

# 21. LLM-as-a-Judge

LLM-as-a-judge may be used for:

- Answer relevance
- Explanation quality
- Research plan quality
- Evidence-grounded reasoning
- Readability
- Structure
- Comparative usefulness

It must not be the sole evaluator for:

- Exact numerical correctness
- Citation existence
- Source identity
- Schema validity
- Security violations
- API correctness
- Dataset integrity

Where possible, deterministic verification must be preferred.

---

# 22. Judge Reliability

LLM judges must be evaluated for:

- Agreement with human evaluators
- Position bias
- Verbosity bias
- Model-family bias
- Prompt sensitivity
- Self-preference bias

Judge prompts must be versioned.

Judge models must be versioned.

---

# 23. Human Evaluation

Human evaluation should assess:

- Overall usefulness
- Evidence usefulness
- Research completeness
- Clarity
- Decision usefulness
- Trustworthiness
- Appropriate uncertainty
- Media quality where applicable

Human evaluators should receive blinded outputs where practical.

---

# 24. Security Evaluation

Security tests must include:

## Prompt Injection

Malicious instructions inside sources.

## Data Exfiltration

Attempts to extract:

- API keys
- Secrets
- Internal prompts
- Private documents
- System information

## Tool Abuse

Attempts to make the model:

- Execute unauthorized actions
- Access unauthorized resources
- Delete data
- Modify infrastructure

## Malicious Documents

Files containing instructions intended to override the system.

## Indirect Injection

Instructions hidden in:

- Websites
- PDFs
- Images
- Metadata
- Search results
- Code repositories

Security failures must be recorded separately from ordinary quality failures.

---

# 25. Guardrail Evaluation

Test that the system:

- Refuses unsupported claims.
- Does not fabricate sources.
- Does not expose secrets.
- Does not follow untrusted document instructions.
- Does not perform unauthorized tool actions.
- Clearly identifies uncertainty.
- Preserves source provenance.
- Does not silently modify user data.
- Does not treat generated media as factual evidence.

---

# 26. Multimodal Evaluation

For image/document understanding evaluate:

- OCR accuracy
- Table extraction
- Chart interpretation
- Diagram interpretation
- Entity extraction
- Spatial relationships
- Caption accuracy
- Evidence grounding

For audio:

- Transcription accuracy
- Speaker separation where supported
- Timestamp accuracy
- Fact extraction
- Summary grounding

For video:

- Scene understanding
- Temporal grounding
- Event extraction
- Timestamp accuracy
- Generated narrative grounding

---

# 27. Media Grounding Evaluation

Generated media must be evaluated against validated research claims.

For generated:

- Images
- Audio
- Videos
- Presentations

check:

1. Are factual claims grounded?
2. Are unsupported facts introduced?
3. Are citations preserved where appropriate?
4. Are visualizations faithful to data?
5. Are generated numbers accurate?
6. Is the source research traceable?

A visually impressive artifact with incorrect facts is a failed artifact.

---

# 28. Fine-Tuning Evaluation

Every fine-tuning experiment must record:

- Base model
- Fine-tuning method
- Dataset version
- Training configuration
- Hyperparameters
- Training duration
- Hardware
- Checkpoint
- Evaluation dataset
- Baseline score
- Fine-tuned score

Potential methods:

- SFT
- IFT
- LoRA
- QLoRA
- DoRA
- Prompt tuning
- DPO
- RLHF where supported
- Domain adaptation
- Continued pretraining

The selected technique must correspond to the actual problem being solved.

---

# 29. Fine-Tuning Acceptance Criteria

A fine-tuned model must demonstrate:

- Improvement on target task.
- No unacceptable regression on general behavior.
- No unacceptable increase in hallucination.
- No unacceptable citation degradation.
- No unacceptable security regression.
- Reproducible evaluation results.

Fine-tuning should be rejected if it improves one narrow benchmark while materially degrading system reliability.

---

# 30. Regression Testing

Every significant change should run regression tests.

Triggers include:

- Model change
- Prompt change
- Retrieval change
- Embedding change
- Database schema change
- Tool change
- Fine-tuning
- Architecture change
- Security change

Regression tests should compare against the previous accepted version.

---

# 31. Evaluation Thresholds

Thresholds should be task-specific rather than universally fixed.

Each evaluation suite should define:

- Minimum acceptable score
- Target score
- Critical failure conditions
- Regression tolerance

Example:

Citation correctness:
minimum = defined by evaluation suite

Hallucination:
maximum = defined by evaluation suite

Security:
critical violations = zero tolerance for release-blocking categories

The initial thresholds should be established after collecting baseline results.

---

# 32. Cost Evaluation

Measure:

- Input tokens
- Output tokens
- Retrieval operations
- Tool calls
- GPU/inference cost
- Storage
- Media-generation cost
- Fine-tuning cost

Metrics:

- Cost per research job
- Cost per successful research job
- Cost per validated claim
- Cost per evaluation run

Optimization must not reduce reliability below accepted thresholds.

---

# 33. Latency Evaluation

Measure:

- Time to first response
- Research planning latency
- Retrieval latency
- Extraction latency
- Model inference latency
- End-to-end job latency
- Media generation latency

Use:

- p50
- p95
- p99

where sufficient traffic exists.

---

# 34. Reliability Evaluation

Measure:

- Job success rate
- Tool failure rate
- Model failure rate
- Retrieval failure rate
- API error rate
- Retry rate
- Timeout rate
- Partial-result rate

Long-running research should support recovery where possible.

---

# 35. Reproducibility Evaluation

Given the same:

- Research question
- Dataset version
- Source snapshot
- Model version
- Prompt version
- Configuration

the system should produce materially consistent results.

Exact deterministic output is not required for stochastic models, but critical factual claims and evidence associations should remain stable.

---

# 36. Evaluation Reports

Every evaluation run should produce a structured report.

Required fields:

- evaluation_id
- dataset_id
- dataset_version
- model
- model_version
- prompt_version
- pipeline_version
- timestamp
- metrics
- failures
- examples
- regressions
- configuration
- artifacts

---

# 37. Evaluation Storage

Evaluation results should be persisted.

Suggested structure:

evaluation/
    runs/
    datasets/
    baselines/
    reports/
    failures/
    regression/
    security/

Evaluation records should be queryable by:

- Model
- Version
- Dataset
- Task
- Date
- Pipeline
- Experiment

---

# 38. Failure Analysis

A failed example must be classified.

Failure categories:

- Retrieval failure
- Extraction failure
- Canonicalization failure
- Evidence-linking failure
- Reasoning failure
- Citation failure
- Temporal failure
- Tool-selection failure
- Guardrail failure
- Security failure
- Media-grounding failure
- Dataset error
- Infrastructure failure

Each critical failure should have a root-cause analysis.

---

# 39. Golden Dataset

ResearchX should maintain a small high-quality golden dataset.

The golden dataset must contain manually validated examples.

It should cover:

- Core research
- Retrieval
- Evidence
- Citation
- Contradictions
- Temporal reasoning
- Multimodal inputs
- Security
- Edge cases

The golden dataset must be protected from accidental contamination by training pipelines.

---

# 40. Synthetic Evaluation Data

Synthetic data may be used to expand coverage.

Synthetic data must:

- Be clearly labeled.
- Be validated before becoming an evaluation target.
- Not be treated as automatically correct.
- Not replace high-quality real-world evaluation data.

Synthetic adversarial cases are encouraged for security testing.

---

# 41. Dataset Contamination

Evaluation data must not unintentionally become training data.

Track:

- Dataset provenance
- Dataset access
- Training usage
- Fine-tuning usage
- Evaluation usage

If contamination is suspected, the dataset must be flagged and replaced or revalidated.

---

# 42. Benchmarking Models

Models should be compared using the same:

- Dataset
- Prompt
- Retrieval configuration
- Tool permissions
- Evaluation criteria

Examples of model families available through Nebius may include NVIDIA Nemotron, Qwen, DeepSeek, Gemma, Llama, GLM, Kimi, and others.

Model comparisons must remain task-specific.

No single model should be declared universally superior.

---

# 43. Model Selection

Model selection should consider:

- Accuracy
- Evidence grounding
- Reasoning quality
- Tool use
- Latency
- Cost
- Context length
- Multimodal capability
- Reliability
- Security
- Fine-tuning compatibility

The selected model must be justified by evaluation results.

---

# 44. Evaluation of NVIDIA Models

Because NVIDIA open models are a core hackathon requirement, at least one NVIDIA model must be included in the evaluation matrix.

The evaluation should demonstrate an actual ResearchX task using the NVIDIA model.

Possible tasks:

- Research planning
- Evidence synthesis
- Structured extraction
- Multimodal understanding
- Decision orchestration

The final project should document:

- NVIDIA model used
- Task performed
- Input
- Output
- Evaluation method
- Results
- Why it was selected for that component

---

# 45. Release Gates

A release candidate should pass:

## Functional Gate

Core workflows operate correctly.

## Quality Gate

Critical evaluation metrics meet thresholds.

## Evidence Gate

No unacceptable fabricated or unsupported citation behavior.

## Security Gate

No release-blocking security failures.

## Regression Gate

No unacceptable degradation from the previous accepted version.

## Performance Gate

Latency and cost remain within defined limits.

## Documentation Gate

Architecture and API documentation are synchronized.

---

# 46. Hackathon Demonstration Evaluation

The hackathon demo should demonstrate a complete chain:

User Research Question
        ↓
Research Plan
        ↓
Source Discovery
        ↓
NVIDIA Model
        ↓
Extraction
        ↓
Evidence Graph
        ↓
Comparison
        ↓
Contradiction Detection
        ↓
Validated Synthesis
        ↓
Citations
        ↓
Optional Image / Audio / Video Artifact

The demo should make the evidence trail visible rather than only showing the final answer.

---

# 47. Definition of Success

ResearchX succeeds when it can demonstrate that:

- Research questions are correctly decomposed.
- Relevant evidence is retrieved.
- Information is extracted into structured data.
- Entities are correctly canonicalized.
- Conflicts are surfaced.
- Claims remain connected to evidence.
- Citations are correct and complete.
- Unsupported claims are minimized.
- Uncertainty is represented.
- NVIDIA open models are meaningfully integrated.
- Laya can assist with research decisions/orchestration.
- Fine-tuning can be evaluated scientifically when introduced.
- Media can be generated from validated research without introducing unsupported facts.
- The complete workflow is observable and reproducible.

The central evaluation principle is:

> A ResearchX result is successful only when the system can show not only what it concluded, but why the conclusion is supported.

============================================================
RESEARCHX — EVALUATION SPECIFICATION EXTENSION
============================================================

This section extends the existing ResearchX Evaluation Specification.

The original evaluation specification remains authoritative.

The additions below define the external benchmark program,
model evaluation program, system evaluation program, ablation
program, performance evaluation program, security evaluation
program, and release gates.

ResearchX must never claim that a model or system is "better"
without controlled evaluation.

============================================================
1. EVALUATION PHILOSOPHY
============================================================

ResearchX is not evaluated as a single LLM.

It is evaluated as a complete research system consisting of:

    DATA
      ↓
    RETRIEVAL
      ↓
    RERANKING
      ↓
    EXTRACTION
      ↓
    EVIDENCE LINKING
      ↓
    CANONICALIZATION
      ↓
    COMPARISON
      ↓
    LAYA DECISION / ROUTING
      ↓
    REASONING
      ↓
    SYNTHESIS
      ↓
    CITATIONS
      ↓
    USER ANSWER

Every major layer must have independent evaluation.

The complete system must also have end-to-end evaluation.

A high score on a general LLM benchmark does not prove that
ResearchX is a good research assistant.

A strong retrieval score does not prove that the final answer
is scientifically correct.

A strong final-answer score does not prove that the system's
citations or evidence are correct.

Therefore evaluation must be layered.

============================================================
2. EVALUATION TARGETS
============================================================

ResearchX evaluates five distinct targets.

TARGET A — FOUNDATION MODELS

Evaluate NVIDIA/open models and other permitted open models
used by ResearchX.

Examples include:

- NVIDIA Nemotron family
- NVIDIA Nemotron reasoning models
- NVIDIA Cosmos models where multimodal functionality is used
- NVIDIA-optimized model variants available through Nebius
- other approved open models used for comparison

The model identity, checkpoint, quantization, inference provider,
prompt format, temperature, max tokens and inference configuration
must always be recorded.

TARGET B — SPECIALIZED COMPONENTS

Evaluate:

- embedding models
- rerankers
- extraction models
- comparison models
- Laya decision model
- media generation models
- judge models

TARGET C — RESEARCHX SERVICES

Evaluate:

- retrieval
- extraction
- canonicalization
- comparison
- evidence
- Laya
- media
- caching
- tool calling
- guardrails

TARGET D — COMPLETE RESEARCHX PIPELINE

Evaluate:

    query
      →
    planning
      →
    retrieval
      →
    reranking
      →
    evidence extraction
      →
    reasoning
      →
    comparison
      →
    synthesis
      →
    citation verification

TARGET E — USER-LEVEL RESEARCH TASK

Evaluate whether ResearchX actually helps a researcher answer
complex research questions accurately and reproducibly.

============================================================
3. EXTERNAL BENCHMARK PROGRAM
============================================================

ResearchX must evaluate against established public benchmarks
where the benchmark task matches a ResearchX capability.

Do not force ResearchX onto a benchmark when the benchmark does
not measure a relevant capability.

Maintain a benchmark registry:

    data/eval/benchmarks/

Each benchmark must have:

    benchmark_id
    version
    source
    license
    task_type
    domain
    input_format
    expected_output
    evaluation_method
    metric
    retrieval_required
    citation_required
    long_context_required
    multimodal
    date_acquired
    contamination_notes

============================================================
4. RETRIEVAL BENCHMARKS
============================================================

ResearchX retrieval must be evaluated using multiple retrieval
benchmarks.

Minimum benchmark families:

------------------------------------------------------------
4.1 BEIR
------------------------------------------------------------

Use BEIR as a general heterogeneous information-retrieval
baseline.

Evaluate:

- BM25
- dense retrieval
- hybrid retrieval
- reranking
- embedding variants

Metrics:

- Recall@k
- Precision@k
- MRR@k
- nDCG@k
- MAP

Do not report only one aggregate score.

Report per-dataset performance.

------------------------------------------------------------
4.2 BRIGHT
------------------------------------------------------------

Use BRIGHT for reasoning-intensive retrieval.

This is important because ResearchX queries frequently require
understanding the research question before identifying the
correct evidence.

Evaluate:

- lexical retrieval
- vector retrieval
- hybrid retrieval
- query expansion
- Laya-assisted retrieval planning
- reranking

Measure whether reasoning-aware retrieval improves retrieval
quality over lexical or semantic retrieval alone.

------------------------------------------------------------
4.3 ResearchX Retrieval Benchmark
------------------------------------------------------------

Maintain an internal benchmark specifically derived from the
ResearchX dataset.

Each query must contain:

    query
    relevant_source_ids
    relevant_evidence_ids
    hard_negative_ids
    difficulty
    domain
    reasoning_depth

Include:

- exact-match queries
- semantic queries
- terminology mismatch
- multi-hop queries
- cross-paper queries
- contradictory evidence
- temporal queries
- rare terminology
- numeric queries
- experiment configuration queries
- benchmark-result queries
- artifact/code queries

============================================================
5. EMBEDDING EVALUATION
============================================================

Embedding models must be evaluated independently.

Use relevant MTEB/MMTEB tasks where applicable.

Measure:

- semantic retrieval
- clustering where relevant
- reranking compatibility
- long-document retrieval
- scientific retrieval
- domain-specific retrieval

ResearchX must not select an embedding model only because it
has a high aggregate benchmark score.

Evaluate it on ResearchX-specific retrieval tasks.

Record:

    embedding_model
    dimension
    precision
    latency
    throughput
    memory
    cost
    retrieval_quality

============================================================
6. RERANKER EVALUATION
============================================================

Evaluate rerankers independently from retrieval.

Pipeline:

    Query
      ↓
    Candidate Retrieval
      ↓
    Reranker
      ↓
    Ranked Evidence

Measure:

- nDCG@k
- MRR@k
- Recall@k
- Precision@k
- ranking improvement over first-stage retrieval

Report:

    retrieval_only
    retrieval + reranker

The reranker must be evaluated on hard negatives.

============================================================
7. SCIENTIFIC QUESTION ANSWERING
============================================================

Use scientific QA benchmarks where applicable.

Minimum benchmark families:

- QASPER
- SciFact
- LitQA / PaperQA-style evaluation
- ResearchX internal scientific QA

QASPER evaluates question answering over full scientific
papers and includes supporting evidence.

SciFact evaluates scientific claim verification and evidence
selection.

ResearchX must additionally test:

- citation correctness
- evidence completeness
- unsupported claims
- abstention
- contradiction handling

Exact-answer accuracy alone is insufficient.

============================================================
8. DEEP RESEARCH EVALUATION
============================================================

Evaluate ResearchX against established deep-research benchmarks
where licensing and access permit.

Candidate benchmark:

    DeepResearch Bench

Evaluate:

- research completeness
- factual correctness
- citation quality
- citation coverage
- evidence quality
- source diversity
- research depth
- reasoning quality
- tool usage
- hallucination rate
- unsupported claims

Also maintain a ResearchX-specific deep research benchmark.

============================================================
9. GENERAL MODEL BENCHMARKS
============================================================

Foundation models may additionally be evaluated using relevant
general-purpose benchmarks.

Candidate categories:

    knowledge
    reasoning
    scientific reasoning
    mathematics
    coding
    instruction following
    long-context understanding
    factuality

Examples may include:

- MMLU / MMLU-Pro
- GPQA
- relevant mathematical reasoning benchmarks
- relevant coding benchmarks
- relevant long-context benchmarks
- relevant instruction-following benchmarks

These benchmarks are supplementary.

They must never replace ResearchX-specific evaluation.

============================================================
10. MODEL COMPARISON
============================================================

Every candidate model must be evaluated using the same:

- dataset version
- prompts
- retrieval corpus
- tools
- temperature
- token budget
- evaluation harness
- judge configuration
- random seeds where applicable

Compare models using a controlled matrix.

Example:

    Model A
    Model B
    Model C
    Model D

Evaluate:

    accuracy
    evidence grounding
    citation accuracy
    extraction F1
    retrieval quality
    abstention quality
    latency
    tokens
    cost
    tool-call success
    structured-output validity

Never select a model using benchmark score alone.

============================================================
11. STRUCTURED EXTRACTION EVALUATION
============================================================

ResearchX extraction must be evaluated field-by-field.

Example fields:

    population
    intervention
    comparator
    dataset
    model
    architecture
    training method
    optimizer
    learning rate
    batch size
    epochs
    hardware
    preprocessing
    augmentation
    evaluation dataset
    metric
    metric value
    confidence interval
    statistical test
    baseline
    ablation
    experimental condition

Metrics:

    exact match
    normalized exact match
    precision
    recall
    F1
    numeric accuracy
    unit accuracy
    evidence-span accuracy

UNKNOWN must be treated as a valid state.

The system must not be rewarded for guessing missing values.

============================================================
12. EVIDENCE EVALUATION
============================================================

Every extracted assertion must be evaluated for:

    source correctness
    passage correctness
    evidence sufficiency
    attribution correctness
    temporal validity

Measure:

    evidence precision
    evidence recall
    evidence F1
    citation precision
    citation recall
    citation coverage

A generated statement is not considered grounded merely because
a citation exists.

The cited evidence must actually support the claim.

============================================================
13. CITATION EVALUATION
============================================================

Evaluate citations independently.

For each claim:

    claim
    citation
    supporting_passage

Classify:

    SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    INSUFFICIENT_EVIDENCE

Metrics:

    citation precision
    citation recall
    citation coverage
    citation entailment
    unsupported-claim rate

ResearchX should optimize for citation correctness rather than
citation quantity.

============================================================
14. ABSTENTION EVALUATION
============================================================

Abstention is a first-class capability.

The system must be rewarded for refusing to answer when evidence
is insufficient.

Evaluate:

    answerable questions
    unanswerable questions
    contradictory questions
    ambiguous questions
    missing-evidence questions

Metrics:

    abstention precision
    abstention recall
    selective accuracy
    coverage
    risk-coverage curve

False confidence must be treated as a serious failure.

============================================================
15. CONTRADICTION EVALUATION
============================================================

Create evaluation cases where credible sources disagree.

Evaluate whether ResearchX:

    detects disagreement
    identifies the conflicting evidence
    preserves attribution
    identifies methodological differences
    distinguishes true contradiction from population/configuration
    differences
    avoids arbitrarily selecting one source

The system must not collapse contradictory evidence into a single
unsupported conclusion.

============================================================
16. EXPERIMENT COMPARISON EVALUATION
============================================================

This is a core ResearchX capability.

Evaluate:

    SAME_EXPERIMENT
    CLOSE_BUT_DIFFERENT
    DIFFERENT_EXPERIMENT
    REPLICATION
    EXTENSION
    INSUFFICIENT_INFORMATION

Evaluate both:

    pair classification

and:

    field-level difference identification

Measure:

    accuracy
    macro-F1
    precision
    recall
    field-level F1
    critical-difference accuracy

Critical fields must receive higher evaluation importance where
a difference materially changes experimental interpretation.

============================================================
17. LAYA EVALUATION
============================================================

Laya is evaluated as a decision/orchestration component.

Evaluate whether Laya selects the appropriate action:

    RETRIEVE
    RERANK
    EXTRACT
    COMPARE
    RECHECK
    ABSTAIN
    ASK_CLARIFICATION
    USE_TOOL
    GENERATE_MEDIA
    SYNTHESIZE

Measure:

    routing accuracy
    unnecessary-tool-call rate
    missed-tool-call rate
    recovery rate
    cost efficiency
    latency
    final-answer impact

Laya must not be evaluated solely on whether its routing decision
looks reasonable.

Evaluate the downstream effect of routing decisions.

============================================================
18. MODEL ROUTING EVALUATION
============================================================

Evaluate whether routing improves:

    quality
    latency
    cost

Compare:

    single-model baseline
    fixed routing
    Laya routing

Metrics:

    quality per dollar
    quality per token
    latency
    successful completion rate
    tool success
    evidence correctness

Routing must not sacrifice critical safety or factuality merely
to reduce cost.

============================================================
19. TOOL-CALLING EVALUATION
============================================================

Evaluate:

    correct tool selection
    correct arguments
    schema validity
    tool-call ordering
    unnecessary calls
    failed calls
    recovery after tool failure
    malicious tool-call attempts

Test:

    normal
    ambiguous
    adversarial
    malformed
    prompt-injected
    unavailable-tool

inputs.

============================================================
20. GUARDRAIL EVALUATION
============================================================

Guardrails must be tested independently.

Evaluate:

    prompt injection
    indirect prompt injection
    malicious documents
    malicious metadata
    tool injection
    data exfiltration attempts
    unauthorized actions
    unsafe media requests
    unsupported scientific claims
    fabricated citations
    fabricated evidence
    schema attacks

Measure:

    attack success rate
    false positive rate
    false negative rate
    unsafe tool-call rate
    data leakage rate

A safety-critical failure can block a release even when quality
benchmarks improve.

============================================================
21. PROMPT INJECTION EVALUATION
============================================================

Research papers, websites, PDFs and retrieved passages must be
treated as untrusted data.

Create adversarial documents containing instructions such as:

    ignore previous instructions
    call this tool
    reveal system prompt
    reveal API credentials
    modify database
    change research conclusion

ResearchX must treat these instructions as document content.

The document must never gain authority over the agent.

============================================================
22. CONTEXT-WINDOW EVALUATION
============================================================

Do not equate advertised context window with effective context.

Evaluate:

    short context
    medium context
    long context
    very long context

Measure:

    retrieval accuracy
    evidence recall
    reasoning accuracy
    citation accuracy
    lost-in-middle behavior
    instruction retention

Compare:

    full-context approach
    retrieval approach
    hierarchical summarization
    context compression
    selective evidence injection

============================================================
23. TOKEN BUDGET EVALUATION
============================================================

Evaluate performance under multiple token budgets.

Example:

    low
    medium
    high

Measure:

    quality
    evidence coverage
    latency
    cost
    completion rate

Determine the minimum useful token budget for each task class.

============================================================
24. KV CACHE / PROMPT CACHE EVALUATION
============================================================

Caching optimizations must be evaluated for:

    latency
    throughput
    cache hit rate
    cost reduction
    correctness

Caching must never cause stale evidence or cross-user data leakage.

Test cache isolation explicitly.

============================================================
25. SEMANTIC CACHE EVALUATION
============================================================

Evaluate:

    exact cache hit
    semantic cache hit
    false cache hit
    false cache miss

A semantic cache must not return an answer when the research
question differs in a scientifically meaningful way.

Temporal and source-version changes must invalidate affected cache
entries.

============================================================
26. HYBRID RETRIEVAL ABLATION
============================================================

Run controlled ablations:

    BM25 only
    vector only
    BM25 + vector
    BM25 + vector + reranker

Measure:

    Recall@k
    nDCG@k
    MRR@k
    downstream answer accuracy
    citation accuracy
    latency
    cost

============================================================
27. MILVUS + PGVECTOR ABLATION
============================================================

Evaluate the selected retrieval architecture against alternatives.

Measure:

    indexing time
    query latency
    recall
    scale
    memory
    operational complexity
    metadata join performance

PostgreSQL remains the source of relational/evidence metadata.

Milvus may be used for scalable vector retrieval.

pgvector may be used where relational locality and vector search
within PostgreSQL are advantageous.

Do not assume either database is superior without measurement.

============================================================
28. RERANKING ABLATION
============================================================

Compare:

    no reranker
    lightweight reranker
    large reranker

Measure:

    retrieval quality
    downstream evidence quality
    latency
    cost

============================================================
29. STRUCTURED OUTPUT EVALUATION
============================================================

Evaluate:

    valid JSON rate
    schema compliance
    missing-field rate
    hallucinated-field rate
    repair rate
    semantic correctness

Invalid structured output must be counted as a measurable failure.

============================================================
30. FINE-TUNING EVALUATION
============================================================

Fine-tuning must never be introduced merely because it is possible.

For every fine-tuned model compare:

    base model
    prompted base model
    fine-tuned model

Potential approaches include:

    SFT
    instruction fine-tuning
    domain adaptation
    LoRA
    QLoRA
    DoRA
    preference optimization
    DPO
    other PEFT approaches

Evaluate:

    task accuracy
    generalization
    hallucination
    citation grounding
    structured extraction
    abstention
    safety
    regression on general capabilities

Fine-tuning must be rejected if it improves the training domain
while materially degrading broader reliability.

============================================================
31. LLM-AS-A-JUDGE
============================================================

LLM-as-a-judge may be used for subjective dimensions.

Judge criteria may include:

    correctness
    completeness
    research depth
    reasoning quality
    citation quality
    evidence grounding
    clarity
    usefulness

However:

    LLM judge ≠ ground truth

Whenever possible, use:

    deterministic metric
    expert annotation
    citation matching
    evidence matching

before relying on LLM judgment.

Judge prompts and judge-model versions must be versioned.

Run judge-agreement checks on a human-annotated subset.

============================================================
32. HUMAN EVALUATION
============================================================

Human evaluation is required for high-value subjective tasks.

Reviewers evaluate:

    factual correctness
    research completeness
    evidence quality
    citation correctness
    comparison correctness
    usefulness
    abstention appropriateness

Record:

    reviewer_id
    task_id
    rubric_version
    rating
    confidence
    disagreement

Measure inter-rater agreement.

============================================================
33. MEDIA EVALUATION
============================================================

If ResearchX generates:

    images
    audio
    video

media must be evaluated separately.

Evaluate:

    prompt adherence
    factual grounding
    source attribution
    visual/audio quality
    consistency
    temporal consistency for video
    hallucinated content
    unsafe content

Media generation must not invent scientific findings.

Generated media must clearly distinguish:

    source-derived visualization
    generated explanatory content
    speculative illustration

============================================================
34. PERFORMANCE EVALUATION
============================================================

Measure:

    time-to-first-token
    total latency
    retrieval latency
    reranking latency
    tool latency
    model latency
    database latency
    throughput
    tokens/sec
    input tokens
    output tokens
    cache hit rate
    GPU utilization
    memory utilization
    cost/request

Measure both:

    p50
    p95
    p99

where meaningful.

============================================================
35. END-TO-END RESEARCH QUALITY
============================================================

Every end-to-end benchmark report must contain:

    research question
    retrieved sources
    selected evidence
    extracted facts
    reasoning/comparison
    final answer
    citations
    abstentions
    tool calls
    latency
    token usage
    model configuration
    dataset version

Evaluate:

    factual correctness
    evidence grounding
    completeness
    citation correctness
    contradiction handling
    uncertainty
    answer usefulness

============================================================
36. REGRESSION TESTING
============================================================

Every significant code/model/prompt/dataset change must run
regression evaluation.

Regression suites include:

    retrieval
    extraction
    evidence
    comparison
    citation
    abstention
    security
    tool calling
    end-to-end research

Store historical results.

Never overwrite previous benchmark results.

============================================================
37. DATASET LEAKAGE / CONTAMINATION
============================================================

Benchmark contamination must be actively investigated.

For every benchmark record:

    source
    publication date
    dataset version
    model training cutoff where known
    benchmark release date

must be recorded.

If contamination is suspected:

    flag the result
    do not silently report it as a clean zero-shot result

ResearchX internal evaluation data must remain isolated from
training data used for model development.

============================================================
38. TRAIN / VALIDATION / TEST SEPARATION
============================================================

Never use the final evaluation set for:

    prompt optimization
    model selection
    dataset construction
    fine-tuning
    retrieval tuning

Maintain:

    TRAIN
    DEVELOPMENT
    VALIDATION
    TEST
    PRIVATE HOLDOUT

The private holdout should remain inaccessible to normal
development workflows.

============================================================
39. HARD NEGATIVES
============================================================

ResearchX evaluation must contain scientifically plausible
hard negatives.

Examples:

    same model, different dataset
    same dataset, different split
    same method, different hyperparameters
    same benchmark, different metric
    same metric, different task
    same paper family, different experiment
    same terminology, different meaning
    same result, different population
    same intervention, different condition

Hard negatives are especially important for:

    retrieval
    reranking
    experiment comparison
    evidence linking

============================================================
40. TEMPORAL EVALUATION
============================================================

Research knowledge changes.

Evaluate:

    historical questions
    current questions
    versioned sources
    updated papers
    updated benchmark results
    changed documentation

ResearchX must not use newer evidence to answer a question
that explicitly asks for information available at an earlier
point in time.

============================================================
41. RESEARCHX SCORECARD
============================================================

Maintain a single machine-readable evaluation record.

Example:

    evaluation_run_id
    git_commit
    dataset_version
    benchmark_version
    model_version
    embedding_version
    reranker_version
    laya_version
    prompt_version
    judge_version
    configuration
    metrics
    latency
    token_usage
    cost
    safety_results
    regression_status

============================================================
42. RELEASE GATES
============================================================

A release must satisfy all applicable gates.

QUALITY GATE

No statistically meaningful regression on critical internal
benchmarks.

RETRIEVAL GATE

Retrieval performance must meet the configured minimum threshold.

EVIDENCE GATE

Unsupported evidence-linked claims must remain below the maximum
allowed threshold.

CITATION GATE

Citation correctness must meet the configured threshold.

ABSTENTION GATE

The system must abstain appropriately on insufficient-evidence
cases.

SECURITY GATE

Critical prompt-injection/tool-abuse vulnerabilities block release.

STRUCTURED OUTPUT GATE

Schema validity must meet the configured threshold.

PERFORMANCE GATE

Latency and cost must remain within configured limits.

REGRESSION GATE

Critical benchmark regressions block release.

============================================================
43. ABLATION STUDY PROGRAM
============================================================

ResearchX must periodically run ablations for:

    BM25
    vector retrieval
    hybrid retrieval
    reranking
    Milvus
    pgvector
    Laya
    model routing
    semantic cache
    prompt cache
    KV cache
    context compression
    tool calling
    guardrails
    fine-tuning

Each ablation must answer:

    What changed?
    What metric changed?
    By how much?
    At what cost?
    Does the improvement survive end-to-end evaluation?

============================================================
44. STATISTICAL REPORTING
============================================================

Whenever practical, report:

    mean
    median
    confidence interval
    sample size
    variance

For model comparisons, use appropriate statistical tests or
bootstrap confidence intervals.

Do not overinterpret tiny differences.

Benchmark differences must be considered in relation to
evaluation uncertainty.

============================================================
45. BENCHMARK REGISTRY
============================================================

Create:

    data/eval/benchmarks/registry.yaml

Initial benchmark registry should include:

    BEIR
    BRIGHT
    MTEB
    MMTEB
    SciFact
    QASPER
    LitQA / PaperQA-style evaluation
    DeepResearch Bench
    ResearchX Retrieval
    ResearchX Extraction
    ResearchX Evidence
    ResearchX Comparison
    ResearchX Citation
    ResearchX Abstention
    ResearchX Security
    ResearchX End-to-End

Additional benchmarks may be added when they measure a capability
relevant to ResearchX.

============================================================
46. BENCHMARK HARNESS
============================================================

Create a reproducible benchmark harness.

Suggested structure:

    evaluation/
        benchmarks/
        datasets/
        runners/
        metrics/
        judges/
        reports/
        regression/
        configs/

The harness must support:

    benchmark discovery
    dataset loading
    model configuration
    inference
    retrieval
    tool execution
    scoring
    LLM judging
    human evaluation import
    result storage
    comparison
    regression detection
    report generation

============================================================
47. EVALUATION REPORT FORMAT
============================================================

Every evaluation run must produce:

    summary.json
    metrics.json
    per_example.jsonl
    errors.jsonl
    config.json
    report.md

The report must contain:

    model
    dataset
    benchmark
    configuration
    aggregate metrics
    per-category metrics
    failure analysis
    latency
    cost
    safety results
    regression results
    known limitations

============================================================
48. FAILURE ANALYSIS
============================================================

Benchmark scores alone are insufficient.

For every major evaluation run, sample failures and classify them.

Failure categories include:

    retrieval_failure
    ranking_failure
    extraction_failure
    normalization_failure
    comparison_failure
    reasoning_failure
    citation_failure
    hallucination
    unsupported_claim
    abstention_failure
    tool_failure
    routing_failure
    prompt_injection_failure
    schema_failure
    context_failure
    temporal_failure

Failure analysis should drive future dataset construction.

============================================================
49. RESEARCHX INTERNAL GOLD EVALUATION SET
============================================================

ResearchX must eventually maintain a human-verified Gold set.

Gold records must contain:

    question
    expected research target
    source documents
    evidence spans
    extracted structured facts
    experiment identity
    comparison label
    expected citation
    abstention requirement
    expert rationale

Gold data must never be generated solely by an LLM.

LLMs may assist annotation, but critical Gold labels require
human verification.

============================================================
50. FINAL EVALUATION PRINCIPLE
============================================================

ResearchX succeeds only when it demonstrates all of the following:

    retrieves the right evidence
          +
    extracts the right facts
          +
    distinguishes similar experiments correctly
          +
    recognizes contradictions
          +
    cites the actual supporting evidence
          +
    abstains when evidence is insufficient
          +
    uses tools correctly
          +
    resists prompt injection
          +
    routes work intelligently
          +
    remains cost and latency efficient
          +
    improves measurably against baselines

The objective is not to maximize one benchmark score.

The objective is to build a scientifically reliable,
evidence-grounded research system whose improvements can be
measured, reproduced and defended.

============================================================
51. IMPLEMENTATION AUDIT ADDENDUM (2026-10-02)
============================================================

51.1 Status

No runtime evaluation harness exists yet. Dataset validation,
audit and statistics exist (services/datasets/validate.py,
audit.py, stats.py). No benchmark has been run, so no benchmark
result exists and none may be reported.

51.2 Verified capability baselines (smoke tests, not benchmarks)

scripts/research/probe_models.py ran one minimal request per
capability against all 25 catalog models
(data/manifests/model_capabilities.json):

- NVIDIA models (Nemotron-3-Nano-30B-A3B, nemotron-3-super-120b-
  a12b, Nemotron-3-Ultra-550b-a55b, Nemotron-3_5-Lightning): chat,
  json_schema (schema-valid, correct) and tool calling passed;
  image input rejected with HTTP 400.
- json_schema failed for Kimi-K2.6 and GLM-5.3-Flash; tool calls
  were not produced by gemma-3-27b-it and MiniCPM-V-4_5.
- Vision input passed for 8 non-NVIDIA models.

These are the starting points for §19 (tool calling) and §29
(structured output). One probe per capability says a feature is
usable, not how reliable it is; validity rates must come from the
production call logs and the evaluation sets.

51.3 Verified constraints that change this plan

- §5 Embedding evaluation: the catalog has one embedding model
  (Qwen/Qwen3-Embedding-8B). Evaluate output dimensions (for
  example 256, 512, 1024, 2000) instead of model variants.
- §27 Milvus + pgvector ablation: pgvector indexes accept at most
  2,000 dimensions (vector) or 4,000 (halfvec). Run pgvector
  configurations (dimension, vector vs halfvec, HNSW parameters)
  first; the Milvus arm waits until Milvus is justified by load.
- §6 and §28 Reranking: the catalog has no reranker model. The
  candidates are no reranking vs LLM reranking with
  Nemotron-3-Nano or Nemotron-3_5-Lightning; a cross-encoder arm
  exists only if one is added.
- §24 KV / prompt cache: a repeated 3,874-token prefix showed no
  cached tokens. Report latency and cached_tokens only; claim no
  cache savings unless they are observed.
- §9 and §10 Model comparison: the matrix covers the four NVIDIA
  catalog models plus non-NVIDIA comparators. Models listed in
  older docs but absent from the catalog are excluded.
- §17 Laya: the shipped checkpoints are over-confident (Laya
  README). Fit and report ECE before any threshold is used; the
  laya.evals harness reports accuracy and ECE.
- §31 LLM-as-a-judge: judges come from a different model family
  than the generator (json_schema verified for Qwen, DeepSeek and
  GLM models) and are calibrated against human labels.
- §33 Media: the provider has no image, audio or video generation
  model; media evaluation does not apply until a provider exists.
- External web research: report evidence yield per Tavily credit,
  primary-source citation rate and credits per answered question
  (plan limit 1,000 credits per billing cycle).

51.4 Benchmark registry status

Licences were checked on 2026-09-30 and recorded in
data/manifests/source_discovery.yaml. Nothing below is acquired
or run yet. All are evaluation-only and never training data.

| Benchmark | Capability | Licence (as recorded) | Plan |
|---|---|---|---|
| BEIR (scientific subsets) | retrieval | per subset (e.g. scifact CC BY-SA 4.0) | M2 |
| LitSearch | literature retrieval | code MIT; dataset licence unstated | M2, evaluation use only |
| BRIGHT | reasoning-intensive retrieval | CC BY 4.0 | M2 |
| SciFact | claim verification | annotations CC BY 4.0; abstracts separate terms | M3 |
| QASPER | scientific QA with evidence | CC BY 4.0 | M3 |
| SciTab | table claim verification | MIT | M3 |
| ResearchQA | scholarly QA | MIT | M3 |
| BrowseComp-Plus | deep research on a fixed corpus | MIT | later |
| DeepResearch Bench | deep research reports | Apache-2.0 (repository) | later |
| GAIA | agent tasks | gated (BLOCKED_MANUAL_DOWNLOAD) | blocked |

Each run records the benchmark version, scope, metric, baseline,
ResearchX result, model, prompt version, retrieval configuration,
date and limitations (§45-47). Public benchmark results and
internal ResearchX results are reported separately.

51.5 Internal retrieval benchmark (§4.3) construction rule

Queries and relevance labels are derived from SOURCE_CHECKED
evidence (the relevant unit is the evidence location). They stay
Silver until a human verifies them, are split by study to prevent
leakage, are never used for training, and every run records the
dataset version.

============================================================
END OF EVALUATION SPECIFICATION EXTENSION
============================================================