# PRODUCT_SPEC.md

# ResearchX — Product Specification

## 1. Product Overview

ResearchX is an enterprise-grade AI research and decision-support platform designed to transform fragmented web information, documents, datasets, multimedia, and technical sources into structured, traceable, evidence-backed research.

The system should not behave as a generic chatbot.

It should behave as a research system that:

1. Understands a research objective.
2. Decomposes the objective into research questions.
3. Discovers relevant sources.
4. Extracts structured information.
5. Canonicalizes entities, claims, terminology, and sources.
6. Retrieves relevant evidence.
7. Compares competing information.
8. Detects contradictions and uncertainty.
9. Produces evidence-backed conclusions.
10. Allows users to inspect the evidence behind every important claim.
11. Optionally generates supporting media such as images, audio, and video when appropriate.
12. Continuously improves through evaluation, feedback, and optional model post-training.

The core product principle is:

> ResearchX should optimize for traceability, evidence quality, reproducibility, and decision usefulness rather than simply generating fluent answers.

---

# 2. Problem

Modern research workflows are fragmented across:

- Search engines
- Web pages
- PDFs
- Academic papers
- Documentation
- GitHub repositories
- Databases
- News sources
- Images
- Audio
- Video
- Structured datasets
- Internal organizational knowledge

Users must manually:

- Find sources.
- Read them.
- Extract information.
- Resolve conflicting terminology.
- Compare claims.
- Track citations.
- Determine source reliability.
- Build summaries.
- Create presentations or other research artifacts.

This creates several problems:

- Information overload.
- Repeated manual work.
- Poor source traceability.
- Hallucination risk.
- Difficult comparison across sources.
- Loss of provenance.
- Difficulty reproducing research.
- Difficulty converting research into useful outputs.

ResearchX addresses this by creating a structured research pipeline rather than a single LLM response.

---

# 3. Target Users

## Primary Users

- Researchers
- Engineers
- Developers
- Product teams
- Technical founders
- Analysts
- Students
- Consultants
- Enterprise knowledge teams

## Secondary Users

- Journalists
- Market researchers
- Academic researchers
- Investment research teams
- Legal/technical research teams
- R&D organizations
- Knowledge-management teams

The system must not assume that all users are technical.

---

# 4. Core Product Experience

A user starts with a research objective.

Example:

"Compare the current open-source multimodal models for building an enterprise document intelligence system."

ResearchX converts this into a structured research job.

Example:

Research Objective
    ↓
Research Questions
    ↓
Source Discovery
    ↓
Source Collection
    ↓
Extraction
    ↓
Canonicalization
    ↓
Evidence Graph
    ↓
Comparison
    ↓
Conflict Detection
    ↓
Decision Support
    ↓
Research Report
    ↓
Optional Media / Artifacts

The user should be able to inspect the entire chain.

---

# 5. Research Job

Every research request becomes a ResearchJob.

A ResearchJob should contain:

- job_id
- user_id
- objective
- research questions
- scope
- constraints
- source requirements
- output requirements
- created_at
- updated_at
- status
- configuration
- model configuration
- evaluation configuration

Possible states:

- CREATED
- PLANNING
- DISCOVERY
- EXTRACTION
- CANONICALIZATION
- RETRIEVAL
- COMPARISON
- VALIDATION
- GENERATION
- EVALUATION
- COMPLETED
- FAILED
- CANCELLED

---

# 6. Research Questions

The system should decompose broad objectives into smaller questions.

Example:

Objective:

"Evaluate open-source multimodal models for enterprise document intelligence."

Questions:

1. Which models are currently available?
2. Which organizations maintain them?
3. What modalities do they support?
4. What context lengths are supported?
5. What licensing terms apply?
6. What inference requirements exist?
7. What benchmarks are available?
8. What limitations are documented?
9. What evidence supports each capability?
10. Which claims conflict across sources?

Each question becomes independently researchable.

---

# 7. Source Types

ResearchX should support multiple source classes.

## Web

- Official websites
- Documentation
- Blogs
- News
- Technical articles
- Product pages
- GitHub
- Forums where appropriate

## Academic

- Papers
- Preprints
- Conference publications
- Technical reports

## Documents

- PDF
- DOCX
- TXT
- Markdown
- CSV
- JSON
- XLSX where applicable

## Multimedia

- Images
- Audio
- Video

## Structured Data

- APIs
- Databases
- Public datasets
- User-provided datasets

---

# 8. Source Trust Model

ResearchX should preserve source provenance.

Every extracted claim should maintain:

- Source ID
- Source URL or file reference
- Source type
- Publication timestamp where available
- Retrieval timestamp
- Author/publisher where available
- Evidence location
- Extraction method
- Model used for extraction
- Confidence
- Verification status

The system should distinguish:

- Primary source
- Secondary source
- User-provided source
- Derived source
- Model-generated information

The system must never present generated information as source-derived information.

---

# 9. Evidence-First Architecture

Every important generated claim should be connected to evidence.

Conceptually:

Claim
  ↓
Evidence
  ↓
Source
  ↓
Location

Example:

Claim:
"Model X supports 128K context."

Evidence:
Documentation statement

Source:
Official model documentation

Location:
Specific section/page/URL

The frontend should allow the user to inspect this chain.

---

# 10. Evidence Graph

ResearchX should maintain a graph containing:

- Entities
- Claims
- Sources
- Evidence
- Research questions
- Comparisons
- Contradictions
- Decisions
- Generated artifacts

Example:

Entity
  ├── Claim
  │     ├── Evidence
  │     │     └── Source
  │     └── Confidence
  │
  └── Claim
        ├── Evidence
        └── Source

Contradictions should be represented explicitly rather than silently resolved.

---

# 11. Canonicalization

Different sources may refer to the same object using different terminology.

ResearchX should normalize:

- Entity names
- Organization names
- Model names
- Product names
- Technical concepts
- Dates
- Units
- Versions
- Identifiers
- Terminology

Example:

"GPT OSS"
"gpt-oss"
"GPT-OSS model"

may refer to the same canonical entity.

Canonicalization must preserve the original source terminology.

---

# 12. Retrieval

ResearchX should use retrieval to locate relevant evidence.

Possible retrieval mechanisms:

- Keyword retrieval
- Semantic retrieval
- Hybrid retrieval
- Vector search
- Metadata filtering
- Entity-aware retrieval
- Graph traversal
- Temporal filtering

Embeddings may use models available through the platform such as:

- BAAI BGE models
- Qwen embeddings
- E5-Mistral

Embedding model selection should remain configurable.

---

# 13. Decision Model

ResearchX may use Laya as a decision/orchestration model.

Repository:

https://github.com/NandhaKishorM/laya

Laya should not automatically become the source of truth.

Its role is to assist with:

- Research planning
- Decision decomposition
- Evidence selection
- Tool routing
- Comparison planning
- Next-step selection
- Research-state transitions

Final factual claims must remain grounded in retrieved evidence.

---

# 14. LLM Layer

The architecture must support multiple open models.

Primary model candidates from the Nebius Token Factory environment include NVIDIA models such as:

- Llama-3.1-Nemotron-Ultra-253B-v1
- Nemotron-3-Nano-30B-A3B
- Nemotron-3-Nano-Omni
- Nemotron-3-Super-120b-a12b
- Nemotron-3-Ultra-550b-a55b
- Nemotron-3.5-Lightning
- Nemotron-Nano-V2-12b

Other available open models may be used for specialized tasks.

Model selection must be task-dependent.

Examples:

Planning:
reasoning-oriented model

Extraction:
smaller structured-output model

Embedding:
embedding model

Vision:
multimodal model

Generation:
appropriate text/multimodal model

The architecture must avoid hard-coding a single model.

---

# 15. NVIDIA Model Requirement

The hackathon implementation should meaningfully use NVIDIA open models.

NVIDIA models should be integrated into an actual product workflow rather than included only for demonstration.

Candidate uses:

- Research planning
- Evidence reasoning
- Structured extraction
- Multimodal document understanding
- Decision support
- Research synthesis

The exact model should be selected based on benchmarked performance, latency, cost, and available API capabilities.

---

# 16. Multimodal Research

ResearchX should support multimodal inputs where practical.

## Image

Images may be processed for:

- Charts
- Diagrams
- Screenshots
- Tables
- Technical figures
- Product images
- Visual evidence

Vision models may be used for extraction and interpretation.

Candidate models include:

- NVIDIA Cosmos3-Super-Reasoner
- NVIDIA Nemotron-Nano-V2-12b
- Other compatible vision models available through the platform

---

# 17. Audio

Audio generation and processing should be treated as an optional media layer.

Potential capabilities:

- Research report narration
- Podcast-style summaries
- Audio briefings
- Source transcription
- Meeting/research audio extraction

Audio generation must only be enabled when an appropriate model/service is available.

The core research pipeline must not depend on audio generation.

---

# 18. Video

Video generation should be treated as an optional artifact-generation layer.

Potential outputs:

- Research explainers
- Short research summaries
- Animated presentations
- Visual timelines
- Product/model comparisons
- Executive briefing videos

Video generation should consume structured research output rather than directly inventing content.

Pipeline:

Research
  ↓
Validated facts
  ↓
Narrative/script
  ↓
Storyboard
  ↓
Media generation
  ↓
Video composition
  ↓
Artifact validation

Generated video must retain links to the underlying research.

---

# 19. Media Grounding

Generated image/audio/video content must be grounded in the validated research state.

The media service must receive structured inputs such as:

- Approved claims
- Citations
- Narrative
- Visual instructions
- Safety constraints

The media generator should not independently perform factual research unless explicitly configured to do so.

---

# 20. Comparison Engine

ResearchX should support structured comparisons.

Comparison dimensions may include:

- Capability
- Performance
- Cost
- Latency
- Context length
- Model size
- License
- Hardware requirements
- Availability
- Benchmark results
- Security considerations
- Integration complexity
- Documentation quality

Users may define custom dimensions.

The system must not hide missing data.

Missing information should be represented explicitly.

---

# 21. Contradiction Detection

The system should detect conflicts such as:

Source A:
Context length = 128K

Source B:
Context length = 256K

ResearchX should represent:

CONFLICT DETECTED

Possible explanations:

- Different model versions
- Updated documentation
- Different deployment configuration
- Different measurement
- Source error

The system should not silently choose a value.

---

# 22. Uncertainty

ResearchX must represent uncertainty explicitly.

Possible states:

- VERIFIED
- HIGH_CONFIDENCE
- MEDIUM_CONFIDENCE
- LOW_CONFIDENCE
- CONFLICTED
- UNVERIFIED
- OUTDATED
- MISSING

Confidence should be evidence-derived where possible.

LLM confidence alone must not be treated as factual verification.

---

# 23. Guardrails

ResearchX must have enterprise-grade guardrails.

## Core Guardrails

- No unsupported factual claims presented as facts.
- No fabricated citations.
- No fabricated sources.
- No fabricated benchmark numbers.
- No fabricated URLs.
- No silent source substitution.
- No silent contradiction resolution.
- No hidden data transformations.
- No leakage of secrets.
- No exposure of API keys.
- No unrestricted tool execution.
- No unrestricted code execution.
- No destructive operations without explicit authorization.
- No untrusted content treated as system instructions.

---

# 24. Prompt Injection Defense

External documents and web pages must be treated as untrusted data.

Instructions found inside:

- Web pages
- PDFs
- GitHub repositories
- Emails
- Documents
- Search results
- User-provided datasets

must not override ResearchX system instructions.

The extraction layer must distinguish:

DATA

from

INSTRUCTIONS.

---

# 25. Tool Security

Tools should follow least-privilege principles.

Every tool invocation should be:

- Explicitly scoped
- Logged
- Traceable
- Validated
- Reversible where possible

Sensitive tools require additional checks.

---

# 26. Secrets

Secrets must never be:

- Hard-coded
- Committed to Git
- Included in generated documentation
- Returned in API responses
- Logged
- Included in model prompts unnecessarily

Environment variables or a secure secret-management mechanism must be used.

Examples:

NEBIUS_API_KEY

Other provider keys

Database credentials

Search API credentials

---

# 27. User Data

User-provided research data must be treated as private by default.

The system should support:

- Dataset isolation
- Job isolation
- Access control
- Audit logs
- Deletion workflows
- Retention policies

---

# 28. Fine-Tuning / Post-Training

Fine-tuning is not the default mechanism for adding factual knowledge.

ResearchX should first use:

- Retrieval
- Structured extraction
- RAG
- Prompting
- Tool use

Fine-tuning should be used when the objective is to improve:

- Behavior
- Output format
- Domain style
- Classification
- Extraction consistency
- Instruction following
- Decision policies

Potential approaches:

## Full Fine-Tuning

Use only when justified by infrastructure and dataset scale.

## SFT

Supervised Fine-Tuning using high-quality input/output examples.

Potential use:

Research question → structured research plan

Document → structured extraction

Evidence set → validated synthesis

## IFT

Instruction Fine-Tuning for reliable task following.

## PEFT

Parameter-efficient fine-tuning.

Potential approaches:

- LoRA
- QLoRA
- DoRA
- Prompt tuning

## Preference Alignment

Potential approaches:

- DPO
- RLHF where infrastructure permits

Use preference alignment for:

- Response quality
- Evidence selection
- Research style
- Citation behavior
- Output consistency

## Domain Adaptation

Use domain-specific corpora to improve terminology and representation where appropriate.

## Unsupervised / Continued Pretraining

Only use when there is sufficient high-quality domain corpus and a demonstrated need.

---

# 29. Fine-Tuning Frameworks

The training layer should remain framework-agnostic.

Potential tooling:

- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- Unsloth
- LLaMA-Factory
- Axolotl

Framework selection should depend on:

- Model compatibility
- Dataset format
- Hardware
- Training objective
- Reproducibility
- Deployment requirements

---

# 30. Dataset Philosophy

ResearchX datasets should be treated as first-class product assets.

Every dataset must have:

- Dataset ID
- Version
- Source
- License
- Schema
- Collection timestamp
- Processing history
- Quality checks
- Deduplication status
- Split information
- Evaluation status

Datasets should be versioned.

---

# 31. Training Data Quality

Training examples should be:

- Traceable
- Deduplicated
- Correctly formatted
- Validated
- Representative
- Free of unnecessary secrets
- Free of unsupported claims
- Properly licensed

Synthetic examples must be labeled as synthetic.

Generated training data must not automatically become ground truth.

---

# 32. Evaluation

ResearchX must evaluate both models and the complete system.

Evaluation dimensions:

- Retrieval accuracy
- Evidence recall
- Citation correctness
- Citation completeness
- Claim correctness
- Extraction accuracy
- Entity resolution accuracy
- Contradiction detection
- Answer faithfulness
- Hallucination rate
- Latency
- Cost
- Robustness
- Security
- Tool reliability

---

# 33. Evaluation Before Optimization

No model should be fine-tuned merely because fine-tuning is available.

Baseline:

Base Model + Prompt + Retrieval

must be compared against:

Fine-Tuned Model + Retrieval

and other relevant variants.

Every optimization must have measurable evidence.

---

# 34. Research Artifacts

ResearchX should produce multiple outputs.

## Primary

- Research report
- Evidence-backed answer
- Comparison table
- Source list

## Advanced

- Evidence graph
- Decision matrix
- Timeline
- Dataset
- Structured JSON
- Markdown report

## Media

- Infographic
- Image
- Audio briefing
- Video briefing
- Presentation/storyboard

---

# 35. Enterprise Features

Future enterprise capabilities:

- Organizations
- Teams
- Projects
- Roles
- Permissions
- Audit logs
- Dataset governance
- Model governance
- Source governance
- Usage limits
- Cost tracking
- Evaluation dashboards
- Model registry
- Prompt registry
- Dataset registry
- Research history
- Reproducibility

---

# 36. Observability

Every ResearchJob should generate trace information.

Track:

- Models used
- Prompts/version IDs
- Tools invoked
- Sources retrieved
- Documents processed
- Tokens consumed
- Latency
- Errors
- Evaluation results
- Generated artifacts

Observability integrations may be added through the available development tooling.

---

# 37. Developer Workflow

Claude Code is the implementation worker.

Claude Code must:

1. Read the global CLAUDE.md.
2. Read this project-specific CLAUDE.md.
3. Read relevant documentation before implementing.
4. Follow architecture decisions.
5. Use available skills and MCPs whenever applicable.
6. Avoid inventing undocumented APIs.
7. Verify external APIs before integration.
8. Run tests after meaningful changes.
9. Update documentation when architecture changes.
10. Record important architectural decisions.

Available tooling/skills may include:

- Figma MCP
- Supabase MCP
- Vercel MCP
- Playwright
- Context7
- Chrome DevTools MCP
- Sentry MCP
- n8n
- Codebase Memory MCP
- Graph/codebase tools
- Planning tools
- UI/design skills
- GSAP skills
- Security audit skills
- Prompt engineering skills
- Research/data skills
- Memory tools
- Other installed Claude Code skills

Claude must select tools based on applicability rather than using every tool indiscriminately.

---

# 38. Product UI

The UI should communicate that ResearchX is a serious research system.

Core screens:

1. Dashboard
2. New Research
3. Research Workspace
4. Research Plan
5. Sources
6. Evidence
7. Comparison
8. Conflicts
9. Dataset
10. Evaluation
11. Generated Artifacts
12. Settings

---

# 39. Research Workspace

The primary workspace should show:

Research objective

Research progress

Research questions

Sources

Evidence

Claims

Conflicts

Comparison

Final synthesis

Artifacts

The user should be able to move from:

Answer → Claim → Evidence → Source

without losing context.

Implemented (2026-10-06, `apps/web`): a chat workspace with private conversations (new, list, search, rename,
delete), sign-in with optional TOTP two-factor authentication, private documents, and dictation. Each answer shows:

- a verdict chip from the evidence status: Supported by evidence, Partially supported, Conflicting evidence,
  Depends on the experiment, Insufficient evidence, No relevant source found; plus the configuration it was matched
  against (model on dataset)
- the answer with citation pills that jump to their source, then "Evidence spans": each cited value with its exact
  quote and whether it was verified in the cited source
- source cards: type (paper passage, table, caption, footnote, abstract, dataset record, web, your document), role
  (Supports the answer, or Related · not used as support, with the reason), configuration match, title and link,
  year, section and locator, the values it supports, the table cell for table values, the excerpt with the evidence
  highlighted, and the verification status
- conflicts with every value and its source, and the corpus's own contradiction note when one is recorded
- the live research trace while the run executes (real stage events), and the full trace and its JSON afterwards

---

# 40. API-First Design

The backend must expose clean APIs.

Major domains:

- Research
- Sources
- Extraction
- Canonicalization
- Retrieval
- Evidence
- Comparison
- Laya
- Media
- Datasets
- Evaluation
- Models
- Jobs
- Authentication

APIs should be versioned.

Example:

/api/v1/research

/api/v1/sources

/api/v1/evidence

/api/v1/datasets

/api/v1/evaluations

---

# 41. Async Architecture

Long-running research should not depend on a single synchronous request.

Use asynchronous jobs for:

- Large source collection
- PDF processing
- Video processing
- Dataset processing
- Model inference
- Fine-tuning
- Evaluation
- Media generation

Example:

POST /research
        ↓
job_id
        ↓
worker
        ↓
pipeline
        ↓
status/events
        ↓
result

---

# 42. Reproducibility

A completed research job should be reproducible as far as practical.

Store:

- Research configuration
- Source versions
- Retrieval configuration
- Model identifiers
- Prompt versions
- Dataset versions
- Tool versions
- Processing steps
- Timestamps
- Evaluation results

---

# 43. MVP

The first production-quality MVP should focus on:

1. Research objective input.
2. Research planning.
3. Web/document source ingestion.
4. Structured extraction.
5. Canonicalization.
6. Evidence storage.
7. Retrieval.
8. Comparison.
9. Citation-backed synthesis.
10. NVIDIA open-model integration.
11. Laya decision/orchestration integration.
12. Basic evaluation.
13. Secure API layer.
14. Research workspace UI.

Media generation should be modular and added after the evidence pipeline is reliable.

---

# 44. Expansion

After the core research engine is stable:

- Multimodal research
- Audio transcription
- Audio briefings
- Image generation
- Video generation
- Automated presentations
- Advanced datasets
- Fine-tuning
- Preference optimization
- Enterprise governance
- Team collaboration
- Continuous research agents
- Scheduled research
- Research monitoring
- Domain-specific agents

---

# 45. Product Success Criteria

ResearchX should demonstrate:

- Traceable research.
- Evidence-backed claims.
- Reproducible workflows.
- Strong source provenance.
- Reliable structured extraction.
- Useful comparisons.
- Explicit uncertainty.
- Contradiction visibility.
- NVIDIA open-model usage.
- Modular model architecture.
- Secure tool execution.
- Evaluation-driven development.
- Extensibility toward enterprise deployment.

The system should demonstrate that AI can function as a research infrastructure layer rather than merely as a conversational interface.