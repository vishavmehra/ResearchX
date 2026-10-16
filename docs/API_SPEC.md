# API_SPEC.md

# ResearchX — API Specification

## 1. Purpose

ResearchX exposes a unified API for evidence-grounded research, retrieval, extraction, canonicalization, comparison, decision support, and optional multimodal artifact generation.

The API separates:

- Retrieval
- Extraction
- Canonicalization
- Evidence
- Comparison
- Decision modeling
- Text generation
- Image generation
- Audio generation
- Video generation
- Dataset operations
- Evaluation
- Fine-tuning/post-training

Generated content must never automatically become factual evidence.

---

## 2. API Principles

1. Evidence first.
2. Every factual claim should be traceable to its source.
3. Sources retain provenance.
4. Conflicting evidence must not be silently merged.
5. Model reasoning must be distinguishable from retrieved evidence.
6. Generated media must be explicitly marked as generated.
7. Prefer structured outputs over unrestricted free-form output.
8. APIs are versioned.
9. Long-running operations use asynchronous jobs.
10. Model calls are observable and reproducible where possible.
11. Secrets must never appear in API responses, logs, datasets, prompts, or source-controlled files.
12. External content is untrusted input.
13. Prompt injection from retrieved content must never become a system instruction.
14. The system must not fabricate citations, sources, datasets, capabilities, or evaluation results.

---

## 3. API Versioning

Base path:

/api/v1

Breaking API changes require a new API version.

---

## 4. Core Entities

### Research Request

```json
{
  "query": "string",
  "context": "string",
  "constraints": [],
  "requested_outputs": [],
  "media": {},
  "options": {}
}
Research Run
{
  "run_id": "string",
  "status": "queued|running|completed|failed",
  "query": "string",
  "created_at": "datetime",
  "completed_at": "datetime|null",
  "sources": [],
  "claims": [],
  "evidence": [],
  "comparisons": [],
  "decision": {},
  "artifacts": []
}

5. Research API
POST /api/v1/research
Starts a complete research workflow.
Request:
{
  "query": "Compare the available approaches for X",
  "context": "",
  "constraints": [
    "Use primary sources where possible",
    "Identify uncertainty"
  ],
  "requested_outputs": [
    "summary",
    "evidence",
    "comparison"
  ]
}

Response:
{
  "run_id": "research_123",
  "status": "queued"
}

GET /api/v1/research/{run_id}
Returns the current research state.
Response:
{
  "run_id": "research_123",
  "status": "completed",
  "query": "...",
  "sources": [],
  "claims": [],
  "evidence": [],
  "comparisons": [],
  "decision": {},
  "artifacts": []
}

6. Retrieval API
POST /api/v1/retrieval/search
Searches configured information sources.
Request:
{
  "query": "string",
  "limit": 10,
  "filters": {},
  "source_types": [
    "web",
    "documentation",
    "paper",
    "repository"
  ]
}

Response:
{
  "results": [
    {
      "source_id": "source_001",
      "title": "string",
      "url": "string",
      "snippet": "string",
      "source_type": "documentation",
      "retrieved_at": "datetime"
    }
  ]
}

Search results are discovery objects and must not automatically be treated as verified facts.
7. Source API
GET /api/v1/sources/{source_id}
Returns source metadata and content where permitted.
{
  "source_id": "source_001",
  "title": "string",
  "url": "string",
  "source_type": "string",
  "publisher": "string",
  "retrieved_at": "datetime",
  "content_hash": "string",
  "content": "string"
}

8. Extraction API
POST /api/v1/extraction
Extracts structured claims from source material.
Request:
{
  "source_id": "source_001",
  "content": "string",
  "schema": {
    "claims": []
  }
}

Response:
{
  "claims": [
    {
      "claim_id": "claim_001",
      "text": "string",
      "source_id": "source_001",
      "location": {
        "page": null,
        "paragraph": null,
        "character_start": null,
        "character_end": null
      }
    }
  ]
}

The extraction service must not invent unsupported claims.
9. Canonicalization API
POST /api/v1/canonicalization
Maps equivalent entities, terminology, concepts, and claims into canonical representations.
Request:
{
  "items": [],
  "domain": "string"
}

Response:
{
  "canonical_entities": [],
  "aliases": [],
  "relationships": []
}

Original representations and provenance must be preserved.
10. Evidence API
POST /api/v1/evidence
Builds an evidence graph around extracted claims.
Request:
{
  "claims": [
    "claim_001"
  ],
  "sources": [
    "source_001"
  ]
}

Response:
{
  "evidence": [
    {
      "evidence_id": "evidence_001",
      "claim_id": "claim_001",
      "source_id": "source_001",
      "support": "supports|contradicts|contextual",
      "confidence": null,
      "rationale": "string"
    }
  ]
}

Confidence must not be interpreted as a calibrated probability unless the evaluation system establishes that calibration.
11. Comparison API
POST /api/v1/comparison
Produces a structured comparison using canonical entities and evidence.
Request:
{
  "entities": [],
  "criteria": [],
  "evidence": [],
  "constraints": []
}

Response:
{
  "comparison_id": "comparison_001",
  "criteria": [],
  "entities": [],
  "evidence_map": [],
  "tradeoffs": [],
  "uncertainties": []
}

Missing evidence must not automatically be treated as negative evidence.
12. Laya Decision API
Laya is the decision-model layer.
Repository:
https://github.com/NandhaKishorM/laya
POST /api/v1/laya/decide
Request:
{
  "question": "string",
  "options": [],
  "criteria": [],
  "constraints": [],
  "evidence": [],
  "preferences": {}
}

Response:
{
  "decision_id": "decision_001",
  "options": [],
  "criteria": [],
  "reasoning_trace": [],
  "evidence": [],
  "uncertainties": [],
  "decision": {}
}

Laya must operate on structured evidence and explicit criteria.
The decision layer must expose:
- Criteria
- Evidence used
- Assumptions
- Uncertainties
- Tradeoffs
- Inputs
- Decision representation
Unsupported assumptions must not be hidden.
13. Model Gateway
All external model calls should pass through a unified model gateway.
POST /api/v1/models/generate
Request:
{
  "model": "model_id",
  "task": "text|vision|embedding|audio|video",
  "messages": [],
  "parameters": {}
}

The gateway is responsible for:
- Authentication
- Model selection
- Request validation
- Rate limiting
- Retries
- Timeout handling
- Structured-output validation
- Safety checks
- Observability
- Token tracking
- Cost tracking
14. Nebius Token Factory
Nebius Token Factory is the primary inference platform for supported open models.
Base endpoint:
https://api.tokenfactory.nebius.com/v1/
Environment variable:
NEBIUS_API_KEY
OpenAI-compatible usage:
from openai import OpenAIimport osclient = OpenAI(    base_url="https://api.tokenfactory.nebius.com/v1/",    api_key=os.environ["NEBIUS_API_KEY"])response = client.chat.completions.create(    model="MODEL_ID",    messages=[        {            "role": "system",            "content": "SYSTEM_PROMPT"        }    ])print(response.to_json())


Exact model identifiers must be obtained from the active Nebius catalog and stored in configuration.
Do not hard-code model identifiers throughout the application.
15. NVIDIA Open Model Requirement
The project must use an NVIDIA open model where required by the hackathon rules.
Available NVIDIA-family candidates from the current Nebius model catalog include:
- Cosmos3-Super-Reasoner
- Llama-3_1-Nemotron-Ultra-253B-v1
- Nemotron-3-Nano-30B-A3B
- Nemotron-3-Nano-Omni
- Nemotron-3-Super-120b-a12b
- Nemotron-3-Ultra-550b-a55b
- Nemotron-3.5-Lightning
- Nemotron-Nano-V2-12b
- NVIDIA-Nemotron-3-Nano-30B-A3B
The final model must be selected after validating:
- Hackathon eligibility
- Model availability
- Task suitability
- Cost
- Latency
- Context length
- Evaluation performance
The NVIDIA model should have a meaningful role in the architecture rather than being included only nominally.
Verified 2026-10-02 (live catalog for the configured key): only nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B,
nvidia/nemotron-3-super-120b-a12b, nvidia/Nemotron-3-Ultra-550b-a55b and nvidia/Nemotron-3_5-Lightning are present;
all four support chat, json_schema structured output and tool calling, and none accepts image input
(docs/CAPABILITY_ARCHITECTURE.md §2.1). The other names above are unverified.
16. Text Model Abstraction
ResearchX must support interchangeable open models.
Potential models from the current catalog include:
- NVIDIA Nemotron
- Qwen
- Llama
- DeepSeek
- GLM
- Gemma
- Kimi
- MiniMax
- Hermes
- GPT-OSS
Model selection must be configuration-driven.
Application code must not be tightly coupled to a single model.
17. Embedding API
POST /api/v1/embeddings
Request:
{
  "model": "embedding_model",
  "texts": [
    "string"
  ]
}

Response:
{
  "embeddings": [
    {
      "text_index": 0,
      "vector": []
    }
  ]
}

Potential embedding models:
- bge-en-icl
- bge-multilingual-gemma2
- e5-mistral-7b-instruct
- Qwen3-Embedding-8B
Final embedding model selection must be benchmarked against the retrieval dataset.
Verified 2026-10-02: only Qwen/Qwen3-Embedding-8B is in the live catalog (4,096 dimensions by default; the
"dimensions" request parameter is honoured). pgvector indexes accept at most 2,000 dimensions (vector) or 4,000
(halfvec), so requests set "dimensions" accordingly.
18. Vision / Image API
POST /api/v1/media/image
Used for optional generated visuals, diagrams, illustrations, research graphics, or transformations when a supported provider/model is available.
Request:
{
  "prompt": "string",
  "reference_images": [],
  "style": "string",
  "metadata": {}
}

Response:
{
  "artifact_id": "image_001",
  "type": "image",
  "status": "completed|queued|failed",
  "uri": "string",
  "generated": true
}

Generated images must never be treated as factual evidence.
19. Audio API
POST /api/v1/media/audio
Used for:
- Text-to-speech
- Narration
- Transcription
- Audio generation
- Optional research narration
Request:
{
  "task": "tts|transcription|audio_generation",
  "input": "string",
  "model": "string",
  "parameters": {}
}

Response:
{
  "artifact_id": "audio_001",
  "type": "audio",
  "status": "completed|queued|failed",
  "uri": "string",
  "generated": true
}

Audio generation is an optional multimodal output layer.
20. Video API
POST /api/v1/media/video
Used for optional generated research explainers.
Request:
{
  "prompt": "string",
  "storyboard": [],
  "reference_images": [],
  "audio": [],
  "duration": null,
  "parameters": {}
}

Response:
{
  "artifact_id": "video_001",
  "type": "video",
  "status": "queued|processing|completed|failed",
  "uri": "string",
  "generated": true
}

Video generation must be treated as an optional enhancement and must never block the core research pipeline.
21. Multimodal Generation Pipeline
Core research:
Research
→ Retrieval
→ Extraction
→ Canonicalization
→ Evidence
→ Comparison
→ Decision
Optional presentation layer:
Decision
→ Narrative
→ Storyboard
→ Image generation
→ Audio generation
→ Video composition
→ Final artifact
Every generated artifact must retain a reference to its originating research run.
22. Fine-Tuning API
Fine-tuning is an experimental/post-training workflow.
It must not automatically be applied to every research task.
POST /api/v1/training/jobs
Request:
{
  "base_model": "model_id",
  "dataset_id": "dataset_001",
  "method": "sft|ift|lora|qlora|dora|preference",
  "parameters": {}
}

Response:
{
  "job_id": "training_001",
  "status": "queued"
}

23. Fine-Tuning / Post-Training Methods
ResearchX may evaluate:
- Full Fine-Tuning
- Supervised Fine-Tuning (SFT)
- Instruction Fine-Tuning (IFT)
- Domain Adaptation
- LoRA
- QLoRA
- DoRA
- Prompt Tuning
- Preference Optimization
- DPO
- RLHF
- Knowledge Distillation
The system should prefer the least expensive method that provides measurable improvement.
Fine-tuning should only be adopted when evaluation demonstrates improvement over prompting, retrieval, or RAG.
24. Fine-Tuning Tooling
Potential tooling:
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- Unsloth
- LLaMA-Factory
- Axolotl
The final framework must be selected according to:
- Chosen base model
- Dataset format
- Hardware
- GPU memory
- Training cost
- Training stability
- Evaluation requirements
25. Dataset API
POST /api/v1/datasets
Request:
{
  "name": "string",
  "version": "1.0.0",
  "task": "string",
  "schema": {},
  "provenance": {}
}

GET /api/v1/datasets/{dataset_id}
Returns dataset metadata.
POST /api/v1/datasets/{dataset_id}/validate
Validation must check:
- Schema validity
- Missing values
- Duplicate examples
- Invalid examples
- Provenance
- Contamination
- Leakage
- Label consistency
- Train/evaluation overlap
- Sensitive information
- License/usage constraints where applicable
26. Evaluation API
POST /api/v1/evaluations
Request:
{
  "model": "model_id",
  "dataset": "dataset_id",
  "tasks": [
    "retrieval",
    "extraction",
    "citation",
    "comparison",
    "decision"
  ]
}

Response:
{
  "evaluation_id": "eval_001",
  "metrics": {},
  "failures": [],
  "artifacts": []
}

27. Evaluation Categories
ResearchX should evaluate:
- Retrieval quality
- Retrieval recall
- Retrieval precision
- Evidence grounding
- Claim extraction accuracy
- Citation correctness
- Citation completeness
- Entity canonicalization
- Comparison consistency
- Decision consistency
- Hallucination rate
- Unsupported-claim rate
- Tool-use correctness
- Latency
- Token usage
- Cost
- Multimodal generation success rate
- Safety/guardrail violations
28. Health API
GET /api/v1/health
{
  "status": "ok",
  "version": "string"
}

29. Readiness API
GET /api/v1/ready
Checks:
- Database
- Vector store
- Model gateway
- Retrieval provider
- Media providers
- Queue
- Storage
Response:
{
  "ready": true,
  "dependencies": {}
}

30. Authentication
Production endpoints require authentication.
Secrets must be provided through secure environment variables or a secrets manager.
Expected configuration may include:
NEBIUS_API_KEY
DATABASE_URL
VECTOR_DB_URL
STORAGE_URL
MODEL_PROVIDER_KEYS
Never:
- Hard-code API keys
- Commit secrets
- Put secrets in datasets
- Return secrets to clients
- Log secrets
- Expose provider credentials to browser code
31. Rate Limiting
Rate limits must exist for:
- Research requests
- Retrieval
- Model inference
- Embedding generation
- Media generation
- Dataset operations
- Training jobs
Long-running operations must use asynchronous jobs.
32. Observability
Requests should record:
request_id
run_id
model_id
dataset_id
timestamp
latency
token_usage
estimated_cost
status
error_code

Sensitive user content must not be logged unnecessarily.
33. Error Format
All API errors should follow:
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "request_id": "request_123",
    "details": {}
  }
}

Example:
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "Requested model is currently unavailable.",
    "request_id": "request_123",
    "details": {}
  }
}

34. Async Jobs
Long-running operations use job identifiers.
Possible states:
queued
running
processing
completed
failed
cancelled
Endpoint:
GET /api/v1/jobs/{job_id}
35. Provenance Contract
Every externally sourced claim should map to:
claim
→ evidence
→ source
→ source location
→ retrieval timestamp
→ content hash
Every generated artifact should map to:
artifact
→ research_run
→ input/prompt
→ model
→ model version
→ generation timestamp
36. Guardrails
The API layer must enforce:
1. No fabricated sources.
2. No fabricated citations.
3. No unsupported factual claims.
4. No prompt injection from external content becoming system instructions.
5. No secrets in prompts or responses.
6. No unnecessary persistence of sensitive information.
7. Generated media cannot automatically become evidence.
8. Uncertainty must be preserved.
9. Conflicting evidence must remain visible.
10. Training and evaluation data must preserve provenance.
11. External content must be treated as untrusted.
12. User input must not override system safety controls.
13. Tool calls must be scoped to the current task.
14. Destructive operations require explicit authorization.
15. Model-generated code must not receive unrestricted production credentials.
16. Model output must be validated against schemas before downstream use.
17. Critical decisions must be based on structured evidence rather than unsupported free-form reasoning.
18. Any failed validation must stop or downgrade the affected pipeline stage rather than silently continuing.
19. Generated content must be clearly distinguishable from retrieved/source material.
20. The system must never manufacture missing evidence to complete a response.
37. Tool and MCP Integration
Claude Code is the implementation worker.
Before implementation, Claude Code must read:
1. The global CLAUDE.md.
2. researchx/CLAUDE.md.
3. Relevant documents under docs/.
4. Relevant planning files.
5. Relevant existing code.
The project may use the configured skills and MCP tools whenever they materially improve the task.
Available tools/skills include:
- Figma MCP
- Playwright
- Context7
- Chrome DevTools MCP
- Sentry MCP
- Supabase MCP
- Vercel MCP
- Codebase Memory MCP
- n8n
- Claude Code Setup
- Task Observer
- Gstack
- Hallmark
- Impeccable
- Taste
- Emil Kowalski skills
- GSAP skills
- Codegraph
- Haystack
- SuperPowers
- Planning-with-Files
- Claude Mem
- Mem0
- Cognee
- Beads
- ECC
- Headroom
- DeerFlow
- relevant agentic skills
- relevant security skills
- relevant prompt engineering skills
Claude Code must not use a tool simply because it exists.
Tool selection must be based on whether it improves:
- Correctness
- Research
- Architecture
- Implementation
- Testing
- UI/UX
- Security
- Observability
- Deployment
- Documentation
- Codebase understanding
38. MCP Safety Rules
MCP tools must be treated as privileged capabilities.
Claude Code must:
- Inspect tool descriptions before use.
- Use the minimum required permissions.
- Avoid destructive operations unless explicitly required.
- Never expose secrets through MCP calls.
- Never upload private data unnecessarily.
- Never modify production resources during local development unless explicitly authorized.
- Verify destructive commands before execution.
- Prefer read-only operations for investigation.
- Keep changes scoped to the current project.
- Record important architectural/tool decisions in planning/DECISIONS.md.
39. API Contract Rule
The API contract is the boundary between the ResearchX architecture and implementation.
Claude Code may change internal implementation details when technically justified, but must preserve:
- Endpoint semantics
- Input/output contracts
- Provenance
- Evidence traceability
- Safety requirements
- Evaluation requirements
- Model abstraction
- Async job behavior
- Dataset integrity
- Reproducibility requirements
Breaking changes must be recorded in:
planning/DECISIONS.md
and reflected in the relevant documentation before implementation proceeds.

40. Capability API Domains (implementation audit, 2026-10-02)
APIs are organised around capabilities, not providers. Provider-specific code stays behind the interfaces in
docs/CAPABILITY_ARCHITECTURE.md §5. Status: specified; not implemented (apps/api is empty).

| Domain | Purpose | Backing interface |
|---|---|---|
| /api/v1/research | evidence-grounded research runs (§5, §41) | orchestrator |
| /api/v1/search | hybrid candidate search (§42) | RetrievalProvider, RerankerProvider |
| /api/v1/retrieve | fetch documents / passages / evidence units by id | system of record |
| /api/v1/evidence | evidence assertions and verification state (§10) | evidence layer |
| /api/v1/papers | papers, versions, study links (§43) | system of record |
| /api/v1/datasets | dataset versions and validation (§25) | release manifests |
| /api/v1/experiments | experiment records with field-level evidence (§43) | system of record |
| /api/v1/comparisons | structured comparisons (§11) | comparison service |
| /api/v1/citations | claim-to-evidence citation records (§44) | evidence layer |
| /api/v1/web-research | external discovery and extraction (§45) | WebResearchProvider |
| /api/v1/tools | allowlisted tool contracts (§46) | ToolRegistry |
| /api/v1/models | verified model capability registry (§47) | model gateway |
| /api/v1/evaluations | evaluation runs (§26) | EvaluationProvider |
| /api/v1/media | generated artifacts (§18-21) | MediaProvider |
| /api/v1/projects | project scoping (§48) | system of record |

41. Research Answer Contract
GET /api/v1/research/{run_id} returns, in addition to §5:
{
  "answer_status": "ANSWERED|PARTIAL|CONFLICTING|ABSTAINED",
  "answer": "string|null",
  "claims": [
    {
      "claim_id": "string",
      "text": "string",
      "kind": "SOURCE_STATEMENT|INFERENCE|SYNTHESIS",
      "citations": [
        {"evidence_id": "string", "source_id": "string", "location": {}, "quote": "string",
         "support": "SUPPORTS|CONTRADICTS|CONTEXT", "verified": true}
      ]
    }
  ],
  "abstention": {"reason": "MISSING_EVIDENCE|UNRESOLVED_CONFLICT|WEAK_EVIDENCE|LOW_SOURCE_QUALITY|UNVERIFIABLE_SOURCE|EXTRACTION_FAILED|NO_CITATION_SUPPORT|EXTERNAL_RESEARCH_INCONCLUSIVE",
                 "missing_information": [], "conflicting_fields": []},
  "versions": {"dataset": "string", "index": "string", "models": {}, "prompts": {}},
  "usage": {"input_tokens": 0, "output_tokens": 0, "retrieved_tokens": 0, "selected_context_tokens": 0,
            "tool_output_tokens": 0, "web_research_credits": 0, "estimated_cost": null}
}
- "I don't have sufficient evidence" is returned with answer_status ABSTAINED and a reason; it is a valid result,
  not an error.
- A claim of kind SOURCE_STATEMENT needs at least one citation with "verified": true (quote re-found at its
  location). INFERENCE and SYNTHESIS claims list the source claims they depend on.
- Retrieved but uncited content never appears as a citation.

42. Search API
POST /api/v1/search
Request:
{
  "query": "string",
  "mode": "lexical|dense|hybrid",
  "rerank": false,
  "granularity": "document|passage|evidence",
  "filters": {"year_from": null, "year_to": null, "areas": [], "sources": [], "licences": [], "split": null},
  "k": 20
}
Response:
{
  "candidates": [
    {"unit_id": "string", "unit_type": "paper|paragraph|table_cell|caption|footnote|sentence",
     "paper_id": "string", "text": "string", "scores": {"lexical": null, "dense": null, "fused": null, "rerank": null}}
  ],
  "retrieval": {"mode": "hybrid", "fusion": "rrf", "index_version": "string", "embedding_model": "string",
                "dimensions": 0, "reranker": null}
}
Candidates and scores are discovery signals, not evidence.

43. Papers and Experiments API
GET /api/v1/papers/{paper_id} returns the paper, its source records and versions, licence and study links.
GET /api/v1/experiments?paper_id=&dataset=&model=&metric= returns experiment records with field-level status
(KNOWN, UNKNOWN, CONFLICTING), evidence ids, depth tier and quality dimensions. UNKNOWN fields are returned as
unknown, never filled in.

44. Citations API
GET /api/v1/citations/{claim_id} returns each citation with the quote, location, source version, content hash,
verification status and the time it was last re-verified.

45. Web Research API
POST /api/v1/web-research
Request:
{"query": "string", "mode": "search|extract|crawl", "urls": [], "depth": "basic|advanced", "max_credits": 5}
Response:
{"sources": [{"url": "string", "title": "string", "source_type": "primary|secondary|unknown",
              "trust_tier": "string", "retrieved_at": "datetime", "content_sha256": "string|null"}],
 "credits_used": 0, "provider": "tavily"}
Web results are untrusted discovery objects. They become evidence only through extraction and verification.
The research endpoint of the provider is not exposed by default because of its credit cost.

46. Tools API
GET /api/v1/tools lists allowlisted tools with input and output schemas, authentication requirements, timeouts,
retry policies and rate limits. There is no endpoint that executes arbitrary commands, SQL or URLs.

47. Models API
GET /api/v1/models returns the capability registry: model id, provider, capabilities with status
(VERIFIED, NOT_SUPPORTED, FAILED, NOT_OBSERVED, UNTESTED), probe date and task metrics. Capabilities that are not
VERIFIED are reported as such and are not used for routing.

48. Projects API
POST /api/v1/projects and GET /api/v1/projects/{project_id} scope sessions, documents, evaluations and caches to a
project. Authorization is enforced per project; semantic-cache entries never cross project or tenant boundaries.

49. Implemented Runtime API (2026-10-06)
Sections 1-48 are the target contract. This section is what `services/api/app.py` serves today (unversioned
`/api/*` paths; the web app reaches them same-origin through a Next.js rewrite).

Authentication: `Authorization: Bearer <Supabase access token>` on every route except `/api/health`. The token is
verified with Supabase Auth; the user id is taken from it, never from the request. Errors use
`{"detail": {"code", "message"}}` or `{"detail": "message"}`: 401 missing or invalid token, 403 `mfa_required`
(verified TOTP factor, session not AAL2), 403 `reauthentication_required` (account deletion needs a sign-in within
10 minutes), 404 for objects that do not exist or belong to another user (no existence oracle), 422 invalid input
(unknown fields are rejected), 429 rate limit.

| Method and path | Purpose |
|---|---|
| GET /api/health | release, pipeline version, BM25 artifact, auth configured, provider capabilities, database size vs quota |
| GET /api/me, PATCH /api/me | profile (`display_name`) and assurance level |
| DELETE /api/account | body `{"confirm": "DELETE MY ACCOUNT"}`; removes storage objects and the auth user (cascade) |
| POST /api/research | body `{"query", "conversation_id"?, "input_modality": "text"|"voice", "stream"?}`; JSON result, or server-sent events when `stream` is true |
| GET /api/research/{research_run_id}/trace | the caller's run: stages (status, latency, detail, error), retrieval traces, stored answer |
| GET /api/conversations?q=&limit=&before= | the caller's chats, newest first; `q` searches titles and messages |
| POST /api/conversations, PATCH /api/conversations/{id}, DELETE /api/conversations/{id} | create, rename, delete (also deletes the chat's runs and their cache entries) |
| GET /api/conversations/{id}/messages?limit=&before= | messages with the stored result of each research run |
| GET /api/documents, POST /api/documents, DELETE /api/documents/{id} | private documents (raw body upload with `X-Filename`; PDF, plain text or Markdown; 2 MB per file, 10 per user) |
| GET /api/metrics/architecture | the caller's run metrics plus corpus-wide counts |

Streaming (`stream: true`): `event: stage` with `{"type": "stage", "stage", "status": "running|completed|skipped|
not_required|failed", "latency_ms", "detail", "error"}` as each stage starts and ends, then exactly one
`event: result` (the result below) or `event: error`. Headers: `Cache-Control: no-cache, no-transform`.

Research result:
{
  "research_run_id": "rx_<ULID>", "conversation_id": "uuid",
  "status": "ANSWERED|PARTIAL|CONFLICTING|ABSTAINED|FAILED", "mode": "pipeline|fallback", "fallback_reason": "string|null",
  "evidence_status": "SUFFICIENT|PARTIAL|CONFLICTING_EVIDENCE|AMBIGUOUS_EXPERIMENT_MATCH|INSUFFICIENT_EVIDENCE|NO_RELEVANT_SOURCE_FOUND",
  "request": {"model": "string|null", "dataset": "string|null", "fields": ["learning rate", "batch size"]},
  "fields": [{"field": "training.learning_rate", "label": "learning rate", "status": "SUPPORTED|UNKNOWN|CONFLICTING|AMBIGUOUS",
              "close_match": false, "values": ["0.02"], "sources": ["S1"]}],
  "answer": "string", "intent": "protocol|result|comparison|contradiction|prior_art|general",
  "claims": [{"text": "Learning rate = 0.02", "field": "training.learning_rate", "value": 0.02, "sources": ["S1"],
              "quote": "initial learning rate as 0.02", "start": 24, "end": 53, "evidence_id": "ev:…",
              "origin": "corpus evidence|dataset record|runtime extraction", "config": "EXACT|STRONG", "verified": true}],
  "sources": [{"sid": "S1", "kind": "internal|record|external|private", "role": "SUPPORTS|RELATED", "match": "EXACT|STRONG|RELATED|WEAK",
               "title": "string", "year": 2019, "section": "string", "unit_type": "paragraph|table|caption|footnote|paper",
               "locator": "S4.SS2.p3", "url": "https://arxiv.org/html/…#S4.SS2.p3", "text": "string",
               "supports": [{"field": "training.learning_rate", "label": "learning rate", "value": "0.02"}],
               "spans": [{"text": "initial learning rate as 0.02", "start": 24, "end": 53, "field": "training.learning_rate",
                          "value": "0.02", "cell": "row 3 (\"RoBERTa\") column 2 (\"javascript\")|null"}],
               "verification": "source-checked span|imported record (value present in record)|validated extraction|null"}],
  "cited_sources": ["S1"], "contradictions": [{"id", "subject_id", "field", "explanation", "value_a", "value_b"}],
  "rejected_facts": 49, "removed": [], "cached": false, "origin_research_run_id": "string|null",
  "rejected_candidates": [{"field": "training.batch_size", "label": "batch size", "value": "1024", "reason": "different_dataset",
                           "context": "SimCLR (repro)", "span": "using a batch size of 1024", "origin": "corpus evidence",
                           "source": {"title": "string", "paper_id": "paper:…", "section": "string", "url": "https://…", "provider": null}}],
  "unused_sources": [{"title": "string|null", "url": "https://…", "provider": "openalex", "doi": "string|null", "arxiv_id": "string|null",
                      "also_found_by": ["core"], "reason": "not_read|not_cited|configuration_related|configuration_weak|no_requested_value|values_not_usable|closer_match_used|instruction_like_text|empty_content|not_https"}],
  "graph": {"version": "runtime-graph-0.1.0", "research_run_id": "rx_<ULID>", "nodes": [], "edges": [], "rejected": [], "stats": {}},
  "total_ms": 9359, "trace_persisted": true, "stages": [], "usage": {}, "routing": {}
}
- `rejected_candidates` (field-level path, at most 60): values the run found but did not use, with the reason
  (`different_dataset`, `configuration_related|weak`, or a validation code: `span_not_in_source`, `value_not_in_span`,
  `no_field_cue`, `value_not_numeric`, `source_not_found`; for text read from outside the corpus also `relative_value`,
  `value_change`, `wrong_metric`, `no_metric_cue`, `value_set`, `other_variant`, `other_column`, `dataset_not_stated`) and
  the source; the answer shows them in a dropdown.
- `unused_sources` (both paths, at most 40, 2026-10-08): external results the run found but did not use: not read
  (beyond `evidence.max_external_sources`), not cited (passage path), about another configuration, no value for the asked
  fields, values that failed validation, a closer match used instead, or rejected by the source gate (an
  instruction-like result's title is never echoed). Duplicates of an accepted work are folded into its `also_found_by`.
- External sources read from a paper's arXiv full text carry `unit_type` `paragraph|table`, `section`, `locator` (the
  LaTeXML id) and a `url` anchored to it (`https://arxiv.org/html/<id>#<locator>`); abstracts carry `unit_type: abstract`.
- Trace stage `external_extraction` (field-level path, after `source_validation`): runtime evidence extraction from the
  accepted external sources, before evidence assessment and the answer. Detail: `extraction` (one row per model call:
  `read` abstracts|abstracts + full text, model, status, latency_ms, validation, facts, valid, usable), `fulltext`
  (arxiv_id, status completed|failed, latency_ms, units_total, units_read, license, or why it was not read),
  `other_version`, and counts.
- Trace stage `external_sources` (2026-10-09, `docs/ARCHITECTURE.md` §61.2): `question` (as asked), `queries` (the
  keyword queries built from the request, most specific first), `deadline_s`, `results`, and per scholarly provider
  `{"status": "completed|failed|cut_off|cache_hit|skipped", "query": "<query that produced the results>", "results", "latency_ms",
  "queries_tried": [{"query", "status": "completed|failed|cut_off|not_sent", "results"|"error"|"reason", "latency_ms"}]}`
  (`queries_tried` only when the provider made more than one attempt). The `terms` key is gone. Trace stage
  `source_validation` adds `selected`: the accepted results `[{"provider", "title", "url"}]` (at most 10).
- Passage path, 2026-10-10: a run whose passages were graded sufficient but did not answer records a second
  `evidence_gap` row (`second_look: true`, `reason`), followed by the external stages, `evidence_selection` and
  `synthesis` again. Stages can therefore repeat in `stages`; their order is execution order.
- Model-output validation reports (2026-10-09, `docs/ARCHITECTURE.md` §60.1): `validation` = `{"status":
  "OK|PARTIAL|EMPTY|INVALID|UNAVAILABLE", "errors": ["facts[0].span: Input should be a valid string", ...], "dropped": 1}`
  in the extraction rows, `evidence_extraction.detail.runtime_extraction`, `synthesis.detail` and `fallback.detail`.
  Gateway call statuses gain `INVALID_SHAPE` (a top-level JSON value that is not an object).
- `graph`: the run's runtime research graph (§50); built after the answer, never changes it.
- `evidence_status`, `request`, `fields` and `rejected_facts` are present for parameter and score questions
  (field-level grounding, `docs/ARCHITECTURE.md` §61); other questions use the passage path and omit them.
- `spans[].start/end` are character offsets into the source text when the evidence comes from the corpus; the
  answer cites only sources with role SUPPORTS.
- A model-only fallback is returned only when retrieval itself fails: `mode: "fallback"`, status FAILED, and the
  answer starts with "Fallback response — ResearchX retrieval pipeline unavailable".

## 50. Knowledge Map and Runtime Research Graph (2026-10-08)

All routes require `Authorization: Bearer <access token>`. The canonical graph is the shared public corpus (same for
every user); a run graph is readable only by the run's owner (RLS + owner filter; another user's run returns the same
404 as a missing one). `docs/ARCHITECTURE.md` §64-65.

Node ids: `paper:…`, `exp:…`, `ent:…` (canonical entity: model, method, dataset/benchmark, metric, topic = task),
`res:<exp>#<outcome index>`, `cfg:<exp>`, `ev:…`; run graphs add `run:<research_run_id>`, `rt:paper:doi:…`,
`rt:paper:arxiv:…`, `rt:web:<hash>`, `rt:exp:…`, `rt:cfg:…`, `rt:res:…`, `rt:ev:…`, `doc:<document id>` (private).

| Route | Purpose |
|---|---|
| `GET /api/graph/stats` | counts of the canonical graph (papers, experiments, results, entities by type, citations, drawn study links, contradictions) |
| `GET /api/graph/search?q=&types=&limit≤50` | entities and papers by name or id (substring, exact first, then by number of experiments) |
| `GET /api/graph/landscape?root=` | a research area at a glance: experiment-record counts by topic, dataset, model, method, metric; papers per year; name-derived model families |
| `GET /api/graph/neighborhood?node=&node_types=&edge_types=&limit≤100&offset=` | one expansion page: neighbours (filtered server-side, ordered conflict → own records → context, varied within a block) and the edges to them, `total`, `has_more` |
| `GET /api/graph/node?id=` | node detail (paper metadata; experiment fields, configuration, results; entity surface forms and co-occurrences; evidence spans; recorded contradictions) and its Ask questions |
| `GET /api/graph/edge?type=&source=&target=` | why the edge exists, its derivation and evidence spans; for experiment pairs the full field comparison |
| `GET /api/graph/diff?a=&b=&run=` | field-by-field comparison of two experiments (comparison engine); `run` is required when one side is a runtime experiment |
| `GET /api/graph/ask?node=&q=&run=` | deterministic answers about a node (no model call); `run:` nodes: `discoveries`, `evidence`, `rejected`; runtime nodes: `evidence`, `closest` |
| `GET /api/graph/run/{research_run_id}` | the run's graph (≤250 nodes), canonical nodes decorated with corpus labels, external nodes with their review state |
| `GET /api/graph/runs?limit≤50` | the caller's runs that have a graph, newest first |
| `GET /api/graph/discoveries?days=1..30` | what the caller's runs found outside the corpus (and in private documents) in the period |
| `POST /api/graph/promotions` `{research_run_id, node_id}` | propose one external source of an own run for review: checks, then `rx.runtime_sources.state` RUNTIME → CANDIDATE (no user linkage); private and canonical nodes are `NOT_ELIGIBLE` |

Node: `{id, type, label, meta, flags: ["conflict"], origin?, status?, provenance?}`; edge: `{id, type, source, target,
strength: EXACT|STRONG|RELATED|POSSIBLE|CONTRADICTORY, label, why, derivation, evidence_ids, origin?, status?,
provenance?}`. Run-graph `origin`: canonical | runtime_external | private_document | user_input | derived; `status`:
validated | unverified | related | promoted | rejected. Errors: 401 no session, 404 unknown node / run, 422 invalid id,
type or limit, 429 promotion rate limit.

## 51. Research Analyses (Features A-H, 2026-10-10 / 14)

All routes require `Authorization: Bearer <access token>` and compute from the shared corpus (`docs/ARCHITECTURE.md`
§66). Every analysis except the selector lists runs as a traced research run of the caller (`research_run_id`, `stages`,
`total_ms`, `trace_persisted` in the response; the run is visible only to its owner). Field values carry a status:
REPORTED (in the experiment record), DERIVED (paper-level statement or full text read at request time, with the validated
quote and its location), CONFLICTING, or UNKNOWN; UNKNOWN is never shown as a value. Analyses that may read a full text
with the extraction model (recipe, paper diff, discrepancy, claim) use the per-user research-run rate limit; the others
use the analysis rate limit.

| Route | Feature | Purpose | Run status |
|---|---|---|---|
| `GET /api/analysis/recipe?experiment=` (or `?paper=` for the experiment list) | C | reproduction checklist of one experiment: every setting with status, evidence and location; settings still unknown are read from the full text at request time; Markdown export | ANSWERED, or PARTIAL when settings are missing or conflicting |
| `GET /api/analysis/paper-diff?papers=a,b[,c,d]` | A | 2-4 corpus papers side by side: overview, shared results, settings, fairness verdict (comparison engine) | ANSWERED |
| `GET /api/analysis/discrepancy?a=&b=` (or `?benchmark=` / one side for candidate pairs) | B | why two experiments report different outcomes: differing and held-equal settings, name components, recorded conflicts | ANSWERED / PARTIAL (undetermined, unexplained) / CONFLICTING |
| `GET /api/analysis/benchmarks` | H | benchmarks with the most results and their metrics (selector, no run) | — |
| `GET /api/analysis/benchmark?benchmark=&metric=[&reference=&family=&year_from=&year_to=]` | H | every reported result with its comparability class against a reference; CSV / JSON export in the UI | ANSWERED, or PARTIAL when nothing ranks |
| `POST /api/analysis/claim` `{claim}` (8-600 chars) | E | supporting, contradicting and inconclusive corpus evidence for a stated claim (comparison, value or setting form); verdict SUPPORTED / CONTRADICTED / MIXED / INSUFFICIENT_EVIDENCE / NO_EVIDENCE | ANSWERED / CONFLICTING (mixed) / ABSTAINED (no assessable evidence) |
| `GET /api/analysis/transfer?model=[&source=&target=]` | F | where a model has been evaluated and what changed between a benchmark and its shifted variants; untested benchmarks of the same task; limitation passages | ANSWERED (observed) / PARTIAL (no direct transfer) / ABSTAINED |
| `GET /api/analysis/plan?experiment=` or `?query=<model> on <dataset>` `[&goal=]` | D | next experiments: reproduce, ablation, missing baseline, robustness, reporting, each with the evidence it rests on | ANSWERED / PARTIAL |
| `POST /api/analysis/review` `{text, runtime_search=false}` (20-20,000 chars) | G | coverage of a research question or draft review: missing benchmarks and model families, thin evidence, conflicts, recency, limitations; `runtime_search` (opt-in) sends only recognised entity names to the literature providers | ANSWERED (gaps found) / PARTIAL |

Errors: 401 no session; 404 unknown experiment, paper or benchmark (or no experiment matches `query`); 422 invalid
input or a claim that cannot be assessed; 429 rate limit. The chat composer's modes (Check a claim, Transferability,
Plan next experiment, Review coverage) call the same routes and render the same views.

## 52. Model Routing and Cache Fields in Research Responses (2026-10-14, pipeline-0.3.6)

`POST /api/research` responses (and the persisted run) carry `routing`: the models resolved for `embedding`,
`reranking` and the preferred synthesis route, `policy` (`routing-0.1.0`), `mode` (`dynamic` in the API), and
`decisions`: one summary per routed model call `{task_type, route, model, reason, fallback_used}`. The trace stages
that call a model (`reranking`, `synthesis`, `evidence_extraction.runtime_extraction`, `external_extraction` steps,
`fallback`) hold the full decision under `routing`: `decision_id, task_type, selected_route, selected_model, provider,
selection_reason, complexity, input_tokens, latency_budget_ms, fallback_route, fallback_model, fallback_strategy,
structured_output_required, verification_required, privacy_scope, mode, routing_policy_version, attempts[], model_used,
fallback_used`; each attempt records `attempt, route, model, event (selected | fallback | fallback skipped), status, error,
latency_ms, input_tokens, output_tokens, outcome`. No cost is reported: the configuration holds no verified provider
pricing (token counts are). Clients cannot choose a model or a route: no request field does.

Cache indicators: `cache_lookup.hit` (per-user answer cache), `pgvector.embedding_cache` (`hit` / `miss`, per-user
in-process query-embedding cache), `reranking.cache` (`hit` / `miss`, per-user in-process reranker grades),
`external_sources.providers.<name>.status = cache_hit` (per-user provider-result cache). Uploading or deleting a private
document (`POST /api/documents`, `DELETE /api/documents/{id}`) empties the caller's answer cache.