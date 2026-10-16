# ResearchX Data

The ResearchX dataset is a versioned data product (DATASET_SPEC.md). It represents research
at the experiment level, not the document level:

    SourceRecord -> Paper / Study -> Experiment (condition) -> OutcomeObservation -> EvidenceAssertion
                                  -> StudyLink · Comparison (field-level) · Contradiction · AbstentionCase
    schema 0.2.0 adds:  ResultRecord <-association-> Experiment · Artifact -> ArtifactEvidence (paper vs repository)
                        CanonicalEntity · GraphNode / GraphEdge · ResearchTask · HardCase · DiscoveryCandidate

Current counts, distributions and quality numbers are always in `reports/statistics.md` and
the release manifest `manifests/releases/v<version>.json`. Both are generated, never hand-written.

Versions: v0.1.0 is the frozen bootstrap release (snapshot archives in `releases/v0.1.0/`, never modified).
v0.2.0 deepens technical content per experiment (design: `docs/DATASET_DEEPENING.md`).

## Layout (DATASET_SPEC §32)

| Path | Contents | In git |
|---|---|---|
| `raw/<source>/` | Cached API payloads with `.meta.json` sidecars (URL without secrets, UTC time, sha256), plus `acquisition.json` | no |
| `normalized/sources/` | `source_records.jsonl` (one record per source record) and `fulltext_index.jsonl` | yes |
| `normalized/papers/`, `normalized/studies/`, `normalized/citations/` | Identifier-merged papers, provisional studies, intra-corpus citations (BRONZE) | yes |
| `fulltext/permitted/<id>/` | CC0 / CC BY / CC BY-SA originals with `parsed.json` and `provenance.json` | no |
| `fulltext/restricted/<id>/` | Everything else, including the default arXiv licence. Internal research use only, never redistributed | no |
| `raw/github/`, `raw/llm_cache/` | Pinned repository files (gzip, per commit) and exact-key LLM responses (replays are free) | no |
| `normalized/discovery/` | Layer 0 discovery candidates from arXiv OAI-PMH (selection version recorded) | yes |
| `silver/` | Model- or rule-assisted records: experiments, evidence (`evidence/` incl. `deep_evidence`, `artifact_field_evidence`), comparisons, study_links, contradictions, abstention; 0.2.0: `results/` (results, associations), `artifacts/`, `entities/`, `graph/`, `tasks/` (research tasks, hard cases) | yes |
| `releases/<version>/` | Snapshot archives of frozen releases (`scripts/release/snapshot_release.py`) | no |
| `gold/` | Human-verified records only, regenerated from the decision log by `promote_gold.py` | yes |
| `annotations/` | `GUIDELINES.md`, review `queue/`, append-only `decisions/`, `promotions/` | yes |
| `eval/`, `restricted/` | Evaluation-only and restricted datasets. Nothing acquired yet | no |
| `manifests/` | `sources.yaml`, `licenses.yaml`, `contamination.yaml`, and generated `versions.yaml` and `datasets.yaml`, plus `releases/` | yes |
| `schemas/` | JSON Schemas exported from `services/datasets/schemas.py` | yes |
| `runs/` | `runs.jsonl` (every stage run: params, outputs with sha256, counts), `failures.jsonl`, `model_calls.jsonl` | yes |
| `reports/` | Validation, leakage, licence and statistics reports | yes |

## Quality tiers (DATASET_SPEC §25)

- **BRONZE.** Harvested metadata in `normalized/`.
- **SILVER.** Machine-assisted records: rule extraction, Papers With Code imports cross-checked against
  paper tables, and LLM extraction when `NEBIUS_API_KEY` is set. Silver is not ground truth.
- **GOLD.** Only records that humans accepted under `review-policy-0.1.0`. Agent or model decisions are
  logged for audit and never promote. Every promotion is written to `annotations/promotions/`.

Evidence status (DATASET_SPEC §11):

| Status | Meaning |
|---|---|
| MACHINE_EXTRACTED | Produced by an extractor and not yet re-checked. |
| SOURCE_CHECKED | The quoted text or structured value was re-found at the stated location of the stored source. This shows the source says it. It does not confirm which experiment the value belongs to. |
| HUMAN_REVIEWED | A human accepted it. |
| REJECTED | The re-check failed. Rejected evidence can never support a KNOWN field. |

Missing information is `UNKNOWN` and never counts as a match.

## Build (reproducible, DATASET_SPEC §55)

```bash
uv sync                                                       # pinned environment (uv.lock)
uv run python scripts/data/run_pipeline.py --profile smoke    # ~20 papers end to end
uv run python scripts/data/run_pipeline.py --profile initial  # ~1.5k candidates -> ~870 relevant papers
uv run python scripts/data/run_pipeline.py --from normalize_metadata   # offline replay from cached raw data
uv run pytest -q
```

Every acquisition stage is cached, checksummed and resumable, and every write is atomic. The OpenAlex
free allowance is small: without a key it is $0.10/day, while search costs $0.001 per call and a filter
costs $0.0001. When anonymous search is paused, discovery falls back to filter-only topic queries.

Optional environment variables hold only names and keys, never values in files:

| Variable | Purpose |
|---|---|
| `NEBIUS_API_KEY` | LLM extraction through the model gateway (`config/models.yaml`); the `extract_deep` stage needs it unless every call is already cached |
| `TAVILY_API_KEY` | Runtime external research only; the dataset pipeline does not use it |
| `OPENALEX_API_KEY` | $1/day OpenAlex allowance |
| `S2_API_KEY` | Higher Semantic Scholar limits |
| `RESEARCHX_CONTACT_EMAIL` | OpenAlex `mailto` |

## Human review

```bash
uv run python scripts/data/build_review_queue.py --batch-id batch-YYYYMMDD-01     # writes queue/<batch>.jsonl + .md
uv run python scripts/data/record_review.py --reviewer <name> --interactive batch-YYYYMMDD-01
uv run python scripts/data/promote_gold.py
```

Review targets: experiment, evidence, comparison, study_link, contradiction, abstention and, since 0.2.0, association
and artifact_evidence. The queue puts disagreeing machine signals first (artifact conflicts, ambiguous or conflicting
associations, cross-check conflicts), then deep-configuration experiments. `promote_gold.py` also writes
`reports/review_agreement.json` (Cohen's kappa between human annotators and median annotation time per target type);
it stays empty until humans review.

## Licensing rules (DATASET_SPEC §8, §31)

- **Registry.** Each source's licence, training, evaluation and redistribution flags live in
  `manifests/sources.yaml`. Unknown means not allowed.
- **Full text.** It is stored as `permitted` only for CC0, CC BY and CC BY-SA. arXiv's default licence
  allows research use but not redistribution.
- **Public release obligations.** `reports/license_report.json` lists them. Evidence quotes from
  restricted sources must be dropped or truncated. Semantic Scholar-derived fields must be dropped or
  re-sourced. CC BY-SA share-alike applies to Papers With Code-derived data.
- **Evaluation-only benchmarks.** Sources such as GAIA and BrowseComp are never placed in `silver/` or
  `gold/`. Validation fails if a corpus record comes from an EVALUATION_ONLY or RESTRICTED source.
