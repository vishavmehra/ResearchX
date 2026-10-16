# Grounding evaluation v0.2.0

grounding-eval-0.1.0 · grounding-0.3.0 · pipeline-0.3.6 · seed 13 · external search off (corpus only) · 110 runs {'positive': 66, 'negative': 20, 'contradiction': 24} · 0 errors · wall 253.5 s

| Metric | Value | n / of |
|---|---|---|
| request parsed | 1.0 | 66 / 66 |
| false abstention rate | 0.136 | 9 / 66 |
| false abstention rate parsed | 0.136 | 9 / 66 |
| sufficiency accuracy | 0.909 | 100 / 110 |
| parameter accuracy | 0.775 | 31 / 40 |
| parameter accuracy when answered | 1.0 | 31 / 31 |
| parameter gold among values | 0.775 | 31 / 40 |
| score accuracy | 0.615 | 16 / 26 |
| score accuracy when answered | 0.615 | 16 / 26 |
| numerical accuracy | 0.712 | 42 / 59 |
| wrong answer rate | 0.0 | 0 / 57 |
| evidence recall | 0.833 | 55 / 66 |
| gold paper retrieved | 0.909 | 60 / 66 |
| configuration matching accuracy | 1.0 | 60 / 60 |
| gold experiment matched | 1.0 | 60 / 60 |
| citation correctness | 1.0 | 153 / 153 |
| entailment accuracy | 1.0 | 153 / 153 |
| unsupported claim rate | 0.0 | 0 / 153 |
| citation completeness | 1.0 | 80 / 80 |
| negative answered rate | 0.0 | 0 / 20 |
| negative answered as exact | 0.0 | 0 / 20 |
| contradiction accuracy | 0.917 | 22 / 24 |
| contradiction both values reported | 0.917 | 22 / 24 |
| false conflict rate | 0.0 | 0 / 66 |

Status counts: `{"positive": {"SUFFICIENT": 45, "NO_RELEVANT_SOURCE_FOUND": 1, "PARTIAL": 2, "INSUFFICIENT_EVIDENCE": 8, "AMBIGUOUS_EXPERIMENT_MATCH": 10}, "negative": {"NO_RELEVANT_SOURCE_FOUND": 5, "INSUFFICIENT_EVIDENCE": 15}, "contradiction": {"CONFLICTING_EVIDENCE": 22, "PASSAGE_CONFLICTING": 1, "INSUFFICIENT_EVIDENCE": 1}}` · median latency 5904 ms

Excluded from sampling: 162 of 188 result questions whose 'metric' the grounding metric lexicon does not recognise: mostly table-header noise in the silver data (dataset or model names, 'Overall'), plus about a dozen real metrics the lexicon lacks (ACC, EM, CIDEr, Recall@k), which take the passage path.

Positives are answerable by construction (gold evidence located in the corpus); negatives (invented model or dataset) must abstain; recorded contradictions must be reported. Citation correctness: the quoted span is in the cited source text, re-read from the database. Entailment: the span states the field and value (deterministic validator). Completeness: every reported value has a citation. Parameter/score accuracy: a single answered value equal to the gold value; 'gold among values' also credits ambiguous or conflicting answers that list it.
