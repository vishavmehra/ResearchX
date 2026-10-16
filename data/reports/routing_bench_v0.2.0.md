# Routing benchmark v0.2.0

routing-bench-0.1.0 · pipeline-0.3.6 · seed 13 · 144 runs ({'positive': 12, 'contradiction': 4, 'passage': 16, 'gap': 4} x 4 modes) · 0 errors · wall 251.0 s · external search and Laya off · no shared cache

| Metric | fixed_nano | fixed_super | static | dynamic |
|---|---|---|---|---|
| Field answers exactly correct | 0.667 (8/12) | 0.667 (8/12) | 0.667 (8/12) | 0.75 (9/12) |
| Field wrong answers (when answered) | 0.0 (0/10) | 0.0 (0/10) | 0.0 (0/10) | 0.0 (0/11) |
| Field false abstentions | 0.167 (2/12) | 0.167 (2/12) | 0.167 (2/12) | 0.083 (1/12) |
| Invented-entity questions answered | None (0/0) | None (0/0) | None (0/0) | None (0/0) |
| Recorded contradictions flagged | 1.0 (4/4) | 1.0 (4/4) | 1.0 (4/4) | 1.0 (4/4) |
| Field questions beyond corpus evidence answered | 0.25 (1/4) | 0.25 (1/4) | 0.25 (1/4) | 0.25 (1/4) |
| Field citation correctness | 1.0 (of 39) | 1.0 (of 39) | 1.0 (of 39) | 1.0 (of 40) |
| Passage questions answered | 0.812 (13/16) | 0.562 (9/16) | 0.562 (9/16) | 0.75 (12/16) |
| Passage synthesis schema-valid | 0.938 (15/16) | 0.938 (15/16) | 0.938 (15/16) | 0.938 (15/16) |
| Passage claims passing citation validation | 0.746 (of 59) | 0.853 (of 34) | 0.757 (of 37) | 0.884 (of 43) |
| Model outputs parsed and schema-valid | 1.0 (16/16) | 1.0 (16/16) | 1.0 (17/17) | 1.0 (17/17) |
| Runs with a fallback model | 0.0 (0/36) | 0.0 (0/36) | 0.028 (1/36) | 0.056 (2/36) |
| Runs with a failed model stage | 0.0 (0/36) | 0.0 (0/36) | 0.0 (0/36) | 0.0 (0/36) |
| Total latency p50 / p95 (ms) | 7137 / 14338 | 5608 / 9020 | 5851 / 12007 | 5765 / 13004 |
| Answer-model latency p50 / p95 (ms) | 4657 / 11013 | 1200 / 3556 | 3408 / 9436 | 4011 / 7887 |
| LLM tokens in / out (answer models + reranker) | 157861 / 12510 | 157263 / 9552 | 153109 / 10768 | 159472 / 10989 |
| Answer-model calls | 16 | 16 | 17 | 17 |

Models used (answer-model calls): fixed_nano: {'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B': 16}; fixed_super: {'nvidia/nemotron-3-super-120b-a12b': 16}; static: {'nvidia/nemotron-3-super-120b-a12b': 8, 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B': 9}; dynamic: {'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B': 8, 'nvidia/nemotron-3-super-120b-a12b': 9}

Passage questions have no gold answers: their correctness is not measured (no LLM judge). Cost is not estimated: config/models.yaml holds no verified provider pricing; token counts are reported instead.
