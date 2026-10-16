# Cache benchmark v0.2.0

cache-bench-0.1.0 · pipeline-0.3.6 · temporary user (deleted afterwards) · wall 138.9 s · cache write guard raised to 490.0 MB for this run (configured 450 MB)

| Layer | Hits on repeat | Miss p50 (ms) | Hit p50 (ms) | Correct on hit |
|---|---|---|---|---|
| Answer cache (per user, RLS) | 5/6 (5 cacheable) | 5508 | 475 | 5/5 identical answers |
| Query embedding (in-process, per user) | 6/6 | 259 | 0 | same vector by construction |
| Reranker grades (in-process, per user) | 5/6 | 2324 | 0 | 6/6 same evidence selected |
| External sources (per user, RLS) | yes | 2605 | 1448 | sources differ |

Pipeline total p50 with in-process caches cold / warm (answer cache unwritten): 4683 / 2944 ms. Abstained answers are never cached (by design).
External providers first / second run: {'semantic_scholar': 'failed', 'openalex': 'completed', 'arxiv': 'completed', 'core': 'completed', 'openreview': 'completed'} / {'semantic_scholar': 'completed', 'openalex': 'cache_hit', 'arxiv': 'cache_hit', 'core': 'cache_hit', 'openreview': 'cache_hit'}.
