-- 2026-10-08: the research trace records runtime evidence extraction from external sources (abstracts, then the
-- best-matching paper's arXiv full text) as its own stage, 'external_extraction' (services/runtime/orchestrator.py
-- _external_facts; docs/ARCHITECTURE.md §61). Widens the stage whitelist by one name; existing rows are unaffected.
alter table rx.run_events drop constraint run_events_step_known;
alter table rx.run_events add constraint run_events_step_known check (step in (
  'query_understanding','laya_decision','research_strategy','model_routing','cache_lookup','bm25','pgvector',
  'hybrid_fusion','reranking','evidence_assessment','evidence_gap','tavily','external_sources','source_validation',
  'external_extraction','evidence_extraction','experiment_extraction','contradiction_detection','evidence_selection',
  'synthesis','citation_validation','abstention','fallback','answer','graph_enrichment'));
