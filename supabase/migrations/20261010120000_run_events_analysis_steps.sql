-- 2026-10-10: analysis features (services/analysis, docs/ARCHITECTURE.md §66) record their executed stages in the same
-- research trace as research runs. Widens the stage whitelist by the analysis stage names; existing rows are unaffected.
alter table rx.run_events drop constraint run_events_step_known;
alter table rx.run_events add constraint run_events_step_known check (step in (
  'query_understanding','laya_decision','research_strategy','model_routing','cache_lookup','bm25','pgvector',
  'hybrid_fusion','reranking','evidence_assessment','evidence_gap','tavily','external_sources','source_validation',
  'external_extraction','evidence_extraction','experiment_extraction','contradiction_detection','evidence_selection',
  'synthesis','citation_validation','abstention','fallback','answer','graph_enrichment',
  'analysis_request','experiment_retrieval','field_facts','comparison','comparability','runtime_enrichment','assessment',
  'recommendations','coverage'));
