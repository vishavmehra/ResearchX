-- Lexical retrieval is Okapi BM25 over a prebuilt artifact (services/retrieval/search/bm25.py; DECISIONS §35).
-- PostgreSQL ts_rank has no IDF, so the generated tsvector column and its GIN index were never queried, yet they held
-- ~107 MB of the Free plan's 500 MB that pgvector embeddings need. Derived data only: re-adding the generated column
-- recomputes it from chunks.text. Run `vacuum full rx.chunks` afterwards to return the heap space.
alter table rx.chunks drop column tsv;
