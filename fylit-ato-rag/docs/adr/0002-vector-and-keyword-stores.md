# ADR-0002: Vector and keyword store selection

- **Status:** accepted
- **Date:** 2026-08-20 (revised 2026-08-25)

## Context

Hybrid retrieval needs a vector index and a keyword index, both filterable by
financial year / version / active status, both runnable in Docker and local
Kubernetes.

An earlier draft of this ADR proposed Qdrant for vectors plus OpenSearch for
keywords, and Step 0 was built against it. Standing that up made the cost
concrete: two services to run, two schemas to keep in step, two sets of filter
semantics that had to be verified to mean the same thing, and a synchronisation
problem at ingest time — a chunk written to one store and not the other is a
silent retrieval bug with no natural place to catch it.

## Decision

**Postgres with pgvector, serving both halves of retrieval from one table.**

- `embedding vector(N)` with an HNSW index (cosine) for semantic similarity.
- `search_vector tsvector` with a GIN index for exact words and phrases.
- Both are columns on the same `chunks` row, so a chunk is written once, in one
  transaction, and cannot exist in one index but not the other.

Filters (`financial_year`, `version`, `active`) are ordinary indexed columns, so
they mean exactly one thing across both halves — the property that took explicit
cross-store testing under the previous decision.

Two columns are `GENERATED ALWAYS`: `active` from `status`, and `search_vector`
from the chunk text. The derivations are enforced by the database rather than by
convention, so neither can drift from its source.

## Consequences

**Gained:** one container instead of two; one schema; one transaction; filters
that cannot disagree; a store the team already knows how to back up and inspect.
Chunk ids become the natural primary key — Qdrant would have required mapping
our string ids onto UUIDs.

**Given up:** Postgres full-text search is a weaker BM25 than OpenSearch —
no per-field boosting to speak of, coarser ranking control. If evaluation
(`scripts/evaluate.py`) shows the keyword half is the bottleneck, revisit; the
`KeywordIndex` interface in `indexing/` exists to keep that swap cheap.

**Financial year is an array, not a scalar.** Auditing the supplied corpus
(5,588 ATO pages) settled this: 73.9% name no financial year at all — evergreen
guidance such as how CGT works — 17.4% name exactly one, and 8.7% name several
(one rate table names 26). So `financial_year text[]`, empty meaning evergreen,
with a GIN index. The filter predicate lives in `schema.YEAR_FILTER_SQL` rather
than being rewritten per call site, because the obvious version is wrong:
`financial_year = '2023-24'` would discard three quarters of the corpus,
including most of the content that actually answers questions. A chunk matches a
year if it is evergreen **or** tagged with that year.

**Watch:** pgvector builds HNSW indexes only up to 2000 dimensions. That is fine
for `text-embedding-3-small` (1536) but rules out `text-embedding-3-large` (3072)
unless dimensions are reduced at embedding time. `schema.ddl_statements` raises
rather than silently leaving the vector column unindexed.

## Alternatives considered

- **Qdrant + OpenSearch** (previously proposed here): better BM25 and purpose-built
  ANN, at the cost of two services, two schemas and an ingest-time consistency
  problem. Revisit if retrieval quality demands it.
- **Elasticsearch**: near-identical to OpenSearch; licensing pushed us to OpenSearch,
  and this decision moves past both.
- **FAISS + Whoosh (in-process)**: lightest to run, but no service boundary to
  deploy/scale in k8s, and persistence is manual.
