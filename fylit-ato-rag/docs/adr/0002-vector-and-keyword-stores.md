# ADR-0002: Vector and keyword store selection

- **Status:** proposed
- **Date:** 2026-08-20

## Context

Hybrid retrieval needs a vector index and a keyword index, both filterable by
financial year / version / active status, both runnable in Docker and local
Kubernetes.

## Decision (default, pending team confirmation)

Qdrant for vectors, OpenSearch for keywords. Both are containerised in
docker-compose, have solid Python clients, and support payload filtering.

## Alternatives considered

- pgvector + Postgres FTS: one store for both; simpler ops, weaker BM25 tooling.
- Elasticsearch: near-identical to OpenSearch; licensing pushed us to OpenSearch.
- FAISS + Whoosh (in-process): lightest to run, but no service boundary to
  deploy/scale in k8s, and persistence is manual.

Revisit before the retrieval milestone; interfaces in `indexing/` keep the
swap cheap.
