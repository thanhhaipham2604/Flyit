# Fylit ATO RAG System

RMIT Capstone project: a Retrieval-Augmented Generation (RAG) service that answers
general Australian tax questions from official ATO content, with citations, and
refuses honestly when the evidence isn't there.

**One sentence:** take a tax question, find the most relevant official ATO content,
return a plain-English answer with links to its sources — refusing to answer when
the evidence just is not there.

## Architecture

Two halves, running at different times:

- **Zone A — ingestion pipeline** (`src/fylit_rag/ingestion`, `src/fylit_rag/indexing`):
  reads ATO Markdown files, validates them, cleans and chunks them (keeping headings,
  lists and tables intact), tags chunks with metadata (source title, URL, financial year),
  then embeds and indexes into a vector index and a keyword index. Incremental and
  versioned — a new financial year never forces a full rebuild.
- **Zone B — query-time pipeline** (`src/fylit_rag/retrieval`, `generation`, `guardrails`, `api`):
  the FastAPI Query API validates and rate-limits requests, hybrid retrieval blends
  vector + keyword search, reranking sorts the shortlist, and the LLM layer writes a
  grounded answer — or refuses safely when evidence is thin.

See `docs/architecture.md` for diagrams and `docs/adr/` for decision records.

## Repository layout

```
src/fylit_rag/
  ingestion/    load, validate, clean, chunk ATO Markdown; incremental pipeline
  indexing/     embeddings, vector index, keyword index, version management
  retrieval/    hybrid search + reranking
  generation/   LLM client, prompts, grounding, conversation memory
  guardrails/   input/output guards, prompt-injection resistance
  api/          FastAPI app, routes, schemas, rate limiting
scripts/        ingest CLI, retrieval evaluation
tests/          pytest suites
data/ato_corpus/  drop the supplied ATO .md files here (configurable)
docker/         Dockerfiles
deploy/k8s/     local Kubernetes manifests
docs/           architecture notes + ADRs
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # add your OpenAI key
make ingest                # index the corpus in data/ato_corpus
make api                   # run the API on :8000 (docs at /docs)
```

Or with Docker: `docker compose up --build`.

## Quality targets

- **Retrieval quality** — right chunk near the top of the shortlist; tracked via `scripts/evaluate.py`.
- **Grounding** — every factual claim traces to a retrieved passage.
- **Honest refusals** — no evidence, no answer; refusals are a feature.
- **Responsiveness** — fast answers, holds up under concurrent users.

## Non-negotiable rules

Never give personalised tax advice, guarantee a refund, or assume a deduction applies.
Answers come only from retrieved active-source evidence, always with source links and
a general-information disclaimer.
