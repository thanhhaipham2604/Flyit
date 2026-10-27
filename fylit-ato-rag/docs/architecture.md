# Architecture

Three views of the same system:

1. [System architecture](#1-system-architecture) - the components, what they own, and
   what talks to what.
2. [Data flow A: ingestion](#2-data-flow-a--ingestion-zone-a-offline) - how an ATO
   Markdown page becomes a retrievable, cited chunk.
3. [Data flow B: answering a question](#3-data-flow-b--answering-a-question-zone-b-request-time) -
   how a question becomes a grounded answer, or an honest refusal.

The two halves run at different times and never run in the same process. Zone A is
offline and writes the index; Zone B serves requests and only reads it.

---

## 1. System architecture

```mermaid
flowchart TB
    subgraph client["Client"]
        browser["Browser page<br/>GET /"]
        api_client["API client<br/>POST /ask"]
    end

    subgraph service["Query service - Zone B, FastAPI"]
        routes["routes.py<br/>HTTP, rate limit 30/min per IP"]
        svc["service.py<br/>pipeline orchestration"]
        guards["guardrails/<br/>input, injection, output"]
        retrieval["retrieval/<br/>hybrid, rerank, query_expansion, temporal"]
        generation["generation/<br/>prompts, grounding, llm, memory"]
    end

    subgraph ingest["Ingestion - Zone A, CLI job"]
        pipeline["ingestion/<br/>loader, cleaner, metadata, chunker"]
        indexing["indexing/<br/>bootstrap, store, run_embeddings"]
    end

    subgraph stores["Stores"]
        pg[("PostgreSQL + pgvector<br/>chunks table<br/>embedding vector(1536) HNSW<br/>search_vector tsvector GIN<br/>financial_year text[] GIN")]
        cache[("SQLite<br/>embedding cache<br/>INDEX_DIR")]
        files[/"data/processed/<br/>enriched_corpus.jsonl<br/>state.json, manifest.json"/]
    end

    subgraph external["External"]
        openai["OpenAI API<br/>text-embedding-3-small<br/>gpt-4o-mini"]
    end

    corpus[/"data/ato_corpus/<br/>5,588 ATO .md files"/]

    browser --> routes
    api_client --> routes
    routes --> svc
    svc --> guards
    svc --> retrieval
    svc --> generation
    retrieval -- "read only" --> pg
    retrieval --> openai
    generation --> openai

    corpus --> pipeline
    pipeline --> files
    pipeline --> indexing
    indexing -- "write" --> pg
    indexing --> openai
    indexing <--> cache

    classDef store fill:#eef4f1,stroke:#176b57,color:#17211b
    classDef ext fill:#f6f0e6,stroke:#8a6d1f,color:#17211b
    class pg,cache,files,corpus store
    class openai ext
```

**Deployment.** Both halves ship as one image (`docker/api.Dockerfile`); the entry point
decides which half runs. Locally, `docker compose` runs the API and Postgres. On
Kubernetes (`deploy/k8s/`) the API is a 2-replica Deployment behind a ClusterIP Service,
and Postgres is a single-replica StatefulSet with a 10 Gi volume. Configuration comes
from a ConfigMap, credentials from a Secret.

Two consequences of running two replicas, both deliberate and both documented in
`api-deployment.yaml`: the rate limiter counts in process memory, so N replicas allow N
times the limit, and conversation memory is per process, so a follow-up that lands on
the other pod has no history. **Ingestion is not yet a cluster workload** - it is run
manually against the same database.

---

## 2. Data flow A - ingestion (Zone A, offline)

Run with `python scripts/ingest.py`. Nothing here is on the request path.

```mermaid
flowchart TD
    md[/"ATO Markdown file"/] --> parse["loader.parse_header_and_body<br/>title, source URL, menu path"]
    parse -->|"header missing"| invalid[/"invalid_files.json"/]
    parse --> clean["cleaner.clean_body<br/>strip nav and QC boilerplate<br/>keep headings, lists, tables<br/>capture Last updated"]
    clean --> ids["loader.stable_id = SHA-1 of URL<br/>content_hash = SHA-256 of cleaned text"]
    ids --> year["metadata.infer_primary_year<br/>precedence: URL, myTax, applicability,<br/>July-June window, title, heading"]
    year --> enriched[/"enriched_corpus.jsonl"/]

    enriched --> diff{"versioning.diff_against_state<br/>vs previous state.json"}
    diff -->|"unchanged"| skip["skip - no chunking,<br/>no embedding, no cost"]
    diff -->|"deleted"| softdel["store.mark_deleted<br/>status = deleted, still citable"]
    diff -->|"new or changed"| chunk["chunker.chunk_document<br/>~1200 chars, never splits a table<br/>or list, one heading path per chunk"]

    chunk --> upsert["store.upsert_chunks<br/>ON CONFLICT (chunk_id) DO UPDATE<br/>embedding = NULL only when<br/>content_hash changed"]
    upsert --> orphans["store.stale_chunk_ids<br/>purge chunks the document<br/>no longer produces"]
    orphans --> commit["COMMIT"]
    softdel --> commit
    commit --> checkpoint["persist state.json<br/>only after the commit succeeds"]
    checkpoint --> embed["run_embeddings<br/>SELECT WHERE embedding IS NULL<br/>batches of 100"]
    embed <--> cache[("SQLite cache<br/>key = model + SHA-256 of text")]
    embed --> openai["OpenAI embeddings"]
    embed --> pg[("chunks.embedding<br/>committed per batch")]

    classDef skipNode fill:#eef4f1,stroke:#176b57,color:#17211b
    classDef store fill:#f6f0e6,stroke:#8a6d1f,color:#17211b
    class skip,softdel skipNode
    class cache,pg,md,enriched,invalid store
```

**Why the checkpoint sits after the commit** (ADR-0004). If `state.json` advanced first
and indexing then failed, the next run would compare against the new state, conclude
nothing had changed, and index nothing - a silent empty index. Writing it only after a
successful commit means a failed run simply repeats.

**What makes a re-run cheap.** The content hash covers *cleaned* text, so a re-scrape
that only changes the scrape timestamp is not a change. An unchanged chunk keeps its
vector, because the upsert clears `embedding` only when that chunk's own hash moved.
Editing one paragraph of a long page therefore re-embeds one chunk, not the page.

---

## 3. Data flow B - answering a question (Zone B, request time)

`POST /ask`, orchestrated by `api/service.py`. Every red path is a refusal, and a
refusal returns HTTP 200 with `diagnostics.refused` - it is a correct outcome, not an
error.

```mermaid
flowchart TD
    q["POST /ask<br/>question, session_id?, financial_year?"] --> valid{"schema valid?<br/>3 to 2000 chars"}
    valid -->|"no"| r422(["422 Unprocessable"])
    valid --> limit{"rate limit<br/>30/min per IP"}
    limit -->|"exceeded"| r429(["429 + Retry-After"])
    limit --> guard{"input_guards.check_input<br/>unlawful, injection,<br/>personal-figure requests"}
    guard -->|"blocked"| refuse1(["refusal<br/>guardrail = input_guard"])

    guard --> mem["memory.contextualise<br/>rewrite a follow-up to stand alone"]
    mem --> expand["query_expansion.expand_query<br/>taxpayer wording to ATO vocabulary"]
    expand --> both{"known ambiguous term?"}
    both -->|"yes"| one["retrieve with clarified query only"]
    both -->|"no"| two["retrieve twice:<br/>original + expanded"]
    two --> merge["interleave and dedupe<br/>keeps vector/keyword provenance"]
    one --> temporal
    merge --> temporal["temporal.prefer_current<br/>drop guidance older than 3 years<br/>unless the question names a year"]

    subgraph hybrid["hybrid.retrieve - per query, shortlist 30"]
        vec["vector search<br/>cosine, HNSW"]
        kw["keyword search<br/>plainto_tsquery, ts_rank_cd"]
        rrf["reciprocal rank fusion<br/>k = 60"]
        vec --> rrf
        kw --> rrf
    end

    temporal --> rank["rerank<br/>fusion (default) or mmr<br/>top 15 kept as evidence"]
    rank --> gate{"grounding.has_sufficient_evidence<br/>top cosine >= 0.45"}
    gate -->|"no"| refuse2(["refusal<br/>insufficient_evidence<br/>no model call, no cost"])

    gate --> sanitise["injection.sanitise_evidence<br/>delimit passages, redact<br/>instruction-like text"]
    sanitise --> llm["generate_answer<br/>gpt-4o-mini, temperature 0<br/>grounding rules in system prompt"]
    llm -->|"model declines"| refuse3(["refusal<br/>model_refused"])
    llm --> out{"output_guards.check_output<br/>refund promises, assumed<br/>entitlement, personal advice"}
    out -->|"violation"| refuse4(["refusal<br/>SAFETY_REFUSAL wording"])
    out -->|"no sources"| refuse5(["refusal<br/>no_sources"])

    out --> ok(["200 answer<br/>+ useful_resources (max 5)<br/>+ disclaimer<br/>+ diagnostics"])
    ok --> remember["memory.add_turn<br/>answered turns only"]

    db[("chunks<br/>filtered: active, financial_year,<br/>version")] --- hybrid
    expand -.-> hybrid

    classDef refusal fill:#fbeee9,stroke:#a4371f,color:#17211b
    classDef success fill:#eef4f1,stroke:#176b57,color:#17211b
    class refuse1,refuse2,refuse3,refuse4,refuse5,r429,r422 refusal
    class ok success
```

### The refusal branch is the point

Five independent paths end in a refusal, and only one of them involves the model. The
cheap checks run first: a guard costs a regex, and the grounding gate costs nothing
beyond retrieval that already happened. Only a question that survives both is worth
paying a model call for.

A confident wrong answer about someone's tax is far more damaging than "I don't have
enough information to answer that."

Two refusals that look alike are deliberately kept distinct, because they mean different
things: **insufficient evidence** says the corpus did not cover it, while
**SAFETY_REFUSAL** says the answer existed but would have read as personal advice.
Reporting both as "not enough information" once sent a reader hunting for a retrieval
problem that was not there.

### Year-scoped questions read superseded content

When a request names a `financial_year`, the filter switches `active` off, so
superseded guidance becomes visible. A question about 2021-22 should be answered from
the guidance that applied then, which is exactly what `Status.SUPERSEDED` exists to
preserve. Undated questions get the opposite treatment from `prefer_current`.

Note that a year filter still matches evergreen chunks - roughly three quarters of the
corpus carries no year at all, and that is where most answers live (ADR-0002).

---

## Tuning knobs

All are environment variables, read through `config.Settings`:

| Setting | Default | Effect |
|---|---|---|
| `SHORTLIST` | 30 | Candidates retrieved per query before reranking |
| `EVIDENCE_COUNT` | 15 | Passages handed to the model |
| `CITATION_LIMIT` | 5 | Sources shown to the user |
| `FAMILY_CAP` | 2 | Limits near-identical pages in one shortlist |
| `RERANK_STRATEGY` | `fusion` | `fusion`, `mmr` or `llm` (ADR-0003) |
| `RATE_LIMIT_PER_MINUTE` | 30 | Per replica, not per service |

`scripts/probe/sweep_k.py` and `sweep_rerank.py` measure the first and fifth of these
against known-answer questions, so the values can be chosen on evidence.

## Where the decisions are recorded

- **ADR-0002** - one Postgres table serves both retrieval halves; financial year is an
  array, not a scalar.
- **ADR-0003** - three rerankers compared; `fusion` ships because neither alternative
  beat it measurably.
- **ADR-0004** - incremental indexing and the state checkpoint.
