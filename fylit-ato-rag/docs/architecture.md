# Architecture

*(Hand-in requires: an architecture diagram + a data-flow diagram. Draw them
here - mermaid is fine - once the pipeline settles.)*

## Zone A - ingestion (offline)

```
ATO .md folder -> loader (validate, stable ID + hash)
              -> cleaner (preserve headings/lists/tables)
              -> chunker (paragraph-sized, metadata-tagged)
              -> embeddings -> vector index (Qdrant)
                            -> keyword index (OpenSearch)
              -> manifest/versioning (incremental: new/changed/deleted/superseded)
```

## Zone B - query time

```
question -> API validation + rate limit
         -> input guards
         -> memory contextualise (follow-ups)
         -> hybrid retrieval (vector + keyword, filtered: year/version/active)
         -> rerank
         -> grounding confidence check --[insufficient]--> controlled refusal
         -> LLM grounded generation
         -> output guards (disclaimer, sources, no personalised advice)
         -> {answer, useful_resources, diagnostics}
```

The red branch - refusing when evidence is thin - is the most important
behaviour in the system. A confident wrong answer about tax is worse than
"I don't have enough information to answer that."
