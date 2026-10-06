# ADR-0004: Incremental indexing

- **Status:** accepted
- **Date:** 2026-10-01

## Context

The ATO corpus is scanned and cleaned on every indexing run, but chunking and
embedding the whole corpus each time would be wasteful. The implementation
therefore compares stable document IDs and hashes of cleaned content before
writing chunks to Postgres. Postgres is the index: as described in ADR-0002,
vector and keyword retrieval use columns on the same `chunks` row.

## Decision

**Use the source URL as document identity, compare cleaned-content hashes, and
apply the resulting document diff to the Postgres chunk table.** The active
pipeline is `indexing.run_indexing.run()`; it calls
`ingestion.pipeline.run_preprocessing()` and then indexes the returned diff.

- `ingestion.loader.stable_id()` returns the SHA-1 of the source URL. If there
  is no source URL, it hashes `path:<relative file path>` instead. The call is
  made by `ingestion.pipeline.process_file()`.
- `process_file()` hashes the cleaned body with SHA-256 via
  `ingestion.loader.content_hash()`. The hash does not include title, category,
  or other metadata.
- By default, `run_preprocessing()` reads
  `data/processed/state.json`, a JSON object mapping each `doc_id` to that
  cleaned-content hash. `indexing.versioning.load_state()` treats a missing or
  malformed file as an empty state. `diff_against_state()` classifies current
  documents as `new`, `changed`, or `unchanged`, and prior IDs absent from the
  current valid-document set as `deleted`. Its result includes ID lists for
  new, changed, and deleted documents; unchanged is a count.
- `run_preprocessing()` writes the current ID-to-hash map back to `state.json`
  and records the run's counts and diff in `data/processed/manifest.json`. The
  manifest is a report, not the version source of truth. Indexed versions and
  statuses are read from Postgres by `indexing.store.current_version()` and
  stored on chunk rows. Although `indexing.versioning.Manifest` can represent
  richer entries, this production path uses the simple `state.json` map.

The cases handled by `indexing.run_indexing.run()` are:

- **New:** `index_documents()` chunks the document and upserts its chunks. A
  chunk ID is `<doc_id>#<ordinal>` (`ingestion.chunker.chunk_document()`). Its
  initial version is 1, or the current indexed version if the ID already has
  rows. New rows have no embedding; `indexing.run_embeddings.run()` fills NULL
  embeddings, using the model-and-text cache in `indexing.embeddings`. Postgres
  maintains the keyword `search_vector` from the row text.
- **Changed:** the whole document is chunked again and its version advances
  from the maximum version in Postgres. `indexing.store.upsert_chunks()` keeps
  an existing embedding when that chunk ID's `content_hash` is unchanged and
  clears it when the chunk text changed, so only missing/invalidated vectors
  need embedding. Chunk IDs no longer produced by the new version are found by
  `stale_chunk_ids()` and physically removed with `purge_chunks()`.
- **Unchanged:** preprocessing still reads, parses, cleans, and hashes the
  source document, but `index_documents()` does not chunk or upsert it. There
  is no embedding work for its already-embedded chunks.
- **Deleted:** an ID present in the previous state but absent from the current
  valid documents is passed to `store.mark_deleted()`. This updates the status
  of all its chunk rows to `deleted`; it does not delete the rows or their
  vectors. The generated `active` value becomes false, taking those rows out of
  default active retrieval. There is no separate keyword index to clean up.

## Consequences

**Gained:** a normal rerun avoids chunking and upserting unchanged documents;
chunk-level hashes let unchanged chunks of a changed document retain their
vectors. Removed sections are purged, while a removed document is retained as
inactive rows. Vector and keyword data stay together in the Postgres table.

**Limitations of the current implementation:**

- `run_preprocessing()` writes `state.json` before the enriched output is
  complete and before `run_indexing.run()` commits its Postgres changes. A
  later failure can leave the saved state ahead of the index; the next diff may
  call those documents unchanged. The indexer has a recovery path when the
  entire chunks table is empty, but it does not reconcile a partially missing
  index against the state file.
- A missing or malformed state file, or `--full-rebuild`, starts the diff from
  an empty map. Current documents are treated as new, but old database-only IDs
  cannot be identified as deleted in that run and may remain active.
- Since the hash covers only cleaned body text, a metadata-only change does not
  trigger an upsert; stored title, URL, or taxonomy fields can remain stale.
  Conversely, a document rejected during preprocessing is absent from the
  current valid-document set and can be classified as deleted.
- The incremental diff saves downstream chunk/index/embedding work, not the
  corpus scan and cleaning. If identity falls back to a relative path, moving
  or renaming the file changes its ID and looks like a deletion plus a new
  document.
- `run_embeddings()` selects every row with a NULL embedding, without filtering
  by status. A deleted row whose vector was still NULL can therefore still be
  embedded on a later pass.

## Alternatives considered

- **Use the relative path for every ID.** Rejected in favor of the source URL
  where available, so a filename or directory change does not create a new
  identity for the same ATO page. The path remains the fallback when the URL is
  missing.
- **Persist a richer manifest as the indexing authority.**
  `indexing.versioning.Manifest` supports per-document version and status
  fields, but the current preprocessing path persists only ID-to-hash state;
  Postgres remains authoritative for indexed versions and statuses.
- **Physically delete rows for removed documents.** The implementation uses a
  deleted status instead, preserving the indexed record while making it
  inactive. Physical deletion is reserved for obsolete chunk IDs within a
  still-present document or explicit purge operations.
