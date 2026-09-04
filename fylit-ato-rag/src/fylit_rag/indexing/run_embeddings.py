"""Fill in embeddings for chunks that don't have one yet.

Reads chunks whose `embedding` is NULL, embeds their text with
`embed_texts`, and writes the vectors back. Only untouched chunks are
processed, so re-running after new content is cheap - this is the
incremental behaviour.

Note on what "untouched" means: NULL is the only signal this runner has. A row
whose text changed but which kept its old vector is invisible here, so whatever
writes chunk rows must set `embedding = NULL` when `content_hash` changes.
`ChunkRecord.as_row()` already omits `embedding`, which gives an upsert the
opening to do exactly that.
"""

from __future__ import annotations

from pgvector.psycopg import register_vector

from fylit_rag.config import settings
from fylit_rag.indexing.bootstrap import connect
from fylit_rag.indexing.embeddings import embed_texts

# How many chunks to pull from the database and embed at a time. Matches
# `embeddings.BATCH_SIZE` so one fetch is one API request.
FETCH_SIZE = 100


def run() -> None:
    """Embed every chunk that is currently missing an embedding."""
    total_embedded = 0

    # Table name comes from our own settings, never from user input, so
    # interpolating it here is safe - psycopg cannot parameterise identifiers.
    table = settings.chunks_table

    with connect() as conn:
        register_vector(conn)  # so a Python list adapts to pgvector's `vector`

        while True:
            # 1. Fetch a batch of chunks that still need an embedding.
            rows = conn.execute(
                f'SELECT chunk_id, "text" FROM {table} '
                "WHERE embedding IS NULL LIMIT %s",
                (FETCH_SIZE,),
            ).fetchall()

            if not rows:
                break  # nothing left to embed

            chunk_ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]

            # 2. Embed their text.
            vectors = embed_texts(texts)

            # A short result would leave rows NULL, and the next iteration would
            # fetch the very same rows - an infinite loop that looks like progress.
            # Fail loudly instead.
            if len(vectors) != len(rows):
                raise RuntimeError(
                    f"Embedded {len(vectors)} vectors for {len(rows)} chunks; "
                    f"aborting rather than leaving rows silently unembedded."
                )

            # 3. Write the vectors back - one round trip, not one per row.
            with conn.cursor() as cur:
                cur.executemany(
                    f"UPDATE {table} SET embedding = %s WHERE chunk_id = %s",
                    list(zip(vectors, chunk_ids, strict=True)),
                )
            conn.commit()

            total_embedded += len(rows)
            print(f"Embedded {total_embedded} chunks so far...")

    print(f"Done. Embedded {total_embedded} chunks in total.")


if __name__ == "__main__":
    run()
