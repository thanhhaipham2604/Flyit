"""Fill in embeddings for chunks that don't have one yet.

Reads chunks whose `embedding` is NULL, embeds their text with
`embed_texts`, and writes the vectors back. Only untouched chunks are
processed, so re-running after new content is cheap - this is the
incremental behaviour.
"""

from __future__ import annotations

import psycopg

from fylit_rag.config import settings
from fylit_rag.indexing.embeddings import embed_texts

# How many chunks to pull from the database and embed at a time.
FETCH_SIZE = 100


def run() -> None:
    """Embed every chunk that is currently missing an embedding."""
    total_embedded = 0

    with psycopg.connect(settings.database_url) as conn:
        while True:
            # 1. Fetch a batch of chunks that still need an embedding.
            rows = conn.execute(
                'SELECT chunk_id, "text" FROM chunks '
                "WHERE embedding IS NULL LIMIT %s",
                (FETCH_SIZE,),
            ).fetchall()

            if not rows:
                break  # nothing left to embed

            chunk_ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]

            # 2. Embed their text.
            vectors = embed_texts(texts)

            # 3. Write each vector back to its row.
            for chunk_id, vector in zip(chunk_ids, vectors):
                conn.execute(
                    "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                    (str(vector), chunk_id),
                )
            conn.commit()

            total_embedded += len(rows)
            print(f"Embedded {total_embedded} chunks so far...")

    print(f"Done. Embedded {total_embedded} chunks in total.")


if __name__ == "__main__":
    run()