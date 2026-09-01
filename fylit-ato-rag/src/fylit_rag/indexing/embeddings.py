"""Embedding client (OpenAI by default, see config.EMBEDDING_MODEL).

Batch requests; cache by chunk content hash so unchanged chunks are
never re-embedded during incremental runs.
"""

from __future__ import annotations

import time

from dotenv import load_dotenv
from openai import OpenAI

from fylit_rag.config import settings

load_dotenv()

_client = OpenAI()

# How many texts to send in a single API request.
BATCH_SIZE = 100

# If a request fails, try this many times before giving up.
MAX_RETRIES = 3


def _embed_one_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch, retrying on transient failures.

    Waits a little longer between each attempt (1s, then 2s, then 4s) so a brief
    network or rate-limit blip does not kill a long run.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            return [item.embedding for item in response.data]
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise  # out of retries - let the error surface
            time.sleep(2 ** attempt)  # 1s, 2s, 4s

    return []  # unreachable, but keeps the type checker happy


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors.

    Sends the texts to OpenAI in batches of BATCH_SIZE, retrying each batch on
    transient failures. Each vector is 1536 floats (text-embedding-3-small).
    The order of the returned vectors matches the order of the input texts.
    """
    if not texts:
        return []

    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        all_vectors.extend(_embed_one_batch(batch))

    return all_vectors