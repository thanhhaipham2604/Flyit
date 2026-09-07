"""Embed chunk text with OpenAI.

Batch requests; cache by chunk content hash so unchanged chunks are
never re-embedded during incremental runs.

The cache is keyed by (model, sha256(text)), so switching embedding models can
never hand back a vector of the wrong width, and two chunks with identical text
are paid for once. Vectors are stored as float32 - the same precision pgvector's
`vector` type keeps - so the cache is lossless with respect to the database.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from fylit_rag.config import settings
from fylit_rag.indexing.schema import dimensions_for

# How many texts to send in a single API request.
BATCH_SIZE = 100

# If a request fails, try this many times before giving up.
MAX_RETRIES = 3

# Only these are worth retrying. An auth failure, an unknown model or an
# over-long input will fail identically on every attempt, so retrying them just
# delays a certain error - fail fast and let the message surface.
TRANSIENT_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """The OpenAI client, built on first use.

    Deliberately not at import time: `schema.py` is imported by tests that run
    before anything is configured, and this module sits in the same package.
    Constructing the client here means importing `embeddings` never needs an API
    key, and only actually embedding does.

    The key comes from `settings`, which is the one place configuration is
    declared - so a key supplied any way pydantic-settings understands (a real
    environment variable, a CI secret) reaches the client, not just one written
    into a .env file. `load_dotenv()` stays as the fallback for the latter, and
    passing None lets the SDK fall back to OPENAI_API_KEY itself.
    """
    load_dotenv()
    return OpenAI(api_key=settings.openai_api_key or None)


# ---------------------------------------------------------------- cache


def _cache_key(model: str, text: str) -> str:
    """Content hash for one text under one model. The model is part of the key
    so a model switch misses the cache rather than returning a stale width."""
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")  # separator, so model+text cannot collide by concatenation
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _cache() -> sqlite3.Connection | None:
    """Open (and create) the on-disk vector cache.

    Returns None if the cache cannot be opened - a missing cache must degrade to
    "pay the API again", never to a failed run.
    """
    try:
        path = Path(settings.index_dir) / "embedding_cache.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")
        conn.commit()
        return conn
    except (OSError, sqlite3.Error):
        return None


def _cache_get(keys: list[str]) -> dict[str, list[float]]:
    """Look up many keys at once. Missing keys are simply absent from the result."""
    conn = _cache()
    if conn is None or not keys:
        return {}
    found: dict[str, list[float]] = {}
    # Chunked so a long run cannot exceed SQLite's variable limit.
    for start in range(0, len(keys), 500):
        window = keys[start : start + 500]
        placeholders = ",".join("?" * len(window))
        rows = conn.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
            window,
        ).fetchall()
        for key, blob in rows:
            found[key] = list(struct.unpack(f"<{len(blob) // 4}f", blob))
    return found


def _cache_put(items: dict[str, list[float]]) -> None:
    """Store vectors. A cache write failure is logged nowhere and swallowed on
    purpose: it costs money next run, it does not break this one."""
    conn = _cache()
    if conn is None or not items:
        return
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(key, struct.pack(f"<{len(v)}f", *v)) for key, v in items.items()],
        )
        conn.commit()
    except (OSError, sqlite3.Error, struct.error):
        pass


# ---------------------------------------------------------------- embedding


def _embed_one_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch, retrying on transient failures.

    Backs off between attempts (1s, then 2s) so a brief network or rate-limit
    blip does not kill a long run. The final attempt raises rather than sleeping.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _client().embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            return [item.embedding for item in response.data]
        except TRANSIENT_ERRORS:
            if attempt == MAX_RETRIES - 1:
                raise  # out of retries - let the error surface
            time.sleep(2**attempt)

    raise AssertionError("unreachable: the loop either returns or raises")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors.

    Cached texts are served from disk; the rest are sent to OpenAI in batches of
    BATCH_SIZE, retrying each batch on transient failures. The returned list is
    the same length as `texts` and in the same order, and each vector is the
    width `schema.dimensions_for` expects for the configured model.
    """
    if not texts:
        return []

    model = settings.embedding_model
    expected_width = dimensions_for(model)

    keys = [_cache_key(model, text) for text in texts]
    vectors = _cache_get(keys)

    # Unique misses, first-seen order. Duplicate texts collapse to one API call.
    missing: list[str] = []
    seen: set[str] = set()
    for key, text in zip(keys, texts, strict=True):
        if key not in vectors and key not in seen:
            seen.add(key)
            missing.append(text)

    fresh: dict[str, list[float]] = {}
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        embedded = _embed_one_batch(batch)
        if len(embedded) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(embedded)} vectors for {len(batch)} inputs; "
                f"refusing to guess which text each belongs to."
            )
        for text, vector in zip(batch, embedded, strict=True):
            if len(vector) != expected_width:
                raise RuntimeError(
                    f"{model!r} returned a {len(vector)}-d vector but the schema expects "
                    f"{expected_width}-d. The chunks table column would reject this - "
                    f"re-index from scratch if you meant to change models."
                )
            fresh[_cache_key(model, text)] = vector

    _cache_put(fresh)
    vectors.update(fresh)

    return [vectors[key] for key in keys]
