"""Embedding client (OpenAI by default, see config.EMBEDDING_MODEL).

Batch requests; cache by chunk content hash so unchanged chunks are
never re-embedded during incremental runs.
"""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """TODO: batched OpenAI embeddings with retry + hash-keyed cache."""
    raise NotImplementedError
