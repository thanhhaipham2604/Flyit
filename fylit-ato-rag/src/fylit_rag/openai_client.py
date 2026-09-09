"""The one OpenAI client, built on first use.

Three modules needed a client - embeddings, reranking, and the eval-set builder -
and each had grown its own lazy constructor with its own copy of the key-loading
rule. Three copies of "where does the API key come from" is three places for it
to drift, and the answer has to be identical everywhere.

Deliberately at the package root rather than inside `generation`: `indexing`
needs a client too, and Zone A importing from Zone B would invert the layering.
`generation.llm` is about turning evidence into a grounded answer, which is a
different job from owning a connection.

Built lazily so that importing any module that *might* call OpenAI does not
require an API key - the schema and chunker tests run before anything is
configured, and they import through these packages.
"""

from __future__ import annotations

from functools import lru_cache

from fylit_rag.config import settings


@lru_cache(maxsize=1)
def client():
    """The shared OpenAI client.

    The key comes from `settings`, which is the one place configuration is
    declared - so a key supplied any way pydantic-settings understands (a real
    environment variable, a CI secret) reaches the client, not just one written
    into a .env file. `load_dotenv()` remains the fallback for the latter, and
    passing None lets the SDK apply its own OPENAI_API_KEY lookup rather than
    being overridden with an empty string.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI(api_key=settings.openai_api_key or None)
