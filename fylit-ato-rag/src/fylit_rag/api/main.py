"""FastAPI application factory.

OpenAPI docs are served at /docs (a hand-in requirement).

/health deliberately does not touch Postgres or OpenAI. A liveness probe that
depends on a database restarts the pod when the database is slow, which is the
opposite of helpful; /health says "this process is up", and /ready says whether
it can actually serve.
"""

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded

from fylit_rag.api.rate_limit import limiter, rate_limit_handler
from fylit_rag.api.routes import router

app = FastAPI(
    title="Fylit ATO RAG API",
    version="0.1.0",
    description=(
        "General Australian tax information grounded in official ATO content. "
        "Not personal tax advice."
    ),
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    """Liveness: the process is running. No dependencies checked on purpose."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: can this instance actually answer a question?

    Checks that the chunks table exists and holds embedded content, because an
    instance pointed at an empty index will return nothing but refusals - which
    looks like a working service and is not one.
    """
    try:
        from fylit_rag.config import settings
        from fylit_rag.indexing.bootstrap import connect

        with connect() as conn:
            embedded = conn.execute(
                f"SELECT count(*) FROM {settings.chunks_table} WHERE embedding IS NOT NULL"
            ).fetchone()[0]
    except Exception as exc:  # noqa: BLE001 - a readiness probe reports, it does not raise
        return {"status": "not ready", "reason": type(exc).__name__}

    if not embedded:
        return {"status": "not ready", "reason": "index is empty", "embedded_chunks": 0}
    return {"status": "ready", "embedded_chunks": embedded}
