"""FastAPI application factory.

OpenAPI docs are served at /docs (a hand-in requirement).
"""

from fastapi import FastAPI

from fylit_rag.api.routes import router

app = FastAPI(
    title="Fylit ATO RAG API",
    version="0.1.0",
    description=(
        "General Australian tax information grounded in official ATO content. "
        "Not personal tax advice."
    ),
)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
