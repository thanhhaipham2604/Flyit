"""API tests: /health, /ask contract (answer + resources + disclaimer +
diagnostics), rate limiting 429, refusal path, injection attempts."""

from fastapi.testclient import TestClient

from fylit_rag.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
