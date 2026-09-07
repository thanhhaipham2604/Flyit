"""Embedding tests: the guarantees the indexer relies on.

No network and no database - the OpenAI client is replaced with a fake that
records what it was asked for, and the cache is pointed at a tmp directory.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from fylit_rag.config import settings
from fylit_rag.indexing import embeddings
from fylit_rag.indexing.schema import dimensions_for

WIDTH = dimensions_for(settings.embedding_model)


class FakeClient:
    """Stands in for `OpenAI()`. Records every batch it is handed."""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None, width: int = WIDTH):
        self.batches: list[list[str]] = []
        self.fail_times = fail_times
        self.exc = exc or APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        )
        self.width = width
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model, input):
        self.batches.append(list(input))
        if len(self.batches) <= self.fail_times:
            raise self.exc
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self._vector(t)) for t in input]
        )

    def _vector(self, text: str) -> list[float]:
        """A vector that identifies its text, so order errors are visible."""
        return [float(len(text))] * self.width


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Install a fake client and an isolated on-disk cache."""
    monkeypatch.setattr(settings, "index_dir", str(tmp_path))
    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)
    embeddings._cache.cache_clear()  # reopen against tmp_path, not the real index dir

    client = FakeClient()
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    yield client
    embeddings._cache.cache_clear()  # don't leak the tmp connection into other tests


def test_empty_input_makes_no_api_call(fake):
    assert embeddings.embed_texts([]) == []
    assert fake.batches == []


def test_order_and_width_are_preserved(fake):
    texts = ["a", "bb", "ccc"]
    vectors = embeddings.embed_texts(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == WIDTH for v in vectors)
    # The fake encodes len(text) in the vector, so a reordering would show here.
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_texts_are_sent_in_batches(fake, monkeypatch):
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)
    texts = [f"chunk {i}" * (i + 1) for i in range(5)]

    vectors = embeddings.embed_texts(texts)

    assert len(vectors) == 5
    assert [len(b) for b in fake.batches] == [2, 2, 1]


def test_duplicate_texts_are_embedded_once(fake):
    """Identical text is paid for once but still returned for every position."""
    vectors = embeddings.embed_texts(["same", "other", "same"])

    assert fake.batches == [["same", "other"]]
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_cache_survives_a_second_call(fake):
    """The whole point of the hash cache: an incremental run pays nothing."""
    first = embeddings.embed_texts(["alpha", "beta"])
    calls_after_first = len(fake.batches)

    second = embeddings.embed_texts(["alpha", "beta"])

    assert second == first
    assert len(fake.batches) == calls_after_first, "second run should hit the cache"


def test_cache_misses_when_the_model_changes(fake, monkeypatch):
    """A model switch must not serve vectors embedded by the previous model."""
    embeddings.embed_texts(["alpha"])
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-ada-002")

    embeddings.embed_texts(["alpha"])

    assert fake.batches == [["alpha"], ["alpha"]]


def test_transient_failure_is_retried(fake, monkeypatch):
    client = FakeClient(fail_times=2)
    monkeypatch.setattr(embeddings, "_client", lambda: client)

    vectors = embeddings.embed_texts(["retry me"])

    assert len(vectors) == 1
    assert len(client.batches) == 3, "two failures then a success"


def test_permanent_failure_is_not_retried(fake, monkeypatch):
    """An auth error or a bad model fails the same way every time - retrying it
    only delays a certain failure."""
    client = FakeClient(fail_times=99, exc=ValueError("invalid api key"))
    monkeypatch.setattr(embeddings, "_client", lambda: client)

    with pytest.raises(ValueError, match="invalid api key"):
        embeddings.embed_texts(["nope"])

    assert len(client.batches) == 1, "permanent errors must not be retried"


def test_wrong_vector_width_is_rejected(fake, monkeypatch):
    """The chunks column is vector(1536); a mismatch must fail with our message,
    not as an opaque Postgres type error later."""
    client = FakeClient(width=WIDTH - 1)
    monkeypatch.setattr(embeddings, "_client", lambda: client)

    with pytest.raises(RuntimeError, match="expects"):
        embeddings.embed_texts(["wrong width"])


def test_short_response_is_rejected(fake, monkeypatch):
    """Fewer vectors than inputs means we cannot say which text each belongs to."""
    client = FakeClient()

    def short_create(*, model, input):
        client.batches.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * WIDTH)])

    client.embeddings = SimpleNamespace(create=short_create)
    monkeypatch.setattr(embeddings, "_client", lambda: client)

    with pytest.raises(RuntimeError, match="refusing to guess"):
        embeddings.embed_texts(["one", "two"])


def test_configured_api_key_reaches_the_client(monkeypatch):
    """settings.openai_api_key must actually be wired to the client.

    Without this, a key supplied any way other than a .env file - a real
    environment variable, a CI secret, a secrets manager feeding Settings -
    would be silently ignored while the config field suggested otherwise.
    """
    captured = {}

    class Recorder:
        def __init__(self, api_key=None, **kwargs):
            captured["api_key"] = api_key

    monkeypatch.setattr(embeddings, "OpenAI", Recorder)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-a-real-key")
    embeddings._client.cache_clear()

    embeddings._client()

    assert captured["api_key"] == "sk-test-not-a-real-key"
    embeddings._client.cache_clear()  # don't hand the recorder to another test


def test_absent_configured_key_defers_to_the_sdk(monkeypatch):
    """An empty setting must become None, so the SDK's own OPENAI_API_KEY
    lookup still works rather than being overridden with an empty string."""
    captured = {}

    class Recorder:
        def __init__(self, api_key=None, **kwargs):
            captured["api_key"] = api_key

    monkeypatch.setattr(embeddings, "OpenAI", Recorder)
    monkeypatch.setattr(settings, "openai_api_key", "")
    embeddings._client.cache_clear()

    embeddings._client()

    assert captured["api_key"] is None
    embeddings._client.cache_clear()


def test_importing_the_module_needs_no_api_key():
    """Tests and CI must be able to import this without credentials.

    Run in a subprocess: the point is what happens at import time in a fresh
    interpreter, which a reload inside this one would not prove.
    """
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", "import fylit_rag.indexing.embeddings"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
