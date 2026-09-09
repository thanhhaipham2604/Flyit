"""Shared OpenAI client tests: where the API key comes from.

These moved out of test_embeddings.py when the client was pulled up into one
module. That is the point of the refactor - the key-loading rule is now tested
once, in the one place it is implemented.
"""

import os
import subprocess
import sys
from typing import ClassVar

import pytest

from fylit_rag import openai_client
from fylit_rag.config import settings


class Recorder:
    """Stands in for `OpenAI`, capturing the arguments it was constructed with."""

    last: ClassVar[dict] = {}

    def __init__(self, api_key=None, **kwargs):
        Recorder.last = {"api_key": api_key, **kwargs}


@pytest.fixture
def recorder(monkeypatch):
    """Make `client()` build a Recorder instead of a real client."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", Recorder)
    openai_client.client.cache_clear()
    Recorder.last = {}
    yield Recorder
    openai_client.client.cache_clear()  # never hand a Recorder to another test


def test_configured_api_key_reaches_the_client(recorder, monkeypatch):
    """A key supplied any way other than a .env file - a real environment
    variable, a CI secret, a secrets manager feeding Settings - must still
    arrive, or the config field would be a lie."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-a-real-key")

    openai_client.client()

    assert recorder.last["api_key"] == "sk-test-not-a-real-key"


def test_absent_configured_key_defers_to_the_sdk(recorder, monkeypatch):
    """An empty setting must become None, so the SDK's own OPENAI_API_KEY lookup
    still applies rather than being overridden with an empty string."""
    monkeypatch.setattr(settings, "openai_api_key", "")

    openai_client.client()

    assert recorder.last["api_key"] is None


def test_the_client_is_built_once(recorder, monkeypatch):
    """Every caller shares one client rather than opening a connection pool each."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    assert openai_client.client() is openai_client.client()


@pytest.mark.parametrize(
    "module",
    [
        "fylit_rag.openai_client",
        "fylit_rag.indexing.embeddings",
        "fylit_rag.retrieval.rerank",
    ],
)
def test_importing_needs_no_api_key(module):
    """Tests and CI must import these without credentials.

    A subprocess, because the point is what happens at import time in a fresh
    interpreter - a reload inside this one would not prove it.
    """
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
