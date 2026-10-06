import json
from types import SimpleNamespace

import pytest

from scripts import build_eval_set


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_):
        return self

    def fetchall(self):
        return [("chunk-1", "doc-1", "Guidance text " * 40, "Title")]


def fake_client(content=None, error=None):
    def create(**_):
        if error:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_build_instantiates_client_and_writes_generated_question(monkeypatch, tmp_path):
    monkeypatch.setattr(build_eval_set, "connect", FakeConnection)
    monkeypatch.setattr(
        build_eval_set,
        "client",
        lambda: fake_client('{"question": "What expenses can I claim?"}'),
    )
    output = tmp_path / "questions.jsonl"

    build_eval_set.build(n=1, out=str(output), seed=42)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["question"] == "What expenses can I claim?"
    assert rows[0]["chunk_id"] == "chunk-1"


def test_build_logs_and_propagates_generation_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_eval_set, "connect", FakeConnection)
    monkeypatch.setattr(
        build_eval_set,
        "client",
        lambda: fake_client(error=ValueError("API response was invalid")),
    )
    output = tmp_path / "questions.jsonl"

    with pytest.raises(ValueError, match="API response was invalid"):
        build_eval_set.build(n=1, out=str(output), seed=42)

    assert "chunk-1: API response was invalid" in capsys.readouterr().err
    assert not output.exists()