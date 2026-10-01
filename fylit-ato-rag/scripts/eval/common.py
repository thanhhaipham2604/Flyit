"""Shared helpers for the evaluation probes.

Everything that talks to the running API or to docker lives here, so the
probes themselves are only questions and scoring.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request

API = "http://localhost:8000"
ENV_FILE = ".env"
CONTAINER = "fylit-ato-rag-api-1"
TIMEOUT = 180


def ask(question: str, financial_year: str | None = None) -> dict:
    """One /ask call. Returns the parsed body, or {"error": ...}.

    A 422 means input validation rejected the question, which is a refusal as
    far as the probes are concerned - so it is reported, not raised.
    """
    body: dict = {"question": question}
    if financial_year:
        body["financial_year"] = financial_year
    req = urllib.request.Request(
        f"{API}/ask",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"http_error": exc.code}
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, never raises
        return {"error": type(exc).__name__}


def outcome(body: dict) -> tuple[str, str, str]:
    """(got, guardrail, answer) where got is answered | refused | error."""
    if "http_error" in body:
        return "refused", f"HTTP {body['http_error']}", ""
    if "error" in body:
        return "error", body["error"], ""
    diag = body.get("diagnostics") or {}
    answer = body.get("answer") or ""
    return ("refused" if diag.get("refused") else "answered",
            diag.get("guardrail") or "", answer)


def contains_any(answer: str, needles: list[str] | None) -> bool | None:
    """True/False if there is something to check, None if there is not."""
    if not needles:
        return None
    low = answer.lower()
    return any(n.lower() in low for n in needles)


def snippet(text: str, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


# --------------------------------------------------------------- sweeping

def read_env() -> list[str]:
    try:
        with open(ENV_FILE, encoding="utf-8") as fh:
            return fh.read().splitlines()
    except FileNotFoundError:
        return []


def get_env(key: str) -> str | None:
    for line in read_env():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    return None


def set_env(key: str, value: str) -> None:
    lines = [ln for ln in read_env() if not ln.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    with open(ENV_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def restart() -> None:
    """Recreate the api container so it picks up .env. No rebuild needed."""
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "api"],
        check=True, capture_output=True,
    )


def wait_ready(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def container_env(key: str) -> str:
    """What the running container actually has - a set_env that did not take
    effect would otherwise be measured as a real difference."""
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "printenv", key],
        capture_output=True, text=True,
    )
    return out.stdout.strip()
