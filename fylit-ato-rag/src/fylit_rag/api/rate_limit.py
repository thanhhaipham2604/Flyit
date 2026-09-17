"""Rate limiting so the service does not get swamped (slowapi/limits).

Default: settings.rate_limit_per_minute per client IP. Return 429 with a
Retry-After header. Must hold up in the concurrent-user demo.

Keyed on client IP, which is the honest option for a service with no accounts.
Its limits are worth stating plainly: everyone behind one NAT shares a bucket,
and an attacker with many addresses gets many buckets. It is a protection against
being swamped, not an authorisation mechanism.

Storage is in-process. That means each replica counts separately, so N replicas
allow N times the limit - fine for the single-instance demo, wrong for the
Kubernetes deployment, where `storage_uri` should point at Redis.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from fylit_rag.config import settings

limiter = Limiter(key_func=get_remote_address)


def ask_limit() -> str:
    """The limit string for /ask, read at call time so tests can vary it."""
    return f"{settings.rate_limit_per_minute}/minute"


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 with Retry-After, in the same shape as every other response.

    A client that hits the limit gets the standard envelope rather than a bare
    error, so a caller written against /ask does not need a second code path to
    read this one - and `answer` says what happened in words a person can act on.
    """
    retry_after = getattr(exc, "retry_after", None) or 60
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={
            "answer": (
                "Too many requests. Please wait a moment and ask again."
            ),
            "useful_resources": [],
            "disclaimer": (
                "This is general information only, not personal tax advice. "
                "Consider speaking to a registered tax agent about your situation."
            ),
            "diagnostics": {"refused": True, "guardrail": "rate_limited"},
        },
    )
