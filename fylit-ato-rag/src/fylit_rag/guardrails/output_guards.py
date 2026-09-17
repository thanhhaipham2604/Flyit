"""Final checks before a response leaves the API.

Enforce: disclaimer present, sources attached when evidence used,
no personalised advice / refund guarantees / assumed deductions slipping
through, refusal format is the controlled one.

This is the second enforcement of rules the system prompt already states, and it
exists because the first one is advisory. A model can be argued out of a prompt;
it cannot be argued out of a regex that runs after it.

**Fails closed.** Anything this cannot repair becomes the controlled refusal. An
answer that quietly promises a refund is worse than no answer, so the tie-break
always goes to refusing.

Two kinds of problem, handled differently:

- *Repairable*: a missing disclaimer is appended, because its absence does not
  make the answer wrong.
- *Not repairable*: a refund guarantee or personalised advice cannot be edited
  out without changing what the answer claims, so the whole response is replaced.
"""

from __future__ import annotations

import re

from fylit_rag.generation.prompts import DISCLAIMER, REFUSAL_MESSAGE

# Promising an outcome to this user. Distinct from explaining that refunds exist.
REFUND_GUARANTEE = (
    re.compile(r"\byou (will|'ll|are going to) (get|receive|be paid)\b[^.\n]{0,30}"
               r"\b(refund|money back|payment)\b", re.IGNORECASE),
    re.compile(r"\byou (are|'re) (definitely |certainly )?entitled to\b", re.IGNORECASE),
    re.compile(r"\byou will (definitely|certainly)\b", re.IGNORECASE),
    re.compile(r"\bguarantee\w*\b[^.\n]{0,30}\brefund\b", re.IGNORECASE),
)

# Telling this user a concession applies, rather than stating its conditions.
ASSUMED_ENTITLEMENT = (
    re.compile(r"\byou can (definitely |certainly )?claim\b", re.IGNORECASE),
    re.compile(r"\byou qualify for\b", re.IGNORECASE),
    re.compile(r"\byour deduction (is|will be)\b", re.IGNORECASE),
    re.compile(r"\byou (do not|don't) (need to|have to) (declare|report|pay)\b", re.IGNORECASE),
)

# Acting as the user's adviser rather than explaining the rules.
PERSONALISED = (
    re.compile(r"\b(i|we) (recommend|advise|suggest) (that )?you\b", re.IGNORECASE),
    re.compile(r"\byou should (claim|deduct|declare|lodge)\b", re.IGNORECASE),
    re.compile(r"\bin your (case|situation|circumstances)\b", re.IGNORECASE),
)

VIOLATIONS = {
    "refund_guarantee": REFUND_GUARANTEE,
    "assumed_entitlement": ASSUMED_ENTITLEMENT,
    "personalised_advice": PERSONALISED,
}


def find_violations(answer: str) -> list[str]:
    """Which prohibited claims the answer makes, by name."""
    return [
        name
        for name, patterns in VIOLATIONS.items()
        if any(p.search(answer or "") for p in patterns)
    ]


def controlled_refusal(reason: str = "guardrail") -> dict:
    """The one shape a refusal may take, so callers cannot invent variants."""
    return {
        "answer": REFUSAL_MESSAGE,
        "refused": True,
        "sources": [],
        "disclaimer": DISCLAIMER,
        "guardrail": reason,
    }


def check_output(response: dict) -> dict:
    """Validate and repair the outgoing response; fail closed to refusal.

    Returns a response dict that is safe to serialise. `guardrail` names what
    intervened, if anything - the API logs it, and the adversarial tests assert
    on it.
    """
    if not isinstance(response, dict):
        return controlled_refusal("malformed_response")

    answer = (response.get("answer") or "").strip()
    refused = bool(response.get("refused"))
    sources = response.get("sources") or []

    if not answer:
        return controlled_refusal("empty_answer")

    # A refusal must be the controlled wording, not the model's improvisation on
    # the theme - otherwise "I'm not sure, but probably yes" counts as refusing.
    if refused:
        return controlled_refusal(response.get("guardrail") or "insufficient_evidence")

    violations = find_violations(answer)
    if violations:
        return controlled_refusal(",".join(violations))

    # An answer asserting tax facts with nothing behind it is ungrounded by
    # definition, whatever it says about itself.
    if not sources:
        return controlled_refusal("no_sources")

    repaired = dict(response)
    repaired["sources"] = sources
    repaired["refused"] = False
    # Repairable: a missing disclaimer does not make the answer wrong.
    repaired["disclaimer"] = DISCLAIMER
    if DISCLAIMER not in answer:
        repaired["answer"] = f"{answer}\n\n{DISCLAIMER}"
    repaired.setdefault("guardrail", None)
    return repaired
