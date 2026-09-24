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

from fylit_rag.generation.prompts import DISCLAIMER, REFUSAL_MESSAGE, SAFETY_REFUSAL

# Promising an outcome to this user. Distinct from explaining that refunds exist.
REFUND_GUARANTEE = (
    re.compile(r"\byou (will|'ll|are going to) (get|receive|be paid)\b[^.\n]{0,30}"
               r"\b(refund|money back|payment)\b", re.IGNORECASE),
    re.compile(r"\byou (are|'re) (definitely |certainly )?entitled to\b", re.IGNORECASE),
    re.compile(r"\byou will (definitely|certainly)\b", re.IGNORECASE),
    re.compile(r"\bguarantee\w*\b[^.\n]{0,30}\brefund\b", re.IGNORECASE),
)

# Telling this user a concession applies, rather than stating its conditions.
#
# A bare "you can claim" used to be enough to refuse, and that was wrong: it is
# the ATO's own phrasing for stating a general rule ("Expenses you can claim
# include tools you buy for work"), so it fired on most deduction answers and
# replaced them with a refusal. Measured against the live index, it killed
# perfectly good answers to "How much can I claim as a builder?" and "...as a
# software developer?".
#
# What actually distinguishes an assumed entitlement is whether the sentence
# names a condition. "You can claim it if the cost is $300 or less" states the
# rule and leaves the reader to check it; "You can claim the full amount" has
# decided their case. So the object alone is not enough to judge by - the test
# is a definite object AND no condition in the same sentence. Emphatics
# ("definitely", "certainly") assert entitlement whatever follows them.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_DEFINITE_CLAIM = re.compile(
    r"\byou can claim (it|this|that|these|those|them|your|the full|the whole)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(if|when|where|unless|provided|assuming|as long as|subject to|"
    r"conditions?|eligib\w+|must|require\w*)\b",
    re.IGNORECASE,
)


def _asserted_without_conditions(answer: str) -> bool:
    """True when a sentence says the user can claim a specific thing, full stop."""
    return any(
        _DEFINITE_CLAIM.search(sentence) and not _CONDITION.search(sentence)
        for sentence in _SENTENCE_SPLIT.split(answer or "")
    )


ASSUMED_ENTITLEMENT = (
    re.compile(r"\byou can (definitely|certainly|absolutely) claim\b", re.IGNORECASE),
    _asserted_without_conditions,
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


def _matches(rule, answer: str) -> bool:
    """A rule is either a compiled pattern or a predicate over the whole answer.

    Predicates exist because some rules need more than one sentence's worth of
    context - see `_asserted_without_conditions`.
    """
    return bool(rule(answer)) if callable(rule) else bool(rule.search(answer))


def find_violations(answer: str) -> list[str]:
    """Which prohibited claims the answer makes, by name."""
    answer = answer or ""
    return [
        name
        for name, rules in VIOLATIONS.items()
        if any(_matches(rule, answer) for rule in rules)
    ]


def refusal_message_for(reason: str) -> str:
    """Which refusal wording a reason deserves.

    A guardrail block and thin evidence are different failures. Reporting both as
    "I don't have enough information" is a lie in the first case, and the one the
    reader acts on - it points at retrieval when the evidence was fine.
    """
    names = {part.strip() for part in (reason or "").split(",")}
    return SAFETY_REFUSAL if names & set(VIOLATIONS) else REFUSAL_MESSAGE


def controlled_refusal(reason: str = "guardrail") -> dict:
    """The one shape a refusal may take, so callers cannot invent variants."""
    return {
        "answer": refusal_message_for(reason),
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
