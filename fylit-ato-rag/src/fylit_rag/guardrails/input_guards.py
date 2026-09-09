"""Validate and screen incoming questions.

Block/deflect: off-topic requests, personalised-advice requests
("how much refund will *I* get"), unsafe content. On-topic general
questions pass through to retrieval.

Rules are regex, not a classifier, for three reasons: they are testable and
reviewable line by line, they cost nothing on the query path, and a classifier
is itself promptable - an odd thing to put in front of an injection defence.

**Off-topic questions are deliberately not blocked here.** Deciding "is this
about tax?" with patterns either misses paraphrases or blocks legitimate
questions using words we did not anticipate, and the corpus already answers that
question better: an off-topic query retrieves nothing relevant, and
`generation.grounding` refuses before the model is ever called. Screening here is
limited to what patterns can judge reliably - a request for *personal* advice, or
for help doing something unlawful - where the words themselves are the problem
regardless of what the corpus contains.
"""

from __future__ import annotations

import re

# Asking us to apply the rules to *this person's* situation. The giveaway is a
# demand for a personal *number* or outcome, not the mere presence of "I".
#
# Measured against the eval set, an earlier version of this list blocked 24% of
# perfectly ordinary questions - "Can I claim a deduction for my computer?",
# "What kinds of deductions can I claim?" - because it matched
# `(should|can|am) i ... (claim|deduct|declare)`. Those are general-rule
# questions: the honest answer is "you may be able to, if you meet these
# conditions", which is exactly what this system is for. That pattern is gone.
# Phrasing that would turn a general answer into a personal one is caught after
# generation by `guardrails.output_guards`, which is the right place for it -
# it can see the answer, and this cannot.
PERSONALISED_ADVICE = (
    # "get" alone is far too broad - it matched "What do I need to do to *get*
    # my charity endorsed?". The outcome has to be money coming to this person.
    re.compile(r"\b(how much)\b[^?\n]{0,40}\bi\b[^?\n]{0,30}"
               r"\b(get back|owe|receive|be refunded|get as a refund)\b", re.IGNORECASE),
    re.compile(r"\bmy\b[^?\n]{0,30}\b(refund|tax bill|return|liability|assessment)s?\b"
               r"[^?\n]{0,30}\b(be|is|will)\b", re.IGNORECASE),
    re.compile(r"\bwhat('?s| is) my\b[^?\n]{0,30}\b(tax|refund|rate|bracket|liability)\b", re.IGNORECASE),
    re.compile(r"\bfile|lodge\b[^?\n]{0,20}\bmy (tax )?return for me\b", re.IGNORECASE),
)

# Asking for help breaking the law. Refused outright, and never retrieved for.
UNLAWFUL = (
    re.compile(r"\b(avoid|evade|dodge|hide|conceal|not (declare|report))\b[^?\n]{0,30}"
               r"\b(tax|taxes|income|earnings|cash|gst)\b", re.IGNORECASE),
    re.compile(r"\b(fake|falsify|inflate|make up)\b[^?\n]{0,30}"
               r"\b(receipt|deduction|expense|claim|invoice)s?\b", re.IGNORECASE),
    re.compile(r"\bwithout (the )?ato (finding out|knowing|noticing)\b", re.IGNORECASE),
    re.compile(r"\bget away with\b", re.IGNORECASE),
)

# Injection attempts arriving in the user's own message. Caught here as well as
# in retrieved text, because the user turn is not wrapped in evidence markers.
USER_INJECTION = (
    re.compile(r"\b(ignore|disregard|forget)\b[^.\n]{0,40}"
               r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
               r"\b(instruction|prompt|rule)s?\b", re.IGNORECASE),
    re.compile(r"\byou are (now|no longer)\b", re.IGNORECASE),
    re.compile(r"\b(reveal|repeat|print|show)\b[^.\n]{0,30}"
               r"\b(system prompt|your instructions)\b", re.IGNORECASE),
)

MAX_QUESTION_CHARS = 2000

PERSONALISED_REFUSAL = (
    "I can't work out your personal tax position - that depends on your full "
    "circumstances. I can explain the general ATO rules and the conditions that "
    "apply, and a registered tax agent can tell you where you stand."
)
UNLAWFUL_REFUSAL = (
    "I can't help with that. I can explain what the ATO requires and how to get "
    "it right."
)
INJECTION_REFUSAL = (
    "I can only answer questions about Australian tax using official ATO guidance."
)
EMPTY_REFUSAL = "Please ask a question about Australian tax."
TOO_LONG_REFUSAL = (
    f"That question is too long. Please keep it under {MAX_QUESTION_CHARS} characters."
)


def check_input(question: str) -> tuple[bool, str | None]:
    """Return (allowed, refusal_reason).

    `allowed=False` means the question never reaches retrieval or the model, and
    the caller returns the reason verbatim. Order is deliberate: unlawful and
    injection attempts are judged before personalised-advice phrasing, so a
    request that is both is refused for the more serious reason.
    """
    if question is None or not question.strip():
        return False, EMPTY_REFUSAL

    if len(question) > MAX_QUESTION_CHARS:
        return False, TOO_LONG_REFUSAL

    if any(p.search(question) for p in UNLAWFUL):
        return False, UNLAWFUL_REFUSAL

    if any(p.search(question) for p in USER_INJECTION):
        return False, INJECTION_REFUSAL

    if any(p.search(question) for p in PERSONALISED_ADVICE):
        return False, PERSONALISED_REFUSAL

    return True, None
