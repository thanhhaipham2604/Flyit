"""Prompt-injection resistance.

Attacks arrive in the user message OR hidden inside retrieved documents.
Defences: treat all retrieved text as data (delimited, never as instructions),
strip/flag instruction-like patterns in chunks, test with an adversarial suite.

This corpus is *scraped web pages*, which makes the second vector real rather
than theoretical: anything on an ato.gov.au page - including text a third party
managed to get published there, or that survives in a comment, quote or example -
arrives in the prompt. The chunk is retrieved because it is topically relevant,
which is exactly what an attacker would arrange.

Three layers, because no single one is reliable:

1. **Delimiting.** Every passage is wrapped in explicit markers and the system
   prompt names them as untrusted data. This is the load-bearing defence.
2. **Neutralising.** Instruction-shaped lines are rewritten so they cannot read
   as a directive. Redaction is preferred over dropping the chunk: a page that
   happens to contain the words "ignore the above" in ordinary prose should
   still be able to answer a question.
3. **Flagging.** What was neutralised is returned, so the API can log it and an
   adversarial test suite can assert the pattern was caught.

Deliberately *not* done: sending passages to a classifier model. It would add a
call per query to defend against something the delimiting already handles, and a
classifier is itself promptable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fylit_rag.generation.prompts import EVIDENCE_CLOSE, EVIDENCE_OPEN

# Patterns that only appear when text is trying to steer a model. Each is
# anchored to a whole line or a clear phrase, because the cost of a false
# positive is redacting real guidance.
INJECTION_PATTERNS = (
    # Overriding earlier instructions
    re.compile(r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b"
               r"(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
               r"\b(instruction|prompt|rule|direction)s?\b", re.IGNORECASE),
    # Persona replacement
    re.compile(r"\byou are (now|no longer)\b", re.IGNORECASE),
    re.compile(r"\bact as (an?|the)\b[^.\n]{0,40}\b(assistant|model|ai|system)\b", re.IGNORECASE),
    # Fake role headers trying to open a new turn
    re.compile(r"^\s*(system|assistant|user)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*/?\s*(system|assistant|user|im_start|im_end)\s*>", re.IGNORECASE),
    # Prompt exfiltration
    re.compile(r"\b(reveal|repeat|print|show|output)\b[^.\n]{0,30}"
               r"\b(system prompt|your instructions|the prompt)\b", re.IGNORECASE),
    # Attempts to forge our own delimiters and end the data block early
    re.compile(re.escape(EVIDENCE_OPEN), re.IGNORECASE),
    re.compile(re.escape(EVIDENCE_CLOSE), re.IGNORECASE),
)

REDACTION = "[redacted: instruction-like text removed from source page]"


@dataclass
class SanitisedEvidence:
    """Passages ready for the prompt, plus what had to be neutralised."""

    block: str
    findings: list[dict] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings


def neutralise(text: str) -> tuple[str, list[str]]:
    """Redact instruction-shaped spans. Returns (text, what was matched).

    Redaction rather than deletion of the whole chunk: the surrounding sentences
    are usually legitimate ATO guidance, and dropping them would let an attacker
    censor a page by planting one phrase in it.
    """
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        def _record(m):
            matched.append(m.group(0)[:120])
            return REDACTION

        text = pattern.sub(_record, text)
    return text, matched


def sanitise_evidence(chunks) -> SanitisedEvidence:
    """Delimit and neutralise instruction-like content in retrieved chunks.

    Accepts anything with `.result` (a hybrid `FusedResult`) or a bare
    `SearchResult`, so retrieval can be reranked or not without changing this.
    """
    findings: list[dict] = []
    blocks: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        result = getattr(chunk, "result", chunk)
        text, matched = neutralise(result.text or "")
        title, _ = neutralise(result.source_title or "")

        if matched:
            findings.append(
                {
                    "chunk_id": result.chunk_id,
                    "source_url": result.source_url,
                    "patterns": matched,
                }
            )

        blocks.append(
            f"{EVIDENCE_OPEN} id={i} title={title!r} url={result.source_url!r}\n"
            f"{text.strip()}\n"
            f"{EVIDENCE_CLOSE}"
        )

    return SanitisedEvidence(block="\n\n".join(blocks), findings=findings)
