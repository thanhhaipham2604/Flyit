"""Short conversation memory for follow-up questions.

Kept small (last N turns), used only to resolve references like "what about
for 2024?". Memory may rewrite the query - it must never override or add
to the retrieved source content.

The separation that keeps this safe: memory rewrites the *question*, and then
plays no further part. Retrieval runs on the rewritten question, grounding
judges the passages that came back, and the answer is written from those
passages alone. An earlier turn can therefore change what we go looking for, but
it can never become evidence - which is what stops a conversation from drifting
into facts nobody retrieved.

Storage is in-process and bounded. That is honest for a single-instance
deployment and wrong for a scaled one; the note in `Consequences` below says what
to swap in.
"""

from __future__ import annotations

import re
from collections import OrderedDict, deque
from dataclasses import dataclass

# How many past turns are kept. Two is enough to resolve "what about 2024?"
# without letting a long conversation accumulate context that quietly steers
# retrieval away from what the user just asked.
MAX_TURNS = 2

# How many sessions are held before the oldest is dropped. Bounded so a public
# endpoint cannot grow memory without limit.
MAX_SESSIONS = 1000

# A question is treated as a follow-up only if it looks like one. Rewriting
# every question would be a needless LLM call and would let stale context
# contaminate a clearly self-contained query.
FOLLOWUP_MARKERS = (
    re.compile(r"^\s*(what|how) about\b", re.IGNORECASE),
    re.compile(r"^\s*(and|but|so)\b", re.IGNORECASE),
    re.compile(r"\b(it|that|this|those|them|they)\b", re.IGNORECASE),
    re.compile(r"^\s*(for|in)\s+\d{4}", re.IGNORECASE),
    re.compile(r"^\s*why\b|^\s*when\b", re.IGNORECASE),
)

# Short questions are far more likely to be elliptical than self-contained.
FOLLOWUP_MAX_WORDS = 12


@dataclass(slots=True)
class Turn:
    question: str
    answer: str


class ConversationMemory:
    """Last few turns per session, used only to make a follow-up standalone."""

    def __init__(self, max_turns: int = MAX_TURNS, max_sessions: int = MAX_SESSIONS):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[Turn]] = OrderedDict()

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a completed turn, evicting the oldest session when full."""
        if not session_id:
            return
        turns = self._sessions.get(session_id)
        if turns is None:
            turns = deque(maxlen=self.max_turns)
            self._sessions[session_id] = turns
        turns.append(Turn(question=question, answer=answer))
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def history(self, session_id: str) -> list[Turn]:
        return list(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def looks_like_followup(self, question: str) -> bool:
        """Whether this question depends on what came before."""
        if not question:
            return False
        if len(question.split()) > FOLLOWUP_MAX_WORDS:
            return False
        return any(p.search(question) for p in FOLLOWUP_MARKERS)

    def contextualise(self, session_id: str, question: str, *, rewrite=None) -> str:
        """Rewrite a follow-up into a standalone query.

        Returns the question unchanged when there is no history, or when it does
        not look like a follow-up - so a self-contained question never pays for
        a rewrite and never inherits stale context.

        `rewrite` is injectable so tests need no API. It receives the previous
        questions and the current one, and returns a standalone question.
        """
        turns = self.history(session_id)
        if not turns or not self.looks_like_followup(question):
            return question

        rewrite = rewrite or _llm_rewrite
        try:
            standalone = rewrite([t.question for t in turns], question)
        except Exception:  # noqa: BLE001 - a failed rewrite must not fail the query
            return question
        return (standalone or "").strip() or question


def _llm_rewrite(previous_questions: list[str], question: str) -> str:
    """Turn an elliptical follow-up into a standalone question."""
    from fylit_rag.config import settings
    from fylit_rag.openai_client import client

    earlier = "\n".join(f"- {q}" for q in previous_questions)
    response = client().chat.completions.create(
        model=settings.generation_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest question so it stands alone, using the "
                    "earlier questions only to resolve what it refers to. Keep it a "
                    "question about Australian tax, change nothing else, add no facts, "
                    "and answer nothing. Reply with the rewritten question only."
                ),
            },
            {"role": "user", "content": f"Earlier questions:\n{earlier}\n\nLatest: {question}"},
        ],
    )
    return response.choices[0].message.content
