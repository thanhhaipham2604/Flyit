"""Short conversation memory for follow-up questions.

Kept small (last N turns), used only to resolve references like "what about
for 2024?". Memory may rewrite the query - it must never override or add
to the retrieved source content.
"""


class ConversationMemory:
    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        raise NotImplementedError

    def contextualise(self, session_id: str, question: str) -> str:
        """TODO: rewrite follow-up into a standalone query."""
        raise NotImplementedError
