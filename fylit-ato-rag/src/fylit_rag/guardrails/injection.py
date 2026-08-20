"""Prompt-injection resistance.

Attacks arrive in the user message OR hidden inside retrieved documents.
Defences: treat all retrieved text as data (delimited, never as instructions),
strip/flag instruction-like patterns in chunks, test with an adversarial suite.
"""


def sanitise_evidence(chunks):
    """TODO: delimit + neutralise instruction-like content in retrieved chunks."""
    raise NotImplementedError
