"""Grounding checks: every factual claim must trace to a retrieved passage.

Also decides confidence: if top evidence is too weak/thin, trigger the
refusal path BEFORE generation (the red branch in the architecture diagram).

Refusing before the call, rather than after, matters for three reasons: it is
free, it is deterministic, and it removes the temptation entirely - a model
handed weak evidence and asked to be helpful will often produce something
plausible, and detecting that afterwards is much harder than not asking.

The signal used is **vector cosine similarity**, not the fused RRF score. RRF
scores are ranks in disguise: the top result of a hopeless query gets almost the
same fused score as the top result of a perfect one, because rank 1 is rank 1
either way. Cosine similarity is absolute and comparable across queries, which
is what a threshold needs. `ts_rank_cd` is unbounded and corpus-dependent, so
keyword-only hits contribute to coverage but not to the threshold test.
"""

from __future__ import annotations

# Below this, the best semantic match is not really about the question.
#
# Calibrated with `python scripts/evaluate.py calibrate`, which measures the top
# similarity for answerable questions against deliberately off-topic ones. The
# two populations separate cleanly:
#
#     answerable (n=60):  min 0.569   p5 0.610   median 0.706
#     off-topic  (n=10):  median 0.248            max 0.388
#
# Anything in [0.40, 0.56] scores 100% on both sides. 0.45 sits in that gap but
# nearer the off-topic end on purpose: the answerable questions are synthetic
# (generated *from* a chunk, so they share its vocabulary) and therefore score
# higher than a real taxpayer question would. The headroom is for that gap
# between the eval set and reality, not for the measurement.
MIN_TOP_SIMILARITY = 0.45

# How many passages must clear the similarity floor. One good passage is enough
# to answer from - the floor is what makes it "good".
#
# This was 2 measured against a *relative* margin (within 0.10 of the best
# passage), which refused questions with excellent evidence: a question whose
# top chunk scored 0.75 while the next scored 0.60 was thrown out for having
# "too few supporting passages", when in fact one chunk was simply a much better
# answer than the rest. Counting against the absolute floor instead fixed a 20%
# false-refusal rate on the eval set.
MIN_SUPPORTING = 1


def _similarity(candidate) -> float | None:
    """Vector cosine similarity for a candidate, or None if only keyword found it.

    A hybrid result carries `.result`; a bare SearchResult is accepted too, so
    this works with or without reranking in front of it.
    """
    result = getattr(candidate, "result", candidate)
    sources = getattr(candidate, "sources", None)
    if sources is not None and "vector" not in sources:
        return None
    if getattr(result, "retriever", "") == "keyword":
        return None
    return float(result.score)


def evidence_strength(evidence) -> dict:
    """Describe the shortlist: best similarity, how many passages support it.

    Returned rather than just a boolean so the API can log *why* it refused and
    the eval harness can tune the thresholds against real distributions.
    """
    similarities = [s for s in (_similarity(c) for c in evidence) if s is not None]
    if not similarities:
        return {"top": None, "supporting": 0, "n": len(list(evidence))}

    top = max(similarities)
    # Counted against the absolute floor, not relative to the best passage - see
    # the note on MIN_SUPPORTING.
    supporting = sum(1 for s in similarities if s >= MIN_TOP_SIMILARITY)
    return {"top": top, "supporting": supporting, "n": len(similarities)}


def has_sufficient_evidence(
    query: str,
    evidence,
    *,
    min_similarity: float = MIN_TOP_SIMILARITY,
    min_supporting: int = MIN_SUPPORTING,
) -> bool:
    """Is this shortlist worth generating an answer from?

    `query` is accepted for interface stability and future query-aware checks;
    the current heuristic judges the evidence alone, which keeps it free and
    deterministic.
    """
    if not evidence:
        return False

    strength = evidence_strength(evidence)
    if strength["top"] is None:
        # Keyword-only matches: the words appear somewhere, but nothing in the
        # corpus is semantically about this question. Not enough to answer from.
        return False

    return strength["top"] >= min_similarity and strength["supporting"] >= min_supporting


def verify_grounding(answer: str, evidence) -> bool:
    """Post-hoc check that the answer's specifics appear in the passages.

    Deliberately narrow: it checks **numbers**, not prose. Paraphrase detection
    would need another model call and would be wrong often enough to be noise,
    but a figure the model invented - a rate, a threshold, a dollar amount - is
    both the most damaging kind of hallucination here and cheap to catch, because
    a grounded figure must appear verbatim in the evidence.

    Returns True when every number in the answer is found in the passages.
    """
    import re

    corpus = " ".join(
        (getattr(c, "result", c).text or "") for c in evidence
    )
    # Percentages, dollar amounts and bare numbers of three digits or more:
    # small integers ("two conditions", "step 3") are prose, not claims.
    figures = set(re.findall(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d{3,}(?:,\d{3})*\b", answer or ""))
    if not figures:
        return True

    normalise = lambda s: s.replace(",", "").replace("$", "").rstrip("%")
    haystack = normalise(corpus)
    return all(normalise(f) in haystack for f in figures)
