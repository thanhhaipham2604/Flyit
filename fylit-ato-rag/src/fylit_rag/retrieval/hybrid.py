"""Hybrid retrieval: vector + keyword search blended (e.g. RRF).

Always filtered by financial year, version, and active status.
Every result keeps source title, URL and document version attached -
citations depend on it.

Why reciprocal rank fusion rather than adding the two scores: cosine similarity
and ts_rank_cd are not on the same scale and their distributions move with the
query. A question phrased in ATO's own words produces high keyword ranks; a
question phrased in a taxpayer's words produces none at all. Fusing *ranks*
sidesteps the calibration problem entirely - a chunk that both halves put near
the top wins, and neither half can dominate by having larger numbers.

    score(chunk) = sum over halves of 1 / (RRF_K + rank_in_that_half)

RRF_K = 60 is the value from Cormack et al. (2009), which introduced the method
and found it robust across collections without per-collection tuning. It damps
the difference between ranks 1 and 2 so a single confident half cannot outvote
agreement between both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fylit_rag.indexing.bootstrap import connect
from fylit_rag.indexing.keyword_index import KeywordIndex
from fylit_rag.indexing.search import SearchResult
from fylit_rag.indexing.vector_index import VectorIndex

# Cormack et al. (2009). Not tuned here; tuning it needs the labelled question
# set that scripts/evaluate.py is waiting for.
RRF_K = 60

# Each half is asked for more than the caller wants, so fusion has room to move
# a chunk up on the strength of agreement. Costs one larger scan, no extra round
# trips.
CANDIDATE_MULTIPLIER = 3


@dataclass(slots=True)
class FusedResult:
    """A chunk with its fused score and where it came from.

    `sources` is kept because it is the first thing to look at when retrieval
    misbehaves: a shortlist that is entirely 'vector' means the keyword half
    matched nothing, which is usually a query-phrasing problem rather than an
    index problem.
    """

    result: SearchResult
    score: float
    sources: dict[str, int] = field(default_factory=dict)
    # Set by `retrieval.rerank`; kept separate from `score` so the fused score
    # that produced the shortlist is still visible after reranking.
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.result.chunk_id

    def citation(self) -> dict:
        return self.result.citation()


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchResult]],
    k: int = RRF_K,
) -> list[FusedResult]:
    """Blend several ranked lists into one, deduping by chunk_id.

    Pure and connection-free so the ranking behaviour can be tested without a
    database - which matters, because this is the part most likely to be quietly
    wrong.
    """
    fused: dict[str, FusedResult] = {}

    for retriever, results in ranked_lists.items():
        for rank, result in enumerate(results, start=1):
            entry = fused.get(result.chunk_id)
            if entry is None:
                entry = FusedResult(result=result, score=0.0)
                fused[result.chunk_id] = entry
            entry.score += 1.0 / (k + rank)
            entry.sources[retriever] = rank

    # Ties broken by chunk_id so the same query always returns the same order -
    # an unstable shortlist makes evaluation numbers impossible to compare.
    return sorted(fused.values(), key=lambda f: (-f.score, f.chunk_id))


def retrieve(
    query: str,
    top_k: int = 20,
    filters: dict | None = None,
    *,
    conn=None,
    embed=None,
) -> list[FusedResult]:
    """Run both searches, blend with reciprocal rank fusion, dedupe by chunk.

    `conn` and `embed` are injectable so tests and the API can supply their own
    connection or a stubbed embedder; by default this opens a connection and
    uses the real embedding client.
    """
    if not query or not query.strip():
        return []

    if embed is None:
        from fylit_rag.indexing.embeddings import embed_texts

        embed = embed_texts

    pool = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    query_vector = embed([query])[0]

    def _search(connection):
        return {
            "vector": VectorIndex(connection).search(query_vector, pool, filters),
            "keyword": KeywordIndex(connection).search(query, pool, filters),
        }

    if conn is not None:
        ranked = _search(conn)
    else:
        with connect() as connection:
            ranked = _search(connection)

    return reciprocal_rank_fusion(ranked)[:top_k]
