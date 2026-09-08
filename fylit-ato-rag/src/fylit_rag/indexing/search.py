"""Read primitives shared by both halves of retrieval - the counterpart to `store`.

`store` writes the row; this reads it. Both halves query the same table, so the
filters live here once rather than twice: a vector search and a keyword search
that disagreed about which content is current would be a silent correctness bug,
not a performance one (ADR-0002).

The year filter is imported from `schema`, never rewritten. Three quarters of the
ATO corpus names no financial year at all - that is the evergreen guidance which
answers most questions - so `financial_year = '2023-24'` would discard the bulk
of the useful content. A chunk matches a year if it is evergreen OR tagged with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fylit_rag.config import settings
from fylit_rag.indexing.schema import YEAR_FILTER_SQL

# Everything a citation needs, plus what ranking needs. Selected by name rather
# than `*` so a schema change surfaces here instead of silently shifting tuple
# positions underneath the row mapper.
RESULT_COLUMNS = (
    "chunk_id",
    "doc_id",
    "text",
    "heading_path",
    "source_title",
    "source_url",
    "version",
    "financial_year",
    "last_updated",
)


@dataclass(slots=True)
class SearchResult:
    """One retrieved chunk, carrying everything an answer needs to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int = 0
    retriever: str = ""
    heading_path: list[str] = field(default_factory=list)
    source_title: str = ""
    source_url: str = ""
    version: int = 1
    financial_year: list[str] = field(default_factory=list)
    last_updated: date | None = None

    @classmethod
    def from_row(cls, row, score: float, rank: int, retriever: str) -> SearchResult:
        (chunk_id, doc_id, text, heading_path, source_title,
         source_url, version, financial_year, last_updated) = row
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            score=float(score),
            rank=rank,
            retriever=retriever,
            heading_path=list(heading_path or []),
            source_title=source_title or "",
            source_url=source_url or "",
            version=version,
            financial_year=list(financial_year or []),
            last_updated=last_updated,
        )

    def citation(self) -> dict:
        """The fields the API hands back as Useful Resources."""
        return {
            "title": self.source_title,
            "url": self.source_url,
            "version": self.version,
            "heading_path": self.heading_path,
            "chunk_id": self.chunk_id,
        }


def build_where(filters: dict | None = None) -> tuple[str, list]:
    """Turn a filter dict into a SQL fragment and its parameters.

    Recognised keys mirror `schema.FILTERABLE_FIELDS`:

    ``active``
        Defaults to True - retrieval answers with current guidance unless asked
        otherwise. Pass None to search superseded and deleted content too, which
        is what a question scoped to a past year needs.
    ``financial_year``
        Matches evergreen content as well as content tagged with that year.
    ``version`` / ``doc_id``
        Exact matches.

    An unknown key raises rather than being ignored: a filter that silently does
    nothing is how a query quietly returns content it should never have seen.
    """
    filters = dict(filters or {})
    known = {"active", "financial_year", "version", "doc_id"}
    unknown = set(filters) - known
    if unknown:
        raise ValueError(
            f"Unknown retrieval filter(s): {sorted(unknown)}. Known filters: {sorted(known)}"
        )

    clauses: list[str] = []
    params: list = []

    active = filters.get("active", True)
    if active is not None:
        clauses.append("active = %s")
        params.append(bool(active))

    year = filters.get("financial_year")
    if year:
        clauses.append(YEAR_FILTER_SQL)
        params.append(year)

    if filters.get("version") is not None:
        clauses.append("version = %s")
        params.append(filters["version"])

    if filters.get("doc_id"):
        clauses.append("doc_id = %s")
        params.append(filters["doc_id"])

    return (" AND ".join(clauses) if clauses else "TRUE"), params


def table_name(table: str | None = None) -> str:
    """Table names come from our own settings, never user input - psycopg cannot
    parameterise an identifier, so this is the one place it is interpolated."""
    return table or settings.chunks_table
