"""The chunk schema - one definition, one table, both retrieval halves.

A chunk is the unit of both search AND citation. Postgres holds it once and
serves both halves of hybrid retrieval from the same row: `embedding` (pgvector,
HNSW) for semantic similarity, `search_vector` (tsvector, GIN) for exact words
and phrases. See ADR-0002.

Two columns are GENERATED - the database derives them, nothing writes them:

- `active`        = (status = 'active'), so the hot retrieval filter is a cheap
                    indexed boolean that cannot drift from `status`.
- `search_vector` = to_tsvector(text), so the keyword index can never go stale
                    against the text it indexes.

Stdlib only, deliberately: this module is imported by tests that must run
before anyone has a container up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

# Fields retrieval filters on. Each needs a real index or the filters quietly
# degrade into sequential scans - fine on 200 chunks, not on 200k.
FILTERABLE_FIELDS = ("doc_id", "financial_year", "version", "status", "active")

# Postgres computes these. Including them in an INSERT is an error, which is
# exactly the guarantee we want: the derivation has one owner, the database.
GENERATED_COLUMNS = ("active", "search_vector")

# pgvector indexes HNSW/IVFFlat only up to 2000 dimensions (the `vector` column
# type itself goes far higher). Above that there is no ANN index to build, so a
# large-embedding model would silently fall back to sequential scan.
MAX_INDEXABLE_DIMENSIONS = 2000

# Vector width is a property of the embedding model, and here it is baked into
# the column type - vector(1536) - so switching models without recreating the
# table is a mistake Postgres itself will reject.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def dimensions_for(model: str) -> int:
    """Vector width for an embedding model, or a loud error naming the fix."""
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        known = ", ".join(sorted(EMBEDDING_DIMENSIONS))
        raise ValueError(
            f"Unknown embedding model {model!r}; add its width to "
            f"EMBEDDING_DIMENSIONS. Known: {known}"
        ) from None


class Status(StrEnum):
    """Lifecycle of a document's chunks.

    `superseded` is distinct from `deleted` on purpose: last year's guidance
    still exists and is still citable for last year, it just must not surface
    as current advice.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ChunkRecord:
    """One indexed chunk, as the `chunks` table holds it."""

    # identity - chunk_id is the primary key and must be deterministic
    # (doc_id + ordinal), so re-ingesting an unchanged document overwrites via
    # ON CONFLICT rather than duplicating.
    chunk_id: str
    doc_id: str
    chunk_ordinal: int
    content_hash: str

    # content
    text: str
    heading_path: list[str] = field(default_factory=list)

    # citation - these two are what the API hands back as Useful Resources
    source_title: str = ""
    source_url: str = ""

    # filters
    financial_year: str | None = None
    version: int = 1
    status: Status = Status.ACTIVE
    superseded_by: str | None = None
    indexed_at: str = field(default_factory=_now)

    @property
    def active(self) -> bool:
        """Mirrors the generated column, for code that has a record in hand and
        doesn't want a round trip. Derived here and in Postgres from the same
        rule; `status` remains the only writer in both.
        """
        return self.status is Status.ACTIVE

    def as_record(self) -> dict:
        """The full logical view of a chunk, generated columns included.

        For assertions, diagnostics and tests. To write a row, use `as_row()`.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_ordinal": self.chunk_ordinal,
            "content_hash": self.content_hash,
            "text": self.text,
            "heading_path": self.heading_path,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "financial_year": self.financial_year,
            "version": self.version,
            "status": str(self.status),
            "active": self.active,
            "superseded_by": self.superseded_by,
            "indexed_at": self.indexed_at,
        }

    def as_row(self) -> dict:
        """The columns an INSERT may actually set - generated ones removed.

        `embedding` is deliberately absent: it is attached at index time by
        `indexing.embeddings`, not carried on the record.
        """
        return {k: v for k, v in self.as_record().items() if k not in GENERATED_COLUMNS}


# `text` is a type name as well as our column name, so it is quoted everywhere
# it appears in an expression - unquoted it is legal but reads ambiguously.
_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id        text PRIMARY KEY,
    doc_id          text NOT NULL,
    chunk_ordinal   integer NOT NULL,
    content_hash    text NOT NULL,
    "text"          text NOT NULL,
    heading_path    text[] NOT NULL DEFAULT ARRAY[]::text[],
    source_title    text NOT NULL DEFAULT '',
    source_url      text NOT NULL DEFAULT '',
    financial_year  text,
    version         integer NOT NULL DEFAULT 1,
    status          text NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'superseded', 'deleted')),
    superseded_by   text,
    indexed_at      timestamptz NOT NULL DEFAULT now(),
    embedding       vector({dim}),
    active          boolean GENERATED ALWAYS AS (status = 'active') STORED,
    search_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', "text")) STORED
)
"""

# Covers the exact filter shape retrieval uses: active content, for a financial
# year, at an index version. doc_id gets its own index for delete-by-document.
_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS {table}_doc_id_idx ON {table} (doc_id)",
    "CREATE INDEX IF NOT EXISTS {table}_filters_idx ON {table} (financial_year, version, active)",
    "CREATE INDEX IF NOT EXISTS {table}_search_idx ON {table} USING gin (search_vector)",
    (
        "CREATE INDEX IF NOT EXISTS {table}_embedding_idx "
        "ON {table} USING hnsw (embedding vector_cosine_ops)"
    ),
)


def ddl_statements(table: str, dim: int) -> list[str]:
    """The whole schema, as individually executable statements.

    Every statement is IF NOT EXISTS, so applying this to a live database is a
    no-op rather than an error - that is what makes bootstrap re-runnable.
    """
    if dim > MAX_INDEXABLE_DIMENSIONS:
        raise ValueError(
            f"{dim}-d vectors exceed pgvector's {MAX_INDEXABLE_DIMENSIONS}-d limit for "
            f"HNSW indexes; retrieval would fall back to a sequential scan. Use a "
            f"smaller embedding model, or reduce dimensions at embedding time."
        )
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        _TABLE_DDL.format(table=table, dim=dim).strip(),
        *(stmt.format(table=table) for stmt in _INDEX_DDL),
    ]
