"""Apply the schema in `schema.py` to Postgres. Idempotent by design.

Run it as often as you like - on every `docker compose up`, at the top of an
ingest, from a test fixture. Every statement is CREATE ... IF NOT EXISTS, so it
creates what is missing and leaves what exists alone. `--recreate` is the
destructive escape hatch, and says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg

from fylit_rag.config import settings
from fylit_rag.indexing.schema import ddl_statements, dimensions_for


@dataclass
class BootstrapReport:
    table: str
    created: bool
    vector_size: int

    def summary(self) -> str:
        state = "created" if self.created else "already present"
        return f"Postgres {self.table}: {state} (embedding dim={self.vector_size}, cosine/HNSW)"


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def wait_for_postgres(*, attempts: int = 30, delay: float = 2.0) -> None:
    """Poll until Postgres answers a real query.

    The container accepts TCP connections well before it will serve SQL, so we
    retry an actual round trip rather than trusting an open port.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with connect() as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready yet"
            last = exc
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(
        f"Postgres never became ready after {attempts} attempts "
        f"({attempts * delay:.0f}s). Last error: {last}"
    )


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    return row is not None and row[0] is not None


def _assert_vector_size(conn: psycopg.Connection, table: str, dim: int) -> None:
    """Catch the silent killer: EMBEDDING_MODEL changed, table didn't.

    Postgres would reject the mismatched insert anyway, but it would do it deep
    inside the first ingest batch. Better to say so up front, and name the fix.
    """
    row = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding'
        """,
        (table,),
    ).fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return  # dimension not recorded; let the first insert be the judge
    existing = row[0]
    if existing != dim:
        raise RuntimeError(
            f"Table {table!r} holds {existing}-d vectors but "
            f"{settings.embedding_model!r} produces {dim}-d. Either restore the old "
            f"EMBEDDING_MODEL or re-index from scratch: "
            f"python scripts/bootstrap.py --recreate"
        )


def bootstrap(*, recreate: bool = False) -> BootstrapReport:
    """Bring the database up to the schema in `schema.py`."""
    dim = dimensions_for(settings.embedding_model)
    table = settings.chunks_table

    wait_for_postgres()
    with connect() as conn:
        if recreate:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        existed = _table_exists(conn, table)
        if existed:
            _assert_vector_size(conn, table, dim)

        # Table name comes from our own settings, never from user input, so
        # interpolating it here is safe - psycopg cannot parameterise identifiers.
        for statement in ddl_statements(table, dim):
            conn.execute(statement)  # type: ignore[arg-type]
        conn.commit()

    return BootstrapReport(table=table, created=not existed, vector_size=dim)
