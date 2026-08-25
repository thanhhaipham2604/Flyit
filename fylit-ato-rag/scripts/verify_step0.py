"""Verify Step 0: the container is up and the schema behaves as designed.

Usage: python scripts/verify_step0.py [--start]

Checks the half that unit tests cannot reach - that Postgres actually accepts
the schema and enforces its guarantees. Every probe that needs rows runs inside
a transaction that is rolled back, so this is safe to run against a populated
database: it never deletes anything.

Exit code is 0 only if every check passes, so it works in CI or a pre-demo
sanity pass.
"""

from __future__ import annotations

import subprocess
import sys
import time

import psycopg
import typer

from fylit_rag.config import settings
from fylit_rag.indexing.bootstrap import bootstrap
from fylit_rag.indexing.schema import YEAR_FILTER_SQL, ChunkRecord, Status, dimensions_for

app = typer.Typer(help="End-to-end verification of the Step 0 storage layer")

EXPECTED_COLUMNS = {
    "chunk_id", "doc_id", "chunk_ordinal", "content_hash", "text", "heading_path",
    "source_title", "source_url", "financial_year", "version", "status",
    "superseded_by", "indexed_at", "embedding", "active", "search_vector",
}
EXPECTED_INDEXES = {
    "chunks_pkey", "chunks_doc_id_idx", "chunks_year_idx",
    "chunks_filters_idx", "chunks_search_idx", "chunks_embedding_idx",
}

results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Run a probe, record pass/fail, never let one failure hide the rest."""
    def wrap(fn):
        try:
            results.append((name, True, fn()))
        except Exception as exc:  # noqa: BLE001 - a failed probe is a result, not a crash
            results.append((name, False, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"))
        return fn
    return wrap


def _sh(*cmd: str) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    return out.stdout.strip()


def wait_healthy(timeout: float = 90.0) -> str:
    """Poll compose until the healthcheck passes.

    `docker compose up -d` returns once the container has *started*, which is
    seconds before Postgres will answer anything - the compose healthcheck alone
    has a 10s start_period. Reading the status immediately just catches
    'health: starting', so poll it the same way bootstrap polls the database.
    """
    deadline = time.monotonic() + timeout
    line = ""
    while True:
        out = _sh("docker", "compose", "ps", "--format", "{{.Service}} {{.Status}}")
        line = next((x for x in out.splitlines() if x.startswith("postgres")), "")
        if "healthy" in line:
            return line.split(" ", 1)[1]
        if time.monotonic() >= deadline:
            raise RuntimeError(f"not healthy after {timeout:.0f}s: {line or 'no container'}")
        time.sleep(2)


def sample_rows() -> list[ChunkRecord]:
    """Four chunks covering every filter branch that matters."""
    return [
        ChunkRecord(chunk_id="_v_evergreen", doc_id="_v", chunk_ordinal=1, content_hash="h1",
                    text="Capital gains tax applies when you sell an asset.", financial_year=[]),
        ChunkRecord(chunk_id="_v_fy2324", doc_id="_v", chunk_ordinal=2, content_hash="h2",
                    text="The tax-free threshold is $18,200.", financial_year=["2023-24"]),
        ChunkRecord(chunk_id="_v_multi", doc_id="_v", chunk_ordinal=3, content_hash="h3",
                    text="Historical rates table.", financial_year=["2023-24", "2024-25"]),
        ChunkRecord(chunk_id="_v_old", doc_id="_v", chunk_ordinal=4, content_hash="h4",
                    text="Superseded guidance.", financial_year=["2024-25"],
                    status=Status.SUPERSEDED),
    ]


def insert(conn: psycopg.Connection, rows: list[ChunkRecord]) -> None:
    for r in rows:
        d = r.as_row()
        cols = ", ".join(f'"{c}"' for c in d)
        marks = ", ".join(["%s"] * len(d))
        conn.execute(
            f"INSERT INTO {settings.chunks_table} ({cols}) VALUES ({marks})", list(d.values())
        )


@app.command()
def run(
    start: bool = typer.Option(False, "--start", help="Start the postgres container if it is down"),
) -> None:
    dim = dimensions_for(settings.embedding_model)
    table = settings.chunks_table

    if start:
        typer.echo("starting postgres ...")
        subprocess.run(["docker", "compose", "up", "-d", "postgres"], timeout=300, check=False)

    @check("docker daemon reachable")
    def _():
        v = _sh("docker", "version", "--format", "{{.Server.Version}}")
        if not v:
            raise RuntimeError("no docker server - is Docker Desktop running?")
        return f"server {v}"

    @check("postgres container healthy")
    def _():
        return wait_healthy()

    @check("bootstrap is idempotent")
    def _():
        first, second = bootstrap(), bootstrap()
        if second.created:
            raise RuntimeError("second run reported 'created' - not idempotent")
        return f"1st={'created' if first.created else 'present'}, 2nd=present"

    with psycopg.connect(settings.database_url) as conn:

        @check("pgvector extension installed")
        def _():
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
            if not row:
                raise RuntimeError("extension 'vector' missing")
            return f"v{row[0]}"

        @check("all columns present")
        def _():
            found = {r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))}
            missing = EXPECTED_COLUMNS - found
            extra = found - EXPECTED_COLUMNS
            if missing or extra:
                raise RuntimeError(f"missing={sorted(missing)} unexpected={sorted(extra)}")
            return f"{len(found)} columns"

        @check("all indexes present")
        def _():
            found = {r[0] for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename=%s", (table,))}
            missing = EXPECTED_INDEXES - found
            if missing:
                raise RuntimeError(f"missing indexes: {sorted(missing)}")
            return f"{len(EXPECTED_INDEXES)} indexes (hnsw, 2x gin, 2x btree, pk)"

        @check("embedding column has the model's width")
        def _():
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid=to_regclass(%s) "
                "AND attname='embedding'", (table,)).fetchone()
            if row[0] != dim:
                raise RuntimeError(f"column is {row[0]}-d, {settings.embedding_model} is {dim}-d")
            return f"{dim}-d matches {settings.embedding_model}"

        @check("generated columns reject direct writes")
        def _():
            try:
                conn.execute(
                    f"INSERT INTO {table} (chunk_id,doc_id,chunk_ordinal,content_hash,text,active)"
                    " VALUES ('_v_x','_v',9,'h','t',true)")
            except psycopg.errors.GeneratedAlways:
                conn.rollback()
                return "active is not writable, as designed"
            conn.rollback()
            raise RuntimeError("insert into 'active' succeeded - the guarantee is broken")

        # --- probes needing rows: one transaction, always rolled back ---
        @check("generated columns compute correctly")
        def _():
            insert(conn, sample_rows())
            got = dict(conn.execute(
                "SELECT chunk_id, active FROM chunks WHERE doc_id='_v'").fetchall())
            if got["_v_old"] or not got["_v_evergreen"]:
                raise RuntimeError(f"active derived wrongly: {got}")
            return "active mirrors status; tsvector populated"

        @check("year filter keeps evergreen content visible")
        def _():
            q = f"SELECT chunk_id FROM chunks WHERE doc_id='_v' AND active AND {YEAR_FILTER_SQL}"
            got = {r[0] for r in conn.execute(q, ("2023-24",))}
            want = {"_v_evergreen", "_v_fy2324", "_v_multi"}
            if got != want:
                raise RuntimeError(f"got {sorted(got)}, want {sorted(want)}")
            return "evergreen + tagged matched; superseded excluded"

        @check("keyword search returns ranked results")
        def _():
            rows = conn.execute(
                "SELECT chunk_id, ts_rank(search_vector, plainto_tsquery('english','capital gains'))"
                " FROM chunks WHERE doc_id='_v' AND search_vector @@ "
                "plainto_tsquery('english','capital gains') ORDER BY 2 DESC").fetchall()
            if not rows:
                raise RuntimeError("no full-text match for 'capital gains'")
            return f"{rows[0][0]} rank={rows[0][1]:.4f}"

        @check("vector similarity search works")
        def _():
            vec = "[" + ",".join(["0.1"] * dim) + "]"
            conn.execute("UPDATE chunks SET embedding=%s::vector WHERE chunk_id='_v_evergreen'", (vec,))
            row = conn.execute(
                "SELECT chunk_id, embedding <=> %s::vector AS d FROM chunks "
                "WHERE doc_id='_v' AND embedding IS NOT NULL ORDER BY d LIMIT 1", (vec,)).fetchone()
            if row is None:
                raise RuntimeError("cosine query returned nothing")
            return f"nearest={row[0]} cosine_distance={row[1]:.6f}"

        conn.rollback()  # nothing this script inserted survives

        @check("verification left no rows behind")
        def _():
            n = conn.execute("SELECT count(*) FROM chunks WHERE doc_id='_v'").fetchone()[0]
            if n:
                raise RuntimeError(f"{n} test rows leaked")
            return "transaction rolled back cleanly"

    width = max(len(n) for n, _, _ in results)
    typer.echo("")
    for name, ok, detail in results:
        mark = typer.style("PASS", fg="green") if ok else typer.style("FAIL", fg="red")
        typer.echo(f"  {mark}  {name.ljust(width)}  {detail}")

    failed = [n for n, ok, _ in results if not ok]
    typer.echo("")
    if failed:
        typer.echo(typer.style(f"{len(failed)} of {len(results)} checks FAILED", fg="red"))
        sys.exit(1)
    typer.echo(typer.style(f"all {len(results)} checks passed - Step 0 is sound", fg="green"))


if __name__ == "__main__":
    app()
