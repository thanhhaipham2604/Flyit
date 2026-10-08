"""FastAPI application factory.

OpenAPI docs are served at /docs (a hand-in requirement).

/health deliberately does not touch Postgres or OpenAI. A liveness probe that
depends on a database restarts the pod when the database is slow, which is the
opposite of helpful; /health says "this process is up", and /ready says whether
it can actually serve.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from slowapi.errors import RateLimitExceeded

from fylit_rag.api.rate_limit import limiter, rate_limit_handler
from fylit_rag.api.routes import router

app = FastAPI(
    title="Fylit ATO RAG API",
    version="0.1.0",
    description=(
        "General Australian tax information grounded in official ATO content. "
        "Questions may be asked in any language; answers follow the question's "
        "language when the retrieved ATO evidence supports an answer. "
        "Not personal tax advice."
    ),
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_interface() -> str:
        """Serve the small browser interface for local demos and testing."""
        return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fylit ATO RAG</title>
    <style>
        :root { color-scheme: light; font-family: Georgia, serif; --ink: #17211b; --muted: #5f6e64; --paper: #f6f3eb; --line: #d6ddd5; --accent: #176b57; }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; color: var(--ink); background: radial-gradient(circle at 15% 0%, #e1eee5, transparent 35%), var(--paper); }
        main { width: min(900px, calc(100% - 32px)); margin: 0 auto; padding: 64px 0; }
        header { margin-bottom: 32px; }
        .kicker { color: var(--accent); font: 700 12px/1.2 Arial, sans-serif; letter-spacing: .12em; text-transform: uppercase; }
        h1 { max-width: 650px; margin: 12px 0; font-size: clamp(38px, 7vw, 72px); line-height: .98; font-weight: 500; }
        .intro { max-width: 600px; color: var(--muted); font: 18px/1.5 Arial, sans-serif; }
        form, .answer { border: 1px solid var(--line); background: rgba(255,255,255,.68); box-shadow: 0 18px 45px rgba(23,33,27,.08); }
        form { padding: 20px; }
        label { display: block; margin-bottom: 8px; font: 700 13px Arial, sans-serif; }
        textarea { width: 100%; min-height: 150px; resize: vertical; border: 1px solid var(--line); padding: 14px; color: var(--ink); background: #fff; font: 17px/1.45 Arial, sans-serif; }
        .actions { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 14px; }
        button { border: 0; padding: 12px 18px; color: #fff; background: var(--accent); font: 700 14px Arial, sans-serif; cursor: pointer; }
        button:disabled { opacity: .55; cursor: wait; }
        .hint { color: var(--muted); font: 13px/1.4 Arial, sans-serif; }
        .answer { display: none; margin-top: 24px; padding: 24px; white-space: pre-wrap; font: 17px/1.6 Arial, sans-serif; }
        .answer.visible { display: block; }
        .error { color: #9b3d2d; }
        @media (max-width: 600px) { main { padding: 36px 0; } .actions { align-items: flex-start; flex-direction: column; } }
    </style>
</head>
<body>
    <main>
        <header>
            <div class="kicker">Fylit / ATO guidance</div>
            <h1>Ask the ATO corpus in your language.</h1>
            <p class="intro">Write your question in any language. Answers are grounded in retrieved Australian Taxation Office content and stay general, not personal tax advice.</p>
        </header>
        <form id="ask-form">
            <label for="question">Your question</label>
            <textarea id="question" required minlength="3" placeholder="For example: ¿Qué debo saber sobre la reforma del IBOR?"></textarea>
            <div class="actions">
                <span class="hint">The answer uses the same language as your question when the evidence supports it.</span>
                <button type="submit">Ask question</button>
            </div>
        </form>
        <section id="answer" class="answer" aria-live="polite"></section>
    </main>
    <script>
        const form = document.getElementById('ask-form');
        const question = document.getElementById('question');
        const answer = document.getElementById('answer');
        const button = form.querySelector('button');
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            button.disabled = true;
            answer.className = 'answer visible';
            answer.textContent = 'Looking up official ATO guidance...';
            try {
                const response = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: question.value}) });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'The request could not be completed.');
                answer.textContent = data.answer;
            } catch (error) {
                answer.className = 'answer visible error';
                answer.textContent = error.message;
            } finally { button.disabled = false; }
        });
    </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict:
    """Liveness: the process is running. No dependencies checked on purpose."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: can this instance actually answer a question?

    Checks that the chunks table exists and holds embedded content, because an
    instance pointed at an empty index will return nothing but refusals - which
    looks like a working service and is not one.
    """
    try:
        from fylit_rag.config import settings
        from fylit_rag.indexing.bootstrap import connect

        with connect() as conn:
            embedded = conn.execute(
                f"SELECT count(*) FROM {settings.chunks_table} WHERE embedding IS NOT NULL"
            ).fetchone()[0]
    except Exception as exc:  # noqa: BLE001 - a readiness probe reports, it does not raise
        return {"status": "not ready", "reason": type(exc).__name__}

    if not embedded:
        return {"status": "not ready", "reason": "index is empty", "embedded_chunks": 0}
    return {"status": "ready", "embedded_chunks": embedded}
