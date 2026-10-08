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

# Offered in the year filter, newest first. Only real Australian financial years
# with meaningful coverage in the index - the `financial_year` column also holds
# junk pairs such as "2025-30", because `ingestion.pipeline.extract_financial_years`
# matches any YYYY-NN and never checks the halves are consecutive. Offering those
# would hand users filters that quietly match almost nothing.
FILTER_YEARS = (
    "2026-27",
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
    "2019-20",
    "2018-19",
    "2017-18",
    "2016-17",
    "2015-16",
)

_YEAR_OPTIONS = "\n".join(
    f'                    <option value="{year}">{year}</option>' for year in FILTER_YEARS
)

_PAGE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fylit ATO RAG</title>
    <style>
        :root {
            color-scheme: light;
            font-family: Georgia, serif;
            --ink: #17211b; --muted: #5f6e64; --paper: #f6f3eb;
            --line: #d6ddd5; --accent: #176b57; --accent-soft: #e6f1ec;
        }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; color: var(--ink);
               background: radial-gradient(circle at 15% 0%, #e1eee5, transparent 35%), var(--paper); }
        main { width: min(900px, calc(100% - 32px)); margin: 0 auto; padding: 56px 0 80px; }
        header { margin-bottom: 28px; }
        .kicker { color: var(--accent); font: 700 12px/1.2 Arial, sans-serif;
                  letter-spacing: .12em; text-transform: uppercase; }
        h1 { max-width: 650px; margin: 12px 0; font-size: clamp(34px, 6vw, 60px);
             line-height: 1; font-weight: 500; }
        .intro { max-width: 620px; color: var(--muted); font: 18px/1.5 Arial, sans-serif; }
        form, .answer { border: 1px solid var(--line); background: rgba(255,255,255,.72);
                        box-shadow: 0 18px 45px rgba(23,33,27,.08); }
        form { padding: 20px; display: grid; gap: 16px; }
        label { display: block; margin-bottom: 8px; font: 700 13px Arial, sans-serif; }
        textarea, select { border: 1px solid var(--line); padding: 12px; color: var(--ink);
                           background: #fff; font: 17px/1.45 Arial, sans-serif; width: 100%; }
        textarea { min-height: 140px; resize: vertical; }
        select { max-width: 260px; font-size: 15px; }
        .field-hint { margin: 6px 0 0; color: var(--muted); font: 13px/1.45 Arial, sans-serif; }
        .actions { display: flex; align-items: center; justify-content: space-between;
                   gap: 16px; flex-wrap: wrap; }
        button { border: 0; padding: 12px 18px; color: #fff; background: var(--accent);
                 font: 700 14px Arial, sans-serif; cursor: pointer; }
        button:disabled { opacity: .55; cursor: wait; }
        :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .hint { color: var(--muted); font: 13px/1.4 Arial, sans-serif; }
        .answer { display: none; margin-top: 24px; padding: 24px 24px 20px;
                  font: 17px/1.6 Arial, sans-serif; }
        .answer.visible { display: block; }
        .answer-text { margin: 0; white-space: pre-wrap; }
        .scope { display: inline-block; margin: 0 0 14px; padding: 4px 10px;
                 background: var(--accent-soft); color: var(--accent);
                 font: 700 12px/1.5 Arial, sans-serif; letter-spacing: .04em; }
        .resources { margin: 22px 0 0; padding: 18px 0 0; border-top: 1px solid var(--line); }
        .resources h2 { margin: 0 0 10px; font: 700 12px/1.2 Arial, sans-serif;
                        letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
        .resources ul { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
        .resources a { color: var(--ink); font: 15px/1.45 Arial, sans-serif;
                       text-decoration-color: var(--line); text-underline-offset: 3px;
                       overflow-wrap: anywhere; }
        .resources a:hover { text-decoration-color: var(--accent); }
        .disclaimer { margin: 18px 0 0; color: var(--muted); font: 13px/1.5 Arial, sans-serif; }
        .error { color: #9b3d2d; }
        @media (max-width: 600px) {
            main { padding: 32px 0 60px; }
            .actions { align-items: flex-start; flex-direction: column; }
            select { max-width: 100%; }
        }
    </style>
</head>
<body>
    <main>
        <header>
            <div class="kicker">Fylit / ATO guidance</div>
            <h1>Ask a question about Australian tax.</h1>
            <p class="intro">Answers come from official Australian Taxation Office content, with
            links to the pages they were drawn from. General information only, not personal tax
            advice.</p>
        </header>
        <form id="ask-form">
            <div>
                <label for="question">Your question</label>
                <textarea id="question" name="question" required minlength="3" maxlength="2000"
                          placeholder="For example: What can I claim for working from home?"></textarea>
            </div>
            <div>
                <label for="financial-year">Financial year</label>
                <select id="financial-year" name="financial_year">
                    <option value="">All years</option>
__YEAR_OPTIONS__
                </select>
                <p class="field-hint">Choosing a year narrows year-specific pages. Guidance that
                applies to every year, which is most of the corpus, is always included.</p>
            </div>
            <div class="actions">
                <span class="hint">Answers are written only from the ATO pages retrieved for your
                question.</span>
                <button type="submit">Ask question</button>
            </div>
        </form>
        <section id="answer" class="answer" aria-live="polite"></section>
    </main>
    <script>
        var form = document.getElementById('ask-form');
        var question = document.getElementById('question');
        var yearField = document.getElementById('financial-year');
        var panel = document.getElementById('answer');
        var button = form.querySelector('button');

        function show(text, isError) {
            panel.className = isError ? 'answer visible error' : 'answer visible';
            panel.textContent = text;
        }

        function safeHref(url) {
            return typeof url === 'string' && /^https?:\\/\\//i.test(url) ? url : null;
        }

        function render(data, year) {
            panel.className = 'answer visible';
            panel.textContent = '';

            if (year) {
                var scope = document.createElement('p');
                scope.className = 'scope';
                scope.textContent = 'Financial year ' + year;
                panel.appendChild(scope);
            }

            var body = document.createElement('p');
            body.className = 'answer-text';
            body.textContent = data.answer || '';
            panel.appendChild(body);

            var resources = (data.useful_resources || []).filter(function (item) {
                return safeHref(item && item.url);
            });
            if (resources.length) {
                var section = document.createElement('div');
                section.className = 'resources';
                var heading = document.createElement('h2');
                heading.textContent = 'Useful resources';
                section.appendChild(heading);

                var list = document.createElement('ul');
                resources.forEach(function (item) {
                    var row = document.createElement('li');
                    var link = document.createElement('a');
                    link.href = safeHref(item.url);
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = item.title || item.url;
                    row.appendChild(link);
                    list.appendChild(row);
                });
                section.appendChild(list);
                panel.appendChild(section);
            }

            // The answer already carries the disclaimer, so only add it when it
            // is genuinely missing - printing it twice reads as a glitch.
            var disclaimer = data.disclaimer || '';
            if (disclaimer && (data.answer || '').indexOf(disclaimer) === -1) {
                var note = document.createElement('p');
                note.className = 'disclaimer';
                note.textContent = disclaimer;
                panel.appendChild(note);
            }
        }

        form.addEventListener('submit', async function (event) {
            event.preventDefault();
            button.disabled = true;
            var year = yearField.value;
            show('Looking up official ATO guidance...', false);

            var payload = { question: question.value };
            if (year) { payload.financial_year = year; }

            try {
                var response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                var data = await response.json();
                if (!response.ok && !data.answer) {
                    throw new Error(typeof data.detail === 'string'
                        ? data.detail
                        : 'The request could not be completed.');
                }
                render(data, year);
            } catch (error) {
                show(error.message, true);
            } finally {
                button.disabled = false;
            }
        });
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_interface() -> str:
    """Serve the small browser interface for local demos and testing."""
    return _PAGE.replace("__YEAR_OPTIONS__", _YEAR_OPTIONS)


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
