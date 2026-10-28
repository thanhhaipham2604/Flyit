"""FastAPI application factory.

OpenAPI docs are served at /docs (a hand-in requirement).

/health deliberately does not touch Postgres or OpenAI. A liveness probe that
depends on a database restarts the pod when the database is slow, which is the
opposite of helpful; /health says "this process is up", and /ready says whether
it can actually serve.
"""
# Import FastAPI and ratelimit package
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi.errors import RateLimitExceeded

from fylit_rag.api.rate_limit import limiter, rate_limit_handler
from fylit_rag.api.routes import router

# The FastAPI app is created at import time so that the ASGI server can find it.
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
    """Serve the chat interface for local demos and testing.

    One page, no build step and no dependencies: the whole conversation lives in
    this string so the demo works from a bare container.

    The thread scrolls and the composer stays pinned at the bottom, which is what
    makes it read as a conversation rather than a form. Each browser tab keeps
    one `session_id`, sent with every question, so the API can resolve follow-ups
    against the turns that came before.
    """
    return r"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fylit ATO RAG</title>
    <style>
        :root {
            color-scheme: light;
            font-family: Arial, sans-serif;
            --ink: #1b2821;
            --muted: #627168;
            --paper: #f6f4ed;
            --surface: rgba(255, 255, 255, .78);
            --surface-strong: #fff;
            --line: #dce2da;
            --accent: #176b57;
            --accent-hover: #105644;
            --accent-soft: #e7f0e9;
            --error: #9b3d2d;
            --error-surface: #fff0eb;
            --shadow: 0 18px 48px rgba(27, 40, 33, .08);
            --soft-shadow: 0 8px 24px rgba(27, 40, 33, .055);
        }
        :root[data-theme="dark"] {
            color-scheme: dark;
            --ink: #e8eee9;
            --muted: #a4b1a8;
            --paper: #17211c;
            --surface: rgba(34, 47, 39, .92);
            --surface-strong: #202d25;
            --line: #3b4b40;
            --accent: #76c5a4;
            --accent-hover: #92d8b9;
            --accent-soft: #263d32;
            --error: #ffb4a1;
            --error-surface: #3e2926;
            --shadow: 0 18px 48px rgba(0, 0, 0, .2);
            --soft-shadow: 0 8px 24px rgba(0, 0, 0, .14);
        }
        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) {
                color-scheme: dark;
                --ink: #e8eee9; --muted: #a4b1a8; --paper: #17211c;
                --surface: rgba(34, 47, 39, .92); --surface-strong: #202d25;
                --line: #3b4b40; --accent: #76c5a4; --accent-hover: #92d8b9;
                --accent-soft: #263d32; --error: #ffb4a1; --error-surface: #3e2926;
                --shadow: 0 18px 48px rgba(0, 0, 0, .2);
                --soft-shadow: 0 8px 24px rgba(0, 0, 0, .14);
            }
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            /* dvh, not vh: a mobile keyboard must not push the composer off screen. */
            height: 100dvh;
            display: flex;
            flex-direction: column;
            color: var(--ink);
            background: radial-gradient(ellipse at 14% 0%, rgba(211, 231, 216, .72), transparent 38%), var(--paper);
        }
        :root[data-theme="dark"] body,
        :root:not([data-theme="light"]) body { background: radial-gradient(ellipse at 14% 0%, rgba(39, 75, 57, .38), transparent 40%), var(--paper); }
        @media (prefers-color-scheme: light) {
            :root:not([data-theme="dark"]) body { background: radial-gradient(ellipse at 14% 0%, rgba(211, 231, 216, .72), transparent 38%), var(--paper); }
        }
        button, textarea { font: inherit; }
        button:focus-visible, textarea:focus-visible, a:focus-visible { outline: 3px solid color-mix(in srgb, var(--accent) 50%, transparent); outline-offset: 3px; }

        .topbar { flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between; gap: 12px; width: min(820px, calc(100% - 32px)); margin: 0 auto; padding: 16px 0 12px; }
        .brand { display: inline-flex; align-items: center; gap: 10px; color: var(--ink); font-size: 13px; font-weight: 700; letter-spacing: .025em; text-decoration: none; }
        .brand-mark { display: grid; width: 30px; height: 30px; place-items: center; border-radius: 10px; color: #fff; background: #176b57; font-family: Georgia, serif; font-size: 18px; }
        .top-actions { display: flex; align-items: center; gap: 10px; }
        .icon-button, .new-chat { min-height: 40px; border: 1px solid var(--line); border-radius: 999px; padding: 0 14px; color: var(--ink); background: var(--surface); cursor: pointer; transition: background-color .18s ease, border-color .18s ease; }
        .icon-button { width: 40px; padding: 0; font-size: 17px; }
        .icon-button:hover, .new-chat:hover { border-color: var(--accent); background: var(--accent-soft); }
        .new-chat { font-size: 13px; font-weight: 700; }
        .new-chat:disabled, .icon-button:disabled { opacity: .5; cursor: default; }

        /* The thread is the only scrolling region, so the composer stays put. */
        .thread-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; }
        .thread { width: min(820px, calc(100% - 32px)); margin: 0 auto; padding: 4px 0 28px; display: grid; gap: 22px; align-content: start; }

        .hero { padding: clamp(20px, 7vh, 64px) 0 8px; }
        .kicker { color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
        h1 { max-width: 700px; margin: 14px 0 18px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(36px, 6.4vw, 64px); font-weight: 400; letter-spacing: -.045em; line-height: 1; }
        .intro { max-width: 590px; margin: 0 0 22px; color: var(--muted); font-size: 16px; line-height: 1.7; }
        .starters { display: flex; flex-wrap: wrap; gap: 9px; }
        .starter { border: 1px solid var(--line); border-radius: 999px; padding: 9px 15px; color: var(--ink); background: var(--surface); font-size: 13px; line-height: 1.4; text-align: left; cursor: pointer; }
        .starter:hover { border-color: var(--accent); background: var(--accent-soft); }
        body.has-turns .hero { display: none; }

        .turn { display: grid; gap: 14px; animation: appear .22s ease both; }
        @keyframes appear { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .user-message { justify-self: end; max-width: min(88%, 620px); border: 1px solid var(--line); border-radius: 18px 18px 5px 18px; padding: 13px 17px; background: var(--accent-soft); font-size: 15px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
        .assistant-message { border: 1px solid var(--line); border-radius: 5px 18px 18px; padding: 21px 23px; background: var(--surface); box-shadow: var(--soft-shadow); }
        .answer-label, .sources-heading { display: block; margin: 0 0 12px; color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
        .answer-content { font-size: 15px; line-height: 1.75; overflow-wrap: anywhere; }
        .answer-content > :first-child { margin-top: 0; }
        .answer-content > :last-child { margin-bottom: 0; }
        .answer-content p { margin: 0 0 13px; }
        .answer-content h2, .answer-content h3, .answer-content h4 { margin: 20px 0 8px; font-family: Georgia, "Times New Roman", serif; line-height: 1.25; }
        .answer-content h2 { font-size: 21px; }
        .answer-content h3 { font-size: 18px; }
        .answer-content h4 { font-size: 16px; }
        .answer-content ul, .answer-content ol { margin: 9px 0 15px; padding-left: 25px; }
        .answer-content li { margin: 5px 0; padding-left: 3px; }
        .answer-content a, .resource-list a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }
        .answer-content code { border-radius: 5px; padding: 2px 5px; background: var(--accent-soft); font-family: Consolas, monospace; font-size: .9em; }
        .answer-content pre { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; padding: 13px 15px; background: var(--surface-strong); }
        .answer-content pre code { padding: 0; background: transparent; }
        .sources { margin-top: 20px; border-top: 1px solid var(--line); padding-top: 15px; }
        .sources-heading { margin-bottom: 9px; color: var(--muted); letter-spacing: .08em; }
        .resource-list { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
        .resource-list a { display: inline-flex; align-items: baseline; gap: 7px; font-size: 13px; line-height: 1.5; }
        .resource-list a::after { content: "\2197"; font-size: 11px; text-decoration: none; }
        .disclaimer { margin: 17px 0 0; border-left: 2px solid var(--line); padding-left: 11px; color: var(--muted); font-size: 11px; line-height: 1.6; }
        .loading-message { display: inline-flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
        .spinner { width: 15px; height: 15px; border: 2px solid color-mix(in srgb, var(--accent) 30%, transparent); border-top-color: var(--accent); border-radius: 50%; animation: spin .75s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .composer-dock { flex: 0 0 auto; border-top: 1px solid transparent; background: linear-gradient(to top, var(--paper) 62%, transparent); padding-bottom: env(safe-area-inset-bottom); }
        .composer { width: min(820px, calc(100% - 32px)); margin: 0 auto; padding: 12px 0 16px; }
        .composer-shell { overflow: hidden; border: 1px solid var(--line); border-radius: 18px; background: var(--surface-strong); box-shadow: var(--shadow); transition: border-color .18s ease; }
        .composer-shell:focus-within { border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); }
        .composer-row { display: flex; align-items: flex-end; gap: 10px; padding: 12px 12px 12px 18px; }
        textarea {
            flex: 1 1 auto;
            display: block;
            min-height: 26px;
            max-height: 184px;
            resize: none;
            border: 0;
            padding: 6px 0;
            color: var(--ink);
            background: transparent;
            font-size: 16px;
            line-height: 1.55;
        }
        textarea:focus { outline: 0; }
        textarea::placeholder { color: var(--muted); opacity: .76; }
        .send-button { display: inline-grid; place-items: center; flex: 0 0 auto; width: 40px; height: 40px; border: 0; border-radius: 12px; color: #fff; background: #176b57; font-size: 16px; cursor: pointer; transition: background-color .18s ease, opacity .18s ease; }
        .send-button:hover:not(:disabled) { background: #105644; }
        .send-button:disabled { opacity: .45; cursor: default; }
        .composer-note { display: flex; justify-content: space-between; gap: 14px; margin: 9px 2px 0; color: var(--muted); font-size: 12px; line-height: 1.5; }
        .composer-note kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; background: var(--surface); font-family: inherit; font-size: 11px; }
        .error-message { display: none; margin: 10px 0 0; border: 1px solid color-mix(in srgb, var(--error) 24%, transparent); border-radius: 12px; padding: 12px 15px; color: var(--error); background: var(--error-surface); font-size: 14px; line-height: 1.5; }
        .error-message.visible { display: block; }

        @media (max-width: 640px) {
            .topbar, .thread, .composer { width: calc(100% - 24px); }
            h1 { font-size: clamp(34px, 10vw, 48px); }
            .intro { font-size: 15px; }
            .assistant-message { padding: 17px; }
            .composer-note .hide-narrow { display: none; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
        }
    </style>
</head>
<body>
    <nav class="topbar" aria-label="Page controls">
        <a class="brand" href="/" aria-label="Fylit home"><span class="brand-mark" aria-hidden="true">f</span><span>ATO guidance</span></a>
        <div class="top-actions">
            <button id="new-chat" class="new-chat" type="button">New conversation</button>
            <button id="theme-toggle" class="icon-button" type="button" aria-label="Switch color theme" title="Switch color theme">&#9680;</button>
        </div>
    </nav>

    <div class="thread-scroll" id="thread-scroll">
        <div class="thread" id="thread">
            <header class="hero" id="hero">
                <div class="kicker">Fylit / ATO guidance</div>
                <h1>Ask the ATO corpus in your language.</h1>
                <p class="intro">Write your question in any language. Answers are grounded in retrieved Australian Taxation Office content and stay general, not personal tax advice. Ask follow-ups - this conversation keeps its context.</p>
                <div class="starters">
                    <button class="starter" type="button">What can I claim for working from home?</button>
                    <button class="starter" type="button">How does capital gains tax work?</button>
                    <button class="starter" type="button">When is my tax return due?</button>
                </div>
            </header>
            <section id="conversation" aria-label="Conversation" aria-live="polite" aria-relevant="additions" style="display: grid; gap: 22px;"></section>
        </div>
    </div>

    <div class="composer-dock">
        <form id="ask-form" class="composer">
            <div class="composer-shell">
                <div class="composer-row">
                    <label for="question" class="sr-only" style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);">Your question</label>
                    <textarea id="question" rows="1" required minlength="3" maxlength="2000"
                              placeholder="Ask about Australian tax, or follow up on the answer above"
                              aria-describedby="composer-hint"></textarea>
                    <button class="send-button" type="submit" aria-label="Send question" title="Send"><span aria-hidden="true">&#8593;</span></button>
                </div>
            </div>
            <div class="composer-note">
                <span id="composer-hint"><kbd>Enter</kbd> to send, <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line<span class="hide-narrow"> &middot; the answer uses the same language as your question</span></span>
            </div>
            <div id="error-message" class="error-message" role="alert"></div>
        </form>
    </div>

    <script>
        const form = document.getElementById('ask-form');
        const question = document.getElementById('question');
        const conversation = document.getElementById('conversation');
        const threadScroll = document.getElementById('thread-scroll');
        const errorMessage = document.getElementById('error-message');
        const sendButton = form.querySelector('.send-button');
        const newChatButton = document.getElementById('new-chat');
        const themeButton = document.getElementById('theme-toggle');
        const sessionStorageKey = 'fylit-session-id';
        const transcriptStorageKey = 'fylit-transcript';
        const themeStorageKey = 'fylit-theme';

        function newSessionId() {
            if (crypto.randomUUID) return crypto.randomUUID();
            return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
        }

        function readStore(key, fallback) {
            try {
                const raw = sessionStorage.getItem(key);
                return raw === null ? fallback : JSON.parse(raw);
            } catch { return fallback; }
        }

        function writeStore(key, value) {
            try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
        }

        let sessionId = readStore(sessionStorageKey, null) || newSessionId();
        writeStore(sessionStorageKey, sessionId);
        // The transcript is replayed on reload so a refresh does not wipe the
        // conversation. The server keeps its own short memory keyed by session.
        let transcript = readStore(transcriptStorageKey, []);
        let isSubmitting = false;

        function setTheme(theme) {
            document.documentElement.dataset.theme = theme;
            const nextTheme = theme === 'dark' ? 'light' : 'dark';
            themeButton.setAttribute('aria-label', 'Switch to ' + nextTheme + ' mode');
            themeButton.title = 'Switch to ' + nextTheme + ' mode';
        }

        let savedTheme = null;
        try { savedTheme = localStorage.getItem(themeStorageKey); } catch { savedTheme = null; }
        setTheme(savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
        themeButton.addEventListener('click', () => {
            const nextTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
            try { localStorage.setItem(themeStorageKey, nextTheme); } catch { /* ignore */ }
            setTheme(nextTheme);
        });

        function makeElement(tag, className, text) {
            const element = document.createElement(tag);
            if (className) element.className = className;
            if (text !== undefined) element.textContent = text;
            return element;
        }

        function appendInlineMarkdown(parent, source) {
            const tokenPattern = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*[^*\n]+\*|_[^_\n]+_)/g;
            let cursor = 0;
            for (const match of source.matchAll(tokenPattern)) {
                const token = match[0];
                const start = match.index;
                parent.append(document.createTextNode(source.slice(cursor, start)));
                if (token.startsWith('**') || token.startsWith('__')) {
                    parent.append(makeElement('strong', '', token.slice(2, -2)));
                } else if (token.startsWith('`')) {
                    parent.append(makeElement('code', '', token.slice(1, -1)));
                } else if (token.startsWith('[')) {
                    const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/i);
                    if (linkMatch) {
                        const link = makeElement('a', '', linkMatch[1]);
                        link.href = linkMatch[2];
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        parent.append(link);
                    } else {
                        parent.append(document.createTextNode(token));
                    }
                } else {
                    parent.append(makeElement('em', '', token.slice(1, -1)));
                }
                cursor = start + token.length;
            }
            parent.append(document.createTextNode(source.slice(cursor)));
        }

        function renderMarkdown(target, markdown) {
            const lines = markdown.replace(/\r\n?/g, '\n').replace(/\\n/g, '\n').split('\n');
            let paragraph = [];
            let list = null;
            let codeLines = null;

            function flushParagraph() {
                if (!paragraph.length) return;
                const block = makeElement('p');
                paragraph.forEach((line, index) => {
                    if (index) block.append(document.createElement('br'));
                    appendInlineMarkdown(block, line);
                });
                target.append(block);
                paragraph = [];
            }

            function flushList() {
                if (list) target.append(list);
                list = null;
            }

            for (const line of lines) {
                if (line.trim().startsWith('```')) {
                    flushParagraph();
                    flushList();
                    if (codeLines) {
                        const pre = makeElement('pre');
                        pre.append(makeElement('code', '', codeLines.join('\n')));
                        target.append(pre);
                        codeLines = null;
                    } else {
                        codeLines = [];
                    }
                    continue;
                }
                if (codeLines) { codeLines.push(line); continue; }

                const heading = line.match(/^(#{1,4})\s+(.+)$/);
                const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
                const bulleted = line.match(/^\s*[-*+]\s+(.+)$/);
                if (heading || numbered || bulleted || !line.trim()) {
                    flushParagraph();
                    if (numbered || bulleted) {
                        const type = numbered ? 'ol' : 'ul';
                        if (list && list.tagName.toLowerCase() !== type) flushList();
                        if (!list) list = makeElement(type);
                        const item = makeElement('li');
                        appendInlineMarkdown(item, (numbered || bulleted)[1]);
                        list.append(item);
                    } else {
                        flushList();
                        if (heading) {
                            const level = Math.min(heading[1].length + 1, 4);
                            const title = makeElement('h' + level);
                            appendInlineMarkdown(title, heading[2]);
                            target.append(title);
                        }
                    }
                    continue;
                }
                flushList();
                paragraph.push(line);
            }
            flushParagraph();
            flushList();
            if (codeLines) {
                const pre = makeElement('pre');
                pre.append(makeElement('code', '', codeLines.join('\n')));
                target.append(pre);
            }
        }

        function safeHttpUrl(value) {
            try {
                const url = new URL(value);
                return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : null;
            } catch { return null; }
        }

        function atBottom() {
            return threadScroll.scrollHeight - threadScroll.scrollTop - threadScroll.clientHeight < 120;
        }

        function scrollToEnd(smooth) {
            const motionOk = !matchMedia('(prefers-reduced-motion: reduce)').matches;
            threadScroll.scrollTo({ top: threadScroll.scrollHeight, behavior: smooth && motionOk ? 'smooth' : 'auto' });
        }

        function renderTurn(entry) {
            const turn = makeElement('article', 'turn');
            turn.append(makeElement('div', 'user-message', entry.question));

            const reply = makeElement('div', 'assistant-message');
            reply.append(makeElement('span', 'answer-label', 'ATO guidance'));
            const answer = makeElement('div', 'answer-content');
            renderMarkdown(answer, entry.answer);
            reply.append(answer);

            const resources = Array.isArray(entry.resources) ? entry.resources : [];
            if (resources.length) {
                const sourceSection = makeElement('div', 'sources');
                sourceSection.append(makeElement('span', 'sources-heading', 'Useful resources'));
                const sourceList = makeElement('ul', 'resource-list');
                for (const resource of resources) {
                    if (!resource || typeof resource.title !== 'string' || typeof resource.url !== 'string') continue;
                    const item = makeElement('li');
                    const safeUrl = safeHttpUrl(resource.url);
                    if (safeUrl) {
                        const link = makeElement('a', '', resource.title);
                        link.href = safeUrl;
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        item.append(link);
                    } else {
                        item.textContent = resource.title;
                    }
                    sourceList.append(item);
                }
                if (sourceList.childElementCount) {
                    sourceSection.append(sourceList);
                    reply.append(sourceSection);
                }
            }

            // The answer usually carries the disclaimer already; printing it a
            // second time underneath reads as a glitch.
            if (typeof entry.disclaimer === 'string' && entry.disclaimer.trim()
                && !(entry.answer || '').includes(entry.disclaimer.trim())) {
                reply.append(makeElement('p', 'disclaimer', entry.disclaimer));
            }

            turn.append(reply);
            conversation.append(turn);
            return turn;
        }

        function markHasTurns() {
            document.body.classList.toggle('has-turns', conversation.childElementCount > 0);
        }

        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.classList.add('visible');
        }

        function clearError() {
            errorMessage.textContent = '';
            errorMessage.classList.remove('visible');
        }

        function autoGrow() {
            question.style.height = 'auto';
            question.style.height = Math.min(question.scrollHeight, 184) + 'px';
        }

        function setBusy(busy) {
            isSubmitting = busy;
            sendButton.disabled = busy || !question.value.trim();
            newChatButton.disabled = busy;
        }

        // Replay anything from an earlier page load before the first paint.
        transcript.forEach(renderTurn);
        markHasTurns();
        scrollToEnd(false);

        question.addEventListener('input', () => {
            question.setCustomValidity('');
            autoGrow();
            sendButton.disabled = isSubmitting || !question.value.trim();
        });

        question.addEventListener('keydown', event => {
            // Enter sends, Shift+Enter makes a new line - the convention every
            // chat interface uses. IME composition must not be interrupted.
            if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                form.requestSubmit();
            }
        });

        for (const starter of document.querySelectorAll('.starter')) {
            starter.addEventListener('click', () => {
                question.value = starter.textContent;
                autoGrow();
                sendButton.disabled = false;
                form.requestSubmit();
            });
        }

        newChatButton.addEventListener('click', () => {
            if (isSubmitting) return;
            sessionId = newSessionId();
            transcript = [];
            writeStore(sessionStorageKey, sessionId);
            writeStore(transcriptStorageKey, transcript);
            conversation.replaceChildren();
            markHasTurns();
            clearError();
            question.value = '';
            autoGrow();
            setBusy(false);
            question.focus();
        });

        form.addEventListener('submit', async event => {
            event.preventDefault();
            if (isSubmitting) return;

            const questionText = question.value.trim();
            if (questionText.length < 3) {
                question.setCustomValidity('Please enter at least 3 characters.');
                form.reportValidity();
                return;
            }

            setBusy(true);
            clearError();

            const pendingTurn = makeElement('article', 'turn');
            pendingTurn.append(makeElement('div', 'user-message', questionText));
            const pendingReply = makeElement('div', 'assistant-message');
            pendingReply.append(makeElement('span', 'answer-label', 'ATO guidance'));
            const loading = makeElement('div', 'loading-message');
            const loadingSpinner = makeElement('span', 'spinner');
            loadingSpinner.setAttribute('aria-hidden', 'true');
            loading.append(loadingSpinner, document.createTextNode('Looking up official ATO guidance...'));
            pendingReply.append(loading);
            pendingTurn.append(pendingReply);
            conversation.append(pendingTurn);
            markHasTurns();
            question.value = '';
            autoGrow();
            scrollToEnd(true);

            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 60000);
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: questionText, session_id: sessionId }),
                    signal: controller.signal
                });
                if (!response.ok) {
                    if (response.status === 429) throw new Error('rate-limited');
                    if (response.status >= 500) throw new Error('server-error');
                    throw new Error('request-rejected');
                }
                let data;
                try {
                    data = await response.json();
                } catch { throw new Error('invalid-response'); }
                if (typeof data.answer !== 'string' || !data.answer.trim()) throw new Error('empty-answer');

                const stick = atBottom();
                pendingTurn.remove();
                const entry = {
                    question: questionText,
                    answer: data.answer,
                    resources: data.useful_resources,
                    disclaimer: data.disclaimer
                };
                transcript.push(entry);
                writeStore(transcriptStorageKey, transcript);
                renderTurn(entry);
                markHasTurns();
                if (stick) scrollToEnd(true);
            } catch (error) {
                pendingTurn.remove();
                markHasTurns();
                question.value = questionText;
                autoGrow();
                if (error.name === 'AbortError') {
                    showError('This is taking longer than expected. Please try again.');
                } else if (error.message === 'rate-limited') {
                    showError('Too many questions in a short time. Please wait a moment and try again.');
                } else if (error.message === 'server-error') {
                    showError('The answer service is temporarily unavailable. Please try again shortly.');
                } else if (error.message === 'empty-answer') {
                    showError('No answer came back this time. Please try asking in a different way.');
                } else if (error.message === 'request-rejected') {
                    showError('That question could not be accepted. Please check it and try again.');
                } else if (error.message === 'invalid-response') {
                    showError('The answer service returned an unreadable response. Please try again.');
                } else {
                    showError('Could not connect to the answer service. Check your connection and try again.');
                }
            } finally {
                clearTimeout(timeout);
                setBusy(false);
                question.focus();
            }
        });

        setBusy(false);
        autoGrow();
    </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict:
    """Liveness: the process is running. No dependencies checked on purpose."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
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

    except Exception as exc:  # noqa: BLE001 - readiness reports failure
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "reason": type(exc).__name__,
            },
        )

    if not embedded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "reason": "index is empty",
                "embedded_chunks": 0,
            },
        )

    return {"status": "ready", "embedded_chunks": embedded}
