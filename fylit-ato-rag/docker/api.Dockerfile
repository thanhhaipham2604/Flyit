# Two stages: the build stage compiles wheels, the runtime stage gets only the
# installed packages. It keeps pip, setuptools and the build cache out of the
# shipped image, which is both smaller and a smaller attack surface.
#
# Python is pinned to a patch release rather than 3.12-slim: a floating tag means
# the image that passes CI today can be built from different bytes tomorrow, and
# "works on my machine" becomes "worked on my Tuesday".

FROM python:3.12.7-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Install into a virtualenv so the runtime stage can copy one self-contained
# directory rather than picking site-packages out of the system Python.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir .


FROM python:3.12.7-slim AS runtime

# Runs unprivileged. Nothing here needs root, and a container that does not need
# it should not have it.
RUN groupadd --system --gid 1001 fylit \
    && useradd --system --uid 1001 --gid fylit --create-home fylit

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=fylit:fylit scripts ./scripts

USER fylit
EXPOSE 8000

# Uses /health, not /ready: this reports whether the process is alive. A
# container that is up but pointed at an empty index should not be restarted -
# that is a readiness question, and Kubernetes asks it separately.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "fylit_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
