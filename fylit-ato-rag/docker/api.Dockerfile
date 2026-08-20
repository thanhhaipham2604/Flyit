FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "fylit_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
