"""Central configuration, loaded from environment / .env via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    # Which reranker runs on the query path: fusion (RRF order, free),
    # mmr (diversity, free) or llm (gpt-4o-mini, costs a call per query).
    # See ADR-0003 for the comparison behind the default.
    rerank_strategy: str = "fusion"

    # The damping constant in reciprocal rank fusion: score = 1 / (rrf_k + rank).
    # Lower values let one confident half outvote agreement between both halves.
    # 60 is Cormack et al. (2009); ADR-0004 records what the alternatives measured
    # on this corpus. Must be >= 1, since 0 or less can divide by zero.
    rrf_k: int = Field(default=60, ge=1)

    corpus_dir: str = "data/ato_corpus"
    index_dir: str = "data/index"

    # Postgres + pgvector serves both retrieval halves (ADR-0002).
    database_url: str = "postgresql://fylit:fylit@localhost:5432/fylit"
    chunks_table: str = "chunks"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 30


settings = Settings()
