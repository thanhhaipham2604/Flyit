"""Central configuration, loaded from environment / .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    corpus_dir: str = "data/ato_corpus"
    index_dir: str = "data/index"

    # Postgres + pgvector serves both retrieval halves (ADR-0002).
    database_url: str = "postgresql://fylit:fylit@localhost:5432/fylit"
    chunks_table: str = "chunks"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 30


settings = Settings()
