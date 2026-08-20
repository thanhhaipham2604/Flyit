"""Central configuration, loaded from environment / .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    corpus_dir: str = "data/ato_corpus"
    index_dir: str = "data/index"

    qdrant_url: str = "http://localhost:6333"
    opensearch_url: str = "http://localhost:9200"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 30


settings = Settings()
