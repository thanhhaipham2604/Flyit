"""Central configuration, loaded from environment / .env via pydantic-settings."""

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
    # Passages retrieved before reranking, and passages kept as evidence for
    # the prompt. Both are settings rather than constants because tuning K is
    # pointless if changing it costs an image rebuild.
    shortlist: int = 30
    evidence_count: int = 15
    # Citations shown to the user. Lower than evidence_count on purpose: the
    # model needs breadth to answer, the reader needs the few sources that
    # carry the answer. Measured - precision falls from 57% to 37% as the
    # cited list grows, because the tail of the ranking is where the noise is.
    citation_limit: int = 5

    # Distinct near-identical documents allowed into the shortlist. The ATO
    # republishes the same page per income year, and eighteen copies of one
    # page can fill the shortlist so the page that answers the question never
    # gets in. 0 disables the cap.
    family_cap: int = 2

    corpus_dir: str = "data/ato_corpus"
    index_dir: str = "data/index"

    # Postgres + pgvector serves both retrieval halves (ADR-0002).
    database_url: str = "postgresql://fylit:fylit@localhost:5432/fylit"
    chunks_table: str = "chunks"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 30


settings = Settings()
