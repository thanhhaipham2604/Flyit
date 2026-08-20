"""Fylit ATO RAG system.

Zone A (offline): ingestion -> indexing.
Zone B (query time): api -> retrieval -> generation, wrapped in guardrails.
"""

__version__ = "0.1.0"
