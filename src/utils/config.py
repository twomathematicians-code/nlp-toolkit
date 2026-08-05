"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration with environment variable overrides.

    Values can be set via environment variables or .env file.
    """

    # spaCy model to use for NER
    spacy_model: str = "en_core_web_sm"

    # Summarization model
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"

    # Keyword extraction model
    keyword_model: str = "distilbert-base-uncased"

    # Maximum text length accepted by endpoints
    max_text_length: int = 10000

    # Default source language for translation
    default_language: str = "en"

    # CORS origins (comma-separated); "*" allows all
    cors_origins: str = "*"

    # Log level
    log_level: str = "INFO"

    model_config = {"env_prefix": "NLP_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
