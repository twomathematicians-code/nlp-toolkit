"""Model loading utilities for lazy model initialization."""

from __future__ import annotations

import logging
from functools import lru_cache

import spacy

logger = logging.getLogger("nlp-toolkit")


@lru_cache(maxsize=1)
def get_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load a spaCy model with caching. Downloads if missing."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info("Downloading spaCy model '%s'...", model_name)
        spacy.cli.download(model_name)  # type: ignore[attr-defined]
        return spacy.load(model_name)


def get_transformers_pipeline(
    task: str,
    model_name: str,
    **kwargs,
):
    """Load a HuggingFace transformers pipeline with caching."""
    from transformers import pipeline

    key = f"{task}:{model_name}"
    if not hasattr(get_transformers_pipeline, "_cache"):
        get_transformers_pipeline._cache = {}  # type: ignore[attr-defined]

    if key not in get_transformers_pipeline._cache:
        logger.info("Loading transformers pipeline '%s' with model '%s'...", task, model_name)
        get_transformers_pipeline._cache[key] = pipeline(task, model=model_name, **kwargs)  # type: ignore[attr-defined]
    return get_transformers_pipeline._cache[key]  # type: ignore[attr-defined]
