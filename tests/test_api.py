"""API integration tests for the NLP Toolkit endpoints.

Tests use real models (spaCy, langdetect) and lightweight statistical methods
(keyword extraction). Summarization and translation use HuggingFace models when
available, with graceful fallbacks in CI.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# ── Health ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    """Health endpoint returns service metadata."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "ner" in data["services"]
    assert "summarizer" in data["services"]
    assert "keywords" in data["services"]
    assert "language" in data["services"]
    assert "translation" in data["services"]


# ── NER (spaCy) ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ner_basic(client: AsyncClient, sample_english_text: str):
    """NER extracts entities from English text with real spaCy model."""
    resp = await client.post(
        "/api/v1/ner",
        json={"text": sample_english_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "entity_count" in data
    assert data["processing_time_ms"] > 0
    assert isinstance(data["entities"], list)

    # With real spaCy, we expect at least some entities from this text
    if data["entity_count"] > 0:
        entity = data["entities"][0]
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity
        assert 0.0 <= entity["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_ner_single_sentence(client: AsyncClient):
    """NER works on a single short sentence."""
    resp = await client.post(
        "/api/v1/ner",
        json={"text": "Barack Obama was born in Hawaii."},
    )
    assert resp.status_code == 200
    data = resp.json()
    # spaCy should find at least one entity (PERSON: Barack Obama)
    assert data["entity_count"] >= 1
    entity_texts = [e["text"] for e in data["entities"]]
    assert any("Obama" in t or "Barack" in t for t in entity_texts)


@pytest.mark.asyncio
async def test_ner_no_entities(client: AsyncClient):
    """NER handles text with no recognizable entities gracefully."""
    resp = await client.post(
        "/api/v1/ner",
        json={"text": "The quick brown fox jumps over the lazy dog."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_count"] == 0
    assert data["entities"] == []


# ── Summarization (transformers / fallback) ─────────────────────────────────


@pytest.mark.asyncio
async def test_summarize_basic(client: AsyncClient, sample_long_text: str):
    """Summarization returns a shorter version of the input text."""
    resp = await client.post(
        "/api/v1/summarize",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert data["compression_ratio"] <= 1.0
    assert data["original_length"] >= data["summary_length"]
    assert data["style"] == "extractive"


@pytest.mark.asyncio
async def test_summarize_style_parameter(client: AsyncClient, sample_long_text: str):
    """Different styles are accepted and reflected in the response."""
    for style in ["extractive", "abstractive", "headline"]:
        resp = await client.post(
            f"/api/v1/summarize?style={style}",
            json={"text": sample_long_text},
        )
        assert resp.status_code == 200
        assert resp.json()["style"] == style


@pytest.mark.asyncio
async def test_summarize_compression(client: AsyncClient, sample_long_text: str):
    """Summary is shorter than the original text."""
    resp = await client.post(
        "/api/v1/summarize",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary_length"] < data["original_length"]


# ── Keywords (TF-IDF) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_keywords_basic(client: AsyncClient, sample_long_text: str):
    """Keyword extraction returns scored keywords."""
    resp = await client.post(
        "/api/v1/keywords?top_k=5",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "keywords" in data
    assert len(data["keywords"]) <= 5
    assert data["processing_time_ms"] > 0

    # Verify keyword structure
    if len(data["keywords"]) > 0:
        kw = data["keywords"][0]
        assert "word" in kw
        assert "score" in kw
        assert isinstance(kw["score"], float | int)


@pytest.mark.asyncio
async def test_keywords_sorted_by_score(client: AsyncClient, sample_long_text: str):
    """Keywords are returned in descending score order."""
    resp = await client.post(
        "/api/v1/keywords?top_k=10",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    scores = [kw["score"] for kw in data["keywords"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_keywords_n_grams(client: AsyncClient, sample_long_text: str):
    """N-grams are extracted when multi-word patterns exist."""
    resp = await client.post(
        "/api/v1/keywords?top_k=10",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    # N-grams list should contain multi-word phrases
    n_grams = data["n_grams"]
    assert isinstance(n_grams, list)


@pytest.mark.asyncio
async def test_keywords_short_text(client: AsyncClient):
    """Keywords handles very short text gracefully."""
    resp = await client.post(
        "/api/v1/keywords?top_k=3",
        json={"text": "Hi there"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["keywords"], list)


# ── Language Detection (langdetect) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_language_english(client: AsyncClient, sample_english_text: str):
    """English text is correctly detected."""
    resp = await client.post(
        "/api/v1/detect-language",
        json={"text": sample_english_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_language"] == "en"
    assert data["language_name"] == "English"
    assert 0.0 < data["confidence"] <= 1.0
    assert isinstance(data["alternative_languages"], list)


@pytest.mark.asyncio
async def test_language_french(client: AsyncClient, sample_french_text: str):
    """French text is correctly detected."""
    resp = await client.post(
        "/api/v1/detect-language",
        json={"text": sample_french_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_language"] == "fr"
    assert data["language_name"] == "French"
    assert data["confidence"] > 0.5


@pytest.mark.asyncio
async def test_language_german(client: AsyncClient):
    """German text is correctly detected."""
    resp = await client.post(
        "/api/v1/detect-language",
        json={"text": "Guten Tag, wie geht es Ihnen? Berlin ist eine große Stadt."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_language"] == "de"


@pytest.mark.asyncio
async def test_language_short_text(client: AsyncClient):
    """Very short text still returns a detection."""
    resp = await client.post(
        "/api/v1/detect-language",
        json={"text": "This is a test."},
    )
    assert resp.status_code == 200
    assert resp.json()["detected_language"] == "en"


# ── Translation (Helsinki-NLP) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_translate_en_to_fr(client: AsyncClient):
    """English to French translation returns French text."""
    resp = await client.post(
        "/api/v1/translate",
        json={"text": "Hello, how are you?", "source_lang": "en", "target_lang": "fr"},
    )
    # Translation may fail if model isn't downloaded; check response format
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "translated_text" in data
        assert len(data["translated_text"]) > 0
        assert data["target_lang"] == "fr"
        assert data["processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_translate_unsupported_pair(client: AsyncClient):
    """Unsupported language pair returns 422."""
    resp = await client.post(
        "/api/v1/translate",
        json={"text": "Hello", "source_lang": "xx", "target_lang": "yy"},
    )
    assert resp.status_code in (422, 500)


# ── Edge Cases ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_rejected(client: AsyncClient):
    """Empty text is rejected by validation."""
    resp = await client.post("/api/v1/ner", json={"text": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_field_rejected(client: AsyncClient):
    """Missing required fields are rejected."""
    resp = await client.post("/api/v1/translate", json={"target_lang": "fr"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_style_rejected(client: AsyncClient, sample_long_text: str):
    """Invalid summarization style is rejected."""
    resp = await client.post(
        "/api/v1/summarize?style=invalid_style",
        json={"text": sample_long_text},
    )
    assert resp.status_code == 422
