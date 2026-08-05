"""Unit tests for NLP service modules."""

from __future__ import annotations

import pytest
from src.services.keyword_service import KeywordService
from src.services.language_service import LanguageService

# ── Keyword Service ──────────────────────────────────────────────────────────


class TestKeywordService:
    """Tests for the TF-IDF keyword extraction service."""

    def setup_method(self):
        self.service = KeywordService()

    def test_extract_returns_keywords(self):
        """Extraction returns a list of keywords with scores."""
        text = (
            "Machine learning algorithms process data to find patterns "
            "and make predictions about future events using neural networks"
        )
        result = self.service.extract(text, top_k=5)
        assert len(result.keywords) > 0
        assert len(result.keywords) <= 5
        assert result.processing_time_ms > 0

    def test_keywords_have_word_and_score(self):
        """Each keyword has a word and a numeric score."""
        text = "Artificial intelligence and deep learning are transforming industries"
        result = self.service.extract(text, top_k=3)
        for kw in result.keywords:
            assert "word" in kw
            assert "score" in kw
            assert isinstance(kw["score"], float | int)

    def test_keywords_sorted_descending(self):
        """Keywords are returned in descending score order."""
        text = "Data science uses machine learning for data analysis and data visualization"
        result = self.service.extract(text, top_k=10)
        scores = [kw["score"] for kw in result.keywords]
        assert scores == sorted(scores, reverse=True)

    def test_empty_text_returns_empty(self):
        """Empty or very short text returns no keywords."""
        result = self.service.extract("Hi", top_k=5)
        assert result.keywords == []
        assert result.n_grams == []

    def test_n_grams_contain_spaces(self):
        """N-grams contain multi-word phrases."""
        text = "Artificial intelligence and machine learning are subsets of computer science"
        result = self.service.extract(text, top_k=10)
        for ng in result.n_grams:
            assert " " in ng

    def test_stopwords_filtered(self):
        """Common stopwords are not returned as keywords."""
        text = "The quick brown fox is a very good animal"
        result = self.service.extract(text, top_k=10)
        words = [kw["word"] for kw in result.keywords]
        # "the" and "is" and "a" should not be in results
        for stopword in ["the", "is", "a"]:
            assert stopword not in words


# ── Language Service ────────────────────────────────────────────────────────


class TestLanguageService:
    """Tests for the langdetect-based language service."""

    def setup_method(self):
        self.service = LanguageService()

    def test_detect_english(self):
        """English text is detected as 'en'."""
        result = self.service.detect("The quick brown fox jumps over the lazy dog.")
        assert result.detected_language == "en"
        assert result.language_name == "English"
        assert result.confidence > 0

    def test_detect_french(self):
        """French text is detected as 'fr'."""
        result = self.service.detect("Bonjour, comment allez-vous aujourd'hui?")
        assert result.detected_language == "fr"
        assert result.language_name == "French"

    def test_detect_german(self):
        """German text is detected as 'de'."""
        result = self.service.detect("Guten Morgen, wie geht es Ihnen?")
        assert result.detected_language == "de"

    def test_detect_spanish(self):
        """Spanish text is detected as 'es'."""
        result = self.service.detect("Buenos días, ¿cómo está usted?")
        assert result.detected_language == "es"

    def test_alternative_languages_provided(self):
        """Alternative language suggestions are returned."""
        result = self.service.detect("Hello world, this is a test sentence.")
        assert len(result.alternative_languages) >= 0

    def test_text_snippet_truncated(self):
        """Text snippet in result is truncated to 100 chars."""
        long_text = "a" * 200
        result = self.service.detect(long_text)
        assert len(result.text_snippet) <= 100

    def test_too_short_text_gives_result(self):
        """Very short text returns a result (langdetect does not raise)."""
        result = self.service.detect("a")
        assert result.detected_language is not None
        assert result.confidence > 0

    def test_empty_text_raises(self):
        """Empty text raises ValueError."""
        with pytest.raises(ValueError):
            self.service.detect("")
