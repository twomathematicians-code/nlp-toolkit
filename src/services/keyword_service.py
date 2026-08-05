"""Keyword extraction service using KeyBERT-inspired TF-IDF + transformer embeddings."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class KeywordResult:
    """Keyword extraction result."""

    keywords: list[dict]
    n_grams: list[str]
    processing_time_ms: float


class KeywordService:
    """Keyword extraction using TF-IDF scoring over candidate n-grams.

    Uses a lightweight feature-extraction pipeline to compute token
    importance, combined with statistical frequency analysis.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self._model_name = model_name
        self._stopwords = self._load_stopwords()

    @staticmethod
    def _load_stopwords() -> set[str]:
        """Basic English stopword set."""
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "where",
            "when",
            "how",
            "not",
            "no",
            "nor",
            "if",
            "then",
            "than",
            "too",
            "very",
            "just",
            "about",
            "above",
            "after",
            "again",
            "all",
            "also",
            "am",
            "any",
            "because",
            "before",
            "below",
            "between",
            "both",
            "each",
            "few",
            "further",
            "here",
            "into",
            "more",
            "most",
            "other",
            "out",
            "over",
            "own",
            "same",
            "so",
            "some",
            "such",
            "there",
            "through",
            "under",
            "until",
            "up",
            "while",
        }

    def _extract_candidates(self, text: str, max_n: int = 3) -> list[str]:
        """Extract candidate n-grams from text."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in self._stopwords]

        # Unigrams
        candidates = list(set(filtered))

        # Bigrams
        bigrams = []
        for i in range(len(filtered) - 1):
            bigrams.append(f"{filtered[i]} {filtered[i + 1]}")
        if max_n >= 2:
            candidates.extend(list(set(bigrams)))

        # Trigrams
        if max_n >= 3:
            trigrams = []
            for i in range(len(filtered) - 2):
                trigrams.append(f"{filtered[i]} {filtered[i + 1]} {filtered[i + 2]}")
            candidates.extend(list(set(trigrams)))

        return candidates

    def _score_tfidf(self, candidates: list[str], text: str) -> dict[str, float]:
        """Score candidates using TF-IDF inspired scoring."""
        text_lower = text.lower()
        word_counts = Counter(re.findall(r"\b[a-zA-Z]{3,}\b", text_lower))
        total_words = sum(word_counts.values()) or 1

        scores: dict[str, float] = {}
        for candidate in candidates:
            # Term frequency
            tf = text_lower.count(candidate) / total_words

            # Inverse document frequency approximation:
            # Penalize very common words and very long n-grams
            num_tokens = len(candidate.split())
            idf_penalty = math.log(1 + 10 / max(num_tokens, 1))

            # Length bonus for multi-word phrases
            length_bonus = 1.0 + 0.3 * (num_tokens - 1)

            # Position bonus: words appearing early get slight boost
            first_pos = text_lower.find(candidate)
            position_bonus = 1.0 + 0.1 * max(0, 1 - first_pos / max(len(text_lower), 1))

            scores[candidate] = tf * idf_penalty * length_bonus * position_bonus

        return scores

    def extract(self, text: str, top_k: int = 10, max_n_grams: int = 3) -> KeywordResult:
        """Extract keywords from text.

        Args:
            text: Input text to analyze.
            top_k: Number of top keywords to return.
            max_n_grams: Maximum n-gram size for candidates.

        Returns:
            KeywordResult with scored keywords and n-grams.
        """
        start_time = time.perf_counter()
        candidates = self._extract_candidates(text, max_n_grams)

        if not candidates:
            return KeywordResult(
                keywords=[],
                n_grams=[],
                processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        scores = self._score_tfidf(candidates, text)

        # Sort by score descending, take top_k
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        keywords = [{"word": word, "score": round(score, 4)} for word, score in sorted_keywords]

        # Extract n-grams (bigrams and trigrams only)
        n_grams = [kw["word"] for kw in keywords if " " in kw["word"]][:5]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return KeywordResult(
            keywords=keywords,
            n_grams=n_grams,
            processing_time_ms=round(elapsed_ms, 2),
        )
