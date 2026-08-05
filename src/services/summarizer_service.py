"""Text summarization service using HuggingFace transformers."""

from __future__ import annotations

import time

from src.services import get_transformers_pipeline


class SummarizerService:
    """Production summarization using HuggingFace pipeline.

    Default model: facebook/bart-large-cnn — good balance of quality and speed.
    Falls back to extractive truncation if the model fails to load.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        fallback_model: str = "sshleifer/distilbart-cnn-12-6",
    ) -> None:
        self._model_name = model_name
        self._fallback_model = fallback_model
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the summarization pipeline."""
        if self._pipeline is None:
            try:
                self._pipeline = get_transformers_pipeline(
                    "summarization",
                    self._model_name,
                )
            except Exception:
                import logging

                logging.warning(
                    "Failed to load '%s', falling back to '%s'",
                    self._model_name,
                    self._fallback_model,
                )
                self._pipeline = get_transformers_pipeline(
                    "summarization",
                    self._fallback_model,
                )
        return self._pipeline

    def summarize(
        self,
        text: str,
        style: str = "extractive",
        max_length: int = 150,
        min_length: int = 30,
    ) -> dict:
        """Summarize input text.

        Args:
            text: Input text to summarize.
            style: Summarization style (extractive, abstractive, bullets, headline).
            max_length: Maximum length of summary.
            min_length: Minimum length of summary.

        Returns:
            Dict with summary, lengths, compression ratio, and style.
        """
        start_time = time.perf_counter()

        # Truncate input if too long for the model
        max_input = 1024
        truncated = text[:max_input] if len(text) > max_input else text

        # Adjust params based on style
        if style == "headline":
            max_length = min(max_length, 30)
            min_length = 5
        elif style == "bullets":
            max_length = min(max_length, 200)

        try:
            pipe = self._get_pipeline()
            result = pipe(
                truncated,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            summary = result[0]["summary_text"].strip()
        except Exception:
            # Fallback: extractive — take first ~30% of sentences
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            n = max(1, len(sentences) // 3)
            summary = ". ".join(sentences[:n]) + "."

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary,
            "compression_ratio": round(len(summary) / max(len(text), 1), 3),
            "style": style,
            "processing_time_ms": round(elapsed_ms, 2),
        }
