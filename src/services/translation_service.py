"""Translation service using Helsinki-NLP Opus-MT models via HuggingFace."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("nlp-toolkit")


# Map common language codes to Opus-MT model identifiers.
# Helsinki-NLP models follow the pattern: Helsinki-NLP/opus-mt-{src}-{tgt}
_MODEL_MAP: dict[str, dict[str, str]] = {
    "en": {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "it": "Helsinki-NLP/opus-mt-en-it",
        "pt": "Helsinki-NLP/opus-mt-en-pt",
        "nl": "Helsinki-NLP/opus-mt-en-nl",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "ar": "Helsinki-NLP/opus-mt-en-ar",
        "zh": "Helsinki-NLP/opus-mt-en-zh",
        "ja": "Helsinki-NLP/opus-mt-en-ja",
        "ru": "Helsinki-NLP/opus-mt-en-ru",
    },
    "fr": {"en": "Helsinki-NLP/opus-mt-fr-en"},
    "de": {"en": "Helsinki-NLP/opus-mt-de-en"},
    "es": {"en": "Helsinki-NLP/opus-mt-es-en"},
    "it": {"en": "Helsinki-NLP/opus-mt-it-en"},
    "pt": {"en": "Helsinki-NLP/opus-mt-pt-en"},
    "nl": {"en": "Helsinki-NLP/opus-mt-nl-en"},
    "hi": {"en": "Helsinki-NLP/opus-mt-hi-en"},
    "ar": {"en": "Helsinki-NLP/opus-mt-ar-en"},
    "zh": {"en": "Helsinki-NLP/opus-mt-zh-en"},
    "ja": {"en": "Helsinki-NLP/opus-mt-ja-en"},
    "ru": {"en": "Helsinki-NLP/opus-mt-ru-en"},
}

# Supported language pairs for API documentation
SUPPORTED_PAIRS = sorted(
    {f"{src}->{tgt}" for src, targets in _MODEL_MAP.items() for tgt in targets}
)


class TranslationService:
    """Translation using Helsinki-NLP/Opus-MT models.

    Supports multiple language pairs. Models are lazy-loaded and cached.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}

    def _get_pipeline(self, source_lang: str, target_lang: str):
        """Lazy-load the translation pipeline for a language pair."""
        pair_key = f"{source_lang}-{target_lang}"

        if pair_key not in self._pipelines:
            model_name = self._resolve_model(source_lang, target_lang)
            from src.services import get_transformers_pipeline

            self._pipelines[pair_key] = get_transformers_pipeline(
                "translation",
                model_name,
            )

        return self._pipelines[pair_key]

    @staticmethod
    def _resolve_model(source_lang: str, target_lang: str) -> str:
        """Resolve a source-target language pair to a model name."""
        targets = _MODEL_MAP.get(source_lang, {})
        model_name = targets.get(target_lang)

        if not model_name:
            supported = [f"{s}->{t}" for s, tgts in _MODEL_MAP.items() for t in tgts]
            raise ValueError(
                f"Unsupported translation pair '{source_lang}->{target_lang}'. "
                f"Supported pairs: {', '.join(supported[:20])}..."
            )

        return model_name

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> dict:
        """Translate text between languages.

        Args:
            text: Input text to translate.
            target_lang: Target language ISO 639-1 code.
            source_lang: Source language code, or 'auto' to detect.

        Returns:
            Dict with translation result and metadata.
        """
        start_time = time.perf_counter()

        # Auto-detect source language if needed
        effective_source = source_lang
        if source_lang == "auto":
            try:
                from langdetect import detect

                effective_source = detect(text)
            except Exception:
                effective_source = "en"

        pipe = self._get_pipeline(effective_source, target_lang)
        result = pipe(text[:512])  # Truncate for model max length
        translated_text = result[0]["translation_text"].strip()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {
            "original_text": text[:200],
            "translated_text": translated_text,
            "source_lang_detected": effective_source,
            "target_lang": target_lang,
            "processing_time_ms": round(elapsed_ms, 2),
        }
