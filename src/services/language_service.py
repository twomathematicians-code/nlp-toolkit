"""Language detection service using langdetect."""

from __future__ import annotations

import time
from dataclasses import dataclass

from langdetect import LangDetectException, detect_langs


@dataclass
class LanguageResult:
    """Language detection result."""

    text_snippet: str
    detected_language: str
    language_name: str
    confidence: float
    alternative_languages: list[dict]
    processing_time_ms: float = 0.0


# ISO 639-1 code to language name mapping
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh": "Chinese",
}


class LanguageService:
    """Language detection using langdetect probabilistic classifier."""

    def detect(self, text: str) -> LanguageResult:
        """Detect the language of the input text.

        Args:
            text: Input text to analyze.

        Returns:
            LanguageResult with detected language and confidence scores.

        Raises:
            ValueError: If language detection fails for the input.
        """
        start_time = time.perf_counter()

        try:
            probs = detect_langs(text)
        except LangDetectException as exc:
            raise ValueError(f"Cannot detect language: {exc}") from exc

        if not probs:
            raise ValueError("Language detection returned no results")

        # Primary detection
        primary = probs[0]
        detected_code = primary.lang.split("-")[0]
        detected_name = LANGUAGE_NAMES.get(detected_code, detected_code)
        confidence = round(primary.prob, 3)

        # Alternative languages
        alternatives = []
        for p in probs[1:4]:
            code = p.lang.split("-")[0]
            alternatives.append(
                {
                    "language": code,
                    "language_name": LANGUAGE_NAMES.get(code, code),
                    "confidence": round(p.prob, 3),
                }
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return LanguageResult(
            text_snippet=text[:100],
            detected_language=detected_code,
            language_name=detected_name,
            confidence=confidence,
            alternative_languages=alternatives,
            processing_time_ms=round(elapsed_ms, 2),
        )
