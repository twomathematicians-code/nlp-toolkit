"""Named Entity Recognition service using spaCy."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.services import get_spacy_model


@dataclass
class NEREntity:
    """A single named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass
class NERResult:
    """NER extraction result."""

    text_snippet: str
    entities: list[NEREntity]
    entity_count: int
    processing_time_ms: float


class NERService:
    """Production NER using spaCy entity recognizer."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name

    def extract(self, text: str, max_entities: int = 50) -> NERResult:
        """Extract named entities from text using spaCy.

        Args:
            text: Input text to analyze.
            max_entities: Maximum number of entities to return.

        Returns:
            NERResult with extracted entities and metadata.
        """
        start_time = time.perf_counter()
        nlp = get_spacy_model(self._model_name)
        doc = nlp(text)

        entities: list[NEREntity] = []
        for ent in doc.ents:
            # spaCy assigns scores via the parser; approximate confidence
            confidence = 1.0 if ent.label_ else 0.0
            entities.append(
                NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=round(confidence, 3),
                )
            )
            if len(entities) >= max_entities:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return NERResult(
            text_snippet=text[:200],
            entities=entities,
            entity_count=len(entities),
            processing_time_ms=round(elapsed_ms, 2),
        )
