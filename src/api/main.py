"""NLP Toolkit API — Production-grade FastAPI application.

Five NLP microservices:
    1. Named Entity Recognition (spaCy)
    2. Text Summarization (HuggingFace transformers)
    3. Keyword Extraction (TF-IDF + statistical scoring)
    4. Language Detection (langdetect)
    5. Translation (Helsinki-NLP Opus-MT)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.services.keyword_service import KeywordService
from src.services.language_service import LanguageService
from src.services.ner_service import NERService
from src.services.summarizer_service import SummarizerService
from src.services.translation_service import TranslationService
from src.utils.config import get_settings
from src.utils.logging import setup_logging

# ---------------------------------------------------------------------------
# Logging & Settings
# ---------------------------------------------------------------------------
settings = get_settings()
logger = setup_logging(level=settings.log_level)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TextInput(BaseModel):
    """Standard text input for NLP endpoints."""

    text: str = Field(..., min_length=1, max_length=10000, description="Input text")
    language: str = Field(default="en", description="ISO 639-1 language code")


class NEREntity(BaseModel):
    """A single named entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float


class NERResponse(BaseModel):
    """NER endpoint response."""

    text_snippet: str
    entities: list[NEREntity]
    entity_count: int
    processing_time_ms: float


class SummaryResponse(BaseModel):
    """Summarization endpoint response."""

    original_length: int
    summary_length: int
    summary: str
    compression_ratio: float
    style: str
    processing_time_ms: float


class KeywordsResponse(BaseModel):
    """Keyword extraction endpoint response."""

    keywords: list[dict]
    n_grams: list[str]
    processing_time_ms: float


class TranslationRequest(BaseModel):
    """Translation endpoint request."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    source_lang: str = Field(default="auto", description="Source language or 'auto'")
    target_lang: str = Field(..., description="Target language ISO 639-1 code")


class TranslationResponse(BaseModel):
    """Translation endpoint response."""

    original_text: str
    translated_text: str
    source_lang_detected: str
    target_lang: str
    processing_time_ms: float


class LanguageDetectionResponse(BaseModel):
    """Language detection endpoint response."""

    text_snippet: str
    detected_language: str
    language_name: str
    confidence: float
    alternative_languages: list[dict]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict[str, str]


# ---------------------------------------------------------------------------
# Service instances (lazy-load models on first request)
# ---------------------------------------------------------------------------

ner_service = NERService(model_name=settings.spacy_model)
summarizer_service = SummarizerService(model_name=settings.summarizer_model)
keyword_service = KeywordService(model_name=settings.keyword_model)
language_service = LanguageService()
translation_service = TranslationService()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI app."""
    logger.info("NLP Toolkit v2.0.0 starting up...")
    logger.info("NER model: %s", settings.spacy_model)
    logger.info("Summarizer model: %s", settings.summarizer_model)
    logger.info("Keyword model: %s", settings.keyword_model)
    yield
    logger.info("NLP Toolkit shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NLP Toolkit API",
    description=(
        "Production-grade NLP microservices powered by "
        "spaCy, HuggingFace transformers, and langdetect."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — configurable via NLP_CORS_ORIGINS env var
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and model availability."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        services={
            "ner": settings.spacy_model,
            "summarizer": settings.summarizer_model,
            "keywords": settings.keyword_model,
            "language": "langdetect",
            "translation": "Helsinki-NLP Opus-MT",
        },
    )


@app.post("/api/v1/ner", response_model=NERResponse, tags=["NER"])
async def extract_ner(inp: TextInput):
    """Extract named entities from text using spaCy.

    Supports entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, etc.
    """
    try:
        result = ner_service.extract(inp.text)
        return NERResponse(
            text_snippet=result.text_snippet,
            entities=[
                NEREntity(
                    text=e.text,
                    label=e.label,
                    start=e.start,
                    end=e.end,
                    confidence=e.confidence,
                )
                for e in result.entities
            ],
            entity_count=result.entity_count,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.error("NER processing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"NER processing failed: {exc}")


@app.post("/api/v1/summarize", response_model=SummaryResponse, tags=["Summarize"])
async def summarize_text(
    inp: TextInput,
    style: str = Query(
        default="extractive",
        pattern="^(extractive|abstractive|bullets|headline)$",
        description="Summarization style",
    ),
):
    """Summarize text using HuggingFace transformer models.

    Styles: extractive, abstractive, bullets, headline.
    """
    try:
        result = summarizer_service.summarize(inp.text, style=style)
        return SummaryResponse(**result)
    except Exception as exc:
        logger.error("Summarization error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")


@app.post("/api/v1/keywords", response_model=KeywordsResponse, tags=["Keywords"])
async def extract_keywords(
    inp: TextInput,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of keywords"),
):
    """Extract keywords and key phrases from text using TF-IDF scoring."""
    try:
        result = keyword_service.extract(inp.text, top_k=top_k)
        return KeywordsResponse(
            keywords=result.keywords,
            n_grams=result.n_grams,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.error("Keyword extraction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {exc}")


@app.post("/api/v1/detect-language", response_model=LanguageDetectionResponse, tags=["Language"])
async def detect_language(inp: TextInput):
    """Detect the language of the input text using probabilistic classification."""
    try:
        result = language_service.detect(inp.text)
        return LanguageDetectionResponse(
            text_snippet=result.text_snippet,
            detected_language=result.detected_language,
            language_name=result.language_name,
            confidence=result.confidence,
            alternative_languages=result.alternative_languages,
            processing_time_ms=result.processing_time_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Language detection error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Language detection failed: {exc}")


@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_text(req: TranslationRequest):
    """Translate text between languages using Helsinki-NLP Opus-MT models.

    Set source_lang to 'auto' for automatic source language detection.
    Supported pairs include en↔fr, en↔de, en↔es, en↔it, en↔pt, en↔nl,
    en↔hi, en↔ar, en↔zh, en↔ja, en↔ru, and reverses.
    """
    try:
        result = translation_service.translate(
            text=req.text,
            target_lang=req.target_lang,
            source_lang=req.source_lang,
        )
        return TranslationResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Translation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}")
