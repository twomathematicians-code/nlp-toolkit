# 🧠 NLP Microservices Toolkit

[![CI](https://github.com/twomathematicians-code/nlp-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/twomathematicians-code/nlp-toolkit/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09a3d5?logo=spacy)](https://spacy.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://hub.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-grade NLP microservices powered by **spaCy**, **HuggingFace Transformers**, and **langdetect**. Five independent services in one deployable FastAPI application — each endpoint does one thing well with real model inference, not mocks.

---

## 🏗️ Architecture

```
FastAPI (async) → Service Layer → Models (lazy-loaded, cached)
                                  ├─ spaCy en_core_web_sm      (NER)
                                  ├─ facebook/bart-large-cnn    (Summarization)
                                  ├─ TF-IDF statistical        (Keywords)
                                  ├─ langdetect                (Language Detection)
                                  └─ Helsinki-NLP/Opus-MT      (Translation, per language pair)
```

All models are **lazy-loaded** on first request and cached in memory — no startup delay, no wasted resources for unused services.

## 🔧 Services

| Service | Endpoint | Model / Method | Input | Output |
|:--|:--|:--|:--|:--|
| **Named Entity Recognition** | `POST /api/v1/ner` | spaCy `en_core_web_sm` | Raw text | Entities with label, start, end, confidence |
| **Text Summarization** | `POST /api/v1/summarize` | `facebook/bart-large-cnn` (fallback: `distilbart`) | Article + optional style | Condensed summary |
| **Keyword Extraction** | `POST /api/v1/keywords` | TF-IDF statistical scoring | Document | Ranked keywords + n-grams with scores |
| **Language Detection** | `POST /api/v1/detect-language` | `langdetect` probabilistic | Text | ISO code, language name, confidence, alternatives |
| **Translation** | `POST /api/v1/translate` | Helsinki-NLP/Opus-MT | Text + source + target | Translated text |

### Supported Translation Pairs

English ↔ French, German, Spanish, Italian, Portuguese, Dutch, Hindi, Arabic, Chinese, Japanese, Russian (22 pairs total)

## 🚀 Quick Start

### Docker (recommended)

```bash
docker compose up -d
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Local development

```bash
# Prerequisites: Python 3.11+, uv (or pip)
uv venv --python 3.11
source .venv/bin/activate  # .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Run
uvicorn src.api.main:app --reload --port 8000
```

## 📡 API Examples

### Named Entity Recognition
```bash
curl -X POST http://localhost:8000/api/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Barack Obama was born in Honolulu, Hawaii and studied at Harvard University"}'
```

### Text Summarization
```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long article text here...", "style": "bullets"}'
```

### Keyword Extraction
```bash
curl -X POST http://localhost:8000/api/v1/keywords \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence...", "top_k": 10}'
```

### Language Detection
```bash
curl -X POST http://localhost:8000/api/v1/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

### Translation (English → French)
```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "source_lang": "en", "target_lang": "fr"}'
```

## ⚙️ Configuration

Environment variables (prefix `NLP_`):

| Variable | Default | Description |
|:--|:--|:--|
| `NLP_SPACY_MODEL` | `en_core_web_sm` | spaCy model for NER |
| `NLP_SUMMARIZER_MODEL` | `facebook/bart-large-cnn` | HuggingFace model for summarization |
| `NLP_MAX_TEXT_LENGTH` | `100000` | Maximum input text length |
| `NLP_DEFAULT_LANGUAGE` | `en` | Default language code |
| `NLP_CORS_ORIGINS` | `*` | CORS allowed origins |
| `NLP_LOG_LEVEL` | `INFO` | Logging level |

Copy `.env.example` to `.env` for local configuration.

## 🧪 Development

```bash
make test        # Run pytest with coverage
make lint         # Format with black + lint with ruff
make download-models  # Pre-download all models
make clean        # Remove caches and build artifacts
```

## 📊 CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. **Lint** — `black --check` + `ruff check` (must pass, no `--exit-zero`)
2. **Test** — `pytest --cov` with spaCy model download
3. **Build** — Docker Buildx image build (depends on lint + test)

All three jobs must pass for a commit to be considered green.

## 📁 Project Structure

```
nlp-toolkit/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI app, endpoints, Pydantic models
│   ├── services/
│   │   ├── __init__.py          # Lazy model loading utilities
│   │   ├── ner_service.py       # spaCy Named Entity Recognition
│   │   ├── summarizer_service.py # HuggingFace BART summarization
│   │   ├── keyword_service.py    # TF-IDF keyword extraction
│   │   ├── language_service.py  # langdetect language detection
│   │   └── translation_service.py # Helsinki-NLP/Opus-MT translation
│   └── utils/
│       ├── config.py            # Pydantic Settings (env-based)
│       └── logging.py           # Structured logging formatter
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_api.py              # 20 API integration tests
│   └── test_services.py         # 14 service unit tests
├── configs/
│   └── model_config.yaml        # Default model configuration
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml
```

---

<p align="center"><i>By Mahesh Solanki</i></p>
