# 🧠 ML NLP Toolkit

[![CI/CD](https://github.com/twomathematicians-code/ml-nlp-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/twomathematicians-code/ml-nlp-toolkit/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://hub.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-Ready-FFD21E)](https://huggingface.co/)

**Production NLP microservices: Named Entity Recognition, Text Summarization, Keyword Extraction, Language Detection & Translation — all via FastAPI with HuggingFace transformers.**

## 🎯 NLP Modules

| Module | Algorithm | Output |
|---|---|---|
| **Named Entity Recognition** | spaCy + BERT-NER | Persons, Orgs, Locations, Dates |
| **Text Summarization** | BART / T5 | Extractive & Abstractive summaries |
| **Keyword Extraction** | KeyBERT + YAKE | Top-N keywords with scores |
| **Language Detection** | FastText + langdetect | 170+ languages |
| **Machine Translation** | OPUS-MT / NLLB | 50+ language pairs |

## 🚀 Quick Start

```bash
git clone https://github.com/twomathematicians-code/ml-nlp-toolkit.git
cd ml-nlp-toolkit
docker-compose up --build
```

API at `http://localhost:8000/docs`

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ner` | Named entity recognition |
| `POST` | `/api/v1/summarize` | Text summarization |
| `POST` | `/api/v1/keywords` | Keyword extraction |
| `POST` | `/api/v1/detect-language` | Language detection |
| `POST` | `/api/v1/translate` | Machine translation |
| `GET` | `/api/v1/health` | Health check |

## 👤 Author

**Mahesh Solanki** — [LinkedIn](https://linkedin.com/in/maheshsolanki-16b9a6a5) | [GitHub](https://github.com/twomathematicians-code)
