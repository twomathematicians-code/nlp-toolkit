# 🧠 NLP Microservices Toolkit

[![CI](https://github.com/twomathematicians-code/nlp-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/twomathematicians-code/nlp-toolkit/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09a3d5?logo=spacy)](https://spacy.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://hub.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=120&section=header&text=Named+Entity+%7C+Summarize+%7C+Translate+%7C+Keywords" />
</p>

Five independent NLP microservices in one deployable API. Each endpoint does one thing well.

---

## 🔧 Services

| Service | Endpoint | Input | Output |
|:--|:--|:--|:--|
| **Named Entity Recognition** | `POST /api/v1/ner` | Raw text | Persons, Orgs, Locations with confidence |
| **Text Summarization** | `POST /api/v1/summarize` | Article | Condensed summary |
| **Keyword Extraction** | `POST /api/v1/keywords` | Document | Ranked keywords + n-grams |
| **Language Detection** | `POST /api/v1/detect-language` | Text | ISO code + confidence |
| **Translation** | `POST /api/v1/translate` | Text + target | Translated text |

## Deploy

```bash
docker compose up -d
# Docs: http://localhost:8000/docs
```

## Example

```bash
curl -X POST http://localhost:8000/api/v1/ner \
  -d '{"text": "Barack Obama was born in Honolulu, Hawaii and studied at Harvard University"}'
```

---

<p align="center"><i>By Mahesh Solanki</i></p>
