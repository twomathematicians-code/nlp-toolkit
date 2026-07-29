# 🧠 NLP Microservices Toolkit

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
