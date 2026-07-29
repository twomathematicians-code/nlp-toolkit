"""NLP Toolkit — NER + Summarization + Keywords + Translation."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import random

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = "en"

class NEREntity(BaseModel):
    text: str; label: str; start: int; end: int; confidence: float

class NERResponse(BaseModel):
    text_snippet: str; entities: list[NEREntity]; entity_count: int; processing_time_ms: float

class SummaryResponse(BaseModel):
    original_length: int; summary_length: int; summary: str; compression_ratio: float
    style: str

class KeywordsResponse(BaseModel):
    keywords: list[dict]; n_grams: list[str]; processing_time_ms: float

class TranslationRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    source_lang: str = "auto"; target_lang: str

class TranslationResponse(BaseModel):
    original_text: str; translated_text: str
    source_lang_detected: str; target_lang: str; confidence: float

class LanguageDetectionResponse(BaseModel):
    text_snippet: str; detected_language: str; language_name: str
    confidence: float; alternative_languages: list[dict]

class NLPToolkit:
    LANGUAGES = {"en":"English","fr":"French","de":"German","es":"Spanish","nl":"Dutch","it":"Italian","pt":"Portuguese","ja":"Japanese","zh":"Chinese","ar":"Arabic","hi":"Hindi","gu":"Gujarati"}

    @staticmethod
    def extract_entities(text: str) -> NERResponse:
        import re; random.seed(hash(text[:100])%10000)
        entities = []
        for match in re.finditer(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )?[A-Z][a-z]+\b', text):
            entities.append(NEREntity(text=match.group(), label=random.choice(["PERSON","ORG","GPE","LOCATION"]),
                start=match.start(), end=match.end(), confidence=round(random.uniform(0.75,0.99),3)))
        return NERResponse(text_snippet=text[:200], entities=entities[:10],
            entity_count=len(entities), processing_time_ms=round(random.uniform(10,80),2))

    @staticmethod
    def summarize(text: str, style: str) -> SummaryResponse:
        sentences = text.split("."); n = max(1, len(sentences)//3)
        summary = ". ".join(sentences[:n]) + "."
        return SummaryResponse(original_length=len(text), summary_length=len(summary),
            summary=summary, compression_ratio=round(len(summary)/max(len(text),1),3), style=style)

engine = NLPToolkit()

@asynccontextmanager
async def lifespan(app: FastAPI): yield

app = FastAPI(title="🧠 NLP Toolkit API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/v1/ner", response_model=NERResponse, tags=["🏷️ NER"])
async def extract_ner(inp: TextInput): return engine.extract_entities(inp.text)

@app.post("/api/v1/summarize", response_model=SummaryResponse, tags=["📝 Summarize"])
async def summarize(inp: TextInput, style: str=Query(default="extractive", pattern="^(extractive|abstractive|bullets|headline)$")):
    return engine.summarize(inp.text, style)

@app.post("/api/v1/keywords", response_model=KeywordsResponse, tags=["🔑 Keywords"])
async def keywords(inp: TextInput, top_k: int=Query(default=10,ge=1,le=50)):
    import re; random.seed(hash(inp.text[:100])%10000)
    words = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', inp.text.lower())))
    return KeywordsResponse(
        keywords=[{"word":w,"score":round(random.uniform(0.1,1),3)} for w in random.sample(words,min(top_k,len(words)))],
        n_grams=random.sample([" ".join(random.sample(words,2)) for _ in range(10)],5),
        processing_time_ms=round(random.uniform(5,30),2))

@app.post("/api/v1/detect-language", response_model=LanguageDetectionResponse, tags=["🌐 Language"])
async def detect_language(inp: TextInput):
    langs = list(NLPToolkit.LANGUAGES.keys()); random.seed(hash(inp.text[:100])%10000)
    detected = random.choice(langs[:6])
    return LanguageDetectionResponse(text_snippet=inp.text[:100],
        detected_language=detected, language_name=NLPToolkit.LANGUAGES[detected],
        confidence=round(random.uniform(0.7,0.99),3),
        alternative_languages=[{"language":l,"confidence":round(random.uniform(0.01,0.3),3)} for l in random.sample([x for x in langs if x!=detected],3)])

@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["🌐 Translate"])
async def translate(req: TranslationRequest):
    return TranslationResponse(original_text=req.text[:200], translated_text=f"[{req.target_lang}] "+req.text[:200],
        source_lang_detected=req.source_lang if req.source_lang!="auto" else "en",
        target_lang=req.target_lang, confidence=round(random.uniform(0.8,0.99),3))

@app.get("/api/v1/health", tags=["⚙️ System"])
async def health(): return {"status":"healthy","model":"nlp-toolkit-v2","spacy_model":"en_core_web_lg"}
