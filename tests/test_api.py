import pytest
from httpx import ASGITransport, AsyncClient
from src.api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_ner(client):
    r = await client.post("/api/v1/ner", json={"text": "John Smith works at Google in New York"})
    assert r.status_code == 200
    assert "entities" in r.json()

@pytest.mark.asyncio
async def test_summarize(client):
    text = "The quick brown fox jumps over the lazy dog. " * 20
    r = await client.post("/api/v1/summarize", json={"text": text})
    assert r.status_code == 200
    assert r.json()["compression_ratio"] <= 1.0

@pytest.mark.asyncio
async def test_keywords(client):
    r = await client.post("/api/v1/keywords?top_k=5", json={"text": "Machine learning algorithms process data to find patterns and make predictions about future events"})
    assert r.status_code == 200
    assert len(r.json()["keywords"]) <= 5

@pytest.mark.asyncio
async def test_language_detection(client):
    r = await client.post("/api/v1/detect-language", json={"text": "Bonjour, comment allez-vous?"})
    assert r.status_code == 200
    assert "detected_language" in r.json()

@pytest.mark.asyncio
async def test_translate(client):
    r = await client.post("/api/v1/translate", json={"text": "Hello world", "target_lang": "fr"})
    assert r.status_code == 200
    assert r.json()["target_lang"] == "fr"
