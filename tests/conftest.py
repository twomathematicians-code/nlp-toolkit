"""Shared test fixtures for the NLP Toolkit test suite."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.main import app


@pytest.fixture
async def client():
    """Async HTTP client for testing the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def sample_english_text() -> str:
    """Sample English text from a news article style."""
    return (
        "Apple Inc. is planning to open a new office in New York City. "
        "CEO Tim Cook announced that the company will hire over 500 engineers "
        "for the new artificial intelligence division. The office will be "
        "located in Manhattan, near Central Park. Google and Microsoft are "
        "also expanding their operations in the city."
    )


@pytest.fixture
def sample_french_text() -> str:
    """Sample French text for language detection and translation tests."""
    return (
        "Bonjour, comment allez-vous? "
        "Paris est la capitale de la France et est connue pour la tour Eiffel. "
        "La ville accueille des millions de touristes chaque année."
    )


@pytest.fixture
def sample_long_text() -> str:
    """Longer text for summarization testing."""
    return (
        "Artificial intelligence has transformed many industries in recent years. "
        "Machine learning algorithms can now process vast amounts of data to find "
        "patterns and make predictions. Deep learning, a subset of machine learning, "
        "uses neural networks with many layers to learn hierarchical representations "
        "of data. Natural language processing enables computers to understand and generate "
        "human language. Computer vision allows machines to interpret and understand "
        "visual information from the world. Reinforcement learning trains agents to "
        "make sequential decisions by rewarding desired behaviors. These technologies "
        "are being applied in healthcare for disease diagnosis, in finance for fraud "
        "detection, in transportation for autonomous vehicles, and in entertainment for "
        "recommendation systems. The future of AI promises even greater advances as "
        "computing power increases and algorithms become more sophisticated."
    )
