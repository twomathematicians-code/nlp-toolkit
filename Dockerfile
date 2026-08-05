FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml poetry.lock* ./
RUN pip install --quiet poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

# Download spaCy model
RUN poetry run python -m spacy download en_core_web_sm

# Copy application code
COPY src/ src/
COPY configs/ configs/

# Create non-root user
RUN useradd -m nluser && chown -R nluser /app
USER nluser

ENV PYTHONUNBUFFERED=1
ENV NLP_SPACY_MODEL=en_core_web_sm
ENV NLP_SUMMARIZER_MODEL=sshleifer/distilbart-cnn-12-6
ENV NLP_KEYWORD_MODEL=distilbert-base-uncased

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--port", "8000", "--host", "0.0.0.0", "--workers", "1"]
