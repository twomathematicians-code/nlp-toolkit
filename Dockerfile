FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml ./
RUN pip install --quiet poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

COPY src/ src/

RUN useradd -m nluser && chown -R nluser /app
USER nluser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--port", "8000", "--host", "0.0.0.0", "--workers", "3"]
