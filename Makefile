.PHONY: install test lint format run docker-build docker-up docker-down clean

install:
	poetry install --with dev

test:
	poetry run pytest tests/ --cov=src -v --tb=short

lint:
	poetry run black --check src/ tests/
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

run:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

download-models:
	poetry run python -m spacy download en_core_web_sm

docker-build:
	docker build -t nlp-toolkit:latest .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
