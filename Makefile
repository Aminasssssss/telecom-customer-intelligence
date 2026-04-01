.PHONY: help install install-dev data train test test-cov lint format api dashboard mlflow docker-up docker-down clean

help:
	@echo "install        install runtime deps"
	@echo "install-dev    install dev deps and pre-commit"
	@echo "data           generate synthetic dataset"
	@echo "train          train segmentation and churn models"
	@echo "test           run tests"
	@echo "test-cov       run tests with coverage report"
	@echo "lint           run ruff and mypy"
	@echo "format         format code with ruff"
	@echo "api            run FastAPI locally"
	@echo "dashboard      run Streamlit dashboard"
	@echo "mlflow         run MLflow UI"
	@echo "docker-up      bring up full stack"
	@echo "docker-down    stop stack"
	@echo "clean          remove caches and artifacts"

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

data:
	PYTHONPATH=. python scripts/generate_data.py

train:
	PYTHONPATH=. python scripts/train_all.py

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
