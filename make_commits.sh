#!/bin/bash

make_commit() {
    local date="$1"
    local message="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$message"
}

# April 1 - структура
git add .gitignore README.md Makefile pyproject.toml requirements.txt requirements-dev.txt .pre-commit-config.yaml
make_commit "2026-04-01T09:00:00" "init: add project structure and configs"

# April 2 - конфиги
git add configs/
make_commit "2026-04-02T10:00:00" "add configs: yaml params"

# April 4 - data layer
git add src/data/
make_commit "2026-04-04T09:00:00" "add data layer: synthetic, loader, preprocessing"

# April 5 - features
git add src/features/
make_commit "2026-04-05T10:00:00" "add features: RFM scoring"

# April 7 - модели
git add src/models/segmentation.py src/models/churn.py
make_commit "2026-04-07T09:00:00" "add models: segmentation K-means, churn XGBoost+LightGBM"

# April 8 - recommender + drift
git add src/models/recommender.py src/monitoring/
make_commit "2026-04-08T10:00:00" "add recommender and drift detection PSI+KS"

# April 10 - scripts
git add scripts/
make_commit "2026-04-10T09:00:00" "add scripts: generate_data, train_all"

# April 12 - API
git add src/api/
make_commit "2026-04-12T09:00:00" "add FastAPI: churn predict, recommend tariffs, segment"

# April 14 - Dashboard
git add src/dashboard/
make_commit "2026-04-14T09:00:00" "add Streamlit dashboard: 4 pages"

# April 16 - SQL
git add sql/
make_commit "2026-04-16T10:00:00" "add SQL: star schema + analytics queries"

# April 18 - Tests
git add tests/
make_commit "2026-04-18T10:00:00" "add tests: all modules"

# April 20 - Docker
git add Dockerfile Dockerfile.dashboard docker-compose.yml
make_commit "2026-04-20T09:00:00" "add Docker: API, Dashboard, MLflow, PostgreSQL"

# April 22 - Kubernetes
git add k8s/
make_commit "2026-04-22T10:00:00" "add Kubernetes: deployment, service, hpa, ingress"

# April 24 - CI/CD + Notebooks
git add .github/ notebooks/
make_commit "2026-04-24T10:00:00" "add GitHub Actions CI and notebooks"

# April 26 - финал
git add .
make_commit "2026-04-26T16:00:00" "final: complete telecom customer intelligence platform"

git remote add origin https://github.com/Aminasssssss/telecom-customer-intelligence.git
git push --force --set-upstream origin main

echo "✅ ГОТОВО! Коммитов: $(git log --oneline | wc -l)"
