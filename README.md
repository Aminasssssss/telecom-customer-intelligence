# Telecom Customer Intelligence Platform

End-to-end ML platform for a telecom operator. One codebase, four capabilities: customer
segmentation, churn prediction, tariff recommendations, and data drift monitoring.
Production-ready with FastAPI serving, Streamlit dashboard, MLflow experiment tracking,
Docker Compose stack, Kubernetes manifests, and GitHub Actions CI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
│  data/raw/  ──►  src/data/synthetic.py  (5 000 customers, 56k CDRs) │
│                  src/data/loader.py     (+ IBM Telco Kaggle support) │
│                  src/data/preprocessing.py  (sklearn pipeline)       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       Feature Engineering                            │
│  src/features/rfm.py   Recency / Frequency / Monetary scoring        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                          ML Models                                   │
│  segmentation.py   K-means (k=5) + DBSCAN + auto silhouette search  │
│  churn.py          XGBoost + LightGBM + SHAP explanations           │
│  recommender.py    SVD collaborative + cosine content + cold start  │
│  drift.py          PSI + Kolmogorov-Smirnov per feature             │
└──────────┬──────────────────────────┬───────────────────────────────┘
           │  MLflow tracking         │
┌──────────▼──────────┐   ┌──────────▼───────────────────────────────┐
│   FastAPI  :8000    │   │        Streamlit Dashboard  :8501         │
│  /predict/churn     │   │  01 Overview  — KPIs, revenue, cities     │
│  /recommend/tariffs │   │  02 Segments  — profiles, RFM scatter     │
│  /segment           │   │  03 Churn     — risk dist, SHAP, table    │
│  /health            │   │  04 Recsys    — live tariff suggestions   │
│  /metrics (Prom.)   │   └──────────────────────────────────────────┘
└─────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                     Infrastructure                                   │
│  docker-compose.yml  API + Dashboard + MLflow + PostgreSQL           │
│  k8s/                Deployment + Service + HPA + Ingress            │
│  .github/workflows/  CI: lint → test → coverage → docker build      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML models | XGBoost, LightGBM, scikit-learn, scikit-surprise |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic, Prometheus |
| Dashboard | Streamlit, Plotly |
| Data | pandas, numpy, scipy |
| Testing | pytest, pytest-cov |
| Linting | ruff, mypy |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |
| Orchestration | Kubernetes (Deployment + HPA + Ingress) |
| Database | PostgreSQL (star schema) |

---

## Quickstart

```bash
git clone https://github.com/Aminasssssss/telecom-customer-intelligence
cd telecom-customer-intelligence

python -m venv .venv && source .venv/bin/activate
make install-dev       # deps + pre-commit hooks

make data              # generate 5 000 customers + 56k transactions
make train             # segmentation + churn + recommender → models/

make test-cov          # 44 tests, coverage report in htmlcov/

make api               # FastAPI at http://localhost:8000/docs
make dashboard         # Streamlit at http://localhost:8501
make mlflow            # MLflow UI at http://localhost:5000
```

Full stack with Docker:

```bash
make docker-up         # postgres + mlflow + api + dashboard
make docker-down
```

---

## API Endpoints

### `POST /predict/churn`
Predict churn probability for a customer with SHAP explanation.

```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34, "gender": "M", "city": "Almaty",
    "tenure_months": 6, "tariff_id": "T01",
    "contract_type": "monthly", "payment_method": "cash",
    "paperless_billing": 1, "has_internet": 1, "has_tv": 0, "has_roaming": 0,
    "recency": 5, "frequency": 6, "monetary": 9000,
    "avg_data_gb": 4.5, "avg_minutes": 95.0
  }'
```

Response:
```json
{
  "churn_probability": 0.74,
  "churn_predicted": true,
  "risk_level": "high",
  "top_factors": [
    {"feature": "contract_type_monthly", "importance": 0.312},
    {"feature": "tenure_months", "importance": 0.198}
  ]
}
```

### `POST /recommend/tariffs`
Get tariff recommendations. Modes: `cold_start`, `similar`, `hybrid`, `collaborative`.

```bash
curl -X POST http://localhost:8000/recommend/tariffs \
  -H "Content-Type: application/json" \
  -d '{"current_tariff": "T03", "mode": "similar", "n": 3}'
```

### `POST /segment`
Assign a customer to a segment.

```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{"recency": 5, "frequency": 12, "monetary": 48000, "tenure_months": 24, "avg_data_gb": 22}'
```

Full interactive docs at `http://localhost:8000/docs`.

---

## Models

### Segmentation — K-means (k=5)
Trained on RFM + tenure + avg_data_gb features. Auto k-search via silhouette score across k=3..8.

| Segment | Profile |
|---|---|
| Champions | High spend, recent, frequent |
| Loyal | Long tenure, stable spend |
| Potential | Mid spend, growing frequency |
| At Risk | Dropping recency, mid value |
| Hibernating | Low spend, long since active |

### Churn Prediction
XGBoost and LightGBM trained on full customer feature set. Best model saved as `churn_best.joblib`.
SHAP TreeExplainer used for feature attribution.

### Recommender
Three modes:
- **collaborative** — SVD via scikit-surprise on customer-tariff spend matrix. Falls back to cold start for customers with fewer than 3 interactions.
- **similar** — cosine similarity on normalised tariff feature vectors (price, data, minutes, category).
- **hybrid** — weighted blend of collaborative (α=0.6) and content-based scores.

### Drift Detection
PSI and Kolmogorov-Smirnov test per numeric feature. Returns a ranked report with severity labels (low / medium / high).

---

## Data

Synthetic data generated by `scripts/generate_data.py`:
- 5 000 customers with age-correlated tariff assignment and realistic churn drivers
  (contract type, tenure, payment method)
- 56 000+ monthly transactions across 8 tariffs (1 200–12 000 KZT/month)

To use the real IBM Telco Customer Churn dataset:
```bash
# download from kaggle.com/datasets/blastchar/telco-customer-churn
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco.csv
# then call src.data.loader.load_telco_kaggle() in your pipeline
```

---

## SQL Analytics

`sql/schema.sql` — PostgreSQL star schema: `fact_transactions`, `dim_customer`,
`dim_tariff`, `dim_date`, `dim_city`.

`sql/analytics_queries.sql` — 7 production queries using:
- `LAG` / `LEAD` for MoM growth and tariff change detection
- `ROW_NUMBER` for top-N customers per city
- `NTILE` for spend quartiles with running totals
- `RANK` for churn risk ranking by segment
- CTEs for cohort retention analysis

---

## Testing

```bash
make test-cov
```

```
tests/test_api.py           8 tests   FastAPI endpoints
tests/test_churn.py         5 tests   XGBoost + LightGBM + SHAP + save/load
tests/test_drift.py         6 tests   PSI + KS + drift report
tests/test_recommender.py   9 tests   cold start + similar + hybrid + save/load
tests/test_rfm.py           4 tests   RFM computation + merge
tests/test_segmentation.py  7 tests   KMeans + DBSCAN + auto-k + save/load
tests/test_synthetic.py     5 tests   data generation + reproducibility
─────────────────────────────────────
Total                       44 tests
```

---

## Project Structure

```
telecom-customer-intelligence/
├── configs/config.yaml          all hyperparameters in one place
├── src/
│   ├── data/                    synthetic generator, loaders, preprocessing
│   ├── features/                RFM feature engineering
│   ├── models/                  segmentation, churn, recommender
│   ├── monitoring/              drift detection (PSI + KS)
│   ├── api/                     FastAPI app, schemas, model deps
│   └── dashboard/               Streamlit app + 4 pages
├── tests/                       44 tests
├── notebooks/                   eda.py, ab_testing.py
├── sql/                         schema.sql, analytics_queries.sql
├── scripts/                     generate_data.py, train_all.py
├── k8s/                         deployment, service, hpa, ingress
├── .github/workflows/           CI pipeline
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## License

MIT
