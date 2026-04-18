"""API endpoint tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_FEATURES = {
    "age": 34, "gender": "M", "city": "Almaty",
    "tenure_months": 24, "tariff_id": "T03",
    "contract_type": "monthly", "payment_method": "card",
    "paperless_billing": 1, "has_internet": 1,
    "has_tv": 0, "has_roaming": 0,
    "recency": 5.0, "frequency": 12.0,
    "monetary": 48000.0, "avg_data_gb": 22.5, "avg_minutes": 480.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "api_requests_total" in r.text


def test_predict_churn_no_model():
    # without a trained model the endpoint returns 503
    r = client.post("/predict/churn", json=VALID_FEATURES)
    # either 200 (model exists) or 503 (no model) — both are valid responses
    assert r.status_code in (200, 503)


def test_predict_churn_validation_error():
    bad = VALID_FEATURES.copy()
    bad["age"] = -5  # below minimum
    r = client.post("/predict/churn", json=bad)
    assert r.status_code == 422


def test_recommend_tariffs_cold_start():
    r = client.post("/recommend/tariffs", json={"mode": "cold_start", "n": 3})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert data["mode"] == "cold_start"
        assert len(data["recommendations"]) <= 3


def test_recommend_tariffs_invalid_n():
    r = client.post("/recommend/tariffs", json={"mode": "cold_start", "n": 99})
    assert r.status_code == 422


def test_segment_endpoint():
    payload = {
        "recency": 5.0, "frequency": 12.0, "monetary": 48000.0,
        "tenure_months": 24.0, "avg_data_gb": 22.5,
    }
    r = client.post("/segment", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "segment_id" in data
        assert "segment_name" in data


def test_segment_missing_field():
    r = client.post("/segment", json={"recency": 5.0})
    assert r.status_code == 422
