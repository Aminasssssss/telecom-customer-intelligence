"""Telecom Customer Intelligence API."""
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from src.api.dependencies import (
    SEGMENT_NAMES,
    get_churn_model,
    get_recommender,
    get_segmenter,
    models_status,
)
from src.api.schemas import (
    ChurnResponse,
    CustomerFeatures,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    SegmentRequest,
    SegmentResponse,
    TariffRecommendation,
)

app = FastAPI(
    title="Telecom Customer Intelligence API",
    description="Churn prediction, tariff recommendations, and customer segmentation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request duration", ["endpoint"]
)


# ------------------------------------------------------------------ #
# health
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(status="ok", models_loaded=models_status())


@app.get("/metrics", response_class=PlainTextResponse, tags=["system"])
def metrics():
    return generate_latest().decode("utf-8")


# ------------------------------------------------------------------ #
# churn prediction
# ------------------------------------------------------------------ #

@app.post("/predict/churn", response_model=ChurnResponse, tags=["ml"])
def predict_churn(features: CustomerFeatures):
    t0 = time.time()
    model = get_churn_model()

    if model is None:
        REQUEST_COUNT.labels(endpoint="churn", status="503").inc()
        raise HTTPException(503, "churn model not loaded — run make train first")

    df = pd.DataFrame([features.model_dump()])

    try:
        proba = float(model.predict_proba(df)[0])
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="churn", status="500").inc()
        raise HTTPException(500, str(e)) from e

    # SHAP top factors
    try:
        expl = model.explain(df, sample_size=1)
        paired = sorted(
            zip(expl["feature_names"], expl["importance"]),
            key=lambda x: x[1],
            reverse=True,
        )
        top_factors = [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in paired[:5]
        ]
    except Exception:
        top_factors = []

    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    REQUEST_COUNT.labels(endpoint="churn", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="churn").observe(time.time() - t0)

    return ChurnResponse(
        churn_probability=round(proba, 4),
        churn_predicted=proba >= 0.5,
        risk_level=risk,
        top_factors=top_factors,
    )


# ------------------------------------------------------------------ #
# recommendations
# ------------------------------------------------------------------ #

@app.post("/recommend/tariffs", response_model=RecommendResponse, tags=["ml"])
def recommend_tariffs(req: RecommendRequest):
    t0 = time.time()
    rec = get_recommender()

    if rec is None:
        REQUEST_COUNT.labels(endpoint="recommend", status="503").inc()
        raise HTTPException(503, "recommender not loaded — run make train first")

    try:
        if req.mode == "similar" and req.current_tariff:
            results = rec.recommend_similar(req.current_tariff, n=req.n)
        elif req.mode == "hybrid" and req.customer_id:
            results = rec.recommend_hybrid(req.customer_id, n=req.n)
        elif req.mode == "collaborative" and req.customer_id:
            results = rec.recommend_collaborative(req.customer_id, n=req.n)
        else:
            results = rec.cold_start(n=req.n, category=req.category)
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="recommend", status="500").inc()
        raise HTTPException(500, str(e)) from e

    recommendations = [
        TariffRecommendation(
            tariff_id=r["tariff_id"],
            name=r.get("name", r["tariff_id"]),
            monthly_price=r.get("monthly_price", 0),
            data_gb=r.get("data_gb", 0),
            category=r.get("category", ""),
            score=round(r["score"], 4),
        )
        for r in results
    ]

    REQUEST_COUNT.labels(endpoint="recommend", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="recommend").observe(time.time() - t0)

    return RecommendResponse(mode=req.mode, recommendations=recommendations)


# ------------------------------------------------------------------ #
# segmentation
# ------------------------------------------------------------------ #

@app.post("/segment", response_model=SegmentResponse, tags=["ml"])
def segment_customer(req: SegmentRequest):
    t0 = time.time()
    segmenter = get_segmenter()

    if segmenter is None:
        REQUEST_COUNT.labels(endpoint="segment", status="503").inc()
        raise HTTPException(503, "segmenter not loaded — run make train first")

    df = pd.DataFrame([req.model_dump()])

    try:
        label = int(segmenter.predict(df)[0])
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="segment", status="500").inc()
        raise HTTPException(500, str(e)) from e

    name = SEGMENT_NAMES.get(label, f"Segment {label}")

    REQUEST_COUNT.labels(endpoint="segment", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="segment").observe(time.time() - t0)

    return SegmentResponse(segment_id=label, segment_name=name)
