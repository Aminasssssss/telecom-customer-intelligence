"""customer segmentation with K-means and DBSCAN."""
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationResult:
    labels: np.ndarray
    silhouette: float
    profile: pd.DataFrame


class CustomerSegmenter:
    """K-means based segmenter with scaling and persistence."""

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame) -> "CustomerSegmenter":
        self.feature_cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.model.fit(X_scaled)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        X_scaled = self.scaler.transform(X[self.feature_cols])
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("model not fitted")
        X_scaled = self.scaler.transform(X[self.feature_cols])
        labels = self.model.predict(X_scaled)
        sample = min(5000, len(X_scaled))
        return float(silhouette_score(X_scaled, labels, sample_size=sample))

    def build_profile(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        profile = X.copy()
        profile["segment"] = labels
        return profile.groupby("segment").agg(["mean", "count"])

    def save(self, path: str) -> None:
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "features": self.feature_cols},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CustomerSegmenter":
        data = joblib.load(path)
        seg = cls(n_clusters=data["model"].n_clusters)
        seg.model = data["model"]
        seg.scaler = data["scaler"]
        seg.feature_cols = data["features"]
        return seg


def run_dbscan(X: pd.DataFrame, eps: float = 0.8, min_samples: int = 30) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return {"labels": labels, "n_clusters": n_clusters, "n_noise": n_noise}


def auto_k_search(
    X: pd.DataFrame, k_range: range = range(3, 9), random_state: int = 42
) -> dict:
    """Search optimal k by silhouette + elbow inertia."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores: dict[int, float] = {}
    inertias: dict[int, float] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)
        sample = min(5000, len(X_scaled))
        scores[k] = float(silhouette_score(X_scaled, km.labels_, sample_size=sample))
        inertias[k] = float(km.inertia_)

    return {"silhouette": scores, "inertia": inertias}


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Heuristic naming based on monetary tier within profile."""
    if ("monetary", "mean") not in profile.columns:
        return {int(i): f"Segment {i}" for i in profile.index}

    monetary = profile[("monetary", "mean")]
    sorted_segments = monetary.sort_values(ascending=False).index.tolist()

    names = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]
    out: dict[int, str] = {}
    for i, seg_id in enumerate(sorted_segments):
        out[int(seg_id)] = names[i] if i < len(names) else f"Other {i}"
    return out
