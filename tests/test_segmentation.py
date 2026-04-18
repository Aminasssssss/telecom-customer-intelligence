import numpy as np
import pandas as pd

from src.models.segmentation import (
    CustomerSegmenter,
    auto_k_search,
    name_segments,
    run_dbscan,
)


def make_features(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "recency": rng.integers(0, 300, n),
        "frequency": rng.integers(1, 50, n),
        "monetary": rng.uniform(1000, 20000, n),
        "tenure": rng.integers(1, 60, n),
    })


def test_segmenter_fit_predict():
    X = make_features()
    seg = CustomerSegmenter(n_clusters=4).fit(X)
    labels = seg.predict(X)
    assert len(labels) == len(X)
    assert set(labels).issubset({0, 1, 2, 3})


def test_segmenter_silhouette():
    X = make_features()
    seg = CustomerSegmenter(n_clusters=4).fit(X)
    score = seg.evaluate(X)
    assert -1.0 <= score <= 1.0


def test_segmenter_save_load(tmp_path):
    X = make_features()
    seg = CustomerSegmenter(n_clusters=3).fit(X)
    path = tmp_path / "seg.joblib"
    seg.save(str(path))
    loaded = CustomerSegmenter.load(str(path))
    assert (seg.predict(X) == loaded.predict(X)).all()


def test_segmenter_not_fitted_raises():
    X = make_features()
    seg = CustomerSegmenter(n_clusters=3)
    import pytest
    with pytest.raises(RuntimeError):
        seg.predict(X)


def test_dbscan_runs():
    X = make_features(n=200)
    res = run_dbscan(X, eps=0.8, min_samples=10)
    assert "n_clusters" in res
    assert len(res["labels"]) == len(X)


def test_auto_k_search():
    X = make_features(n=200)
    res = auto_k_search(X, k_range=range(3, 6))
    assert set(res["silhouette"].keys()) == {3, 4, 5}
    assert all(-1.0 <= v <= 1.0 for v in res["silhouette"].values())


def test_name_segments():
    X = make_features(n=400)
    seg = CustomerSegmenter(n_clusters=5).fit(X)
    labels = seg.predict(X)
    profile = seg.build_profile(X, labels)
    names = name_segments(profile)
    assert len(names) == 5
    assert "Champions" in names.values()
