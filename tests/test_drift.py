import numpy as np
import pandas as pd

from src.monitoring.drift import detect_drift, ks_test, population_stability_index


def test_psi_no_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    psi = population_stability_index(a, b)
    assert psi < 0.1


def test_psi_with_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2, 1, 5000)
    psi = population_stability_index(a, b)
    assert psi > 0.2


def test_ks_no_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0, 1, 1000)
    res = ks_test(a, b)
    assert res["pvalue"] > 0.05


def test_ks_with_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0.5, 1, 1000)
    res = ks_test(a, b)
    assert res["pvalue"] < 0.05


def test_detect_drift_dataframe():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "stable": rng.normal(0, 1, 1000),
        "shifted": rng.normal(5, 1, 1000),
    })
    cur = pd.DataFrame({
        "stable": rng.normal(0, 1, 1000),
        "shifted": rng.normal(8, 1, 1000),
    })
    report = detect_drift(ref, cur)
    assert len(report) == 2
    shifted_row = report[report["feature"] == "shifted"].iloc[0]
    assert shifted_row["drift_detected"]
    stable_row = report[report["feature"] == "stable"].iloc[0]
    assert not stable_row["drift_detected"]


def test_detect_drift_empty():
    ref = pd.DataFrame({"a": []})
    cur = pd.DataFrame({"a": []})
    report = detect_drift(ref, cur, columns=["a"])
    assert len(report) == 0
