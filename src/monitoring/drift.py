"""data drift detection using PSI and Kolmogorov-Smirnov tests."""
import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """PSI between two distributions. < 0.1 stable, 0.1-0.25 moderate, > 0.25 shift."""
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_pct = np.percentile(expected, breakpoints)
    expected_pct[0] = -np.inf
    expected_pct[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=expected_pct)
    actual_counts, _ = np.histogram(actual, bins=expected_pct)

    expected_pct_arr = expected_counts / max(len(expected), 1)
    actual_pct_arr = actual_counts / max(len(actual), 1)

    eps = 1e-6
    expected_pct_arr = np.where(expected_pct_arr == 0, eps, expected_pct_arr)
    actual_pct_arr = np.where(actual_pct_arr == 0, eps, actual_pct_arr)

    psi = np.sum(
        (actual_pct_arr - expected_pct_arr) * np.log(actual_pct_arr / expected_pct_arr)
    )
    return float(psi)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> dict:
    statistic, pvalue = stats.ks_2samp(expected, actual)
    return {"statistic": float(statistic), "pvalue": float(pvalue)}


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run PSI + KS on numeric features, return ranked drift report."""
    if columns is None:
        columns = [
            c for c in reference.select_dtypes(include=[np.number]).columns
            if c in current.columns
        ]

    rows = []
    for col in columns:
        ref = reference[col].dropna().values
        cur = current[col].dropna().values
        if len(ref) == 0 or len(cur) == 0:
            continue

        psi = population_stability_index(ref, cur)
        ks = ks_test(ref, cur)

        drift_flag = bool(psi > psi_threshold or ks["pvalue"] < ks_alpha)
        if psi > 0.25:
            severity = "high"
        elif psi > 0.1:
            severity = "medium"
        else:
            severity = "low"

        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_pvalue": round(ks["pvalue"], 4),
            "drift_detected": drift_flag,
            "severity": severity,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "feature", "psi", "ks_statistic", "ks_pvalue", "drift_detected", "severity"
        ])

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
