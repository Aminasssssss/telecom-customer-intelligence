"""RFM features from transaction history."""
from datetime import datetime

import pandas as pd


def compute_rfm(
    transactions: pd.DataFrame, reference_date: datetime | None = None
) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary plus usage means per customer."""
    if not pd.api.types.is_datetime64_any_dtype(transactions["transaction_date"]):
        transactions = transactions.copy()
        transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    if reference_date is None:
        reference_date = transactions["transaction_date"].max()

    grouped = transactions.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("amount", "sum"),
        avg_data_gb=("data_gb_used", "mean"),
        avg_minutes=("minutes_used", "mean"),
    ).reset_index()

    # higher score = better engagement, use rank to avoid duplicate-bin issues
    grouped["r_score"] = pd.qcut(
        grouped["recency"].rank(method="first", ascending=False),
        5, labels=[1, 2, 3, 4, 5],
    ).astype(int)
    grouped["f_score"] = pd.qcut(
        grouped["frequency"].rank(method="first"),
        5, labels=[1, 2, 3, 4, 5],
    ).astype(int)
    grouped["m_score"] = pd.qcut(
        grouped["monetary"].rank(method="first"),
        5, labels=[1, 2, 3, 4, 5],
    ).astype(int)

    grouped["rfm_score"] = (
        grouped["r_score"] * 100 + grouped["f_score"] * 10 + grouped["m_score"]
    )

    return grouped


def merge_with_customers(customers: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    merged = customers.merge(rfm, on="customer_id", how="left")
    fill_values = {
        "recency": 999, "frequency": 0, "monetary": 0,
        "avg_data_gb": 0, "avg_minutes": 0,
        "r_score": 1, "f_score": 1, "m_score": 1, "rfm_score": 111,
    }
    return merged.fillna(fill_values)
