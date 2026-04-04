"""data loaders for synthetic and IBM Telco Kaggle dataset."""
from pathlib import Path

import pandas as pd


def load_customers(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_transactions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["transaction_date"])


def load_tariffs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_telco_kaggle(path: Path) -> pd.DataFrame:
    """Load IBM Telco Customer Churn dataset from Kaggle if available."""
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())

    if "churn" in df.columns and df["churn"].dtype == object:
        df["churn"] = (df["churn"].str.lower() == "yes").astype(int)

    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].astype(int)

    return df
