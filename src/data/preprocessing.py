"""sklearn preprocessing pipeline for customer features."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = ["gender", "city", "contract_type", "payment_method", "tariff_id"]
NUMERICAL_COLS = [
    "age", "tenure_months", "paperless_billing",
    "has_internet", "has_tv", "has_roaming",
]
RFM_NUMERICAL = ["recency", "frequency", "monetary", "avg_data_gb", "avg_minutes"]


def build_feature_pipeline(include_rfm: bool = True) -> ColumnTransformer:
    num_cols = NUMERICAL_COLS + (RFM_NUMERICAL if include_rfm else [])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def split_features_target(
    df: pd.DataFrame, target: str = "churn", drop: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    drop = drop or []
    excluded = {target, "customer_id", *drop}
    cols = [c for c in df.columns if c not in excluded]
    return df[cols], df[target]
