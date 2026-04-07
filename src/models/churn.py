"""churn prediction with XGBoost and LightGBM, SHAP-based explanations."""
from dataclasses import dataclass, asdict
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.preprocessing import build_feature_pipeline


@dataclass
class ChurnMetrics:
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChurnPredictor:
    """Wraps preprocessing + tree model + SHAP, with joblib persistence."""

    def __init__(self, model_type: str = "xgboost", include_rfm: bool = True, **params):
        self.model_type = model_type
        self.include_rfm = include_rfm
        self.params = params
        self.pipeline = build_feature_pipeline(include_rfm=include_rfm)
        self.model = self._init_model()
        self._fitted = False

    def _init_model(self):
        if self.model_type == "xgboost":
            defaults = {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
            }
            defaults.update(self.params)
            return xgb.XGBClassifier(**defaults)
        if self.model_type == "lightgbm":
            defaults = {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "random_state": 42, "verbose": -1, "n_jobs": -1,
            }
            defaults.update(self.params)
            return lgb.LGBMClassifier(**defaults)
        raise ValueError(f"unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ChurnPredictor":
        X_enc = self.pipeline.fit_transform(X)
        self.model.fit(X_enc, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self.pipeline.transform(X)
        return self.model.predict(X_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self.pipeline.transform(X)
        return self.model.predict_proba(X_enc)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ChurnMetrics:
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        return ChurnMetrics(
            accuracy=float(accuracy_score(y, preds)),
            roc_auc=float(roc_auc_score(y, proba)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1=float(f1_score(y, preds, zero_division=0)),
            confusion=confusion_matrix(y, preds).tolist(),
        )

    def explain(self, X: pd.DataFrame, sample_size: int = 500) -> dict:
        """Return mean absolute SHAP importance per feature."""
        X_enc = self.pipeline.transform(X.head(sample_size))
        feature_names = self.pipeline.get_feature_names_out()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_enc)
        # lightgbm binary returns list[ndarray]
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        importance = np.abs(shap_values).mean(axis=0)
        return {
            "feature_names": list(feature_names),
            "importance": importance.tolist(),
            "shap_values": shap_values,
        }

    def save(self, path: str) -> None:
        joblib.dump({
            "model_type": self.model_type,
            "include_rfm": self.include_rfm,
            "model": self.model,
            "pipeline": self.pipeline,
            "params": self.params,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ChurnPredictor":
        data = joblib.load(path)
        inst = cls(
            model_type=data["model_type"],
            include_rfm=data.get("include_rfm", True),
            **data["params"],
        )
        inst.model = data["model"]
        inst.pipeline = data["pipeline"]
        inst._fitted = True
        return inst


def train_and_compare(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42,
    include_rfm: bool = True,
) -> dict:
    """Train XGBoost and LightGBM, return both with eval metrics on test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results = {}
    for model_type in ["xgboost", "lightgbm"]:
        predictor = ChurnPredictor(model_type=model_type, include_rfm=include_rfm)
        predictor.fit(X_train, y_train)
        results[model_type] = {
            "predictor": predictor,
            "metrics": predictor.evaluate(X_test, y_test),
            "X_test": X_test,
            "y_test": y_test,
        }

    return results
