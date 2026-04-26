"""mlflow setup and logging helpers."""
import os
from contextlib import contextmanager
from typing import Any

import mlflow


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str = "default") -> None:
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None):
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_metrics_dict(metrics: dict[str, Any]) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))


def log_params_dict(params: dict[str, Any]) -> None:
    flat = {k: v for k, v in params.items() if not isinstance(v, (dict, list))}
    if flat:
        mlflow.log_params(flat)
