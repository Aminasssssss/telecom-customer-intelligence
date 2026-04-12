"""model loading with cached singletons."""
from functools import lru_cache
from pathlib import Path

from src.models.churn import ChurnPredictor
from src.models.recommender import TariffRecommender
from src.models.segmentation import CustomerSegmenter

MODELS_DIR = Path("models")

SEGMENT_NAMES = {
    0: "Champions", 1: "Loyal", 2: "Potential", 3: "At Risk", 4: "Hibernating"
}


@lru_cache(maxsize=1)
def get_churn_model() -> ChurnPredictor | None:
    path = MODELS_DIR / "churn_best.joblib"
    if path.exists():
        return ChurnPredictor.load(str(path))
    return None


@lru_cache(maxsize=1)
def get_segmenter() -> CustomerSegmenter | None:
    path = MODELS_DIR / "segmenter.joblib"
    if path.exists():
        return CustomerSegmenter.load(str(path))
    return None


@lru_cache(maxsize=1)
def get_recommender() -> TariffRecommender | None:
    path = MODELS_DIR / "recommender.joblib"
    if path.exists():
        return TariffRecommender.load(str(path))
    return None


def models_status() -> dict[str, bool]:
    return {
        "churn": (MODELS_DIR / "churn_best.joblib").exists(),
        "segmenter": (MODELS_DIR / "segmenter.joblib").exists(),
        "recommender": (MODELS_DIR / "recommender.joblib").exists(),
    }
