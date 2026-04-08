"""
tariff recommendation engine.

collaborative:  SVD via scikit-surprise on customer-tariff spend matrix
content-based:  cosine similarity on tariff feature vectors
cold start:     popularity-based fallback for new/sparse customers
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except Exception:
    SURPRISE_AVAILABLE = False


# minimum interactions to trust collaborative score
MIN_INTERACTIONS = 3


def _build_ratings(transactions: pd.DataFrame) -> pd.DataFrame:
    """Customer-tariff rating matrix via normalised spend per customer."""
    agg = transactions.groupby(["customer_id", "tariff_id"]).agg(
        total_spend=("amount", "sum"),
        months=("transaction_id", "count"),
    ).reset_index()

    # per-customer min-max to 1-5 range so each customer's scale is comparable
    scaler = MinMaxScaler(feature_range=(1, 5))
    agg["rating"] = (
        agg.groupby("customer_id")["total_spend"]
        .transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel())
    )
    return agg[["customer_id", "tariff_id", "rating", "months"]]


def _build_tariff_features(tariffs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric + one-hot tariff feature matrix for cosine similarity."""
    enc = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), ["monthly_price", "data_gb", "minutes"]),
            ("cat", OneHotEncoder(sparse_output=False), ["category"]),
        ]
    )
    features = enc.fit_transform(tariffs)
    tariff_ids = tariffs["tariff_id"].tolist()
    return features, tariff_ids


class TariffRecommender:
    """Hybrid recommender for telecom tariff suggestions."""

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        random_state: int = 42,
        hybrid_alpha: float = 0.6,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.hybrid_alpha = hybrid_alpha  # weight of collaborative vs content

        self.svd_model = None
        self.ratings_df: pd.DataFrame | None = None
        self.tariffs_df: pd.DataFrame | None = None
        self.tariff_features: np.ndarray | None = None
        self.tariff_ids: list[str] = []
        self.cosine_matrix: np.ndarray | None = None
        self.popularity: pd.Series | None = None
        self._fitted = False

    def fit(
        self,
        transactions: pd.DataFrame,
        tariffs: pd.DataFrame,
        customers: pd.DataFrame | None = None,
    ) -> "TariffRecommender":
        self.tariffs_df = tariffs.copy()
        self.tariff_features, self.tariff_ids = _build_tariff_features(tariffs)
        self.cosine_matrix = cosine_similarity(self.tariff_features)

        # popularity: number of distinct customers per tariff
        self.popularity = (
            transactions.groupby("tariff_id")["customer_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        self.ratings_df = _build_ratings(transactions)

        if SURPRISE_AVAILABLE:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.ratings_df[["customer_id", "tariff_id", "rating"]], reader
            )
            trainset = data.build_full_trainset()
            self.svd_model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                random_state=self.random_state,
            )
            self.svd_model.fit(trainset)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # collaborative
    # ------------------------------------------------------------------ #

    def _interaction_count(self, customer_id: str) -> int:
        if self.ratings_df is None:
            return 0
        return int((self.ratings_df["customer_id"] == customer_id).sum())

    def recommend_collaborative(
        self, customer_id: str, n: int = 5, exclude_current: bool = True
    ) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("model not fitted")

        if not SURPRISE_AVAILABLE or self.svd_model is None:
            return self.cold_start(n=n)

        if self._interaction_count(customer_id) < MIN_INTERACTIONS:
            return self.cold_start(n=n)

        # predict score for every tariff
        scores = []
        for tid in self.tariff_ids:
            pred = self.svd_model.predict(uid=customer_id, iid=tid)
            scores.append({"tariff_id": tid, "score": pred.est})

        if exclude_current and self.ratings_df is not None:
            seen = self.ratings_df[
                self.ratings_df["customer_id"] == customer_id
            ]["tariff_id"].tolist()
            scores = [s for s in scores if s["tariff_id"] not in seen]

        scores.sort(key=lambda x: x["score"], reverse=True)
        return self._enrich(scores[:n])

    # ------------------------------------------------------------------ #
    # content-based
    # ------------------------------------------------------------------ #

    def recommend_similar(self, tariff_id: str, n: int = 5) -> list[dict]:
        """Find tariffs most similar to a given tariff."""
        if not self._fitted:
            raise RuntimeError("model not fitted")
        if tariff_id not in self.tariff_ids:
            raise ValueError(f"unknown tariff: {tariff_id}")

        idx = self.tariff_ids.index(tariff_id)
        sims = self.cosine_matrix[idx]
        ranked = np.argsort(sims)[::-1]
        results = []
        for i in ranked:
            if self.tariff_ids[i] != tariff_id:
                results.append({
                    "tariff_id": self.tariff_ids[i],
                    "score": float(sims[i]),
                })
            if len(results) >= n:
                break
        return self._enrich(results)

    # ------------------------------------------------------------------ #
    # hybrid
    # ------------------------------------------------------------------ #

    def recommend_hybrid(self, customer_id: str, n: int = 5) -> list[dict]:
        """Blend collaborative and content-based scores."""
        if not self._fitted:
            raise RuntimeError("model not fitted")

        # content score anchored on current tariff
        current_tariff = None
        if self.ratings_df is not None:
            cust_rows = self.ratings_df[self.ratings_df["customer_id"] == customer_id]
            if not cust_rows.empty:
                current_tariff = (
                    cust_rows.sort_values("months", ascending=False).iloc[0]["tariff_id"]
                )

        collab = self.recommend_collaborative(customer_id, n=len(self.tariff_ids))
        collab_scores = {r["tariff_id"]: r["score"] for r in collab}

        if current_tariff and current_tariff in self.tariff_ids:
            content = self.recommend_similar(current_tariff, n=len(self.tariff_ids))
            content_scores = {r["tariff_id"]: r["score"] for r in content}
        else:
            content_scores = {}

        # normalise both to [0, 1]
        def _norm(scores: dict) -> dict:
            if not scores:
                return {}
            vals = np.array(list(scores.values()), dtype=float)
            mn, mx = vals.min(), vals.max()
            if mx == mn:
                return {k: 0.5 for k in scores}
            return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

        collab_n = _norm(collab_scores)
        content_n = _norm(content_scores)

        combined = {}
        for tid in self.tariff_ids:
            c = collab_n.get(tid, 0.0)
            cb = content_n.get(tid, 0.0)
            combined[tid] = self.hybrid_alpha * c + (1 - self.hybrid_alpha) * cb

        # exclude current tariff
        if current_tariff:
            combined.pop(current_tariff, None)

        sorted_tids = sorted(combined, key=combined.get, reverse=True)[:n]
        results = [{"tariff_id": tid, "score": combined[tid]} for tid in sorted_tids]
        return self._enrich(results)

    # ------------------------------------------------------------------ #
    # cold start
    # ------------------------------------------------------------------ #

    def cold_start(
        self, n: int = 5, category: str | None = None
    ) -> list[dict]:
        """Popular tariffs overall or filtered by category."""
        if not self._fitted:
            raise RuntimeError("model not fitted")

        if self.tariffs_df is None or self.popularity is None:
            return []

        tariffs = self.tariffs_df.copy()
        if category:
            tariffs = tariffs[tariffs["category"] == category]

        tariffs = tariffs.set_index("tariff_id")
        tariffs["popularity"] = tariffs.index.map(self.popularity).fillna(0)
        top = tariffs.sort_values("popularity", ascending=False).head(n)

        return self._enrich([
            {"tariff_id": tid, "score": float(row["popularity"])}
            for tid, row in top.iterrows()
        ])

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _enrich(self, results: list[dict]) -> list[dict]:
        """Attach tariff metadata to recommendation results."""
        if self.tariffs_df is None:
            return results
        meta = self.tariffs_df.set_index("tariff_id")
        for r in results:
            tid = r["tariff_id"]
            if tid in meta.index:
                row = meta.loc[tid]
                r["name"] = row["name"]
                r["monthly_price"] = int(row["monthly_price"])
                r["data_gb"] = int(row["data_gb"])
                r["category"] = row["category"]
        return results

    def save(self, path: str) -> None:
        joblib.dump({
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "random_state": self.random_state,
            "hybrid_alpha": self.hybrid_alpha,
            "svd_model": self.svd_model,
            "ratings_df": self.ratings_df,
            "tariffs_df": self.tariffs_df,
            "tariff_features": self.tariff_features,
            "tariff_ids": self.tariff_ids,
            "cosine_matrix": self.cosine_matrix,
            "popularity": self.popularity,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TariffRecommender":
        data = joblib.load(path)
        inst = cls(
            n_factors=data["n_factors"],
            n_epochs=data["n_epochs"],
            random_state=data["random_state"],
            hybrid_alpha=data["hybrid_alpha"],
        )
        for key in ["svd_model", "ratings_df", "tariffs_df", "tariff_features",
                    "tariff_ids", "cosine_matrix", "popularity"]:
            setattr(inst, key, data[key])
        inst._fitted = True
        return inst
