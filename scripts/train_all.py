"""train segmentation, churn, and recommender models — log everything to MLflow."""
from pathlib import Path

import mlflow
import yaml

from src.data.loader import load_customers, load_transactions, load_tariffs
from src.features.rfm import compute_rfm, merge_with_customers
from src.models.churn import train_and_compare
from src.models.recommender import TariffRecommender
from src.models.segmentation import CustomerSegmenter, auto_k_search, name_segments
from src.utils.mlflow_utils import log_metrics_dict, log_params_dict, setup_mlflow


def main() -> None:
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw = Path(cfg["data"]["raw_dir"])
    processed = Path(cfg["data"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("loading data")
    customers = load_customers(raw / "customers.csv")
    transactions = load_transactions(raw / "transactions.csv")
    tariffs = load_tariffs(raw / "tariffs.csv")

    print("computing RFM")
    rfm = compute_rfm(transactions)
    full = merge_with_customers(customers, rfm)
    full.to_csv(processed / "customers_enriched.csv", index=False)

    setup_mlflow(cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment_name"])

    # segmentation
    print("training segmentation")
    seg_features = cfg["segmentation"]["features"]
    X_seg = full[seg_features].fillna(0)

    with mlflow.start_run(run_name="segmentation_kmeans"):
        search = auto_k_search(X_seg, k_range=range(3, 9))
        for k, score in search["silhouette"].items():
            mlflow.log_metric(f"silhouette_k{k}", score)

        segmenter = CustomerSegmenter(n_clusters=cfg["segmentation"]["n_clusters"]).fit(X_seg)
        labels = segmenter.predict(X_seg)
        sil = segmenter.evaluate(X_seg)
        mlflow.log_metric("silhouette", sil)
        mlflow.log_param("n_clusters", cfg["segmentation"]["n_clusters"])

        profile = segmenter.build_profile(X_seg, labels)
        names = name_segments(profile)
        for seg_id, name in names.items():
            mlflow.log_param(f"segment_{seg_id}_name", name)

        seg_path = models_dir / "segmenter.joblib"
        segmenter.save(str(seg_path))
        mlflow.log_artifact(str(seg_path))
        print(f"  silhouette: {sil:.4f}  segments: {names}")

    full["segment"] = labels
    full["segment_name"] = full["segment"].map(names)
    full.to_csv(processed / "customers_segmented.csv", index=False)

    # churn
    print("training churn")
    drop_cols = ["customer_id", "segment", "segment_name",
                 "r_score", "f_score", "m_score", "rfm_score"]
    feature_cols = [c for c in full.columns if c not in drop_cols + ["churn"]]
    results = train_and_compare(
        full[feature_cols], full["churn"],
        test_size=cfg["churn"]["test_size"],
        random_state=cfg["churn"]["random_state"],
    )

    for model_type, res in results.items():
        with mlflow.start_run(run_name=f"churn_{model_type}"):
            log_params_dict(cfg["churn"]["models"][model_type])
            metrics = res["metrics"].to_dict()
            metrics.pop("confusion")
            log_metrics_dict(metrics)
            model_path = models_dir / f"churn_{model_type}.joblib"
            res["predictor"].save(str(model_path))
            mlflow.log_artifact(str(model_path))
            m = res["metrics"]
            print(f"  {model_type}: roc_auc={m.roc_auc:.4f} f1={m.f1:.4f}")

    best_type, best_res = max(results.items(), key=lambda x: x[1]["metrics"].roc_auc)
    best_res["predictor"].save(str(models_dir / "churn_best.joblib"))
    print(f"  best: {best_type} (roc_auc={best_res['metrics'].roc_auc:.4f})")

    # recommender
    print("training recommender")
    with mlflow.start_run(run_name="recommender_svd"):
        rec = TariffRecommender(n_factors=50, n_epochs=20, random_state=42)
        rec.fit(transactions, tariffs, customers)
        mlflow.log_param("n_factors", 50)
        mlflow.log_param("n_epochs", 20)
        mlflow.log_param("hybrid_alpha", rec.hybrid_alpha)
        mlflow.log_param("n_ratings",
                         len(rec.ratings_df) if rec.ratings_df is not None else 0)

        rec_path = models_dir / "recommender.joblib"
        rec.save(str(rec_path))
        mlflow.log_artifact(str(rec_path))

        recs = rec.cold_start(n=3)
        print(f"  cold start top-3: {[r['tariff_id'] for r in recs]}")
        recs_sim = rec.recommend_similar("T03", n=3)
        print(f"  similar to T03: {[r['tariff_id'] for r in recs_sim]}")

    print("done")


if __name__ == "__main__":
    main()
