from src.data.synthetic import generate_customers, generate_transactions
from src.features.rfm import compute_rfm, merge_with_customers
from src.models.churn import ChurnPredictor, train_and_compare


def _make_dataset(n=400, seed=1):
    customers = generate_customers(n=n, seed=seed)
    tx = generate_transactions(customers, months=4, seed=seed)
    rfm = compute_rfm(tx)
    full = merge_with_customers(customers, rfm)
    drop = ["customer_id", "churn", "r_score", "f_score", "m_score", "rfm_score"]
    feature_cols = [c for c in full.columns if c not in drop]
    return full[feature_cols], full["churn"]


def test_churn_xgboost_fit_predict():
    X, y = _make_dataset()
    model = ChurnPredictor(model_type="xgboost", n_estimators=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_churn_lightgbm_evaluate():
    X, y = _make_dataset()
    model = ChurnPredictor(model_type="lightgbm", n_estimators=50)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    assert 0 <= metrics.roc_auc <= 1
    assert 0 <= metrics.f1 <= 1


def test_train_and_compare():
    X, y = _make_dataset(n=600, seed=2)
    results = train_and_compare(X, y, test_size=0.25)
    assert "xgboost" in results
    assert "lightgbm" in results
    for r in results.values():
        assert 0 <= r["metrics"].roc_auc <= 1


def test_churn_save_load(tmp_path):
    X, y = _make_dataset(n=400, seed=3)
    model = ChurnPredictor(model_type="xgboost", n_estimators=30).fit(X, y)
    path = tmp_path / "churn.joblib"
    model.save(str(path))
    loaded = ChurnPredictor.load(str(path))
    assert (model.predict(X) == loaded.predict(X)).all()


def test_churn_explain():
    X, y = _make_dataset(n=300, seed=4)
    model = ChurnPredictor(model_type="xgboost", n_estimators=30).fit(X, y)
    expl = model.explain(X, sample_size=50)
    assert len(expl["feature_names"]) == len(expl["importance"])
    assert all(i >= 0 for i in expl["importance"])
