import pytest

from src.data.synthetic import generate_customers, generate_tariffs_catalog, generate_transactions
from src.models.recommender import TariffRecommender, _build_ratings


def _get_data(n=300, seed=5):
    customers = generate_customers(n=n, seed=seed)
    transactions = generate_transactions(customers, months=6, seed=seed)
    tariffs = generate_tariffs_catalog()
    return customers, transactions, tariffs


def test_build_ratings():
    customers, transactions, _ = _get_data()
    ratings = _build_ratings(transactions)
    assert {"customer_id", "tariff_id", "rating", "months"}.issubset(ratings.columns)
    assert ratings["rating"].between(1, 5).all()


def test_cold_start_returns_results():
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    results = rec.cold_start(n=3)
    assert len(results) == 3
    for r in results:
        assert "tariff_id" in r
        assert "monthly_price" in r


def test_cold_start_by_category():
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    results = rec.cold_start(n=2, category="basic")
    assert len(results) <= 2
    assert all(r["category"] == "basic" for r in results)


def test_recommend_similar():
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    results = rec.recommend_similar("T03", n=3)
    assert len(results) == 3
    assert all(r["tariff_id"] != "T03" for r in results)
    assert all(0 <= r["score"] <= 1 for r in results)


def test_recommend_similar_unknown_tariff():
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    with pytest.raises(ValueError):
        rec.recommend_similar("T99", n=3)


def test_recommend_collaborative_cold_start_fallback():
    # customer with no history → should fall back to cold start
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    results = rec.recommend_collaborative("C999999", n=3)
    assert len(results) == 3


def test_recommend_hybrid():
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    cid = customers.iloc[0]["customer_id"]
    results = rec.recommend_hybrid(cid, n=4)
    assert len(results) <= 4
    assert all("tariff_id" in r for r in results)


def test_not_fitted_raises():
    rec = TariffRecommender()
    with pytest.raises(RuntimeError):
        rec.recommend_collaborative("C000001")


def test_save_load(tmp_path):
    customers, transactions, tariffs = _get_data()
    rec = TariffRecommender().fit(transactions, tariffs)
    path = tmp_path / "rec.joblib"
    rec.save(str(path))
    loaded = TariffRecommender.load(str(path))
    r1 = rec.recommend_similar("T03", n=3)
    r2 = loaded.recommend_similar("T03", n=3)
    assert [r["tariff_id"] for r in r1] == [r["tariff_id"] for r in r2]
