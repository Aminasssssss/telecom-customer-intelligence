from src.data.synthetic import (
    generate_customers,
    generate_tariffs_catalog,
    generate_transactions,
)


def test_generate_customers_shape():
    df = generate_customers(n=200, seed=1)
    assert len(df) == 200
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_generate_customers_reproducible():
    a = generate_customers(n=100, seed=42)
    b = generate_customers(n=100, seed=42)
    assert a.equals(b)


def test_customer_id_unique():
    df = generate_customers(n=300, seed=7)
    assert df["customer_id"].is_unique


def test_generate_transactions(customers):
    tx = generate_transactions(customers.head(50), months=3, seed=1)
    assert tx["customer_id"].nunique() <= 50
    assert (tx["amount"] > 0).all()
    assert tx["transaction_id"].is_unique


def test_tariffs_catalog():
    cat = generate_tariffs_catalog()
    assert len(cat) >= 5
    assert "monthly_price" in cat.columns
    assert (cat["monthly_price"] > 0).all()
