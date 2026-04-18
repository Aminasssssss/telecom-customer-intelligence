from src.features.rfm import compute_rfm, merge_with_customers


def test_rfm_columns(transactions):
    rfm = compute_rfm(transactions)
    expected = {"customer_id", "recency", "frequency", "monetary",
                "avg_data_gb", "avg_minutes",
                "r_score", "f_score", "m_score", "rfm_score"}
    assert expected.issubset(rfm.columns)


def test_rfm_unique_customers(transactions):
    rfm = compute_rfm(transactions)
    assert rfm["customer_id"].is_unique


def test_rfm_value_ranges(transactions):
    rfm = compute_rfm(transactions)
    assert (rfm["recency"] >= 0).all()
    assert (rfm["frequency"] > 0).all()
    assert (rfm["monetary"] > 0).all()
    assert rfm["r_score"].between(1, 5).all()
    assert rfm["f_score"].between(1, 5).all()
    assert rfm["m_score"].between(1, 5).all()


def test_merge_with_customers(customers, transactions):
    rfm = compute_rfm(transactions)
    merged = merge_with_customers(customers, rfm)
    assert len(merged) == len(customers)
    assert "rfm_score" in merged.columns
    assert merged["recency"].notna().all()
