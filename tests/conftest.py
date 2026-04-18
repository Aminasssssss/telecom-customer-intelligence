"""shared pytest fixtures."""
import pytest

from src.data.synthetic import generate_customers, generate_transactions


@pytest.fixture(scope="session")
def customers():
    return generate_customers(n=500, seed=123)


@pytest.fixture(scope="session")
def transactions(customers):
    return generate_transactions(customers, months=6, seed=123)
