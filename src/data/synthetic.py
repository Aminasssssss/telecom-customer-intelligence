"""synthetic telecom data generator."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


TARIFFS = [
    {"tariff_id": "T01", "name": "Basic 100", "monthly_price": 1500,
     "data_gb": 5, "minutes": 100, "category": "basic"},
    {"tariff_id": "T02", "name": "Basic 200", "monthly_price": 2500,
     "data_gb": 10, "minutes": 200, "category": "basic"},
    {"tariff_id": "T03", "name": "Smart 500", "monthly_price": 4000,
     "data_gb": 25, "minutes": 500, "category": "smart"},
    {"tariff_id": "T04", "name": "Smart Unlim", "monthly_price": 6000,
     "data_gb": 50, "minutes": 1500, "category": "smart"},
    {"tariff_id": "T05", "name": "Pro Family", "monthly_price": 8500,
     "data_gb": 100, "minutes": 3000, "category": "pro"},
    {"tariff_id": "T06", "name": "Pro Business", "monthly_price": 12000,
     "data_gb": 200, "minutes": 5000, "category": "pro"},
    {"tariff_id": "T07", "name": "Youth Social", "monthly_price": 1800,
     "data_gb": 15, "minutes": 50, "category": "youth"},
    {"tariff_id": "T08", "name": "Senior Calls", "monthly_price": 1200,
     "data_gb": 2, "minutes": 400, "category": "senior"},
]


def generate_customers(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"C{i:06d}" for i in range(1, n + 1)]
    ages = rng.integers(18, 75, size=n)
    genders = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])
    cities = rng.choice(
        ["Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe", "Pavlodar"],
        size=n,
        p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.07],
    )

    tenure_months = rng.integers(1, 84, size=n)

    # tariff loosely tied to age bucket
    tariff_ids = []
    for age in ages:
        if age < 25:
            choice = rng.choice(["T07", "T01", "T03"], p=[0.5, 0.3, 0.2])
        elif age < 40:
            choice = rng.choice(["T03", "T04", "T05", "T06"], p=[0.35, 0.30, 0.20, 0.15])
        elif age < 60:
            choice = rng.choice(["T04", "T05", "T06", "T02"], p=[0.3, 0.3, 0.2, 0.2])
        else:
            choice = rng.choice(["T08", "T01", "T02"], p=[0.5, 0.3, 0.2])
        tariff_ids.append(choice)

    contract_types = rng.choice(
        ["monthly", "yearly", "two-year"], size=n, p=[0.55, 0.30, 0.15]
    )
    payment_methods = rng.choice(
        ["card", "bank-transfer", "cash", "e-wallet"],
        size=n,
        p=[0.45, 0.25, 0.15, 0.15],
    )
    paperless = rng.choice([0, 1], size=n, p=[0.3, 0.7])

    has_internet = rng.choice([0, 1], size=n, p=[0.15, 0.85])
    has_tv = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    has_roaming = rng.choice([0, 1], size=n, p=[0.75, 0.25])

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "city": cities,
        "tenure_months": tenure_months,
        "tariff_id": tariff_ids,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "paperless_billing": paperless,
        "has_internet": has_internet,
        "has_tv": has_tv,
        "has_roaming": has_roaming,
    })

    # churn probability with risk factors
    churn_prob = np.full(n, 0.10)
    churn_prob += (df["contract_type"] == "monthly").astype(int) * 0.18
    churn_prob += (df["tenure_months"] < 12).astype(int) * 0.12
    churn_prob -= (df["tenure_months"] > 48).astype(int) * 0.08
    churn_prob += (df["payment_method"] == "cash").astype(int) * 0.05
    churn_prob -= df["has_tv"] * 0.04
    churn_prob -= df["has_internet"] * 0.03
    churn_prob = np.clip(churn_prob, 0.01, 0.85)
    df["churn"] = rng.binomial(1, churn_prob).astype(int)

    return df


def generate_transactions(
    customers: pd.DataFrame, months: int = 12, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    tariffs = pd.DataFrame(TARIFFS).set_index("tariff_id")

    end_date = datetime(2025, 12, 1)
    rows = []

    for _, row in customers.iterrows():
        cid = row["customer_id"]
        tariff = tariffs.loc[row["tariff_id"]]
        base_price = float(tariff["monthly_price"])

        active_months = min(months, int(row["tenure_months"]))

        for m in range(active_months):
            date = end_date - timedelta(days=30 * m)
            amount = base_price * rng.uniform(0.9, 1.4)
            usage_data_gb = float(tariff["data_gb"]) * rng.uniform(0.4, 1.2)
            usage_minutes = float(tariff["minutes"]) * rng.uniform(0.3, 1.1)

            rows.append({
                "transaction_id": f"TX{len(rows):09d}",
                "customer_id": cid,
                "transaction_date": date,
                "amount": round(amount, 2),
                "data_gb_used": round(usage_data_gb, 2),
                "minutes_used": round(usage_minutes, 1),
                "tariff_id": row["tariff_id"],
            })

    return pd.DataFrame(rows)


def generate_tariffs_catalog() -> pd.DataFrame:
    return pd.DataFrame(TARIFFS)


def save_all(
    out_dir: Path, n_customers: int = 5000, months: int = 12, seed: int = 42
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(n_customers, seed=seed)
    transactions = generate_transactions(customers, months=months, seed=seed)
    tariffs = generate_tariffs_catalog()

    customers.to_csv(out_dir / "customers.csv", index=False)
    transactions.to_csv(out_dir / "transactions.csv", index=False)
    tariffs.to_csv(out_dir / "tariffs.csv", index=False)

    return {
        "customers": len(customers),
        "transactions": len(transactions),
        "tariffs": len(tariffs),
    }
