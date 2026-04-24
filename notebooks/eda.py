# %% [markdown]
# # Exploratory Data Analysis
# telecom customer dataset — distributions, correlations, churn drivers

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.synthetic import generate_customers, generate_transactions
from src.features.rfm import compute_rfm, merge_with_customers

# %%
customers = generate_customers(n=5000, seed=42)
transactions = generate_transactions(customers, months=12, seed=42)
rfm = compute_rfm(transactions)
df = merge_with_customers(customers, rfm)

print(df.shape)
print(df.dtypes)
print(df.describe().round(2))


# %% [markdown]
# ## Churn rate by key features

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# contract type
ct = df.groupby("contract_type")["churn"].mean()
axes[0, 0].bar(ct.index, ct.values, color=["#636EFA", "#EF553B", "#00CC96"])
axes[0, 0].set_title("Churn by Contract Type")
axes[0, 0].set_ylabel("Churn Rate")

# tenure distribution
axes[0, 1].hist(df[df["churn"] == 0]["tenure_months"], bins=30,
                alpha=0.6, label="retained", color="#2ecc71")
axes[0, 1].hist(df[df["churn"] == 1]["tenure_months"], bins=30,
                alpha=0.6, label="churned", color="#e74c3c")
axes[0, 1].set_title("Tenure Distribution")
axes[0, 1].set_xlabel("Tenure (months)")
axes[0, 1].legend()

# payment method
pm = df.groupby("payment_method")["churn"].mean().sort_values(ascending=False)
axes[0, 2].barh(pm.index, pm.values)
axes[0, 2].set_title("Churn by Payment Method")
axes[0, 2].set_xlabel("Churn Rate")

# city
city_churn = df.groupby("city")["churn"].mean().sort_values(ascending=False)
axes[1, 0].bar(city_churn.index, city_churn.values)
axes[1, 0].set_title("Churn by City")
axes[1, 0].set_xticklabels(city_churn.index, rotation=30)

# age distribution
axes[1, 1].hist(df[df["churn"] == 0]["age"], bins=25,
                alpha=0.6, label="retained", color="#2ecc71")
axes[1, 1].hist(df[df["churn"] == 1]["age"], bins=25,
                alpha=0.6, label="churned", color="#e74c3c")
axes[1, 1].set_title("Age Distribution")
axes[1, 1].legend()

# monetary vs churn
axes[1, 2].boxplot(
    [df[df["churn"] == 0]["monetary"], df[df["churn"] == 1]["monetary"]],
    labels=["retained", "churned"],
)
axes[1, 2].set_title("Total Spend vs Churn")
axes[1, 2].set_ylabel("Total Spend (KZT)")

plt.tight_layout()
plt.savefig("notebooks/eda_churn_drivers.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved eda_churn_drivers.png")


# %% [markdown]
# ## Correlation heatmap

# %%
num_cols = ["age", "tenure_months", "has_internet", "has_tv", "has_roaming",
            "recency", "frequency", "monetary", "churn"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax)
ax.set_title("Feature Correlations")
plt.tight_layout()
plt.savefig("notebooks/eda_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved eda_correlations.png")


# %% [markdown]
# ## RFM score distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col, title in zip(axes,
                           ["r_score", "f_score", "m_score"],
                           ["Recency Score", "Frequency Score", "Monetary Score"]):
    df[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="#636EFA")
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Customers")

plt.tight_layout()
plt.savefig("notebooks/eda_rfm_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved eda_rfm_distribution.png")


# %% [markdown]
# ## Summary stats

# %%
print("\nDataset Summary")
print(f"  customers         : {len(df):,}")
print(f"  transactions      : {len(transactions):,}")
print(f"  churn rate        : {df['churn'].mean():.1%}")
print(f"  avg tenure        : {df['tenure_months'].mean():.1f} months")
print(f"  avg monthly spend : {df['monetary'].mean() / 12:,.0f} KZT")
print(f"  monthly contract  : {(df['contract_type'] == 'monthly').mean():.1%}")
