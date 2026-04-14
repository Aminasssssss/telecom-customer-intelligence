"""overview page — platform-wide KPIs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.synthetic import generate_customers, generate_transactions

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")


@st.cache_data
def load_data():
    processed = Path("data/processed/customers_segmented.csv")
    if processed.exists():
        customers = pd.read_csv(processed)
    else:
        customers = generate_customers(n=1000, seed=42)

    tx_path = Path("data/raw/transactions.csv")
    if tx_path.exists():
        transactions = pd.read_csv(tx_path, parse_dates=["transaction_date"])
    else:
        transactions = generate_transactions(customers, months=6, seed=42)

    return customers, transactions


customers, transactions = load_data()

# KPIs
total = len(customers)
churn_rate = customers["churn"].mean() if "churn" in customers.columns else 0
monthly_revenue = transactions.groupby(
    transactions["transaction_date"].dt.to_period("M")
)["amount"].sum()
arpu = monthly_revenue.mean() / max(total, 1)
segment_col = "segment_name" if "segment_name" in customers.columns else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total:,}")
col2.metric("Churn Rate", f"{churn_rate:.1%}")
col3.metric("Avg Monthly Revenue", f"{monthly_revenue.mean():,.0f} KZT")
col4.metric("ARPU", f"{arpu:,.0f} KZT")

st.markdown("---")

# monthly revenue chart
st.subheader("Monthly Revenue")
rev_df = monthly_revenue.reset_index()
rev_df.columns = ["month", "revenue"]
rev_df["month"] = rev_df["month"].astype(str)
fig = px.bar(rev_df, x="month", y="revenue", labels={"revenue": "Revenue (KZT)"})
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns(2)

# churn by contract type
with col_left:
    st.subheader("Churn by Contract Type")
    if "contract_type" in customers.columns and "churn" in customers.columns:
        ct = customers.groupby("contract_type")["churn"].mean().reset_index()
        ct.columns = ["contract_type", "churn_rate"]
        fig2 = px.bar(ct, x="contract_type", y="churn_rate",
                      labels={"churn_rate": "Churn Rate"})
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

# customers by city
with col_right:
    st.subheader("Customers by City")
    if "city" in customers.columns:
        city_ct = customers["city"].value_counts().reset_index()
        city_ct.columns = ["city", "count"]
        fig3 = px.pie(city_ct, names="city", values="count")
        st.plotly_chart(fig3, use_container_width=True)
