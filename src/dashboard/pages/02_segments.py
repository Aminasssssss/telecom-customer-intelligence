"""segments page — cluster profiles and RFM breakdown."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.synthetic import generate_customers, generate_transactions
from src.features.rfm import compute_rfm, merge_with_customers

st.set_page_config(page_title="Segments", layout="wide")
st.title("Customer Segments")


@st.cache_data
def load_segmented():
    path = Path("data/processed/customers_segmented.csv")
    if path.exists():
        return pd.read_csv(path)
    # generate + compute RFM on the fly for demo
    customers = generate_customers(n=1000, seed=42)
    tx = generate_transactions(customers, months=6, seed=42)
    rfm = compute_rfm(tx)
    return merge_with_customers(customers, rfm)


df = load_segmented()

seg_col = "segment_name" if "segment_name" in df.columns else "segment"
if seg_col not in df.columns:
    st.warning("Run make train to generate segment labels.")
    st.stop()

# segment distribution
col1, col2 = st.columns(2)
with col1:
    st.subheader("Segment Distribution")
    counts = df[seg_col].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    fig = px.pie(counts, names="segment", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn Rate by Segment")
    if "churn" in df.columns:
        churn_by_seg = df.groupby(seg_col)["churn"].mean().reset_index()
        churn_by_seg.columns = ["segment", "churn_rate"]
        churn_by_seg = churn_by_seg.sort_values("churn_rate", ascending=True)
        fig2 = px.bar(churn_by_seg, x="churn_rate", y="segment",
                      orientation="h", labels={"churn_rate": "Churn Rate"})
        fig2.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Segment Profiles")

if "monetary" in df.columns and "recency" in df.columns:
    profile = df.groupby(seg_col).agg(
        avg_monetary=("monetary", "mean"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_tenure=("tenure_months", "mean"),
        count=(seg_col, "count"),
    ).reset_index()
    profile = profile.round(1)
    st.dataframe(profile, use_container_width=True, hide_index=True)

    # scatter: recency vs monetary per segment
    st.subheader("Recency vs Monetary Value")
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig3 = px.scatter(
        sample, x="recency", y="monetary",
        color=seg_col, opacity=0.6,
        labels={"recency": "Recency (days)", "monetary": "Total Spend (KZT)"},
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("RFM features not found — run make data && make train first.")

st.markdown("---")

# segment filter
st.subheader("Explore Segment")
selected = st.selectbox("Select segment", options=sorted(df[seg_col].unique()))
subset = df[df[seg_col] == selected]
st.write(f"{len(subset):,} customers in this segment")
st.dataframe(
    subset[["customer_id", "age", "city", "tenure_months",
            "contract_type", "churn"] if "churn" in df.columns
           else ["customer_id", "age", "city", "tenure_months", "contract_type"]
           ].head(50),
    use_container_width=True,
    hide_index=True,
)
