"""recommendations page — live tariff suggestions per customer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("Tariff Recommendations")


@st.cache_resource
def load_recommender():
    path = Path("models/recommender.joblib")
    if not path.exists():
        return None
    from src.models.recommender import TariffRecommender
    return TariffRecommender.load(str(path))


@st.cache_data
def load_customers():
    path = Path("data/processed/customers_segmented.csv")
    if path.exists():
        return pd.read_csv(path)
    from src.data.synthetic import generate_customers
    return generate_customers(n=500, seed=42)


rec = load_recommender()
customers = load_customers()

if rec is None:
    st.warning("No trained recommender found. Run `make train` first.")
    st.stop()

st.markdown("---")

# --- mode selector ---
mode = st.radio(
    "Recommendation mode",
    ["cold_start", "similar", "hybrid"],
    horizontal=True,
    help="cold_start: popularity-based | similar: content-based | hybrid: blended",
)

n_recs = st.slider("Number of recommendations", 1, 8, 5)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Input")

    if mode == "cold_start":
        category = st.selectbox(
            "Category filter (optional)",
            ["All", "basic", "smart", "pro", "youth", "senior"],
        )
        category = None if category == "All" else category

        if st.button("Get Recommendations"):
            results = rec.cold_start(n=n_recs, category=category)
            st.session_state["recs"] = results

    elif mode == "similar":
        tariff_ids = rec.tariff_ids if rec.tariff_ids else []
        selected_tariff = st.selectbox("Current tariff", tariff_ids)
        if st.button("Find Similar"):
            results = rec.recommend_similar(selected_tariff, n=n_recs)
            st.session_state["recs"] = results

    else:  # hybrid
        customer_ids = customers["customer_id"].tolist()
        selected_customer = st.selectbox(
            "Customer ID", customer_ids[:200]
        )
        if st.button("Get Hybrid Recommendations"):
            results = rec.recommend_hybrid(selected_customer, n=n_recs)
            st.session_state["recs"] = results

with col_right:
    st.subheader("Results")
    if "recs" in st.session_state and st.session_state["recs"]:
        recs_list = st.session_state["recs"]
        recs_df = pd.DataFrame(recs_list)

        if "name" in recs_df.columns:
            display_cols = ["tariff_id", "name", "monthly_price", "data_gb", "category", "score"]
            display_cols = [c for c in display_cols if c in recs_df.columns]
            st.dataframe(recs_df[display_cols], use_container_width=True, hide_index=True)

            if "monthly_price" in recs_df.columns:
                fig = px.bar(
                    recs_df, x="name", y="monthly_price",
                    color="category",
                    labels={"monthly_price": "Price (KZT/month)", "name": ""},
                    title="Price comparison",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(recs_df, use_container_width=True, hide_index=True)
    else:
        st.info("Choose a mode and click the button to get recommendations.")

st.markdown("---")

# tariff catalog overview
st.subheader("Full Tariff Catalog")
if rec.tariffs_df is not None:
    tariffs_display = rec.tariffs_df.copy()
    if rec.popularity is not None:
        tariffs_display["customers"] = tariffs_display["tariff_id"].map(rec.popularity).fillna(0).astype(int)
    st.dataframe(tariffs_display, use_container_width=True, hide_index=True)
