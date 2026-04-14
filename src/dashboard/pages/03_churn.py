"""churn page — risk distribution, top-risk customers, feature importance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn", layout="wide")
st.title("Churn Analysis")


@st.cache_data
def load_customers():
    path = Path("data/processed/customers_segmented.csv")
    if path.exists():
        return pd.read_csv(path)
    from src.data.synthetic import generate_customers
    return generate_customers(n=1000, seed=42)


@st.cache_resource
def load_model():
    model_path = Path("models/churn_best.joblib")
    if not model_path.exists():
        return None
    from src.models.churn import ChurnPredictor
    return ChurnPredictor.load(str(model_path))


df = load_customers()
model = load_model()

if model is None:
    st.warning("No trained churn model found. Run `make train` first.")
    st.stop()

# compute probabilities
drop = ["customer_id", "churn", "segment", "segment_name",
        "r_score", "f_score", "m_score", "rfm_score"]
feature_cols = [c for c in df.columns if c not in drop]

@st.cache_data
def compute_probas(_model, data_hash):
    df_local = load_customers()
    drop_local = ["customer_id", "churn", "segment", "segment_name",
                  "r_score", "f_score", "m_score", "rfm_score"]
    feature_cols_local = [c for c in df_local.columns if c not in drop_local]
    return _model.predict_proba(df_local[feature_cols_local])

probas = compute_probas(model, len(df))
df = df.copy()
df["churn_probability"] = probas
df["risk_level"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("High Risk Customers",
            f"{(df['risk_level'] == 'High').sum():,}")
col2.metric("Medium Risk",
            f"{(df['risk_level'] == 'Medium').sum():,}")
col3.metric("Avg Churn Probability",
            f"{df['churn_probability'].mean():.1%}")
actual_churn = df["churn"].mean() if "churn" in df.columns else 0
col4.metric("Actual Churn Rate", f"{actual_churn:.1%}")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(df, x="churn_probability", nbins=40,
                       color="risk_level",
                       category_orders={"risk_level": ["Low", "Medium", "High"]},
                       labels={"churn_probability": "Churn Probability"})
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Risk Level Breakdown")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig2 = px.pie(risk_counts, names="risk_level", values="count",
                  color="risk_level",
                  color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# SHAP feature importance
st.subheader("Feature Importance (SHAP)")
try:
    expl = model.explain(df[feature_cols], sample_size=300)
    importance_df = pd.DataFrame({
        "feature": expl["feature_names"],
        "importance": expl["importance"],
    }).sort_values("importance", ascending=True).tail(15)
    fig3 = px.bar(importance_df, x="importance", y="feature", orientation="h",
                  labels={"importance": "Mean |SHAP|", "feature": ""})
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.info(f"SHAP unavailable: {e}")

st.markdown("---")

# high-risk customers table
st.subheader("Top High-Risk Customers")
threshold = st.slider("Churn probability threshold", 0.3, 0.9, 0.6, 0.05)
high_risk = (
    df[df["churn_probability"] >= threshold]
    .sort_values("churn_probability", ascending=False)
    [["customer_id", "churn_probability", "risk_level",
      "tenure_months", "contract_type", "city"]]
    .head(100)
)
st.write(f"{len(high_risk)} customers above threshold {threshold:.0%}")
st.dataframe(high_risk, use_container_width=True, hide_index=True)
