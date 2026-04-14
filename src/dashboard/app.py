"""Streamlit dashboard entry point."""
import streamlit as st

st.set_page_config(
    page_title="Telecom Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Telecom Customer Intelligence Platform")
st.markdown(
    "End-to-end ML platform for customer segmentation, churn prediction, "
    "and tariff recommendations."
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.page_link("pages/01_overview.py", label="Overview", icon=None)
col2.page_link("pages/02_segments.py", label="Segments", icon=None)
col3.page_link("pages/03_churn.py", label="Churn", icon=None)
col4.page_link("pages/04_recommendations.py", label="Recommendations", icon=None)

st.info("Use the sidebar to navigate between pages.")
