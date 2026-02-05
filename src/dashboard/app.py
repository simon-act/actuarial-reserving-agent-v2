"""
Actuarial Reserving Dashboard - Multi-Page App
===============================================

Pages:
1. Reported Claims - Triangles, Anomaly Detection, Model Selection
2. Summary - Ultimates & Reported Table, Economic Scenarios

Run with: streamlit run app.py
"""

import streamlit as st

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Actuarial Reserving Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stSelectbox > div > div {
        font-size: 14px;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main page content - redirect to Reported Claims
st.title("📊 Actuarial Reserving Dashboard")
st.markdown("---")
st.info("👈 Select a page from the sidebar to get started.")

st.markdown("""
### Available Pages:

**1. 📋 Reported Claims**
- Loss Development Triangles (Cumulative, Incremental, Lag Factors)
- Anomaly Detection
- Automatic Model Selection with GBM
- Model Diagnostics (sidebar)

**2. 📊 Summary**
- Ultimates & Reported Detail Table
- Economic Scenario Analysis
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Actuarial Reserving Dashboard | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
