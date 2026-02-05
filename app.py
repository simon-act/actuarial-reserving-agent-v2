"""
Intelligent Reserving Agent - Multi-Page App
=============================================

An AI-powered actuarial reserving system with intelligent agents
that reason about data instead of applying hardcoded rules.

Pages:
1. Reported Claims - Triangles, Intelligent Model Selection
2. Summary - Ultimates & Reported Table, Economic Scenarios

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Intelligent Reserving Agent",
    page_icon="🧠",
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

# Main page content
st.title("🧠 Intelligent Reserving Agent")
st.markdown("---")
st.info("👈 Select a page from the sidebar to get started.")

st.markdown("""
### About This System

This is an **AI-powered actuarial reserving system** where agents reason about data
intelligently instead of applying hardcoded rules.

**Key Features:**
- 🧠 **Intelligent Method Selection** - LLM analyzes patterns and decides the best approach
- 🔍 **Pattern Analysis** - Detects anomalies and applies smoothing when appropriate
- 💭 **Transparent Reasoning** - Every decision comes with full explanation
- 🔎 **Self-Critique** - Agents evaluate their own decisions

### Available Pages:

**1. 📋 Reported Claims**
- Loss Development Triangles (Cumulative, Incremental, Lag Factors)
- Intelligent Model Selection with reasoning
- Model Diagnostics (sidebar)

**2. 📊 Summary**
- Ultimates & Reported Detail Table
- Economic Scenario Analysis
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Intelligent Reserving Agent | AI-Powered Actuarial Analysis | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
