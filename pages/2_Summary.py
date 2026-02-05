"""
Page 2: Summary
================
- Ultimates & Reported Detail Table
- Economic Scenario Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Summary | Reserving",
    page_icon="📊",
    layout="wide"
)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chain_ladder import ChainLadder
from scenario_analysis.economic_scenario_generator import EconomicScenarioGenerator
from data_loader import TriangleLoader, TriangleInfo

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_project_root() -> Path:
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "data").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent.parent


@st.cache_data
def load_sample_triangle(name: str) -> pd.DataFrame:
    loader = TriangleLoader()
    if name == 'swiss_re':
        data_dir = get_project_root() / "data" / "processed"
        try:
            triangle = pd.read_csv(data_dir / "reported_absolute_losses.csv", index_col=0)
            triangle.columns = triangle.columns.astype(int)
        except FileNotFoundError:
            triangle = loader.load_sample('mack')
    else:
        triangle = loader.load_sample(name)
    return triangle


SAMPLE_TRIANGLES = {
    'swiss_re': 'Swiss Re Property Reinsurance (16×16)',
    'mack': 'Mack GenIns - Mack 1993 (10×10)',
    'taylor_ashe': 'Taylor/Ashe Benchmark (10×10)',
    'abc': 'ABC Insurance Incurred (10×10)',
    'uk_motor': 'UK Motor Claims (7×7)'
}


# ============================================
# MAIN PAGE
# ============================================
st.title("📊 Summary")

# --------------------------------------------
# SIDEBAR - Data Selection
# --------------------------------------------
st.sidebar.header("📁 Data Selection")

data_source = st.sidebar.radio("Data source:", ["Sample Triangles", "Upload File"], key="summary_data")

triangle = None

if data_source == "Sample Triangles":
    triangle_name = st.sidebar.selectbox(
        "Select Triangle:",
        options=list(SAMPLE_TRIANGLES.keys()),
        format_func=lambda x: SAMPLE_TRIANGLES[x],
        key="summary_triangle"
    )
    triangle = load_sample_triangle(triangle_name)
    st.sidebar.success(f"✅ {SAMPLE_TRIANGLES[triangle_name]}")
else:
    uploaded = st.sidebar.file_uploader("Upload Triangle", type=['csv', 'xlsx'], key="summary_upload")
    if uploaded:
        if uploaded.name.endswith('.csv'):
            triangle = pd.read_csv(uploaded, index_col=0)
        else:
            triangle = pd.read_excel(uploaded, index_col=0)
        try:
            triangle.columns = triangle.columns.astype(int)
        except:
            pass
        triangle = triangle.apply(pd.to_numeric, errors='coerce')
        st.sidebar.success(f"✅ {uploaded.name}")

if triangle is None:
    st.info("👈 Select or upload a triangle from the sidebar.")
    st.stop()

# Get model factors from session state or use default CL
if 'model_factors' in st.session_state and st.session_state['model_factors']:
    model_factors = st.session_state['model_factors']
    available_models = list(model_factors.keys())
else:
    # Default to Chain-Ladder
    cl = ChainLadder(triangle)
    cl.run_full_analysis()
    model_factors = {"Chain-Ladder": cl.selected_factors}
    available_models = ["Chain-Ladder"]

# Model selection for ultimates
selected_model = st.sidebar.selectbox(
    "Model for Ultimates:",
    available_models,
    key="ultimate_model"
)

ultimate_factors = model_factors[selected_model]

# ============================================
# 1. ULTIMATES & REPORTED DETAIL TABLE
# ============================================
st.header("📋 Ultimates & Reported Detail Table")
st.markdown(f"**Model:** {selected_model}")

# Calculate cumulative factors
cum_factors = np.cumprod(ultimate_factors.values[::-1])[::-1]

# Initialize manual IBNR in session state if not present
if 'manual_ibnr' not in st.session_state:
    st.session_state['manual_ibnr'] = {}

# Build detail table
detail_data = []
for i, year in enumerate(triangle.index):
    row = triangle.loc[year]
    reported = row.dropna().iloc[-1]
    dev_age = len(row.dropna())
    dev_period = triangle.columns[dev_age - 1]

    # Calculate ultimate
    age_idx = dev_age - 1
    if age_idx < len(cum_factors):
        remaining_factor = cum_factors[age_idx]
        model_ultimate = reported * remaining_factor
    else:
        model_ultimate = reported

    model_ibnr = model_ultimate - reported

    # Get manual IBNR from session state (default to 0)
    manual_ibnr = st.session_state['manual_ibnr'].get(str(year), 0.0)

    # Total IBNR = Model IBNR + Manual adjustment
    total_ibnr = model_ibnr + manual_ibnr
    final_ultimate = reported + total_ibnr
    pct_reported = (reported / final_ultimate * 100) if final_ultimate > 0 else 100

    detail_data.append({
        'Accident Year': year,
        'Dev. Age': dev_period,
        'Reported': reported,
        'Model IBNR': model_ibnr,
        'Manual IBNR': manual_ibnr,
        'Total IBNR': total_ibnr,
        'Ultimate': final_ultimate,
        '% Reported': pct_reported
    })

detail_df = pd.DataFrame(detail_data)

# Ensure Manual IBNR is float type for editing
detail_df['Manual IBNR'] = detail_df['Manual IBNR'].astype(float)

# Display options
col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    show_all = st.checkbox("Show all years (including fully developed)", value=True)
with col_opt2:
    if st.button("🗑️ Clear All Manual IBNR"):
        st.session_state['manual_ibnr'] = {}
        st.rerun()

if not show_all:
    detail_df = detail_df[detail_df['Total IBNR'].apply(lambda x: isinstance(x, (int, float)) and abs(x) > 1)]

# Editable table with totals
st.caption("💡 Click on 'Manual IBNR' cells to edit. Press Enter to confirm.")

edited = st.data_editor(
    detail_df,
    column_config={
        'Accident Year': st.column_config.TextColumn("Accident Year", disabled=True),
        'Dev. Age': st.column_config.TextColumn("Dev. Age", disabled=True),
        'Reported': st.column_config.NumberColumn("Reported", format="$%,.0f", disabled=True),
        'Model IBNR': st.column_config.NumberColumn("Model IBNR", format="$%,.0f", disabled=True),
        'Manual IBNR': st.column_config.NumberColumn(
            "Manual IBNR",
            help="Enter manual IBNR adjustments (positive or negative)",
            default=0.0,
            step=1000.0,
        ),
        'Total IBNR': st.column_config.NumberColumn("Total IBNR", format="$%,.0f", disabled=True),
        'Ultimate': st.column_config.NumberColumn("Ultimate", format="$%,.0f", disabled=True),
        '% Reported': st.column_config.NumberColumn("% Reported", format="%.1f%%", disabled=True),
    },
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="ibnr_editor"
)

# Update session state and recalculate
for idx, row in edited.iterrows():
    year = str(row['Accident Year'])
    st.session_state['manual_ibnr'][year] = row['Manual IBNR']
    manual = row['Manual IBNR']
    model = row['Model IBNR']
    reported = row['Reported']
    edited.at[idx, 'Total IBNR'] = model + manual
    edited.at[idx, 'Ultimate'] = reported + model + manual
    if (reported + model + manual) > 0:
        edited.at[idx, '% Reported'] = (reported / (reported + model + manual)) * 100

# Calculate totals
totals = {
    'Accident Year': 'TOTAL',
    'Dev. Age': '-',
    'Reported': edited['Reported'].sum(),
    'Model IBNR': edited['Model IBNR'].sum(),
    'Manual IBNR': edited['Manual IBNR'].sum(),
    'Total IBNR': edited['Total IBNR'].sum(),
    'Ultimate': edited['Ultimate'].sum(),
    '% Reported': edited['Reported'].sum() / edited['Ultimate'].sum() * 100 if edited['Ultimate'].sum() > 0 else 100
}

# Display totals row
st.dataframe(
    pd.DataFrame([totals]).style.format({
        'Reported': '${:,.0f}',
        'Model IBNR': '${:,.0f}',
        'Manual IBNR': '${:,.0f}',
        'Total IBNR': '${:,.0f}',
        'Ultimate': '${:,.0f}',
        '% Reported': '{:.1f}%'
    }),
    use_container_width=True,
    hide_index=True
)

# Build display_df for export
display_df = pd.concat([edited, pd.DataFrame([totals])], ignore_index=True)

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reported", f"${totals['Reported']:,.0f}")
col2.metric("Total IBNR", f"${totals['Total IBNR']:,.0f}",
            delta=f"Manual: {totals['Manual IBNR']:+,.0f}" if totals['Manual IBNR'] != 0 else None)
col3.metric("Final Ultimate", f"${totals['Ultimate']:,.0f}")
col4.metric("% Reported", f"{totals['% Reported']:.1f}%")

# Export button
csv = display_df.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"ultimates_{selected_model.replace(' ', '_')}.csv",
    mime="text/csv"
)

st.markdown("---")

# ============================================
# 2. ECONOMIC SCENARIOS
# ============================================
st.header("🌍 Economic Scenario Analysis")

scenario_tabs = st.tabs(["📊 Pre-defined Scenarios", "🎲 Monte Carlo Simulation"])

with scenario_tabs[0]:
    st.subheader("Pre-defined Economic Scenarios")
    st.markdown("""
    Analyzes reserves under different economic scenarios including:
    - **Base Case** - Current assumptions
    - **High Inflation** - Elevated claims costs
    - **Low Interest Rates** - Lower discount rates
    - **Recession** - Economic downturn effects
    - **Expansion** - Strong economic growth
    """)

    if st.button("🚀 Run Scenario Analysis", key="scenario_btn"):
        with st.spinner("Analyzing scenarios..."):
            try:
                esg = EconomicScenarioGenerator(triangle)
                comparison = esg.run_all_scenarios()
                esg_summary = esg.summary()
                weighted = esg.calculate_weighted_reserve()

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Base Reserve", f"${esg_summary['base_reserve']:,.0f}")
                col2.metric("Weighted Expected", f"${weighted['weighted_reserve']:,.0f}",
                           delta=f"{(weighted['weighted_reserve'] - esg_summary['base_reserve']) / esg_summary['base_reserve'] * 100:+.1f}%")
                col3.metric("Weighted Std Dev", f"${weighted['weighted_std']:,.0f}")

                # Bar chart
                fig = go.Figure()
                colors = ['green' if r < esg_summary['base_reserve'] else 'red' for r in comparison['Adjusted_Reserve']]
                fig.add_trace(go.Bar(
                    x=comparison['Scenario'],
                    y=comparison['Adjusted_Reserve'],
                    marker_color=colors,
                    text=[f"${x:,.0f}" for x in comparison['Adjusted_Reserve']],
                    textposition='outside'
                ))
                fig.add_hline(y=esg_summary['base_reserve'], line_dash="dash", line_color="blue",
                             annotation_text=f"Base: ${esg_summary['base_reserve']:,.0f}")
                fig.update_layout(
                    title="Reserve by Economic Scenario",
                    height=450,
                    xaxis_tickangle=-45,
                    yaxis_title="Reserve"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Details table
                st.subheader("Scenario Details")
                scenario_display = comparison[['Scenario', 'Probability', 'Inflation', 'Interest_Rate', 'Adjusted_Reserve', 'Change_Pct']].copy()
                st.dataframe(
                    scenario_display.style.format({
                        'Probability': '{:.0%}',
                        'Inflation': '{:.1%}',
                        'Interest_Rate': '{:.1%}',
                        'Adjusted_Reserve': '${:,.0f}',
                        'Change_Pct': '{:+.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:
                st.error(f"Error in scenario analysis: {e}")

with scenario_tabs[1]:
    st.subheader("Monte Carlo Simulation")
    st.markdown("""
    Generates reserve distribution through stochastic simulation of economic variables.
    Uses correlated random draws for inflation and interest rates.
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, 1000, key="mc_sims")
    with col2:
        confidence = st.slider("Confidence Level (%)", 75, 99, 95, key="mc_conf")

    if st.button("🎲 Run Monte Carlo", key="mc_btn"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            try:
                esg = EconomicScenarioGenerator(triangle)
                mc_results = esg.monte_carlo_simulation(n_simulations=n_sims)

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Reserve", f"${mc_results['mean_adjusted']:,.0f}")
                col2.metric("Std Dev", f"${mc_results['std_adjusted']:,.0f}")
                cv = mc_results['std_adjusted'] / mc_results['mean_adjusted'] * 100
                col3.metric("CV", f"{cv:.1f}%")

                percentile_key = f"P{confidence}"
                if percentile_key in mc_results['percentiles_adjusted']:
                    col4.metric(f"{percentile_key}", f"${mc_results['percentiles_adjusted'][percentile_key]:,.0f}")

                # Distribution histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=mc_results['distribution']['adjusted'],
                    nbinsx=50,
                    marker_color='lightblue',
                    name='Distribution'
                ))

                # Add percentile lines
                for pct, label in [(0.50, 'P50'), (0.75, 'P75'), (0.95, 'P95')]:
                    val = mc_results['distribution']['adjusted'].quantile(pct)
                    fig.add_vline(x=val, line_dash="dash", line_color="red",
                                 annotation_text=f"{label}: ${val:,.0f}",
                                 annotation_position="top")

                fig.update_layout(
                    title=f"Reserve Distribution ({n_sims:,} simulations)",
                    xaxis_title="Reserve",
                    yaxis_title="Frequency",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

                # Percentiles table
                st.subheader("Reserve Percentiles")
                pct_df = pd.DataFrame({
                    'Percentile': list(mc_results['percentiles_adjusted'].keys()),
                    'Adjusted Reserve': list(mc_results['percentiles_adjusted'].values()),
                    'Discounted Reserve': list(mc_results['percentiles_discounted'].values())
                })
                st.dataframe(
                    pct_df.style.format({
                        'Adjusted Reserve': '${:,.0f}',
                        'Discounted Reserve': '${:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:
                st.error(f"Error in Monte Carlo: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Actuarial Reserving Dashboard | Summary Page</p>
</div>
""", unsafe_allow_html=True)
