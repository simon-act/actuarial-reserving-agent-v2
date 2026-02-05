"""
Page 1: Reported Claims
=======================
- Loss Development Triangles (Cumulative, Incremental, Lag Factors + Anomalies)
- Anomaly Detection (auto-run)
- Automatic Model Selection (auto-run) with GBM
- Model Diagnostics (sidebar)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Reported Claims | Reserving",
    page_icon="📋",
    layout="wide"
)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chain_ladder import ChainLadder
from stochastic_reserving.mack_model import MackChainLadder
from diagnostics.diagnostic_tests import DiagnosticTests
from diagnostics.residual_analysis import ResidualAnalyzer
from ml_models.anomaly_detector import TriangleAnomalyDetector
from ml_models.gradient_boosting_factors import GBMFactorPredictor
from model_selection.model_selector import ModelSelector
from model_selection.windowed_estimators import generate_windowed_estimators, get_optimal_window_by_method
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


@st.cache_data
def load_premium_data() -> pd.DataFrame:
    data_dir = get_project_root() / "data" / "processed"
    try:
        return pd.read_csv(data_dir / "earned_premium.csv", index_col=0)
    except:
        return None


def calculate_incremental_triangle(triangle: pd.DataFrame) -> pd.DataFrame:
    """Calculate incremental triangle from cumulative."""
    incremental = triangle.copy()
    for j in range(1, len(triangle.columns)):
        col_curr = triangle.columns[j]
        col_prev = triangle.columns[j - 1]
        incremental[col_curr] = triangle[col_curr] - triangle[col_prev]
    return incremental


def calculate_lag_factors(triangle: pd.DataFrame) -> pd.DataFrame:
    """Calculate age-to-age (lag) factors."""
    factors = pd.DataFrame(index=triangle.index, columns=triangle.columns[:-1], dtype=float)
    for j in range(len(triangle.columns) - 1):
        col_curr = triangle.columns[j]
        col_next = triangle.columns[j + 1]
        for year in triangle.index:
            curr = triangle.loc[year, col_curr]
            next_val = triangle.loc[year, col_next]
            if pd.notna(curr) and pd.notna(next_val) and curr > 0:
                factors.loc[year, col_curr] = next_val / curr
    return factors


def factors_to_pattern(factors):
    """Convert development factors to % of ultimate reported."""
    factors_arr = np.array(factors) if not isinstance(factors, np.ndarray) else factors
    factors_arr = np.nan_to_num(factors_arr, nan=1.0)
    cum_factors = np.cumprod(factors_arr[::-1])[::-1]
    return (1 / cum_factors) * 100


def format_number(val):
    """Format number for display in heatmap."""
    if pd.isna(val):
        return ""
    if abs(val) >= 1e6:
        return f"{val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.0f}K"
    else:
        return f"{val:.0f}"


@st.cache_resource
def run_anomaly_detection(_triangle: pd.DataFrame):
    """Run anomaly detection (cached)."""
    detector = TriangleAnomalyDetector(anomaly_threshold=2.5)
    detector.fit(_triangle, epochs=300)
    return detector


@st.cache_data
def run_model_selection(_triangle_hash, triangle: pd.DataFrame, min_window: int, max_window: int):
    """Run model selection (cached based on parameters)."""
    windowed_estimators = generate_windowed_estimators(
        triangle, min_window=min_window, max_window=max_window, recent_only=True
    )
    selector = ModelSelector(
        triangle=triangle, estimators=windowed_estimators,
        validation_method='holdout', verbose=False
    )
    validation_results = selector.run_validation()
    return selector.comparison_table, validation_results, windowed_estimators


def run_cross_model_validation(triangle: pd.DataFrame, top3_estimators: list = None):
    """
    Run cross-model validation: Chain-Ladder, Mack, GBM + top 3 CL patterns.
    ALL models validated on SAME holdout positions for fair comparison.
    Note: This function is not cached because top3_estimators contains unhashable objects.
    The parent function run_model_selection is already cached.
    """
    # Create reduced triangle by removing last non-NaN value per row
    reduced_triangle = triangle.copy()
    holdout_positions = {}  # (year, col) -> actual_value

    for year in triangle.index:
        row = triangle.loc[year]
        non_nan = row.dropna()
        if len(non_nan) > 1:  # Need at least 2 values to have prev + holdout
            last_col = non_nan.index[-1]
            holdout_positions[(year, last_col)] = non_nan.iloc[-1]
            reduced_triangle.loc[year, last_col] = np.nan

    model_results = {}
    model_factors = {}

    # Chain-Ladder (standard, all data)
    try:
        cl = ChainLadder(reduced_triangle)
        cl.run_full_analysis()
        model_factors["Chain-Ladder"] = cl.selected_factors
        preds, acts = _predict_holdout(reduced_triangle, holdout_positions, cl.selected_factors)
        if len(preds) > 0:
            errors = np.array(acts) - np.array(preds)
            model_results["Chain-Ladder"] = {"RMSE": np.sqrt(np.mean(errors**2)), "MAE": np.mean(np.abs(errors))}
    except:
        pass

    # Mack
    try:
        mack = MackChainLadder(reduced_triangle)
        mack.fit()
        model_factors["Mack Model"] = mack.development_factors
        preds, acts = _predict_holdout(reduced_triangle, holdout_positions, mack.development_factors)
        if len(preds) > 0:
            errors = np.array(acts) - np.array(preds)
            model_results["Mack Model"] = {"RMSE": np.sqrt(np.mean(errors**2)), "MAE": np.mean(np.abs(errors))}
    except:
        pass

    # GBM
    try:
        gbm = GBMFactorPredictor(n_estimators=100, learning_rate=0.1, max_depth=3)
        gbm.fit(reduced_triangle)
        gbm_pred = gbm.predict_factors()
        gbm_factors = pd.Series(gbm_pred['GBM_Factor'].values, index=reduced_triangle.columns[:-1])
        model_factors["GBM"] = gbm_factors
        preds, acts = _predict_holdout(reduced_triangle, holdout_positions, gbm_factors)
        if len(preds) > 0:
            errors = np.array(acts) - np.array(preds)
            model_results["GBM"] = {"RMSE": np.sqrt(np.mean(errors**2)), "MAE": np.mean(np.abs(errors))}
    except:
        pass

    # Top 3 CL patterns (validated on SAME holdout positions)
    if top3_estimators:
        for estimator in top3_estimators:
            try:
                # Estimate factors on reduced triangle
                factors = estimator.estimate(reduced_triangle)
                model_factors[estimator.name] = factors
                preds, acts = _predict_holdout(reduced_triangle, holdout_positions, factors)
                if len(preds) > 0:
                    errors = np.array(acts) - np.array(preds)
                    model_results[estimator.name] = {"RMSE": np.sqrt(np.mean(errors**2)), "MAE": np.mean(np.abs(errors))}
            except:
                pass

    return model_results, model_factors


def _predict_holdout(reduced_triangle, holdout_positions, factors):
    """
    Predict holdout values using factors.
    Same logic as ModelSelector validation_framework.
    """
    predictions = []
    actuals = []

    for (year, holdout_col), actual in holdout_positions.items():
        # Find the last available value in reduced triangle for this row
        row = reduced_triangle.loc[year]
        non_nan = row.dropna()

        if len(non_nan) == 0:
            continue

        last_available_col = non_nan.index[-1]
        last_available_value = non_nan.iloc[-1]

        # Use label-based indexing to get the correct factor
        # Factor at column k represents transition from period k to k+1
        if last_available_col in factors.index:
            factor = factors.loc[last_available_col]
            pred = last_available_value * factor
            predictions.append(pred)
            actuals.append(actual)

    return predictions, actuals


# ============================================
# SAMPLE TRIANGLES
# ============================================
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
st.title("📋 Reported Claims")

# --------------------------------------------
# SIDEBAR - Data Selection + Diagnostics
# --------------------------------------------
st.sidebar.header("📁 Data Selection")

data_source = st.sidebar.radio("Data source:", ["Sample Triangles", "Upload File"])

triangle = None
earned_premium = None

if data_source == "Sample Triangles":
    triangle_name = st.sidebar.selectbox(
        "Select Triangle:",
        options=list(SAMPLE_TRIANGLES.keys()),
        format_func=lambda x: SAMPLE_TRIANGLES[x]
    )
    triangle = load_sample_triangle(triangle_name)
    if triangle_name == 'swiss_re':
        earned_premium = load_premium_data()
    st.sidebar.success(f"✅ {SAMPLE_TRIANGLES[triangle_name]}")
else:
    uploaded = st.sidebar.file_uploader("Upload Triangle", type=['csv', 'xlsx'])
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

# Triangle info
info = TriangleInfo(triangle)
st.sidebar.markdown(f"""
**Triangle Info:**
- Years: {info.n_years} ({info.origin_start} - {info.origin_end})
- Periods: {info.n_periods}
""")

# Create hash for caching
triangle_hash = hash(triangle.to_json())

# Run base Chain-Ladder
cl = ChainLadder(triangle)
cl.run_full_analysis()

# --------------------------------------------
# AUTO-RUN: Anomaly Detection
# --------------------------------------------
with st.spinner("Running anomaly detection..."):
    detector = run_anomaly_detection(triangle)
    anomalies = detector.detect_anomalies()
    anomaly_heatmap = detector.get_anomaly_heatmap()
    anomaly_summary = detector.summary()

# --------------------------------------------
# AUTO-RUN: Model Selection
# --------------------------------------------
min_window = st.session_state.get('min_window', 3)
max_window = st.session_state.get('max_window', min(10, len(triangle)))

with st.spinner("Running model selection..."):
    comparison_table, validation_results, windowed_estimators = run_model_selection(
        triangle_hash, triangle, min_window, max_window
    )

    # Get top 3 CL estimators
    top3_cl_names = comparison_table.head(3).index.tolist()
    top3_estimators = [e for e in windowed_estimators if e.name in top3_cl_names]

    # Run cross-model validation with ALL models on SAME holdout
    model_results, model_factors = run_cross_model_validation(
        triangle, top3_estimators
    )

# Store in session state for diagnostics
# model_factors now contains ALL models (CL, Mack, GBM + top 3 CL patterns)
st.session_state['model_factors'] = model_factors
st.session_state['triangle'] = triangle
st.session_state['cl'] = cl

# ============================================
# 1. LOSS DEVELOPMENT TRIANGLES
# ============================================
st.header("📐 Loss Development Triangles")

incremental = calculate_incremental_triangle(triangle)
lag_factors = calculate_lag_factors(triangle)

# Prepare text matrices for display
triangle_text = triangle.applymap(format_number).values
incremental_text = incremental.applymap(format_number).values

tab1, tab2, tab3 = st.tabs(["Cumulative Triangle", "Incremental Triangle", "Lag Factors + Anomalies"])

with tab1:
    fig = go.Figure(data=go.Heatmap(
        z=triangle.values,
        x=[str(c) for c in triangle.columns],
        y=[str(i) for i in triangle.index],
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Cumulative"),
        hoverongaps=False,
        text=triangle_text,
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), xaxis_title="Development Period", yaxis_title="Accident Year")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure(data=go.Heatmap(
        z=incremental.values,
        x=[str(c) for c in incremental.columns],
        y=[str(i) for i in incremental.index],
        colorscale="Greens",
        showscale=True,
        colorbar=dict(title="Incremental"),
        hoverongaps=False,
        text=incremental_text,
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), xaxis_title="Development Period", yaxis_title="Accident Year")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Lag factors colored by column (each column has its own color scale)
    # Normalize each column to [0, 1] for coloring
    lag_normalized = lag_factors.copy()
    for col in lag_factors.columns:
        col_data = lag_factors[col].dropna()
        if len(col_data) > 0:
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                lag_normalized[col] = (lag_factors[col] - col_min) / (col_max - col_min)
            else:
                lag_normalized[col] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=lag_normalized.values,
        x=[str(c) for c in lag_factors.columns],
        y=[str(i) for i in lag_factors.index],
        colorscale="RdYlBu_r",
        showscale=True,
        colorbar=dict(title="Relative<br>(per column)"),
        hoverongaps=False,
        text=lag_factors.round(4).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), xaxis_title="Development Period", yaxis_title="Accident Year",
                      title="Lag Factors (colored by relative position within each column)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================
# 2. ANOMALY DETECTION
# ============================================
st.header("🔍 Anomaly Detection")

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Cells Analyzed", anomaly_summary['total_cells'])
col2.metric("Anomalies Found", anomaly_summary['anomalies_detected'])
col3.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate']:.1f}%")

# Anomaly heatmap (z-scores)
st.subheader("Anomaly Heatmap (Z-Scores)")
fig_anomaly = go.Figure(data=go.Heatmap(
    z=anomaly_heatmap.values,
    x=[str(c) for c in anomaly_heatmap.columns],
    y=[str(i) for i in anomaly_heatmap.index],
    colorscale="RdYlGn_r",
    zmid=0,
    showscale=True,
    colorbar=dict(title="Z-Score"),
    hoverongaps=False,
    text=anomaly_heatmap.round(2).values,
    texttemplate="%{text}",
    textfont={"size": 9}
))
fig_anomaly.update_layout(
    height=400,
    yaxis=dict(autorange="reversed"),
    xaxis_title="Development Period",
    yaxis_title="Accident Year"
)
st.plotly_chart(fig_anomaly, use_container_width=True)

# Anomaly table
if anomalies:
    st.subheader("Detected Anomalies")
    anomaly_df = detector.get_anomaly_summary()
    st.dataframe(anomaly_df, use_container_width=True)
else:
    st.success("✅ No significant anomalies detected!")

# Re-run button
if st.button("🔄 Re-run Anomaly Detection"):
    st.cache_resource.clear()
    st.rerun()

st.markdown("---")

# ============================================
# 3. AUTOMATIC MODEL SELECTION
# ============================================
st.header("🎯 Automatic Model Selection")

# CL Pattern Selection
st.subheader("Chain-Ladder Pattern Selection")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    new_min = st.slider("Min Window", 2, 5, min_window, key="min_w")
with col2:
    new_max = st.slider("Max Window", 5, 15, max_window, key="max_w")
with col3:
    if st.button("🔄 Re-run Selection"):
        st.session_state['min_window'] = new_min
        st.session_state['max_window'] = new_max
        st.cache_data.clear()
        st.rerun()

st.success(f"✅ Tested {len(windowed_estimators)} CL combinations")

# Best by method table
best_by_method = get_optimal_window_by_method(validation_results, criterion='RMSE')
display_df = best_by_method.copy()
display_df['Window'] = display_df['Window_Length'].apply(lambda x: f"{int(x)}y" if pd.notna(x) else "All")
display_df['RMSE'] = display_df['Error'].apply(lambda x: f"{x:.4f}")
st.dataframe(display_df[['Method', 'Window', 'RMSE']].head(7), use_container_width=True, hide_index=True)

st.markdown("---")

# Cross-Model Comparison
st.subheader("📊 Cross-Model Comparison (including GBM + Top 3 CL)")

# All models now validated on SAME holdout positions
if model_results:
    # Ranking table
    results_df = pd.DataFrame(model_results).T.sort_values('RMSE')
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df = results_df[['Rank', 'RMSE', 'MAE']].head(7)

    # Add medal emojis
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    results_df['Rank'] = results_df['Rank'].apply(lambda x: f"{medals.get(x, '')} {x}")
    results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    results_df['MAE'] = results_df['MAE'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)

    st.dataframe(results_df, use_container_width=True)

    # Settlement Speed Chart
    st.subheader("📈 Settlement Speed (% of Ultimate Reported)")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Plot all models including top 3 CL
    for i, (model_name, factors) in enumerate(model_factors.items()):
        pattern = factors_to_pattern(factors.values)
        periods = list(range(1, len(pattern) + 1))

        # Shorten name for legend
        display_name = model_name
        if len(model_name) > 25:
            display_name = model_name[:22] + "..."

        fig.add_trace(go.Scatter(
            x=periods,
            y=pattern,
            mode='lines+markers',
            name=display_name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        height=450,
        xaxis_title="Development Period",
        yaxis_title="% of Ultimate Reported",
        xaxis=dict(tick0=1, dtick=1),
        yaxis=dict(range=[0, 105]),
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No models validated successfully.")

# ============================================
# SIDEBAR: MODEL DIAGNOSTICS
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Diagnostics")

available_models = list(model_factors.keys()) if model_factors else ["Chain-Ladder"]
selected_model = st.sidebar.selectbox("Select Model:", available_models, key="diag_model")

# Get factors
if selected_model in model_factors:
    diag_factors = model_factors[selected_model]
else:
    diag_factors = cl.selected_factors

# Run diagnostics
diag = DiagnosticTests(triangle, diag_factors)
score = diag.get_model_adequacy_score()

# Adequacy gauge (simplified for sidebar)
adequacy_color = "🟢" if score['adequacy_score'] >= 70 else "🟡" if score['adequacy_score'] >= 40 else "🔴"
st.sidebar.metric("Adequacy Score", f"{adequacy_color} {score['adequacy_score']:.0f}%")
st.sidebar.write(f"**Rating:** {score['rating']}")

if score['issues']:
    st.sidebar.warning(f"Issues: {len(score['issues'])}")
    with st.sidebar.expander("View Issues"):
        for issue in score['issues']:
            st.write(f"• {issue}")

# Residuals histogram
with st.sidebar.expander("📊 Residuals Distribution"):
    res = ResidualAnalyzer(triangle, diag_factors)
    res.fit()
    residuals = res.standardized_residuals.values.flatten()
    residuals = residuals[~np.isnan(residuals)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=15, marker_color='steelblue'))
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Residual", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# Statistical tests
with st.sidebar.expander("📋 Statistical Tests"):
    all_tests = diag.run_all_tests()

    tests_status = []
    tests_status.append(("Calendar Year", "✅" if not all_tests['calendar_year_effect'].get('calendar_effect_detected', False) else "❌"))
    tests_status.append(("Accident Year", "✅" if all_tests['accident_year_effect'].get('n_anomalous', 0) == 0 else "❌"))
    tests_status.append(("Independence", "✅" if not all_tests['independence'].get('independence_assumption_violated', False) else "❌"))
    tests_status.append(("Proportionality", "✅" if all_tests['proportionality'].get('proportionality_holds', True) else "❌"))
    tests_status.append(("Variance", "✅" if all_tests['variance_structure'].get('variance_assumption_holds', True) else "❌"))

    for test, status in tests_status:
        st.write(f"{status} {test}")

# Residuals triangle
with st.sidebar.expander("🔲 Residuals Triangle"):
    res_triangle = pd.DataFrame(index=triangle.index, columns=triangle.columns[:-1], dtype=float)
    for j in range(len(triangle.columns) - 1):
        col_curr = triangle.columns[j]
        col_next = triangle.columns[j + 1]
        factor = diag_factors.iloc[j] if j < len(diag_factors) else 1.0
        for year in triangle.index:
            curr = triangle.loc[year, col_curr]
            next_v = triangle.loc[year, col_next]
            if pd.notna(curr) and pd.notna(next_v) and curr > 0:
                expected = curr * factor
                res_triangle.loc[year, col_curr] = (next_v - expected) / np.sqrt(curr)

    # Standardize
    vals = res_triangle.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) > 0:
        res_triangle = (res_triangle - np.mean(vals)) / np.std(vals)

    fig = go.Figure(data=go.Heatmap(
        z=res_triangle.values,
        x=[str(c) for c in res_triangle.columns],
        y=[str(i) for i in res_triangle.index],
        colorscale="RdBu_r",
        zmid=0,
        showscale=False
    ))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
