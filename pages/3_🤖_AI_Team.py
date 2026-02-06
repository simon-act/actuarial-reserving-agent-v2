import streamlit as st
import time
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.orchestrator import Orchestrator
from agents.schemas import ReservingInput, AgentRole
from data_loader import TriangleLoader, TriangleInfo

st.set_page_config(page_title="AI Actuarial Team", page_icon="🤖", layout="wide")

# Custom CSS to control scroll and layout
st.markdown("""
<style>
    /* Limit chat message container height to prevent auto-scroll issues */
    section[data-testid="stChatMessageContainer"] {
        max-height: 400px;
        overflow-y: auto;
    }

    /* Make sure tabs stay visible */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        background: var(--background-color);
        z-index: 100;
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Actuarial Team")
st.markdown("---")

# ============================================
# HELPER FUNCTIONS
# ============================================

SAMPLE_TRIANGLES = {
    'swiss_re': 'Swiss Re Property Reinsurance (16×16)',
    'mack': 'Mack GenIns - Mack 1993 (10×10)',
    'taylor_ashe': 'Taylor/Ashe Benchmark (10×10)',
    'abc': 'ABC Insurance Incurred (10×10)',
    'uk_motor': 'UK Motor Claims (7×7)'
}

def get_project_root() -> Path:
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "data").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent

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

# ============================================
# SIDEBAR
# ============================================

# --- Data Selection ---
st.sidebar.header("📁 Data Selection")

data_source = st.sidebar.radio("Data source:", ["Sample Triangles", "Upload File"], key="ai_data_source")

selected_triangle = None
selected_premium = None
triangle_label = ""

if data_source == "Sample Triangles":
    triangle_name = st.sidebar.selectbox(
        "Select Triangle:",
        options=list(SAMPLE_TRIANGLES.keys()),
        format_func=lambda x: SAMPLE_TRIANGLES[x],
        key="ai_triangle_select"
    )
    selected_triangle = load_sample_triangle(triangle_name)
    if triangle_name == 'swiss_re':
        selected_premium = load_premium_data()
    triangle_label = SAMPLE_TRIANGLES[triangle_name]
    st.sidebar.success(f"✅ {triangle_label}")
else:
    uploaded = st.sidebar.file_uploader("Upload Triangle (CSV/XLSX)", type=['csv', 'xlsx'], key="ai_upload")
    if uploaded:
        if uploaded.name.endswith('.csv'):
            selected_triangle = pd.read_csv(uploaded, index_col=0)
        else:
            selected_triangle = pd.read_excel(uploaded, index_col=0)
        try:
            selected_triangle.columns = selected_triangle.columns.astype(int)
        except:
            pass
        selected_triangle = selected_triangle.apply(pd.to_numeric, errors='coerce')
        triangle_label = uploaded.name
        st.sidebar.success(f"✅ {uploaded.name}")

if selected_triangle is not None:
    info = TriangleInfo(selected_triangle)
    st.sidebar.markdown(f"""
**Triangle Info:**
- Years: {info.n_years} ({info.origin_start} - {info.origin_end})
- Periods: {info.n_periods}
- Type: {'Cumulative' if info.is_cumulative else 'Incremental'}
""")

st.sidebar.markdown("---")

# --- Team Members ---
st.sidebar.markdown("### 👥 Team Members")
st.sidebar.markdown("""
- **🧠 Methodology**: Strategy & Planning
- **🎯 Selection**: LLM Method Optimizer
- **⚙️ Actuary**: Calculations & Models
- **🔍 Validator**: Quality Assurance
- **📢 Reporter**: Communication
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "final_result" not in st.session_state:
    st.session_state.final_result = None

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Guard: no triangle selected
if selected_triangle is None:
    st.info("👈 Select or upload a triangle from the sidebar to get started.")
    st.stop()

# User Input
if prompt := st.chat_input("Tell the research team what to analyze..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine Mode -> Unified by Orchestrator
    with st.chat_message("assistant"):
        orch = Orchestrator()

        # Build ReservingInput from selected triangle
        # Save selected triangle to a temp CSV so the orchestrator can load it
        tmp_dir = get_project_root() / "data" / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_triangle_path = tmp_dir / "_ai_team_triangle.csv"
        selected_triangle.to_csv(tmp_triangle_path)

        premium_path = None
        if selected_premium is not None:
            tmp_premium_path = tmp_dir / "_ai_team_premium.csv"
            selected_premium.to_csv(tmp_premium_path)
            premium_path = str(tmp_premium_path)

        inputs = ReservingInput(
            triangle_path=str(tmp_triangle_path),
            premium_path=premium_path
        )

        # Containers (Lazy initialization)
        containers = {}
        def get_container(name, expanded=False):
            if name not in containers:
                containers[name] = st.status(f"⚙️ {name.capitalize()} Agent", expanded=expanded)
            return containers[name]

        # Stream Updates from Unified Router
        full_response = ""
        current_context = st.session_state.final_result

        # If no analysis yet, provide basic triangle info so Q&A can answer simple questions
        if current_context is None and selected_triangle is not None:
            tri = selected_triangle
            current_context = {
                "triangle_preview": {
                    "accident_years": [int(y) for y in tri.index.tolist()],
                    "development_periods": [int(p) for p in tri.columns.tolist()],
                    "shape": f"{tri.shape[0]} accident years × {tri.shape[1]} development periods",
                    "first_year": int(tri.index.min()),
                    "last_year": int(tri.index.max()),
                    "label": triangle_label,
                }
            }

        reasoning_capture = {}  # Capture agent thought data for reasoning tab

        for update in orch.route_request(prompt, current_result=current_context, inputs=inputs):
            step = update["step"]

            # 1. Router Decision
            if step == "router":
                st.caption(update["message"])

            # 2. Analysis Steps (Methodology, Selection, Execution, etc.)
            elif step in ["methodology", "selection", "execution", "validation", "reporting"]:
                cont = get_container(step, expanded=(update["status"]=="running"))
                if update["status"] == "running":
                    cont.write(update["message"])
                elif update["status"] == "done":
                    cont.update(state="complete", expanded=False)
                    # Specific summaries
                    if step == "methodology":
                        cont.write(f"**Plan:** {update['data'].analysis_type.value}")
                    elif step == "selection":
                        sel = update.get("data", {})
                        cont.write(f"**Selected:** {sel.get('selected_estimator', 'N/A')}")
                        if sel.get("bf_years"):
                            cont.write(f"**BF Years:** {', '.join(sel.get('bf_years', []))}")
                    elif step == "execution":
                        cont.write(f"**Reserves:** ${update['data'].chain_ladder.total_reserve:,.0f}")
                    elif step == "validation":
                        cont.write(f"**Score:** {update['data'].overall_confidence_score}/100")

            # 2b. Selection Agent Thought Process (streaming reasoning)
            elif step.startswith("selection_thought_"):
                phase = step.replace("selection_thought_", "")
                cont = get_container("selection", expanded=True)
                cont.write(update["message"])
                # Capture completed thought data for the reasoning tab
                if update.get("thought_data") and update["status"] == "complete":
                    reasoning_capture[phase] = {
                        "message": update["message"],
                        "data": update["thought_data"]
                    }

            # 3. Final Result Bundle
            elif step == "complete":
                update["result"]["agent_reasoning"] = reasoning_capture
                st.session_state.final_result = update["result"]
                st.success("Analysis Updated!")

            # 4. Code Execution (NEW)
            elif step == "code":
                if update["status"] == "running":
                    st.caption("🐍 " + update["message"])
                elif update["status"] == "done":
                    full_response = update["data"]
                    st.markdown(full_response)

            # 5. Answers (Synthesis or Direct Q&A)
            elif step in ["final_synthesis", "qa"]:
                if update["status"] == "done":
                    full_response = update["data"]
                    st.markdown(full_response)

        # Persist the final text response
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show Final Artifacts (Always show if available)
    if st.session_state.final_result:
        res = st.session_state.final_result
        st.divider()

        # Reset Button to Start Over
        if st.button("🔄 Start New Analysis"):
            st.session_state.final_result = None
            st.session_state.messages = []
            st.rerun()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📄 Final Report",
            "🎯 Method Selection",
            "📊 Charts",
            "🧬 Structured Data",
            "📜 Audit Trail",
            "🧠 Agent Reasoning"
        ])

        with tab1:
            st.markdown(res["report"])
            st.download_button("Download Report", res["report"], file_name="actuarial_report.md")

        with tab2:
            # Method Selection Results
            sel = res.get("method_selection", {})
            if sel:
                st.subheader("🧠 LLM Method Selection")
                st.success(f"**Selected Estimator:** {sel.get('selected_estimator', 'N/A')}")
                st.write(f"**Reason:** {sel.get('estimator_reason', '')}")

                immature = sel.get("immature_years") or sel.get("bf_years")
                if immature:
                    st.warning(f"**Immature Years (Need Extra Scrutiny):** {', '.join(immature)}")
                    reason = sel.get("immature_reason") or sel.get("bf_reason", '')
                    st.write(f"**Reason:** {reason}")

                st.divider()
                st.markdown("**Summary:**")
                st.info(sel.get("summary", "No summary available"))
            else:
                st.info("Method selection not available for this analysis.")

        with tab3:
            # === CHARTS TAB ===
            sel = res.get("method_selection", {})
            structured = res.get("structured_results")

            if sel:
                st.subheader("📊 Visual Analysis")

                # --- 1. ESTIMATORS COMPARISON BAR CHART ---
                all_est = sel.get("all_estimators", {})
                if all_est:
                    st.markdown("### Reserve Estimates by Estimator")

                    df_est = pd.DataFrame([
                        {"Estimator": name, "Reserve": reserve,
                         "Selected": "✓ Selected" if name == sel.get("selected_estimator") else ""}
                        for name, reserve in sorted(all_est.items(), key=lambda x: x[1])
                    ])

                    # Color the selected estimator differently
                    colors = ["#1f77b4" if name != sel.get("selected_estimator") else "#2ca02c"
                              for name in df_est["Estimator"]]

                    fig_est = go.Figure(data=[
                        go.Bar(
                            x=df_est["Reserve"],
                            y=df_est["Estimator"],
                            orientation='h',
                            marker_color=colors,
                            text=[f"${r:,.0f}" for r in df_est["Reserve"]],
                            textposition='outside'
                        )
                    ])
                    fig_est.update_layout(
                        title=f"All Estimators Comparison (Selected: {sel.get('selected_estimator', 'N/A')})",
                        xaxis_title="Total Reserve ($)",
                        yaxis_title="Estimator",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_est, use_container_width=True)

                # --- 2. MATURITY BY YEAR CHART ---
                maturity = sel.get("maturity_by_year", {})
                if maturity:
                    st.markdown("### Maturity by Accident Year")

                    bf_years = set(sel.get("bf_years", []))
                    df_mat = pd.DataFrame([
                        {"Year": str(year), "Maturity": mat,
                         "Method": "BF Recommended" if str(year) in bf_years else "Chain Ladder"}
                        for year, mat in sorted(maturity.items())
                    ])

                    fig_mat = px.bar(
                        df_mat,
                        x="Year",
                        y="Maturity",
                        color="Method",
                        color_discrete_map={"Chain Ladder": "#2ca02c", "BF Recommended": "#ff7f0e"},
                        text=[f"{m:.0f}%" for m in df_mat["Maturity"]]
                    )
                    fig_mat.add_hline(y=70, line_dash="dash", line_color="red",
                                      annotation_text="BF Threshold (70%)")
                    fig_mat.add_hline(y=90, line_dash="dash", line_color="green",
                                      annotation_text="Mature (90%)")
                    fig_mat.update_layout(
                        title="Year Maturity Analysis (LLM Decision: CL vs BF)",
                        xaxis_title="Accident Year",
                        yaxis_title="Maturity %",
                        height=400
                    )
                    st.plotly_chart(fig_mat, use_container_width=True)

                # --- 3. VALIDATION METRICS HEATMAP ---
                val_metrics = sel.get("validation_metrics", {})
                if val_metrics:
                    st.markdown("### Validation Metrics by Estimator")

                    # Create DataFrame for heatmap
                    metrics_list = []
                    for est_name, metrics in val_metrics.items():
                        row = {"Estimator": est_name}
                        for metric_name, value in metrics.items():
                            if metric_name in ['MSE', 'MAE', 'RMSE', 'MAPE', 'R2']:
                                row[metric_name] = value
                        metrics_list.append(row)

                    if metrics_list:
                        df_metrics = pd.DataFrame(metrics_list).set_index("Estimator")

                        # Normalize for display (lower is better for most, higher for R2)
                        col1, col2 = st.columns(2)

                        with col1:
                            # MSE/MAE comparison
                            if 'MSE' in df_metrics.columns and 'MAE' in df_metrics.columns:
                                df_err = df_metrics[['MSE', 'MAE']].reset_index()
                                df_err_melted = df_err.melt(id_vars='Estimator', var_name='Metric', value_name='Value')

                                fig_err = px.bar(
                                    df_err_melted,
                                    x="Estimator",
                                    y="Value",
                                    color="Metric",
                                    barmode="group",
                                    title="MSE vs MAE by Estimator (Lower is Better)"
                                )
                                fig_err.update_layout(height=350, xaxis_tickangle=-45)
                                st.plotly_chart(fig_err, use_container_width=True)

                        with col2:
                            # R2 comparison
                            if 'R2' in df_metrics.columns:
                                df_r2 = df_metrics[['R2']].reset_index()

                                fig_r2 = px.bar(
                                    df_r2,
                                    x="Estimator",
                                    y="R2",
                                    title="R² Score by Estimator (Higher is Better)",
                                    color="R2",
                                    color_continuous_scale="Greens"
                                )
                                fig_r2.update_layout(height=350, xaxis_tickangle=-45)
                                st.plotly_chart(fig_r2, use_container_width=True)

                # --- 4. RESERVES BY YEAR (from structured results) ---
                if structured and structured.detailed_data:
                    reserves_by_year = structured.detailed_data.reserves_by_year
                    if reserves_by_year:
                        st.markdown("### Reserves by Accident Year")

                        df_res = pd.DataFrame([
                            {"Year": str(year), "Reserve": reserve}
                            for year, reserve in sorted(reserves_by_year.items())
                        ])

                        fig_res = px.bar(
                            df_res,
                            x="Year",
                            y="Reserve",
                            title="Reserve Distribution by Accident Year",
                            color="Reserve",
                            color_continuous_scale="Blues"
                        )
                        fig_res.update_traces(text=[f"${r:,.0f}" for r in df_res["Reserve"]], textposition='outside')
                        fig_res.update_layout(height=400)
                        st.plotly_chart(fig_res, use_container_width=True)

                # --- 5. DEVELOPMENT FACTORS CHART ---
                if structured and structured.detailed_data:
                    dev_factors = structured.detailed_data.development_factors
                    if dev_factors:
                        st.markdown("### Development Factors")

                        df_dev = pd.DataFrame([
                            {"Period": period, "Factor": factor}
                            for period, factor in dev_factors.items()
                        ])

                        fig_dev = px.line(
                            df_dev,
                            x="Period",
                            y="Factor",
                            title="Age-to-Age Development Factors",
                            markers=True
                        )
                        fig_dev.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                          annotation_text="No Development (1.0)")
                        fig_dev.update_layout(height=350)
                        st.plotly_chart(fig_dev, use_container_width=True)

                # --- 6. PIE CHART: RESERVE DISTRIBUTION ---
                if structured and structured.detailed_data and structured.detailed_data.reserves_by_year:
                    st.markdown("### Reserve Distribution (Pie)")

                    reserves_by_year = structured.detailed_data.reserves_by_year
                    # Only show years with positive reserves
                    positive_reserves = {k: v for k, v in reserves_by_year.items() if v > 0}

                    if positive_reserves:
                        df_pie = pd.DataFrame([
                            {"Year": str(year), "Reserve": reserve}
                            for year, reserve in sorted(positive_reserves.items())
                        ])

                        fig_pie = px.pie(
                            df_pie,
                            values="Reserve",
                            names="Year",
                            title="Reserve Share by Accident Year",
                            hole=0.4
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Run an analysis to see charts.")

        with tab4:
            st.json(res["structured_results"].model_dump(mode='json'))

        with tab5:
            for log in res["audit_trail"]:
                st.text(f"[{log.timestamp.strftime('%H:%M:%S')}] {log.agent.value}: {log.action}")
            # Append Q&A logs if any exist in the local orch instance

        with tab6:
            st.subheader("🧠 Agent Reasoning Process")
            st.caption("Transparent view into how each AI agent reasoned during the analysis.")

            reasoning = res.get("agent_reasoning", {})
            sel_data = res.get("method_selection", {})

            if not reasoning:
                st.info("💡 Agent reasoning is captured during live analysis. Run a new analysis to see the full thought process here.")
            else:
                # ── SELECTION AGENT ─────────────────────────────
                st.markdown("### 🎯 Selection Agent — Step-by-Step Reasoning")

                # Step 1: Data Collection
                if "data_collection" in reasoning:
                    with st.expander("📊 Step 1 — Data Collection (All Estimators)", expanded=False):
                        data = reasoning["data_collection"].get("data", {})
                        if data and data.get("reserves"):
                            for name, reserve in sorted(data["reserves"].items(), key=lambda x: x[1]):
                                st.write(f"**{name}:** ${reserve:,.0f}")
                        else:
                            st.write("No estimator data captured.")

                # Step 2: Cross-Validation
                if "validation" in reasoning:
                    with st.expander("📈 Step 2 — Cross-Validation", expanded=False):
                        data = reasoning["validation"].get("data", {})
                        if data and data.get("best"):
                            for metric, best_est in data["best"].items():
                                st.success(f"Best by **{metric}:** {best_est}")
                        if data and data.get("table") is not None:
                            try:
                                st.dataframe(data["table"], use_container_width=True)
                            except Exception:
                                st.write("Validation table available in Structured Data tab.")

                # Step 3: Diagnostics
                if "diagnostics" in reasoning:
                    with st.expander("🔬 Step 3 — Diagnostic Tests", expanded=False):
                        data = reasoning["diagnostics"].get("data", {})
                        if data:
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.metric("Model Adequacy", f"{data.get('adequacy_score', 'N/A')}/100")
                            with col_d2:
                                st.metric("Rating", data.get("rating", "N/A"))
                            issues = data.get("issues", [])
                            if issues:
                                st.warning("Issues detected:")
                                for issue in issues:
                                    st.write(f"- {issue}")
                            else:
                                st.success("No diagnostic issues found.")

                # Step 4: Pattern Analysis (with full AgentThought)
                if "pattern_analysis" in reasoning:
                    with st.expander("🔍 Step 4 — Pattern Analysis (LLM Reasoning)", expanded=True):
                        data = reasoning["pattern_analysis"].get("data", {})
                        if data:
                            smoothing = data.get("smoothing_applied", False)
                            if smoothing:
                                st.success(f"✅ Smoothing applied: **{data.get('smoothing_method', 'N/A')}** (weight: {data.get('smoothing_weight', 'N/A')})")
                            else:
                                st.info("ℹ️ No smoothing applied — raw development patterns deemed adequate.")

                            # Display the full AgentThought if available
                            tp = data.get("thought_process")
                            if tp and hasattr(tp, 'analysis'):
                                st.divider()

                                # ── Analysis Phase ──
                                st.markdown("**🔍 Analysis Phase**")
                                st.write(tp.analysis.reasoning)
                                col_a1, col_a2 = st.columns(2)
                                with col_a1:
                                    if tp.analysis.observations:
                                        st.markdown("*Observations:*")
                                        for obs in tp.analysis.observations:
                                            st.write(f"- {obs}")
                                    if tp.analysis.patterns:
                                        st.markdown("*Patterns detected:*")
                                        for p in tp.analysis.patterns:
                                            st.write(f"- {p}")
                                with col_a2:
                                    if tp.analysis.anomalies:
                                        st.markdown("*⚠️ Anomalies:*")
                                        for a in tp.analysis.anomalies:
                                            st.write(f"- {a}")
                                    st.write(f"*Confidence:* **{tp.analysis.confidence.value}**")

                                st.divider()

                                # ── Decision Phase ──
                                st.markdown("**🎯 Decision Phase**")
                                st.write(f"*Choice:* **{tp.decision.choice}**")
                                st.write(tp.decision.reasoning)
                                st.write(f"*Confidence:* **{tp.decision.confidence.value}**")
                                col_b1, col_b2, col_b3 = st.columns(3)
                                with col_b1:
                                    if tp.decision.evidence:
                                        st.markdown("*Evidence:*")
                                        for e in tp.decision.evidence:
                                            st.write(f"- {e}")
                                with col_b2:
                                    if tp.decision.alternatives:
                                        st.markdown("*Alternatives considered:*")
                                        for a in tp.decision.alternatives:
                                            st.write(f"- {a}")
                                with col_b3:
                                    if tp.decision.risks:
                                        st.markdown("*Risks:*")
                                        for r in tp.decision.risks:
                                            st.write(f"- {r}")

                                st.divider()

                                # ── Self-Critique Phase ──
                                st.markdown("**🔎 Self-Critique Phase**")
                                st.write(tp.critique.recommendation)
                                st.write(f"*Confidence:* **{tp.critique.confidence.value}**")
                                col_c1, col_c2 = st.columns(2)
                                with col_c1:
                                    if tp.critique.weaknesses:
                                        st.markdown("*Weaknesses identified:*")
                                        for w in tp.critique.weaknesses:
                                            st.write(f"- {w}")
                                with col_c2:
                                    if tp.critique.alternatives:
                                        st.markdown("*Alternative approaches:*")
                                        for a in tp.critique.alternatives:
                                            st.write(f"- {a}")

                # Step 5: Tail Fitting
                if "tail_fitting" in reasoning:
                    with st.expander("📐 Step 5 — Tail Factor Estimation", expanded=False):
                        data = reasoning["tail_fitting"].get("data", {})
                        if data:
                            st.metric("Tail Factor", f"{data.get('tail_factor', 1.0):.4f}")

                # Step 6: LLM Final Decision
                if "llm_decision" in reasoning:
                    with st.expander("🧠 Step 6 — LLM Final Decision", expanded=True):
                        data = reasoning["llm_decision"].get("data", {})
                        if data:
                            st.success(f"**Selected Method:** {data.get('choice', 'N/A')}")
                            st.write(f"**Confidence:** {data.get('confidence', 'N/A')}")
                            st.markdown("---")
                            st.markdown("**Full Reasoning:**")
                            st.info(data.get("reasoning", "No reasoning available."))

                # Step 7: Prudential Adjustments
                if "prudence" in reasoning:
                    with st.expander("⚖️ Step 7 — Prudential Adjustments", expanded=False):
                        data = reasoning["prudence"].get("data", {})
                        if data:
                            adjustments = data.get("adjustments", [])
                            if adjustments:
                                for adj in adjustments:
                                    st.write(f"- **{adj.get('type', '')}** (period {adj.get('period', '')}): {adj.get('old', 0):.4f} → {adj.get('new', 0):.4f}")
                            else:
                                st.success("No prudential adjustments needed.")

                # ── VALIDATION AGENT — PEER REVIEW ──────────────
                st.markdown("---")
                st.markdown("### 🔍 Validation Agent — Peer Review")

                vf = sel_data.get("validation_feedback") if sel_data else None
                if vf:
                    with st.expander("📋 Peer Review Feedback", expanded=True):
                        agrees = vf.get("agrees", True)
                        if agrees:
                            st.success("✅ Validator **AGREES** with the selection")
                        else:
                            st.error("❌ Validator **DISAGREES** with the selection")

                        st.write(f"**Confidence:** {vf.get('confidence', 'N/A')}")

                        if vf.get("reasoning"):
                            st.markdown("**Reasoning:**")
                            st.info(vf["reasoning"])

                        if vf.get("concerns"):
                            st.warning("**Concerns:**")
                            for concern in vf["concerns"]:
                                st.write(f"- {concern}")

                        if vf.get("suggestions"):
                            st.markdown("**Suggestions:**")
                            for suggestion in vf["suggestions"]:
                                st.write(f"- {suggestion}")

                        if vf.get("alternative"):
                            st.error(f"**Alternative Recommendation:** {vf['alternative']}")
                else:
                    st.info("Peer review feedback not available for this analysis.")

                # ── FINAL QUALITY SCORE ─────────────────────────
                st.markdown("---")
                st.markdown("### ✅ Final Quality Score")

                validation = res.get("validation")
                if validation:
                    col_q1, col_q2 = st.columns([1, 2])
                    with col_q1:
                        st.metric("Confidence Score", f"{validation.overall_confidence_score}/100")
                        status_icons = {"PASSED": "🟢", "WARNING": "🟡", "REJECTED": "🔴"}
                        st.write(f"{status_icons.get(validation.status.value, '⚪')} **{validation.status.value}**")
                    with col_q2:
                        if validation.issues:
                            for issue in validation.issues:
                                if issue.severity == "WARNING":
                                    st.warning(f"[{issue.component}] {issue.message}")
                                elif issue.severity == "CRITICAL":
                                    st.error(f"[{issue.component}] {issue.message}")
                                else:
                                    st.info(f"[{issue.component}] {issue.message}")
                        else:
                            st.success("No validation issues found.")
                        st.caption(f"**Summary:** {validation.comparison_summary}")
