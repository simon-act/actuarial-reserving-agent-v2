import streamlit as st
import time
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.orchestrator import Orchestrator
from agents.schemas import ReservingInput, AgentRole

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

# Sidebar
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

# User Input
if prompt := st.chat_input("Tell the research team what to analyze..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine Mode -> Unified by Orchestrator
    with st.chat_message("assistant"):
        orch = Orchestrator()

        # Containers (Lazy initialization)
        containers = {}
        def get_container(name, expanded=False):
            if name not in containers:
                containers[name] = st.status(f"⚙️ {name.capitalize()} Agent", expanded=expanded)
            return containers[name]

        # Stream Updates from Unified Router
        full_response = ""
        current_context = st.session_state.final_result

        for update in orch.route_request(prompt, current_result=current_context):
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

            # 3. Final Result Bundle
            elif step == "complete":
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

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Final Report",
            "🎯 Method Selection",
            "📊 Charts",
            "🧬 Structured Data",
            "📜 Audit Trail"
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
