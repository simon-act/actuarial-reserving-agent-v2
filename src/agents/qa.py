from agents.schemas import ReservingOutput, ValidationReport, AgentRole, AgentLog
from agents.llm_utils import LLMClient
from typing import Tuple, List, Dict, Optional
import json


class QASpecialistAgent:
    """
    ANALYTICAL Q&A AGENT (Enhanced with Conversation Memory)

    - Specialized in reserving results analysis.
    - Capabilities: Qualitative analysis, trends, patterns, critical reading.
    - Constraints: No new calculations, no data invention.
    - NEW: Uses conversation history for context continuity.
    """

    def __init__(self):
        self.role = AgentRole.REPORTING
        self.llm = LLMClient()

    def answer_query(
        self,
        query: str,
        context: dict,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Tuple[str, AgentLog]:
        """
        Answer a query using the provided context (results + validation).

        Args:
            query: User question
            context: Results dictionary with structured_results and validation
            conversation_history: Optional list of recent messages for context continuity
        """
        results: ReservingOutput = context.get("structured_results")
        validation: ValidationReport = context.get("validation")

        # Format conversation history for context
        history_context = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:200]  # Truncate
                history_lines.append(f"{role}: {content}")
            history_context = "\n".join(history_lines)

        # Try LLM first - using the enhanced "Analytical" persona
        llm_error = None
        if self.llm.is_available():
            try:
                # Include conversation history in prompt
                history_section = ""
                if history_context:
                    history_section = f"\nRECENT CONVERSATION:\n{history_context}\n"

                # Check if we have full analysis results or just triangle preview
                triangle_preview = context.get("triangle_preview")

                if results is not None:
                    # Full analysis context available
                    results_json = results.model_dump_json(exclude={"timestamp"})
                    validation_json = validation.model_dump_json() if validation else "{}"

                    system_prompt = (
                        "You are an Analytical Q&A Agent specialized in actuarial reserving results.\n"
                        "You have access to COMPLETE DATA in JSON format:\n\n"
                        "AVAILABLE DATA:\n"
                        "- triangle_info: n_accident_years, n_development_periods, first/last year\n"
                        "- detailed_data.triangle: FULL triangle {year: {period: value}}\n"
                        "- detailed_data.development_factors: link ratios by period\n"
                        "- detailed_data.reserves_by_year: reserve for each accident year\n"
                        "- detailed_data.ultimates_by_year: ultimate loss for each year\n"
                        "- detailed_data.latest_diagonal: most recent value per year\n"
                        "- chain_ladder, mack, bootstrap, cape_cod: method results\n"
                        "- diagnostics: model quality assessment\n"
                        "- validation: issues and confidence score\n\n"
                        "CAPABILITIES:\n"
                        "- Answer ANY question about the data (structure, values, years, trends)\n"
                        "- Provide specific numbers from any year or period\n"
                        "- Analyze patterns and explain results\n"
                        "- Compare methods and values\n"
                        "- Understand follow-up questions using conversation history\n\n"
                        "CONSTRAINT: Only use data present in the JSON. Do not invent numbers.\n\n"
                        "Respond in the same language as the user's question.\n"
                        f"{history_section}\n"
                        "CONTEXT (JSON):\n"
                        f"RESULTS: {results_json}\n"
                        f"VALIDATION: {validation_json}\n"
                    )

                elif triangle_preview is not None:
                    # No analysis yet, but we have basic triangle info
                    import json
                    preview_json = json.dumps(triangle_preview, indent=2)

                    system_prompt = (
                        "You are an Analytical Q&A Agent specialized in actuarial reserving.\n"
                        "No analysis has been run yet, but you have basic information about the selected triangle.\n\n"
                        "AVAILABLE DATA:\n"
                        f"{preview_json}\n\n"
                        "CAPABILITIES:\n"
                        "- Answer questions about the triangle structure (years, periods, shape)\n"
                        "- Describe what data is available\n"
                        "- If the user asks about reserves or analysis results, explain that they need to run an analysis first\n\n"
                        "Respond in the same language as the user's question.\n"
                        f"{history_section}\n"
                    )

                else:
                    raise ValueError(
                        "No data available. Please select a triangle from the sidebar."
                    )

                response = self.llm.get_completion(system_prompt, query)

                # Only treat as error if it starts with our specific error prefix
                if not response.startswith("Error:") and not response.startswith(
                    "Error calling"
                ):
                    return response, AgentLog(
                        agent=self.role,
                        action="Q&A (LLM)",
                        details="Answered via OpenAI",
                    )
                else:
                    llm_error = response
            except Exception as e:
                llm_error = str(e)
                print(f"LLM Failed: {e}. Falling back to rules.")

        # Fallback to Rule-Based Logic
        fallback_preamble = ""
        if llm_error:
            fallback_preamble = f"⚠️ *Note: LLM error ({llm_error}). Using basic keyword rules instead.*\n\n"
        elif not self.llm.is_available():
            error_msg = (
                self.llm.last_error
                if hasattr(self.llm, "last_error")
                else "API key not configured"
            )
            fallback_preamble = f"⚠️ *Note: OpenAI API not available ({error_msg}). Using basic keyword rules.*\n\n"

        query_lower = query.lower()
        response = ""

        # Handle case where no analysis has been run yet
        if results is None:
            triangle_preview = context.get("triangle_preview")
            if triangle_preview:
                years = triangle_preview.get("accident_years", [])
                label = triangle_preview.get("label", "Selected triangle")
                response = (
                    f"**{label}**\n\n"
                    f"- Accident Years: {', '.join(str(y) for y in years)}\n"
                    f"- Shape: {triangle_preview.get('shape', 'N/A')}\n"
                    f"- First Year: {triangle_preview.get('first_year', 'N/A')}\n"
                    f"- Last Year: {triangle_preview.get('last_year', 'N/A')}\n\n"
                    "💡 *Run an analysis to get reserve estimates, diagnostics, and more.*"
                )
            else:
                response = "No triangle selected and no analysis results available. Please select a triangle from the sidebar."

            if fallback_preamble and not response.startswith("⚠️"):
                response = fallback_preamble + response

            log = AgentLog(
                agent=self.role,
                action="Q&A (Triangle Info)",
                details=f"Query: {query[:20]}... | Answered: Triangle preview",
            )
            return response, log

        # 1. Validation / Warnings
        if (
            "warning" in query_lower
            or "valid" in query_lower
            or "issue" in query_lower
            or "problem" in query_lower
        ):
            if validation.issues:
                response = (
                    f"The Validation Agent found {len(validation.issues)} issues:\n"
                )
                for issue in validation.issues:
                    response += f"- **{issue.severity}**: {issue.message}\n"
                response += f"\nOverall Confidence Score: {validation.overall_confidence_score}/100"
            else:
                response = (
                    "✅ No validation issues were found. The results passed all checks."
                )

        # 2. IBNR / Reserve questions
        elif (
            "ibnr" in query_lower or "reserve" in query_lower or "riserv" in query_lower
        ):
            response = f"**Total IBNR Reserve (Chain Ladder)**: ${results.chain_ladder.total_reserve:,.0f}\n"
            if results.mack:
                response += f"**Mack Model**: ${results.mack.total_reserve:,.0f} (SE: ${results.mack.standard_error:,.0f}, CV: {results.mack.cv:.1%})\n"
            if results.bootstrap:
                response += f"**Bootstrap**: ${results.bootstrap.total_reserve:,.0f} (95th percentile: ${results.bootstrap.percentiles.get('95%', 0):,.0f})\n"
            if results.cape_cod:
                response += f"**Cape Cod**: ${results.cape_cod.total_reserve:,.0f}"

        # 3. Ultimate losses
        elif (
            "ultimate" in query_lower
            or "final" in query_lower
            or "totale" in query_lower
        ):
            response = f"**Ultimate Loss (Chain Ladder)**: ${results.chain_ladder.ultimate_loss:,.0f}\n"
            if results.cape_cod:
                response += (
                    f"**Ultimate (Cape Cod)**: ${results.cape_cod.ultimate_loss:,.0f}"
                )

        # 4. Mack specific
        elif (
            "mack" in query_lower
            or "stochastic" in query_lower
            or "uncertainty" in query_lower
            or "incertezza" in query_lower
        ):
            if results.mack:
                response = (
                    f"**Mack Model Results**:\n"
                    f"- Total Reserve: ${results.mack.total_reserve:,.0f}\n"
                    f"- Standard Error: ${results.mack.standard_error:,.0f}\n"
                    f"- Coefficient of Variation: {results.mack.cv:.1%}"
                )
            else:
                response = (
                    "The Mack model was not included in this analysis configuration."
                )

        # 5. Bootstrap / Simulation
        elif (
            "bootstrap" in query_lower
            or "simulat" in query_lower
            or "distribut" in query_lower
            or "percentil" in query_lower
        ):
            if results.bootstrap:
                response = (
                    f"**Bootstrap Results**:\n"
                    f"- Mean Reserve: ${results.bootstrap.total_reserve:,.0f}\n"
                    f"- Standard Error: ${results.bootstrap.standard_error:,.0f}\n"
                    f"- 75th Percentile: ${results.bootstrap.percentiles.get('75%', 0):,.0f}\n"
                    f"- 95th Percentile: ${results.bootstrap.percentiles.get('95%', 0):,.0f}\n"
                    f"- 99.5th Percentile: ${results.bootstrap.percentiles.get('99.5%', 0):,.0f}"
                )
            else:
                response = "Bootstrap simulation was not included in this analysis."

        # 6. Chain Ladder specific
        elif (
            "chain" in query_lower
            or "ladder" in query_lower
            or "cl " in query_lower
            or "development" in query_lower
        ):
            response = (
                f"**Chain Ladder Results**:\n"
                f"- Total Reserve: ${results.chain_ladder.total_reserve:,.0f}\n"
                f"- Ultimate Loss: ${results.chain_ladder.ultimate_loss:,.0f}"
            )

        # 7. Cape Cod
        elif "cape" in query_lower or "cod" in query_lower:
            if results.cape_cod:
                response = (
                    f"**Cape Cod Results**:\n"
                    f"- Total Reserve: ${results.cape_cod.total_reserve:,.0f}\n"
                    f"- Ultimate Loss: ${results.cape_cod.ultimate_loss:,.0f}\n"
                    f"- Implicit ELR: {results.cape_cod.model_params.get('elr', 'N/A')}"
                )
            else:
                response = "Cape Cod was not included in this analysis."

        # 9. Diagnostics / Quality
        elif (
            "diagnos" in query_lower
            or "quality" in query_lower
            or "adequacy" in query_lower
            or "qualità" in query_lower
        ):
            if results.diagnostics:
                response = (
                    f"**Model Diagnostics**:\n"
                    f"- Adequacy Score: {results.diagnostics.adequacy_score}/100\n"
                    f"- Rating: {results.diagnostics.rating}\n"
                )
                if results.diagnostics.issues:
                    response += (
                        f"- Issues Found: {', '.join(results.diagnostics.issues)}"
                    )
            else:
                response = "Diagnostics were not run for this analysis."

        # 10. Methodology / Config
        elif (
            "method" in query_lower
            or "config" in query_lower
            or "metod" in query_lower
            or "approach" in query_lower
        ):
            response = (
                f"**Analysis Configuration**:\n"
                f"- Type: {results.config_used.analysis_type.value}\n"
                f"- Bootstrap: {'Yes' if results.config_used.run_bootstrap else 'No'}\n"
                f"- Diagnostics: {'Yes' if results.config_used.run_diagnostics else 'No'}\n"
                f"- Stress Testing: {'Yes' if results.config_used.run_stress_testing else 'No'}"
            )

        # 11. Summary / Overview
        elif (
            "summary" in query_lower
            or "overview" in query_lower
            or "riepilogo" in query_lower
            or "all" in query_lower
        ):
            response = f"**Analysis Summary**:\n\n"
            response += f"📊 **Chain Ladder**: Reserve ${results.chain_ladder.total_reserve:,.0f} | Ultimate ${results.chain_ladder.ultimate_loss:,.0f}\n"
            if results.mack:
                response += f"📈 **Mack**: Reserve ${results.mack.total_reserve:,.0f} (CV: {results.mack.cv:.1%})\n"
            if results.bootstrap:
                response += f"🎲 **Bootstrap**: Reserve ${results.bootstrap.total_reserve:,.0f} (P95: ${results.bootstrap.percentiles.get('95%', 0):,.0f})\n"
            if results.cape_cod:
                response += (
                    f"🏖️ **Cape Cod**: Reserve ${results.cape_cod.total_reserve:,.0f}\n"
                )
            if validation:
                response += f"\n✅ **Validation Score**: {validation.overall_confidence_score}/100"

        # 12. Compare methods
        elif (
            "compar" in query_lower
            or "differenc" in query_lower
            or "vs" in query_lower
            or "confronto" in query_lower
        ):
            response = "**Method Comparison**:\n\n"
            response += (
                f"| Method | Reserve | Ultimate |\n|--------|---------|----------|\n"
            )
            response += f"| Chain Ladder | ${results.chain_ladder.total_reserve:,.0f} | ${results.chain_ladder.ultimate_loss:,.0f} |\n"
            if results.mack:
                response += f"| Mack | ${results.mack.total_reserve:,.0f} | - |\n"
            if results.cape_cod:
                response += f"| Cape Cod | ${results.cape_cod.total_reserve:,.0f} | ${results.cape_cod.ultimate_loss:,.0f} |\n"

        # Default - now more helpful
        else:
            response = (
                f"{fallback_preamble}"
                "🤔 I couldn't match your question to a specific topic.\n\n"
                "**Try asking about:**\n"
                "- `reserves` or `IBNR` - Total reserve estimates\n"
                "- `ultimate` - Ultimate loss projections\n"
                "- `summary` - Full overview of results\n"
                "- `compare` - Method comparison table\n"
                "- `mack` or `uncertainty` - Stochastic results\n"
                "- `bootstrap` - Simulation percentiles\n"
                "- `diagnostics` - Model quality checks\n"
                "- `warnings` - Validation issues\n"
                "- `method` - Configuration used"
            )

        if fallback_preamble and not response.startswith("⚠️"):
            response = fallback_preamble + response

        log = AgentLog(
            agent=self.role,
            action="Q&A (Fallback)",
            details=f"Query: {query[:20]}... | Answered: Via Rules",
        )

        return response, log
