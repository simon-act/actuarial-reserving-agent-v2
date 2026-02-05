"""
Intelligent Reporting Agent

Generates actuarial reports using LLM reasoning instead of hardcoded templates.
The LLM sees ALL the data and writes a comprehensive, contextual report.
"""

from agents.schemas import ReservingOutput, ValidationReport, AgentRole, AgentLog
from agents.llm_utils import LLMClient
from typing import Tuple, Dict, Any
import json


class ReportingAgent:
    """
    INTELLIGENT REPORTING AGENT

    Instead of hardcoded templates, this agent:
    1. Formats ALL results into structured data
    2. Passes everything to LLM
    3. LLM writes a contextual, intelligent report
    """

    def __init__(self):
        self.role = AgentRole.REPORTING
        self.llm = LLMClient()

    def _prepare_data_for_llm(self, results: ReservingOutput, validation: ValidationReport) -> str:
        """Convert all results to a format the LLM can understand."""

        data = {
            "metadata": {
                "timestamp": results.timestamp.strftime('%Y-%m-%d %H:%M'),
                "status": validation.status.value,
                "confidence_score": validation.overall_confidence_score,
                "methodology": results.config_used.analysis_type.value if results.config_used else "standard"
            }
        }

        # Triangle info
        if results.triangle_info:
            ti = results.triangle_info
            data["triangle"] = {
                "accident_years": ti.n_accident_years,
                "first_year": ti.first_accident_year,
                "last_year": ti.last_accident_year,
                "development_periods": ti.n_development_periods,
                "currency": ti.currency,
                "units": ti.units
            }

        # Method selection (from intelligent SelectionAgent)
        if results.method_selection:
            ms = results.method_selection
            data["ai_selection"] = {
                "selected_estimator": ms.selected_estimator,
                "estimator_reason": ms.estimator_reason,
                "immature_years": getattr(ms, 'immature_years', None) or getattr(ms, 'bf_years', None),
                "immature_reason": getattr(ms, 'immature_reason', None) or getattr(ms, 'bf_reason', None),
                "summary": ms.summary,
                "all_estimators": ms.all_estimators,
                "validation_metrics": ms.validation_metrics,
                "maturity_by_year": ms.maturity_by_year,
                "prudential_adjustments": ms.prudential_adjustments,
                "adjusted_factors": ms.adjusted_factors
            }

        # Reserve results by method
        data["reserves"] = {}

        data["reserves"]["chain_ladder"] = {
            "total_reserve": results.chain_ladder.total_reserve,
            "ultimate_loss": results.chain_ladder.ultimate_loss
        }

        if results.mack:
            data["reserves"]["mack"] = {
                "total_reserve": results.mack.total_reserve,
                "ultimate_loss": results.mack.ultimate_loss,
                "standard_error": results.mack.standard_error,
                "cv": f"{results.mack.cv:.1%}"
            }

        if results.bootstrap:
            data["reserves"]["bootstrap"] = {
                "total_reserve": results.bootstrap.total_reserve,
                "ultimate_loss": results.bootstrap.ultimate_loss,
                "standard_error": results.bootstrap.standard_error,
                "percentiles": results.bootstrap.percentiles
            }

        if results.cape_cod:
            data["reserves"]["cape_cod"] = {
                "total_reserve": results.cape_cod.total_reserve,
                "ultimate_loss": results.cape_cod.ultimate_loss,
                "elr": results.cape_cod.model_params.get('elr', 'N/A')
            }

        # Development factors
        if results.detailed_data and results.detailed_data.development_factors:
            data["development_factors"] = results.detailed_data.development_factors

        # Reserves by year
        if results.detailed_data and results.detailed_data.reserves_by_year:
            data["reserves_by_year"] = results.detailed_data.reserves_by_year
            data["ultimates_by_year"] = results.detailed_data.ultimates_by_year
            data["latest_diagonal"] = results.detailed_data.latest_diagonal

        # Validation issues
        if validation.issues:
            data["validation_issues"] = [
                {"severity": i.severity, "component": i.component, "message": i.message}
                for i in validation.issues
            ]

        # Diagnostics
        if results.diagnostics:
            data["diagnostics"] = {
                "adequacy_score": results.diagnostics.adequacy_score,
                "rating": results.diagnostics.rating,
                "issues": results.diagnostics.issues
            }

        return json.dumps(data, indent=2, default=str)

    def _generate_with_llm(self, data_json: str) -> str:
        """Ask LLM to generate the report."""

        system_prompt = """You are an expert actuarial report writer.
Generate a comprehensive, professional actuarial reserving report based on the data provided.

WRITING STYLE:
- Write in clear, professional prose (NOT bullet points or templates)
- Explain the findings and their implications
- Highlight key insights and concerns
- Be specific with numbers but also explain what they mean
- Use markdown formatting for structure

REPORT STRUCTURE:
1. Executive Summary (2-3 paragraphs summarizing key findings)
2. AI Method Selection Analysis (explain WHY the AI chose this method)
3. Reserve Estimates Discussion (compare methods, explain differences)
4. Risk Assessment (uncertainties, concerns, immature years)
5. Recommendations (what the actuary should focus on)

IMPORTANT:
- Don't just list numbers - INTERPRET them
- Explain any concerns or anomalies
- If validation issues exist, discuss their implications
- If prudential adjustments were applied (e.g., factors capped at 1.0 or tail fitting), explicitly mention them and why
- Be honest about uncertainties
- Write as if advising a senior actuary

Format the report in markdown with proper headers (##, ###)."""

        user_prompt = f"""Generate a professional actuarial reserving report based on this analysis data:

{data_json}

Write the report now. Be thorough but focused on insights, not just data regurgitation."""

        try:
            report = self.llm.get_completion(system_prompt, user_prompt)
            return report
        except Exception as e:
            print(f"[{self.role}] ⚠️ LLM failed, using fallback: {e}")
            return self._fallback_report(data_json)

    def _fallback_report(self, data_json: str) -> str:
        """Simple fallback if LLM is not available."""
        data = json.loads(data_json)

        report = ["# Actuarial Reserving Report\n"]
        report.append(f"**Date:** {data.get('metadata', {}).get('timestamp', 'N/A')}")
        report.append(f"**Status:** {data.get('metadata', {}).get('status', 'N/A')}\n")

        reserves = data.get('reserves', {})
        if 'chain_ladder' in reserves:
            cl = reserves['chain_ladder']
            report.append(f"## Reserve Estimate\n")
            report.append(f"Total Reserve (Chain Ladder): **${cl['total_reserve']:,.0f}**\n")

        if data.get('ai_selection'):
            sel = data['ai_selection']
            report.append(f"## AI Selection\n")
            report.append(f"Selected Method: **{sel.get('selected_estimator', 'N/A')}**")
            report.append(f"\nReason: {sel.get('estimator_reason', 'N/A')}\n")

        report.append("\n---\n*Report generated by AI Actuarial Team*")

        return "\n".join(report)

    def generate_report(self, results: ReservingOutput, validation: ValidationReport) -> Tuple[str, AgentLog]:
        """Generate an intelligent report using LLM."""
        print(f"[{self.role}] 🧠 Generating intelligent report with LLM...")

        # Prepare all data
        data_json = self._prepare_data_for_llm(results, validation)

        # Generate with LLM
        if self.llm.is_available():
            report = self._generate_with_llm(data_json)
            method = "LLM"
        else:
            report = self._fallback_report(data_json)
            method = "Fallback"

        log = AgentLog(
            agent=self.role,
            action=f"Generate Intelligent Report ({method})",
            details=f"Generated report with {method} ({len(report)} chars)"
        )

        return report, log
