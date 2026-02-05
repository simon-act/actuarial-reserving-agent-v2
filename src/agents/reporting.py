"""
Intelligent Reporting Agent

Generates actuarial reports using LLM reasoning instead of hardcoded templates.
The LLM sees ALL the data and writes a comprehensive, contextual report.
"""

import json
from typing import Tuple, Dict

from agents.llm_utils import LLMClient
from agents.schemas import ReservingOutput, ValidationReport, AgentRole, AgentLog


# =====================================================
# Reporting Agent
# =====================================================


class CleanReportingAgent:
    def __init__(self):
        self.role = AgentRole.REPORTING
        self.llm = LLMClient()

    # =================================================
    # Public API
    # =================================================

    def generate_report(
        self, results: ReservingOutput, validation: ValidationReport
    ) -> Tuple[str, AgentLog]:
        print(f"[{self.role}] Generating report...")

        data = self._prepare_data(results, validation)

        if self.llm.is_available():
            report = self._generate_llm(data)
            mode = "LLM"
        else:
            report = self._fallback(data)
            mode = "Fallback"

        log = AgentLog(
            agent=self.role,
            action=f"Generate Report ({mode})",
            details=f"{len(report)} chars",
        )

        return report, log

    # =================================================
    # Data preparation
    # =================================================

    def _prepare_data(
        self, results: ReservingOutput, validation: ValidationReport
    ) -> Dict:
        data = {
            "meta": {
                "date": results.timestamp.strftime("%Y-%m-%d"),
                "status": validation.status.value,
                "confidence": validation.overall_confidence_score,
            },
            "reserves": {},
            "selection": {},
            "diagnostics": {},
            "issues": [],
        }

        # Triangle
        if results.triangle_info:
            t = results.triangle_info

            data["triangle"] = {
                "years": t.n_accident_years,
                "first": t.first_accident_year,
                "last": t.last_accident_year,
                "periods": t.n_development_periods,
                "currency": t.currency,
            }

        # AI Selection
        if results.method_selection:
            s = results.method_selection

            data["selection"] = {
                "method": s.selected_estimator,
                "reason": s.estimator_reason,
                "factors": s.adjusted_factors,
                "adjustments": s.prudential_adjustments,
                "validation": s.validation_metrics,
            }

        # Reserves
        data["reserves"]["chain_ladder"] = {
            "reserve": results.chain_ladder.total_reserve,
            "ultimate": results.chain_ladder.ultimate_loss,
        }

        if results.mack:
            data["reserves"]["mack"] = {
                "reserve": results.mack.total_reserve,
                "cv": results.mack.cv,
            }

        if results.bootstrap:
            data["reserves"]["bootstrap"] = {
                "reserve": results.bootstrap.total_reserve,
                "percentiles": results.bootstrap.percentiles,
            }

        if results.cape_cod:
            data["reserves"]["cape_cod"] = {
                "reserve": results.cape_cod.total_reserve,
                "elr": results.cape_cod.model_params.get("elr"),
            }

        # Diagnostics
        if results.diagnostics:
            data["diagnostics"] = {
                "score": results.diagnostics.adequacy_score,
                "rating": results.diagnostics.rating,
                "issues": results.diagnostics.issues,
            }

        # Validation issues
        if validation.issues:
            data["issues"] = [
                {"severity": i.severity, "component": i.component, "msg": i.message}
                for i in validation.issues
            ]

        return data

    # =================================================
    # LLM Generation
    # =================================================

    def _generate_llm(self, data: Dict) -> str:
        system = """
You are a professional actuarial report writer.

Write a clear, structured reserving report.

Use markdown.

Include:
- Executive summary
- Method selection
- Reserve discussion
- Risks
- Recommendations

Explain numbers. Discuss uncertainty.
"""

        user = f"""
Generate an actuarial reserving report from this data:

{json.dumps(data, indent=2)}
"""

        return self.llm.get_completion(system, user)

    # =================================================
    # Fallback
    # =================================================

    def _fallback(self, data: Dict) -> str:
        lines = []

        lines.append("# Actuarial Reserving Report\n")

        meta = data.get("meta", {})

        lines.append(f"Date: {meta.get('date')}")
        lines.append(f"Status: {meta.get('status')}")
        lines.append("")

        # Reserves
        cl = data["reserves"].get("chain_ladder", {})

        lines.append("## Reserve Estimate")
        lines.append(f"Chain Ladder: ${cl.get('reserve', 0):,.0f}")
        lines.append("")

        # Selection
        sel = data.get("selection", {})

        if sel:
            lines.append("## AI Selection")
            lines.append(f"Method: {sel.get('method')}")
            lines.append(f"Reason: {sel.get('reason')}")
            lines.append("")

        # Diagnostics
        diag = data.get("diagnostics", {})

        if diag:
            lines.append("## Diagnostics")
            lines.append(f"Score: {diag.get('score')}")
            lines.append(f"Rating: {diag.get('rating')}")
            lines.append("")

        lines.append("---")
        lines.append("*Generated in fallback mode*")

        return "\n".join(lines)


# Backward compatibility alias
ReportingAgent = CleanReportingAgent
