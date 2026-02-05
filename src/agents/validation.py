"""
Intelligent Validation Agent
============================

Reviews SelectionAgent's choices and provides INTELLIGENT feedback.
Uses LLM reasoning instead of hardcoded rules.

The workflow is:
1. SelectionAgent makes initial selection
2. ValidationAgent reviews and provides feedback
3. SelectionAgent considers feedback and FINALIZES (NO more iterations)
"""

from agents.schemas import (
    ReservingOutput, ValidationReport, ValidationStatus, ValidationIssue, AgentRole, AgentLog
)
from agents.llm_utils import LLMClient
from agents.intelligent_base import IntelligentAgent, ConfidenceLevel
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class ValidationFeedback:
    """Feedback from ValidationAgent to SelectionAgent."""
    agrees_with_selection: bool
    concerns: List[str]
    suggestions: List[str]
    alternative_recommendation: Optional[str]
    reasoning: str
    confidence: ConfidenceLevel


class ValidationAgent(IntelligentAgent):
    """
    INTELLIGENT VALIDATION AGENT

    Reviews SelectionAgent's choices and provides feedback.
    Uses LLM reasoning - NO hardcoded thresholds.

    After providing feedback ONCE, it has no more say (no loops).
    """

    def __init__(self):
        super().__init__(
            role=AgentRole.VALIDATION,
            name="Intelligent_Validation_Agent"
        )

    def _get_system_prompt(self) -> str:
        return """You are a SENIOR ACTUARIAL PEER REVIEWER with expertise in:
- Loss reserving methodology validation
- Statistical model validation
- Diagnostic interpretation
- Risk assessment

YOUR ROLE:
You review another actuary's method selection and provide constructive feedback.
You are NOT adversarial - you help improve the analysis.

WHEN REVIEWING:
1. Consider whether the selection is REASONABLE given the data
2. Identify any concerns or risks that may have been overlooked
3. Suggest improvements if appropriate
4. If you disagree with the selection, explain WHY with specific reasons

IMPORTANT:
- Don't just repeat what the SelectionAgent said
- Add VALUE by identifying risks or alternatives
- Be specific with numbers and evidence
- You can AGREE with the selection but still note concerns
- Your feedback will be considered in the final decision
"""

    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format data for validation review."""
        text_parts = []

        if 'selection' in data:
            sel = data['selection']
            text_parts.append("=" * 60)
            text_parts.append("SELECTION TO REVIEW")
            text_parts.append("=" * 60)
            text_parts.append(f"Selected Method: {sel.get('selected_estimator', 'N/A')}")
            text_parts.append(f"Confidence: {sel.get('confidence', 'N/A')}")
            text_parts.append("")

            if sel.get('reasoning'):
                text_parts.append("Selection Reasoning:")
                text_parts.append(sel['reasoning'][:500] + "..." if len(sel.get('reasoning', '')) > 500 else sel.get('reasoning', ''))
                text_parts.append("")

        if 'estimator_results' in data:
            text_parts.append("=" * 60)
            text_parts.append("ALL ESTIMATOR RESULTS")
            text_parts.append("=" * 60)
            for name, reserve in data['estimator_results'].items():
                text_parts.append(f"  {name}: ${reserve:,.0f}")
            text_parts.append("")

        if 'diagnostics' in data:
            diag = data['diagnostics']
            text_parts.append("=" * 60)
            text_parts.append("DIAGNOSTIC RESULTS")
            text_parts.append("=" * 60)
            text_parts.append(f"Adequacy Score: {diag.get('adequacy_score', 'N/A')}/100")
            text_parts.append(f"Rating: {diag.get('rating', 'N/A')}")
            if diag.get('issues'):
                text_parts.append("Issues:")
                for issue in diag['issues']:
                    text_parts.append(f"  • {issue}")
            text_parts.append("")

        if 'pattern_analysis' in data:
            pa = data['pattern_analysis']
            text_parts.append("=" * 60)
            text_parts.append("PATTERN ANALYSIS")
            text_parts.append("=" * 60)
            text_parts.append(f"Smoothing Applied: {pa.get('smoothing_applied', False)}")
            if pa.get('smoothing_method'):
                text_parts.append(f"Smoothing Method: {pa['smoothing_method']}")
            text_parts.append("")

        return "\n".join(text_parts)

    def review_selection(
        self,
        selection_result: Dict,
        estimator_results: Dict[str, float],
        diagnostics: Dict,
        pattern_analysis: Dict,
        verbose: bool = True
    ) -> ValidationFeedback:
        """
        Review SelectionAgent's selection and provide feedback.

        This is called ONCE. After feedback is given, ValidationAgent
        has no more say (no loops).

        Args:
            selection_result: The selection made by SelectionAgent
            estimator_results: All estimator reserve results
            diagnostics: Diagnostic test results
            pattern_analysis: Pattern analysis results
            verbose: Print progress

        Returns:
            ValidationFeedback with concerns, suggestions, and recommendation
        """
        if verbose:
            self._log("🔍 Reviewing SelectionAgent's choice...")

        # Prepare data for review
        review_data = {
            'selection': {
                'selected_estimator': selection_result.get('selected_estimator'),
                'confidence': selection_result.get('confidence'),
                'reasoning': selection_result.get('reasoning', '')
            },
            'estimator_results': estimator_results,
            'diagnostics': diagnostics,
            'pattern_analysis': pattern_analysis
        }

        formatted_data = self._format_data_for_analysis(review_data)

        prompt = f"""
{self._get_system_prompt()}

=== DATA FOR REVIEW ===
{formatted_data}

=== YOUR TASK ===
Review this method selection and provide your feedback.

1. Do you AGREE with the selection? Why or why not?
2. What CONCERNS do you have (even if you agree)?
3. What SUGGESTIONS would you make?
4. If you disagree, what would YOU recommend instead?

Be specific. Cite numbers. Add value beyond what SelectionAgent said.

Respond in JSON format:
{{
    "agrees_with_selection": true/false,
    "concerns": ["concern 1 with specific reason", "concern 2", ...],
    "suggestions": ["suggestion 1", "suggestion 2", ...],
    "alternative_recommendation": "Your recommended method if different, or null",
    "reasoning": "Your detailed reasoning for this feedback",
    "confidence": "high|medium|low|uncertain"
}}
"""

        if not self.llm.is_available():
            if verbose:
                self._log("⚠️ LLM not available, using basic validation")
            return self._fallback_review(selection_result, diagnostics)

        try:
            response = self.llm.get_completion(
                system_prompt="You are an expert actuarial reviewer. Respond in valid JSON.",
                user_prompt=prompt
            )

            result = self._parse_json_response(response)

            feedback = ValidationFeedback(
                agrees_with_selection=result.get('agrees_with_selection', True),
                concerns=result.get('concerns', []),
                suggestions=result.get('suggestions', []),
                alternative_recommendation=result.get('alternative_recommendation'),
                reasoning=result.get('reasoning', ''),
                confidence=ConfidenceLevel(result.get('confidence', 'medium'))
            )

            if verbose:
                status = "✓ AGREES" if feedback.agrees_with_selection else "✗ DISAGREES"
                self._log(f"   {status} with selection")
                if feedback.concerns:
                    self._log(f"   Concerns: {len(feedback.concerns)}")
                if feedback.alternative_recommendation:
                    self._log(f"   Alternative: {feedback.alternative_recommendation}")

            return feedback

        except Exception as e:
            if verbose:
                self._log(f"✗ Review failed: {e}")
            return self._fallback_review(selection_result, diagnostics)

    def _fallback_review(self, selection_result: Dict, diagnostics: Dict) -> ValidationFeedback:
        """Basic review when LLM is not available."""
        concerns = []

        # Check diagnostic score
        adequacy = diagnostics.get('adequacy_score', 100)
        if adequacy < 60:
            concerns.append(f"Low diagnostic adequacy score ({adequacy}/100) suggests model assumptions may be violated")

        # Check for issues
        if diagnostics.get('issues'):
            for issue in diagnostics['issues'][:2]:
                concerns.append(f"Diagnostic issue: {issue}")

        return ValidationFeedback(
            agrees_with_selection=True,  # Default to agree in fallback
            concerns=concerns,
            suggestions=["Consider reviewing diagnostic issues manually"],
            alternative_recommendation=None,
            reasoning="Basic validation (LLM not available)",
            confidence=ConfidenceLevel.UNCERTAIN
        )

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        return json.loads(response)

    # Keep old validate method for backward compatibility
    def validate(self, results: ReservingOutput) -> Tuple[ValidationReport, AgentLog]:
        """
        Legacy validation method for backward compatibility.
        Uses rule-based validation (not LLM).
        """
        print(f"[{self.role}] 🔍 Validating results (legacy mode)...")

        issues = []
        score = 100

        # 1. Check Model Adequacy
        if results.diagnostics:
            diag = results.diagnostics
            if diag.adequacy_score < 60:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    message=f"Model Adequacy Score is low ({diag.adequacy_score}/100)",
                    component="Diagnostics"
                ))
                score -= 20

            if diag.issues:
                for issue in diag.issues:
                    issues.append(ValidationIssue(
                        severity="INFO",
                        message=f"Diagnostic issue: {issue}",
                        component="Diagnostics"
                    ))

        # 2. Check CV (Volatility)
        if results.mack:
            if results.mack.cv > 0.25:
                 issues.append(ValidationIssue(
                    severity="WARNING",
                    message=f"High Volatility detected (CV = {results.mack.cv:.1%})",
                    component="Mack Model"
                ))
                 score -= 10

        # 3. Compare Methods
        cl_res = results.chain_ladder.total_reserve
        msg_parts = [f"CL: ${cl_res:,.0f}"]

        if results.mack:
            mack_res = results.mack.total_reserve
            diff = abs(cl_res - mack_res) / cl_res if cl_res > 0 else 0
            msg_parts.append(f"Mack: ${mack_res:,.0f}")
            if diff > 0.1:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    message=f"Large divergence between CL and Mack ({diff:.1%})",
                    component="Cross-Check"
                ))
                score -= 10

        if results.bootstrap:
             boot_res = results.bootstrap.total_reserve
             msg_parts.append(f"Boot: ${boot_res:,.0f}")

        # Determine Status
        status = ValidationStatus.PASSED
        if score < 70:
            status = ValidationStatus.WARNING
        if score < 40:
            status = ValidationStatus.REJECTED

        report = ValidationReport(
            status=status,
            overall_confidence_score=max(0, score),
            issues=issues,
            comparison_summary=" | ".join(msg_parts)
        )

        log = AgentLog(
            agent=self.role,
            action="Validate Results",
            details=f"Status: {status}, Score: {score}/100. Issues found: {len(issues)}"
        )

        print(f"[{self.role}] ✅ Validation complete. Status: {status} (Score: {score})")
        return report, log
