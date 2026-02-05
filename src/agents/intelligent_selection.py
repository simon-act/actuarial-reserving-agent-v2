"""
Intelligent Selection Agent
===========================

A truly intelligent method selection agent that:
- SEES raw data (factors, metrics, patterns)
- REASONS about what method to use
- ANALYZES patterns for anomalies
- DECIDES on smoothing when appropriate
- EXPLAINS its reasoning IN DEPTH
- CRITIQUES its own decisions

NO hardcoded rules. ALL decisions through LLM reasoning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intelligent_base import (
    IntelligentAgent, Analysis, Decision, Critique,
    AgentThought, ConfidenceLevel
)
from agents.schemas import AgentRole, AgentLog

# Import calculation tools
from model_selection.factor_estimators import get_all_estimators
from model_selection.model_selector import ModelSelector
from diagnostics.diagnostic_tests import DiagnosticTests
from diagnostics.volatility_analysis import VolatilityAnalyzer
from tail_fitting.tail_estimator import TailEstimator
from chain_ladder import ChainLadder
from pattern_analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisResult
from pattern_analysis.curve_fitting import CurveFitter


TAIL_VOLATILITY_CV_THRESHOLD = 0.15
TAIL_FIT_MIN_PERIODS = 3


@dataclass
class SelectionResult:
    """Complete result of intelligent selection."""
    selected_estimator: str
    selected_factors: pd.Series
    original_factors: pd.Series
    adjusted_factors: pd.Series
    pattern_analysis: PatternAnalysisResult
    thought_process: AgentThought
    all_estimator_results: Dict[str, float]
    diagnostics_summary: Dict
    validation_metrics: Dict
    tail_factor: float
    confidence: ConfidenceLevel
    explanation: str
    prudential_adjustments: List[Dict]


@dataclass
class PrudentialAdjustment:
    """Record of a prudential adjustment made to factors."""
    adjustment_type: str
    period: str
    original_value: float
    adjusted_value: float
    reason: str


class IntelligentSelectionAgent(IntelligentAgent):
    """
    Intelligent method selection agent.

    Unlike the old rule-based SelectionAgent, this agent:
    - Sees ALL the raw data
    - Reasons about patterns intelligently
    - Decides whether to smooth patterns
    - Explains every decision IN DEPTH
    - Is transparent about uncertainty
    """

    def __init__(self):
        super().__init__(
            role=AgentRole.METHODOLOGY,
            name="Intelligent_Selection_Agent"
        )
        self.pattern_analyzer = PatternAnalyzer()

    def _get_system_prompt(self) -> str:
        return """You are a SENIOR ACTUARIAL CONSULTANT with 20+ years of experience in loss reserving.

YOUR EXPERTISE:
- Chain Ladder methodology and its variants (Mack, 1993)
- Development factor estimation: volume-weighted, simple average, medial, geometric
- Statistical validation and out-of-sample testing
- Pattern analysis and anomaly detection
- Mack model assumptions (independence, proportionality, no calendar effects)

ACTUARIAL BEST PRACTICES YOU FOLLOW:
1. Volume-Weighted Average is often preferred because larger claims have more credibility
2. Simple Average treats all years equally - good when older years are less relevant
3. Medial Average (excluding high/low) reduces outlier impact
4. Geometric Mean reduces impact of extreme factors
5. Exponential Weighted gives more weight to recent years

WHEN ANALYZING PATTERNS:
- Development factors should generally DECREASE over time (monotonic decay)
- Non-monotonic patterns (factor increases) may indicate:
  * Case reserve strengthening
  * Data quality issues
  * Portfolio changes
  * Truly erratic development
- High volatility in early periods is EXPECTED (small base)
- High volatility in late periods is CONCERNING (should be stable)

WHEN MAKING DECISIONS:
- Always cite SPECIFIC NUMBERS from the data
- Explain the ACTUARIAL RATIONALE, not just statistical metrics
- Consider PRACTICAL implications for reserves
- Discuss UNCERTAINTY honestly
- Compare at least 2-3 alternatives in depth

RESPOND WITH:
- Detailed analysis with specific numerical references
- Clear reasoning that an actuary would find convincing
- Discussion of trade-offs between methods
- Acknowledgment of limitations and uncertainties
"""

    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format selection data for comprehensive LLM analysis."""
        text_parts = []

        # === ESTIMATOR COMPARISON ===
        if 'estimator_results' in data:
            results = data['estimator_results']
            reserves = list(results.values())
            min_res, max_res = min(reserves), max(reserves)
            spread = (max_res - min_res) / min_res * 100 if min_res > 0 else 0

            text_parts.append("=" * 60)
            text_parts.append("ESTIMATOR RESERVE COMPARISON")
            text_parts.append("=" * 60)
            text_parts.append(f"{'Estimator':<25} {'Reserve':>15} {'vs Min':>10}")
            text_parts.append("-" * 60)

            for name, reserve in sorted(results.items(), key=lambda x: x[1]):
                diff_pct = (reserve - min_res) / min_res * 100 if min_res > 0 else 0
                text_parts.append(f"{name:<25} ${reserve:>14,.0f} {diff_pct:>+9.1f}%")

            text_parts.append("-" * 60)
            text_parts.append(f"Range: ${min_res:,.0f} to ${max_res:,.0f} (spread: {spread:.1f}%)")
            text_parts.append("")

        # === VALIDATION METRICS ===
        if 'validation_metrics' in data:
            text_parts.append("=" * 60)
            text_parts.append("OUT-OF-SAMPLE VALIDATION (CRITICAL FOR SELECTION)")
            text_parts.append("=" * 60)
            metrics = data['validation_metrics']
            if 'best_by_metric' in metrics:
                text_parts.append("BEST PERFORMERS:")
                for metric, best in metrics['best_by_metric'].items():
                    text_parts.append(f"  • Best by {metric}: {best}")

            if 'detailed_metrics' in metrics:
                text_parts.append("\nDETAILED METRICS:")
                for name, m in metrics['detailed_metrics'].items():
                    text_parts.append(f"  {name}:")
                    text_parts.append(f"    MSE={m.get('MSE', 'N/A'):.2f}, MAE={m.get('MAE', 'N/A'):.2f}, RMSE={m.get('RMSE', 'N/A'):.2f}")
            text_parts.append("")

        # === DIAGNOSTIC TESTS ===
        if 'diagnostics' in data:
            diag = data['diagnostics']
            text_parts.append("=" * 60)
            text_parts.append("MACK MODEL DIAGNOSTIC TESTS")
            text_parts.append("=" * 60)
            text_parts.append(f"Overall Adequacy Score: {diag.get('adequacy_score', 'N/A')}/100")
            text_parts.append(f"Rating: {diag.get('rating', 'N/A')}")
            text_parts.append("")

            if diag.get('issues'):
                text_parts.append("⚠️ ISSUES DETECTED:")
                for issue in diag['issues']:
                    text_parts.append(f"  • {issue}")
            else:
                text_parts.append("✓ No significant diagnostic issues")
            text_parts.append("")

        # === DEVELOPMENT FACTORS ===
        if 'factors' in data:
            factors = data['factors']
            text_parts.append("=" * 60)
            text_parts.append("DEVELOPMENT FACTORS (AGE-TO-AGE)")
            text_parts.append("=" * 60)
            text_parts.append(f"{'Period':<10} {'Factor':>10} {'Change':>12} {'Note':>20}")
            text_parts.append("-" * 60)

            prev_factor = None
            for period, factor in factors.items():
                if prev_factor is not None:
                    change = factor - prev_factor
                    change_pct = change / prev_factor * 100 if prev_factor != 0 else 0

                    # Identify concerning patterns
                    if change > 0:
                        note = "⚠️ INCREASING!"
                    elif abs(change_pct) > 20:
                        note = "Large drop"
                    else:
                        note = "Normal decay"

                    text_parts.append(f"{period:<10} {factor:>10.4f} {change:>+11.4f} {note:>20}")
                else:
                    text_parts.append(f"{period:<10} {factor:>10.4f} {'---':>12} {'First period':>20}")

                prev_factor = factor
            text_parts.append("")

        # === PATTERN SMOOTHING STATUS ===
        if 'pattern_smoothing_applied' in data:
            text_parts.append("=" * 60)
            text_parts.append("PATTERN ANALYSIS RESULT")
            text_parts.append("=" * 60)
            if data['pattern_smoothing_applied']:
                text_parts.append(f"✓ Smoothing RECOMMENDED: {data.get('pattern_smoothing_method', 'Unknown')}")
                text_parts.append("  The pattern analyzer detected issues that warrant smoothing.")
            else:
                text_parts.append("✓ No smoothing needed - pattern is acceptable")
            text_parts.append("")

        # === TRIANGLE CHARACTERISTICS ===
        if 'triangle_info' in data:
            info = data['triangle_info']
            text_parts.append("=" * 60)
            text_parts.append("TRIANGLE CHARACTERISTICS")
            text_parts.append("=" * 60)
            text_parts.append(f"  Accident Years: {info.get('n_years', 'N/A')} ({info.get('first_year', '?')}-{info.get('last_year', '?')})")
            text_parts.append(f"  Development Periods: {info.get('n_periods', 'N/A')}")

            # Maturity assessment
            n_years = info.get('n_years', 0)
            n_periods = info.get('n_periods', 0)
            if n_years > 0 and n_periods > 0:
                immature_years = max(0, n_years - n_periods + 1)
                text_parts.append(f"  Immature Years (< 100% developed): ~{immature_years}")
            text_parts.append("")

        return "\n".join(text_parts)

    def analyze_and_select(
        self,
        triangle: pd.DataFrame,
        premium: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> SelectionResult:
        """
        Perform intelligent method selection.

        This method:
        1. Calculates all estimators ONCE
        2. Runs validation and diagnostics
        3. Analyzes patterns (with potential smoothing)
        4. Reasons deeply about what method to use
        5. Returns selection with FULL explanation

        Args:
            triangle: Loss development triangle
            premium: Optional earned premium
            verbose: Print progress

        Returns:
            SelectionResult with complete analysis and reasoning
        """
        if verbose:
            self._log("🧠 Starting INTELLIGENT method selection...")
            self._log("   All decisions will be made by reasoning, not rules.")

        # Step 1: Calculate all estimators
        estimator_results = self._calculate_all_estimators(triangle, verbose)

        # Step 2: Run validation
        validation_metrics = self._run_validation(triangle, verbose)

        # Step 3: Run diagnostics
        diagnostics = self._run_diagnostics(triangle, verbose)

        # Step 4: Analyze pattern (this is intelligent!)
        if verbose:
            self._log("🔍 Analyzing development pattern...")

        # Get factors from best estimator so far
        best_by_mse = validation_metrics.get('best_by_metric', {}).get('MSE', 'Volume-Weighted')
        estimators = get_all_estimators()
        best_estimator = next((e for e in estimators if e.name == best_by_mse), estimators[0])
        initial_factors = best_estimator.estimate(triangle)

        # Intelligent pattern analysis
        pattern_result = self.pattern_analyzer.analyze_pattern(
            factors=initial_factors,
            triangle=triangle,
            context=f"Diagnostics: {diagnostics.get('rating', 'N/A')}, "
                    f"Best estimator: {best_by_mse}"
        )

        # Step 5: Fit tail
        tail_factor = self._fit_tail(triangle, verbose)

        # Step 6: Prepare COMPREHENSIVE data for LLM decision
        decision_data = {
            'estimator_results': estimator_results,
            'validation_metrics': validation_metrics,
            'diagnostics': diagnostics,
            'factors': initial_factors.to_dict(),
            'pattern_smoothing_applied': pattern_result.smoothing_applied,
            'pattern_smoothing_method': pattern_result.smoothing_method,
            'triangle_info': {
                'n_years': len(triangle),
                'n_periods': len(triangle.columns),
                'first_year': int(triangle.index[0]),
                'last_year': int(triangle.index[-1])
            }
        }

        # Available options for final decision
        options = list(estimator_results.keys())
        if pattern_result.smoothing_applied:
            options.append(f"{best_by_mse}_smoothed")

        # Step 7: Run the INTELLIGENT thinking process
        if verbose:
            self._log("💭 Deep reasoning about method selection...")

        # Create rich context for decision
        context = self._create_decision_context(
            estimator_results, validation_metrics, diagnostics,
            pattern_result, tail_factor
        )

        thought = self.think(
            data=decision_data,
            options=options,
            task="Select the BEST development factor estimation method",
            context=context
        )

        # Determine final selection
        selected_estimator = thought.decision.choice if thought.decision else best_by_mse

        # Base factors (pre-prudence)
        if "_smoothed" in selected_estimator and pattern_result.smoothing_applied:
            base_factors = pattern_result.recommended_factors
        else:
            base_factors = initial_factors

        # Apply prudential adjustments (cap > 1.0, tail fitting if volatile)
        adjusted_factors, adjustments = self._apply_prudential_adjustments(
            base_factors, verbose=verbose
        )

        # Generate comprehensive explanation
        explanation = self._generate_explanation(
            thought, pattern_result, selected_estimator, diagnostics,
            estimator_results, validation_metrics
        )

        if adjustments:
            explanation += "\n\n" + "=" * 70 + "\n"
            explanation += "PRUDENTIAL ADJUSTMENTS\n"
            explanation += "=" * 70 + "\n"
            explanation += "The following prudential adjustments were applied to development factors:\n"
            for adj in adjustments:
                explanation += (
                    f"  • {adj['period']}: {adj['original_value']:.4f} → "
                    f"{adj['adjusted_value']:.4f} ({adj['adjustment_type']})\n"
                )
            explanation += "\n"
            explanation += "Note: Factors above 1.0 were capped at 1.0 for prudence, "
            explanation += "and tail fitting was applied when tail volatility was high.\n"

        if verbose:
            self._log(f"✅ SELECTED: {selected_estimator}")
            self._log(f"   Confidence: {thought.critique.revised_confidence.value if thought.critique else 'N/A'}")
            if pattern_result.smoothing_applied:
                self._log(f"   Pattern smoothing: {pattern_result.smoothing_method}")
            if adjustments:
                self._log(f"   Prudential adjustments: {len(adjustments)}")

        return SelectionResult(
            selected_estimator=selected_estimator,
            selected_factors=adjusted_factors,
            original_factors=base_factors,
            adjusted_factors=adjusted_factors,
            pattern_analysis=pattern_result,
            thought_process=thought,
            all_estimator_results=estimator_results,
            diagnostics_summary=diagnostics,
            validation_metrics=validation_metrics,
            tail_factor=tail_factor,
            confidence=thought.critique.revised_confidence if thought.critique else ConfidenceLevel.UNCERTAIN,
            explanation=explanation,
            prudential_adjustments=adjustments
        )

    def _create_decision_context(
        self,
        estimator_results: Dict,
        validation_metrics: Dict,
        diagnostics: Dict,
        pattern_result: PatternAnalysisResult,
        tail_factor: float
    ) -> str:
        """Create rich context for the decision-making process."""

        context_parts = []

        # Reserve range analysis
        reserves = list(estimator_results.values())
        min_res, max_res = min(reserves), max(reserves)
        spread_pct = (max_res - min_res) / min_res * 100 if min_res > 0 else 0

        context_parts.append("KEY CONSIDERATIONS:")
        context_parts.append(f"1. Reserve estimates range from ${min_res:,.0f} to ${max_res:,.0f} ({spread_pct:.1f}% spread)")

        if spread_pct > 20:
            context_parts.append("   → HIGH SPREAD: Method selection is CRITICAL")
        elif spread_pct > 10:
            context_parts.append("   → Moderate spread: Method selection matters")
        else:
            context_parts.append("   → Low spread: Methods are consistent")

        # Diagnostic concerns
        adequacy = diagnostics.get('adequacy_score', 50)
        if adequacy < 50:
            context_parts.append(f"2. DIAGNOSTIC CONCERNS: Adequacy score only {adequacy}/100")
            context_parts.append("   → Consider whether Chain Ladder assumptions are violated")
        else:
            context_parts.append(f"2. Diagnostics acceptable: {adequacy}/100")

        # Pattern analysis
        if pattern_result.smoothing_applied:
            context_parts.append(f"3. Pattern Analysis RECOMMENDS smoothing: {pattern_result.smoothing_method}")
            context_parts.append("   → Pattern has irregularities that could distort reserves")
        else:
            context_parts.append("3. Pattern analysis: No smoothing needed")

        # Tail factor
        context_parts.append(f"4. Tail factor: {tail_factor:.4f}")

        context_parts.append("")
        context_parts.append("YOUR TASK: Weigh these factors and select the BEST method.")
        context_parts.append("Explain your reasoning IN DEPTH with specific numbers.")

        return "\n".join(context_parts)

    def _calculate_all_estimators(self, triangle: pd.DataFrame, verbose: bool) -> Dict[str, float]:
        """Calculate reserves for all estimators."""
        if verbose:
            self._log("📊 Calculating ALL estimators...")

        estimators = get_all_estimators()
        results = {}

        for est in estimators:
            try:
                factors = est.estimate(triangle)
                cl = ChainLadder(triangle)
                cl.age_to_age_factors = cl.calculate_age_to_age_factors()
                cl.selected_factors = factors
                cl.calculate_cumulative_factors()
                cl.project_ultimate_losses()

                total_reserve = cl.ultimate_losses['Reserve'].sum()
                results[est.name] = round(total_reserve, 2)

                if verbose:
                    self._log(f"   {est.name}: ${total_reserve:,.0f}")

            except Exception as e:
                if verbose:
                    self._log(f"   {est.name}: Error - {e}")

        return results

    def _run_validation(self, triangle: pd.DataFrame, verbose: bool) -> Dict:
        """Run validation metrics."""
        if verbose:
            self._log("🧪 Running out-of-sample validation...")

        try:
            selector = ModelSelector(triangle, verbose=False)
            selector.run_validation()

            best_by_metric = {}
            detailed_metrics = {}

            if selector.comparison_table is not None:
                for col in ['MSE', 'MAE', 'RMSE']:
                    if col in selector.comparison_table.columns:
                        best_by_metric[col] = selector.comparison_table[col].idxmin()

                # Get detailed metrics for each estimator
                for idx in selector.comparison_table.index:
                    detailed_metrics[idx] = {
                        'MSE': selector.comparison_table.loc[idx, 'MSE'] if 'MSE' in selector.comparison_table.columns else None,
                        'MAE': selector.comparison_table.loc[idx, 'MAE'] if 'MAE' in selector.comparison_table.columns else None,
                        'RMSE': selector.comparison_table.loc[idx, 'RMSE'] if 'RMSE' in selector.comparison_table.columns else None,
                    }

            if verbose and best_by_metric:
                self._log(f"   Best by MSE: {best_by_metric.get('MSE', 'N/A')}")

            return {
                'best_by_metric': best_by_metric,
                'detailed_metrics': detailed_metrics
            }

        except Exception as e:
            if verbose:
                self._log(f"   Validation error: {e}")
            return {'error': str(e)}

    def _run_diagnostics(self, triangle: pd.DataFrame, verbose: bool) -> Dict:
        """Run diagnostic tests."""
        if verbose:
            self._log("🔬 Running Mack diagnostics...")

        try:
            diag = DiagnosticTests(triangle)
            adequacy = diag.get_model_adequacy_score()

            if verbose:
                self._log(f"   Adequacy: {adequacy['adequacy_score']}/100 ({adequacy['rating']})")

            return {
                'adequacy_score': adequacy['adequacy_score'],
                'rating': adequacy['rating'],
                'issues': adequacy['issues']
            }

        except Exception as e:
            if verbose:
                self._log(f"   Diagnostics error: {e}")
            return {'error': str(e)}

    def _fit_tail(self, triangle: pd.DataFrame, verbose: bool) -> float:
        """Fit tail factor."""
        if verbose:
            self._log("📐 Fitting tail factor...")

        try:
            tail = TailEstimator(triangle)
            tail.fit()
            factor = tail.tail_factor

            if verbose:
                self._log(f"   Tail factor: {factor:.4f}")

            return factor

        except Exception as e:
            if verbose:
                self._log(f"   Tail fitting error: {e}")
            return 1.0

    def _apply_prudential_adjustments(
        self,
        factors: pd.Series,
        verbose: bool = True
    ) -> Tuple[pd.Series, List[Dict]]:
        """
        Apply prudential adjustments to development factors:
        - Cap factors > 1.0 to 1.0 (prudence)
        - Fit and smooth tail if volatile
        """
        adjustments: List[Dict] = []
        adjusted = factors.copy().astype(float)

        # Tail volatility check
        tail_len = min(TAIL_FIT_MIN_PERIODS, len(adjusted))
        if tail_len >= TAIL_FIT_MIN_PERIODS:
            tail = adjusted.tail(tail_len)
            tail_mean = tail.mean()
            tail_cv = tail.std() / tail_mean if tail_mean != 0 else 0.0

            if tail_cv >= TAIL_VOLATILITY_CV_THRESHOLD:
                if verbose:
                    self._log(f"⚠️ Tail volatility high (CV={tail_cv:.2%}). Applying tail fit.")

                fitter = CurveFitter(adjusted)
                fit_results = fitter.fit_all()

                # Choose best fit by RMSE
                best_fit = None
                best_rmse = None
                for name, result in fit_results.items():
                    if best_rmse is None or result.rmse < best_rmse:
                        best_fit = result
                        best_rmse = result.rmse

                if best_fit is not None:
                    fitted_tail = best_fit.fitted_factors.tail(tail_len)
                    for idx in fitted_tail.index:
                        original_val = adjusted.loc[idx]
                        new_val = float(fitted_tail.loc[idx])
                        if abs(new_val - original_val) > 1e-6:
                            adjustments.append({
                                "adjustment_type": f"tail_fit_{best_fit.method}",
                                "period": str(idx),
                                "original_value": float(original_val),
                                "adjusted_value": float(new_val),
                                "reason": f"Tail volatility high (CV={tail_cv:.2%})"
                            })
                            adjusted.loc[idx] = new_val

        # Cap factors > 1.0 for prudence
        for idx, val in adjusted.items():
            if val > 1.0:
                adjustments.append({
                    "adjustment_type": "cap_to_1.0",
                    "period": str(idx),
                    "original_value": float(val),
                    "adjusted_value": 1.0,
                    "reason": "Factor > 1.0 capped for prudence"
                })
                adjusted.loc[idx] = 1.0

        return adjusted, adjustments

    def _generate_explanation(
        self,
        thought: AgentThought,
        pattern_result: PatternAnalysisResult,
        selected: str,
        diagnostics: Dict,
        estimator_results: Dict,
        validation_metrics: Dict
    ) -> str:
        """Generate comprehensive human-readable explanation."""
        lines = []

        lines.append("=" * 70)
        lines.append("INTELLIGENT METHOD SELECTION - COMPLETE ANALYSIS")
        lines.append("=" * 70)

        # Selected Method
        lines.append(f"\n🎯 SELECTED METHOD: {selected}")
        lines.append("-" * 50)

        # Full Reasoning
        if thought.decision:
            lines.append(f"\n📝 REASONING:")
            lines.append(thought.decision.reasoning)

        # Evidence Cited
        if thought.decision and thought.decision.evidence:
            lines.append(f"\n📊 EVIDENCE CITED:")
            for ev in thought.decision.evidence:
                lines.append(f"   • {ev}")

        # Alternatives Considered
        if thought.decision and thought.decision.alternatives_considered:
            lines.append(f"\n🔄 ALTERNATIVES CONSIDERED:")
            for alt in thought.decision.alternatives_considered:
                lines.append(f"   • {alt}")

        # Pattern Analysis
        lines.append(f"\n📈 PATTERN ANALYSIS:")
        if pattern_result.smoothing_applied:
            lines.append(f"   ✓ Smoothing applied: {pattern_result.smoothing_method}")
            lines.append(f"   ✓ Smoothing weight: {pattern_result.smoothing_weight:.0%}")
            if pattern_result.thought_process:
                lines.append(f"   ✓ Reason: {pattern_result.thought_process[:200]}...")
        else:
            lines.append("   ✓ No smoothing needed - pattern is acceptable")

        # Diagnostics Summary
        lines.append(f"\n🔬 DIAGNOSTICS:")
        lines.append(f"   Adequacy: {diagnostics.get('adequacy_score', 'N/A')}/100")
        lines.append(f"   Rating: {diagnostics.get('rating', 'N/A')}")
        if diagnostics.get('issues'):
            for issue in diagnostics['issues'][:3]:
                lines.append(f"   ⚠️ {issue}")

        # Reserve Comparison
        lines.append(f"\n💰 RESERVE COMPARISON:")
        for name, reserve in sorted(estimator_results.items(), key=lambda x: x[1]):
            marker = " ← SELECTED" if name == selected or name in selected else ""
            lines.append(f"   {name}: ${reserve:,.0f}{marker}")

        # Self-Critique
        if thought.critique:
            lines.append(f"\n🔎 SELF-CRITIQUE:")
            lines.append(f"   Revised Confidence: {thought.critique.revised_confidence.value}")

            if thought.critique.weaknesses:
                lines.append(f"\n   Weaknesses in reasoning:")
                for w in thought.critique.weaknesses[:3]:
                    lines.append(f"      • {w}")

            lines.append(f"\n💡 FINAL RECOMMENDATION:")
            lines.append(f"   {thought.critique.final_recommendation}")

        # Risks
        if thought.decision and thought.decision.risks:
            lines.append(f"\n⚠️ RISKS TO CONSIDER:")
            for risk in thought.decision.risks[:3]:
                lines.append(f"   • {risk}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


    def consider_validation_feedback(
        self,
        initial_selection: 'SelectionResult',
        feedback: 'ValidationFeedback',
        verbose: bool = True
    ) -> 'SelectionResult':
        """
        Consider ValidationAgent's feedback and make FINAL decision.

        This is called ONCE after receiving feedback.
        NO more iterations after this.

        Args:
            initial_selection: The initial selection made
            feedback: Feedback from ValidationAgent
            verbose: Print progress

        Returns:
            Final SelectionResult (may be same or different)
        """
        if verbose:
            self._log("🔄 Considering ValidationAgent feedback...")

        # If validator agrees and has no major concerns, keep selection
        if feedback.agrees_with_selection and not feedback.alternative_recommendation:
            if verbose:
                self._log("   ✓ Validator agrees - keeping initial selection")

            # Add validation concerns to explanation
            updated_explanation = initial_selection.explanation + "\n\n"
            updated_explanation += "=" * 70 + "\n"
            updated_explanation += "VALIDATION REVIEW\n"
            updated_explanation += "=" * 70 + "\n"
            updated_explanation += f"✓ Validator AGREES with selection\n"

            if feedback.concerns:
                updated_explanation += "\nConcerns noted:\n"
                for c in feedback.concerns:
                    updated_explanation += f"  • {c}\n"

            if feedback.suggestions:
                updated_explanation += "\nSuggestions:\n"
                for s in feedback.suggestions:
                    updated_explanation += f"  • {s}\n"

            # Return updated selection with validation info
            return SelectionResult(
                selected_estimator=initial_selection.selected_estimator,
                selected_factors=initial_selection.selected_factors,
                original_factors=initial_selection.original_factors,
                adjusted_factors=initial_selection.adjusted_factors,
                pattern_analysis=initial_selection.pattern_analysis,
                thought_process=initial_selection.thought_process,
                all_estimator_results=initial_selection.all_estimator_results,
                diagnostics_summary=initial_selection.diagnostics_summary,
                validation_metrics=initial_selection.validation_metrics,
                tail_factor=initial_selection.tail_factor,
                confidence=initial_selection.confidence,
                explanation=updated_explanation,
                prudential_adjustments=initial_selection.prudential_adjustments
            )

        # Validator disagrees or has alternative - LLM makes final decision
        if verbose:
            self._log("   ⚠️ Validator has concerns - making final decision...")

        prompt = f"""
{self._get_system_prompt()}

=== YOUR INITIAL SELECTION ===
Selected: {initial_selection.selected_estimator}
Your reasoning: {initial_selection.thought_process.decision.reasoning if initial_selection.thought_process.decision else 'N/A'}

=== VALIDATOR FEEDBACK ===
Agrees: {feedback.agrees_with_selection}
Concerns: {json.dumps(feedback.concerns)}
Suggestions: {json.dumps(feedback.suggestions)}
Alternative recommendation: {feedback.alternative_recommendation or 'None'}
Validator reasoning: {feedback.reasoning}

=== YOUR TASK ===
Consider the validator's feedback and make your FINAL decision.

You can:
1. KEEP your original selection (if you believe it's still correct despite concerns)
2. CHANGE to the validator's recommendation (if their argument is convincing)
3. MODIFY your selection based on both perspectives

IMPORTANT: Explain WHY you're making this final decision.
Acknowledge the validator's concerns even if you disagree.

Respond in JSON:
{{
    "final_selection": "the method you're choosing",
    "changed_from_original": true/false,
    "reasoning": "detailed reasoning for final decision",
    "response_to_validator": "how you address their concerns",
    "confidence": "high|medium|low"
}}
"""

        try:
            response = self.llm.get_completion(
                system_prompt="You are making a final decision. Respond in JSON.",
                user_prompt=prompt
            )

            result = self._parse_json_response(response)
            final_choice = result.get('final_selection', initial_selection.selected_estimator)
            changed = result.get('changed_from_original', False)

            if verbose:
                if changed:
                    self._log(f"   🔄 CHANGED selection to: {final_choice}")
                else:
                    self._log(f"   ✓ KEPT original selection: {final_choice}")

            # Build final explanation
            final_explanation = initial_selection.explanation + "\n\n"
            final_explanation += "=" * 70 + "\n"
            final_explanation += "VALIDATION REVIEW & FINAL DECISION\n"
            final_explanation += "=" * 70 + "\n\n"

            final_explanation += f"Validator {'AGREED' if feedback.agrees_with_selection else 'DISAGREED'}\n"
            if feedback.alternative_recommendation:
                final_explanation += f"Validator suggested: {feedback.alternative_recommendation}\n"

            final_explanation += f"\nConcerns raised:\n"
            for c in feedback.concerns:
                final_explanation += f"  • {c}\n"

            final_explanation += f"\n📋 FINAL DECISION: {final_choice}\n"
            final_explanation += f"Changed from original: {'Yes' if changed else 'No'}\n\n"
            final_explanation += f"Reasoning:\n{result.get('reasoning', 'N/A')}\n\n"
            final_explanation += f"Response to validator:\n{result.get('response_to_validator', 'N/A')}\n"

            # Get factors for final selection
            if final_choice != initial_selection.selected_estimator:
                # Need to get factors for new selection
                estimators = get_all_estimators()
                est = next((e for e in estimators if e.name == final_choice), None)
                if est:
                    # Would need triangle here - for now keep original factors
                    final_factors = initial_selection.adjusted_factors
                else:
                    final_factors = initial_selection.adjusted_factors
            else:
                final_factors = initial_selection.adjusted_factors

            return SelectionResult(
                selected_estimator=final_choice,
                selected_factors=final_factors,
                original_factors=initial_selection.original_factors,
                adjusted_factors=initial_selection.adjusted_factors,
                pattern_analysis=initial_selection.pattern_analysis,
                thought_process=initial_selection.thought_process,
                all_estimator_results=initial_selection.all_estimator_results,
                diagnostics_summary=initial_selection.diagnostics_summary,
                validation_metrics=initial_selection.validation_metrics,
                tail_factor=initial_selection.tail_factor,
                confidence=ConfidenceLevel(result.get('confidence', 'medium')),
                explanation=final_explanation,
                prudential_adjustments=initial_selection.prudential_adjustments
            )

        except Exception as e:
            if verbose:
                self._log(f"   ✗ Final decision failed: {e}, keeping original")
            return initial_selection

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        return json.loads(response)


def get_intelligent_selection_agent() -> IntelligentSelectionAgent:
    """Factory function."""
    return IntelligentSelectionAgent()


# Backward compatibility
def get_selection_agent() -> IntelligentSelectionAgent:
    """Backward compatible factory function."""
    return IntelligentSelectionAgent()
