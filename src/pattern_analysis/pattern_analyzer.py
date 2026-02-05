"""
Pattern Analyzer - Intelligent Analysis of Development Patterns
================================================================

Uses LLM to analyze development factor patterns, detect anomalies,
and decide whether/how to apply smoothing.

This is NOT a rule-based system. The LLM:
1. SEES the raw pattern data
2. ANALYZES it intelligently
3. DECIDES what to do
4. EXPLAINS its reasoning
5. CRITIQUES its own decision
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intelligent_base import (
    IntelligentAgent, Analysis, Decision, Critique,
    AgentThought, ConfidenceLevel
)
from agents.schemas import AgentRole
from pattern_analysis.curve_fitting import CurveFitter, FitResult


@dataclass
class PatternAnalysisResult:
    """Complete result of pattern analysis."""
    original_factors: pd.Series
    recommended_factors: pd.Series
    smoothing_applied: bool
    smoothing_method: Optional[str]
    smoothing_weight: Optional[float]
    thought_process: AgentThought
    fit_comparison: Optional[pd.DataFrame]
    all_fits: Optional[Dict[str, FitResult]]


class PatternAnalyzer(IntelligentAgent):
    """
    Intelligent pattern analyzer that uses LLM to reason about
    development factor patterns.

    Unlike rule-based approaches, this agent:
    - Sees the actual numbers
    - Considers context (triangle size, volatility, etc.)
    - Weighs evidence for/against smoothing
    - Can distinguish noise from structural changes
    - Explains its reasoning
    """

    def __init__(self):
        super().__init__(
            role=AgentRole.METHODOLOGY,
            name="Pattern_Analyzer"
        )

    def _get_system_prompt(self) -> str:
        return """You are an expert actuarial analyst specializing in loss reserving and development pattern analysis.

Your expertise includes:
- Chain ladder methodology and development factor analysis
- Statistical pattern recognition
- Distinguishing statistical noise from real structural changes
- Curve fitting methods (exponential decay, inverse power, Weibull, splines)
- Actuarial judgment in uncertain situations

When analyzing patterns, you consider:
- Whether deviations are isolated or systematic
- The magnitude of deviations relative to overall volatility
- Whether the triangle has characteristics that might explain anomalies
- The trade-off between smoothing noise and removing real information

You are thoughtful, evidence-based, and transparent about uncertainty."""

    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format pattern data for LLM analysis."""
        factors = data.get('factors', pd.Series())
        triangle_info = data.get('triangle_info', {})
        fit_results = data.get('fit_results', {})

        # Format factors with analysis
        factors_text = "DEVELOPMENT FACTORS:\n"
        factors_text += "-" * 60 + "\n"
        factors_text += f"{'Period':<10} {'Factor':<12} {'Δ from prev':<15} {'Direction':<12}\n"
        factors_text += "-" * 60 + "\n"

        prev_factor = None
        for i, (period, factor) in enumerate(factors.items()):
            delta = ""
            direction = ""
            if prev_factor is not None:
                delta_val = factor - prev_factor
                delta = f"{delta_val:+.4f}"
                if delta_val > 0:
                    direction = "⚠️ INCREASE"
                elif delta_val < 0:
                    direction = "decrease"
                else:
                    direction = "unchanged"

            factors_text += f"{period:<10} {factor:<12.4f} {delta:<15} {direction:<12}\n"
            prev_factor = factor

        factors_text += "-" * 60 + "\n"

        # Statistics
        factors_arr = factors.values
        factors_text += f"\nSTATISTICS:\n"
        factors_text += f"  Mean: {np.mean(factors_arr):.4f}\n"
        factors_text += f"  Std Dev: {np.std(factors_arr):.4f}\n"
        factors_text += f"  CV (Coef of Variation): {np.std(factors_arr)/np.mean(factors_arr):.2%}\n"
        factors_text += f"  Min: {np.min(factors_arr):.4f}\n"
        factors_text += f"  Max: {np.max(factors_arr):.4f}\n"
        factors_text += f"  Range: {np.max(factors_arr) - np.min(factors_arr):.4f}\n"

        # Count non-monotonic points
        non_monotonic = sum(1 for i in range(1, len(factors_arr)) if factors_arr[i] > factors_arr[i-1])
        factors_text += f"  Non-monotonic points: {non_monotonic}\n"

        # Triangle context
        if triangle_info:
            factors_text += f"\nTRIANGLE CONTEXT:\n"
            for key, value in triangle_info.items():
                factors_text += f"  {key}: {value}\n"

        # Curve fitting results
        if fit_results:
            factors_text += f"\nCURVE FITTING RESULTS:\n"
            factors_text += "-" * 60 + "\n"
            factors_text += f"{'Method':<20} {'R²':<10} {'RMSE':<10}\n"
            factors_text += "-" * 60 + "\n"
            for name, result in fit_results.items():
                factors_text += f"{name:<20} {result.r_squared:<10.4f} {result.rmse:<10.4f}\n"

        return factors_text

    def analyze_pattern(
        self,
        factors: pd.Series,
        triangle: pd.DataFrame = None,
        context: str = ""
    ) -> PatternAnalysisResult:
        """
        Analyze a development factor pattern and decide on smoothing.

        Args:
            factors: Development factors (age-to-age)
            triangle: Original triangle (for context)
            context: Additional context string

        Returns:
            PatternAnalysisResult with recommendation and reasoning
        """
        self._log("🔍 Starting intelligent pattern analysis...")

        # Prepare curve fitting tools
        fitter = CurveFitter(factors)
        fit_results = fitter.fit_all()
        fit_comparison = fitter.get_comparison_table(fit_results)

        # Build triangle context
        triangle_info = {}
        if triangle is not None:
            triangle_info = {
                'n_accident_years': len(triangle),
                'n_development_periods': len(triangle.columns),
                'first_year': int(triangle.index[0]),
                'last_year': int(triangle.index[-1]),
                'total_triangle_value': f"{triangle.sum().sum():,.0f}",
            }

            # Calculate volatility
            incremental = triangle.diff(axis=1)
            volatility = incremental.std().mean() / incremental.mean().mean()
            triangle_info['overall_volatility'] = f"{abs(volatility):.2%}" if pd.notna(volatility) else "N/A"

        # Prepare data for analysis
        data = {
            'factors': factors,
            'triangle_info': triangle_info,
            'fit_results': fit_results
        }

        # Build options for decision
        options = ["no_smoothing"]
        for method_name in fit_results.keys():
            options.append(f"smooth_{method_name}_100")  # Full smoothing
            options.append(f"smooth_{method_name}_70")   # 70% smooth, 30% original
            options.append(f"smooth_{method_name}_50")   # 50/50 blend

        # Run intelligent thinking process
        thought = self.think(
            data=data,
            options=options,
            task="Pattern Analysis and Smoothing Decision",
            focus="anomalies, non-monotonicity, and whether smoothing is appropriate",
            context=context
        )

        # Parse decision
        smoothing_applied = False
        smoothing_method = None
        smoothing_weight = None
        recommended_factors = factors.copy()

        if thought.decision and thought.decision.choice != "no_smoothing":
            choice = thought.decision.choice

            if choice.startswith("smooth_"):
                parts = choice.split("_")
                # Parse: smooth_methodname_weight
                weight_str = parts[-1]
                method_name = "_".join(parts[1:-1])

                smoothing_weight = int(weight_str) / 100.0
                smoothing_method = method_name
                smoothing_applied = True

                if method_name in fit_results:
                    fitted = fit_results[method_name].fitted_factors
                    recommended_factors = fitter.blend_with_original(
                        fitted, weight=smoothing_weight
                    )

                    # Ensure same index as original
                    recommended_factors = recommended_factors.reindex(factors.index)

        # Log summary
        if smoothing_applied:
            self._log(f"✓ Recommending smoothing: {smoothing_method} at {smoothing_weight:.0%}")
        else:
            self._log("✓ Recommending no smoothing - pattern is acceptable")

        return PatternAnalysisResult(
            original_factors=factors,
            recommended_factors=recommended_factors,
            smoothing_applied=smoothing_applied,
            smoothing_method=smoothing_method,
            smoothing_weight=smoothing_weight,
            thought_process=thought,
            fit_comparison=fit_comparison,
            all_fits=fit_results
        )

    def explain_decision(self, result: PatternAnalysisResult) -> str:
        """
        Generate human-readable explanation of the decision.

        Args:
            result: The analysis result to explain

        Returns:
            Formatted explanation string
        """
        thought = result.thought_process

        explanation = []
        explanation.append("=" * 60)
        explanation.append("PATTERN ANALYSIS REPORT")
        explanation.append("=" * 60)

        # Analysis summary
        if thought.analysis:
            explanation.append("\n📊 OBSERVATIONS:")
            for obs in thought.analysis.observations[:5]:  # Top 5
                explanation.append(f"  • {obs}")

            if thought.analysis.anomalies:
                explanation.append("\n⚠️ ANOMALIES DETECTED:")
                for anom in thought.analysis.anomalies[:3]:  # Top 3
                    explanation.append(f"  • {anom}")
            else:
                explanation.append("\n✅ No significant anomalies detected")

            explanation.append(f"\n🎯 Analysis Confidence: {thought.analysis.confidence.value}")

        # Decision
        if thought.decision:
            explanation.append("\n" + "-" * 60)
            explanation.append("📋 DECISION:")
            explanation.append(f"  Choice: {thought.decision.choice}")
            explanation.append(f"\n  Reasoning: {thought.decision.reasoning[:500]}...")

            if thought.decision.risks:
                explanation.append("\n  Risks:")
                for risk in thought.decision.risks[:3]:
                    explanation.append(f"    • {risk}")

            explanation.append(f"\n  Decision Confidence: {thought.decision.confidence.value}")

        # Critique
        if thought.critique:
            explanation.append("\n" + "-" * 60)
            explanation.append("🔎 SELF-CRITIQUE:")

            if thought.critique.weaknesses:
                explanation.append("  Potential weaknesses:")
                for weak in thought.critique.weaknesses[:2]:
                    explanation.append(f"    • {weak}")

            explanation.append(f"\n  Final Confidence: {thought.critique.revised_confidence.value}")
            explanation.append(f"\n  Final Recommendation: {thought.critique.final_recommendation[:300]}...")

        # Result summary
        explanation.append("\n" + "=" * 60)
        explanation.append("RESULT SUMMARY:")
        if result.smoothing_applied:
            explanation.append(f"  ✓ Smoothing Applied: {result.smoothing_method}")
            explanation.append(f"  ✓ Smoothing Weight: {result.smoothing_weight:.0%}")
        else:
            explanation.append("  ✓ No Smoothing Applied")
            explanation.append("  ✓ Using original factors")

        explanation.append("=" * 60)

        return "\n".join(explanation)

    def compare_factors(self, result: PatternAnalysisResult) -> pd.DataFrame:
        """
        Create comparison table of original vs recommended factors.

        Args:
            result: The analysis result

        Returns:
            DataFrame with comparison
        """
        df = pd.DataFrame({
            'Original': result.original_factors,
            'Recommended': result.recommended_factors,
        })

        df['Change'] = df['Recommended'] - df['Original']
        df['Change_Pct'] = (df['Change'] / df['Original'] * 100).round(2)

        return df


def get_pattern_analyzer() -> PatternAnalyzer:
    """Factory function to get PatternAnalyzer instance."""
    return PatternAnalyzer()
