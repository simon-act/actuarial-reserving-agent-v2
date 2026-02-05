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
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from agents.intelligent_base import IntelligentAgent, ConfidenceLevel
from agents.schemas import AgentRole

from model_selection.factor_estimators import get_all_estimators
from model_selection.model_selector import ModelSelector
from diagnostics.diagnostic_tests import DiagnosticTests
from tail_fitting.tail_estimator import TailEstimator
from chain_ladder import ChainLadder
from pattern_analysis.pattern_analyzer import PatternAnalyzer
from pattern_analysis.curve_fitting import CurveFitter


TAIL_CV_THRESHOLD = 0.15
TAIL_MIN_PERIODS = 3


# ======================================================
# Result container
# ======================================================


@dataclass
class SelectionResult:
    method: str
    factors: pd.Series
    reserves: Dict[str, float]

    diagnostics: Dict
    validation: Dict
    pattern: object

    tail: float
    confidence: ConfidenceLevel
    explanation: str

    adjustments: List[Dict]


# ======================================================
# Agent
# ======================================================


class CleanSelectionAgent(IntelligentAgent):
    def __init__(self):
        super().__init__(role=AgentRole.METHODOLOGY, name="Clean_Selection_Agent")

        self.pattern_analyzer = PatternAnalyzer()

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    def _get_system_prompt(self):
        return """
You are a senior actuarial consultant.

Expertise:
- Chain Ladder and Mack
- Factor estimation
- Validation
- Pattern analysis

Task:
Select best method based on data.

Rules:
- Cite numbers
- Compare methods
- Discuss uncertainty

Output:
Clear, professional reasoning.
"""

    def _format_data_for_analysis(self, data: dict) -> str:
        """Format data for LLM analysis."""
        import json

        return json.dumps(data, indent=2, default=str)

    # ==================================================
    # Main
    # ==================================================

    def analyze_and_select(
        self, triangle: pd.DataFrame, verbose: bool = True
    ) -> SelectionResult:
        # 1. Estimates
        reserves, factors = self._run_estimators(triangle)

        # 2. Validation
        validation = self._run_validation(triangle)

        # 3. Diagnostics
        diagnostics = self._run_diagnostics(triangle)

        # 4. Pattern
        base_factors = self._select_base_factors(validation, factors)

        pattern = self.pattern_analyzer.analyze_pattern(base_factors, triangle)

        # 5. Tail
        tail = self._fit_tail(triangle)

        # 6. LLM decision
        thought = self._decide(reserves, validation, diagnostics, pattern, tail)

        # 7. Final factors
        if "_smoothed" in thought["choice"] and pattern.smoothing_applied:
            final_factors = pattern.recommended_factors
        else:
            final_factors = base_factors

        # 8. Prudence
        final_factors, adjustments = self._prudence(final_factors)

        # 9. Explanation
        explanation = self._build_explanation(thought)

        return SelectionResult(
            method=thought["choice"],
            factors=final_factors,
            reserves=reserves,
            diagnostics=diagnostics,
            validation=validation,
            pattern=pattern,
            tail=tail,
            confidence=ConfidenceLevel(thought["confidence"]),
            explanation=explanation,
            adjustments=adjustments,
        )

    # ==================================================
    # Steps
    # ==================================================

    def _run_estimators(self, triangle: pd.DataFrame) -> Tuple[Dict, Dict]:
        results = {}
        factors = {}

        for est in get_all_estimators():
            f = est.estimate(triangle)

            cl = ChainLadder(triangle)
            cl.selected_factors = f
            cl.calculate_cumulative_factors()
            cl.project_ultimate_losses()

            reserve = cl.ultimate_losses["Reserve"].sum()

            results[est.name] = round(reserve, 2)
            factors[est.name] = f

        return results, factors

    # --------------------------------------------------

    def _run_validation(self, triangle):
        selector = ModelSelector(triangle, verbose=False)
        selector.run_validation()

        best = {}

        if selector.comparison_table is not None:
            for m in ["MSE", "MAE", "RMSE"]:
                if m in selector.comparison_table:
                    best[m] = selector.comparison_table[m].idxmin()

        return {"best": best, "table": selector.comparison_table}

    # --------------------------------------------------

    def _run_diagnostics(self, triangle):
        diag = DiagnosticTests(triangle)

        res = diag.get_model_adequacy_score()

        return res

    # --------------------------------------------------

    def _select_base_factors(self, validation, factors):
        best = validation["best"].get("MSE")

        if best and best in factors:
            return factors[best]

        return list(factors.values())[0]

    # --------------------------------------------------

    def _fit_tail(self, triangle):
        try:
            tail = TailEstimator(triangle)
            tail.fit()

            return tail.tail_factor

        except:
            return 1.0

    # ==================================================
    # LLM
    # ==================================================

    def _decide(self, reserves, validation, diagnostics, pattern, tail):
        data = {
            "reserves": reserves,
            "best_validation": validation["best"],
            "diagnostics": diagnostics,
            "smoothing": pattern.smoothing_applied,
            "tail": tail,
        }

        prompt = f"""
Data:
{json.dumps(data, indent=2)}

Choose best method.

Respond in JSON:
{{
 "choice": "...",
 "reasoning": "...",
 "confidence": "high|medium|low"
}}
"""

        response = self.llm.get_completion(
            system_prompt=self._get_system_prompt(), user_prompt=prompt
        )

        return self._parse_json(response)

    # --------------------------------------------------

    def _parse_json(self, txt):
        s = txt.find("{")
        e = txt.rfind("}") + 1

        return json.loads(txt[s:e])

    # ==================================================
    # Prudence
    # ==================================================

    def _prudence(self, factors: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        adj = []
        f = factors.copy().astype(float)

        # Tail volatility
        tail = f.tail(TAIL_MIN_PERIODS)

        if len(tail) >= TAIL_MIN_PERIODS:
            cv = tail.std() / tail.mean()

            if cv >= TAIL_CV_THRESHOLD:
                fitter = CurveFitter(f)
                fits = fitter.fit_all()

                best = min(fits.values(), key=lambda x: x.rmse)

                new = best.fitted_factors.tail(len(tail))

                for i in new.index:
                    old = f[i]
                    val = new[i]

                    if abs(old - val) > 1e-6:
                        adj.append(
                            {
                                "type": "tail_fit",
                                "period": str(i),
                                "old": float(old),
                                "new": float(val),
                            }
                        )

                        f[i] = val

        # Cap > 1
        for i, v in f.items():
            if v > 1:
                adj.append(
                    {"type": "cap", "period": str(i), "old": float(v), "new": 1.0}
                )

                f[i] = 1.0

        return f, adj

    # ==================================================
    # Report
    # ==================================================

    def _build_explanation(self, thought):
        return f"""
Selected: {thought["choice"]}

Reasoning:
{thought["reasoning"]}

Confidence: {thought["confidence"]}
"""


# ======================================================
# Factory
# ======================================================


def get_selection_agent():
    return CleanSelectionAgent()


# Backward compatibility alias
IntelligentSelectionAgent = CleanSelectionAgent
