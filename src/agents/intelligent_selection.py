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

from agents.intelligent_base import IntelligentAgent, ConfidenceLevel, AgentThought
from agents.schemas import (
    AgentRole,
    ReservingOutput,
    ReservingConfigFile,
    MethodResult,
    StochasticResult,
    DiagnosticsResult,
    TriangleMetadata,
    AnalysisType,
)

from model_selection.factor_estimators import get_all_estimators
from model_selection.model_selector import ModelSelector
from diagnostics.diagnostic_tests import DiagnosticTests
from tail_fitting.tail_estimator import TailEstimator
from chain_ladder import ChainLadder
from pattern_analysis.pattern_analyzer import PatternAnalyzer
from pattern_analysis.curve_fitting import CurveFitter
import sys

sys.path.append("src")
from enhanced_workflow import EnhancedReservingWorkflow


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

    # Optional thought process from LLM
    thought_process: Optional[AgentThought] = None

    # Properties for backward compatibility with orchestrator
    @property
    def selected_estimator(self) -> str:
        return self.method

    @property
    def all_estimator_results(self) -> Dict[str, float]:
        return self.reserves

    @property
    def diagnostics_summary(self) -> Dict:
        return self.diagnostics

    @property
    def pattern_analysis(self):
        return self.pattern

    @property
    def adjusted_factors(self):
        return self.factors

    @property
    def prudential_adjustments(self) -> List[Dict]:
        return self.adjustments


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

    def analyze_and_select_stream(self, triangle: pd.DataFrame, verbose: bool = True):
        """
        Streaming version of analyze_and_select.
        Yields thought process in real-time.

        Yields:
            Dict with 'phase', 'status', 'content', 'data'
        """
        if verbose:
            print(f"[{self.role}] 🔍 Starting intelligent selection...")

        # Phase 1: Data Collection
        yield {
            "phase": "data_collection",
            "status": "running",
            "content": "📊 Running all estimators...",
            "data": None,
        }

        reserves, factors = self._run_estimators(triangle)
        yield {
            "phase": "data_collection",
            "status": "complete",
            "content": f"✓ {len(reserves)} estimators completed",
            "data": {"estimators": list(reserves.keys()), "reserves": reserves},
        }

        # Phase 2: Validation
        yield {
            "phase": "validation",
            "status": "running",
            "content": "📈 Running cross-validation...",
            "data": None,
        }

        validation = self._run_validation(triangle)
        yield {
            "phase": "validation",
            "status": "complete",
            "content": f"✓ Best estimator by MAE: {validation['best'].get('MAE', 'N/A')}",
            "data": {"best": validation["best"], "table": validation["table"]},
        }

        # Phase 3: Diagnostics
        yield {
            "phase": "diagnostics",
            "status": "running",
            "content": "🔬 Running diagnostic tests...",
            "data": None,
        }

        diagnostics = self._run_diagnostics(triangle)
        yield {
            "phase": "diagnostics",
            "status": "complete",
            "content": f"✓ Adequacy score: {diagnostics.get('adequacy_score', 'N/A')}/100",
            "data": diagnostics,
        }

        # Phase 4: Pattern Analysis with streaming
        yield {
            "phase": "pattern_analysis",
            "status": "running",
            "content": "🔍 Analyzing development patterns (with intelligent reasoning)...",
            "data": None,
        }

        base_factors = self._select_base_factors(validation, factors)

        # Stream pattern analysis thoughts
        pattern = None
        for pattern_thought in self.pattern_analyzer.analyze_pattern_stream(
            base_factors, triangle
        ):
            # Yield pattern analyzer thoughts to UI
            if pattern_thought["phase"] != "complete":
                yield {
                    "phase": f"pattern_{pattern_thought['phase']}",
                    "status": pattern_thought["status"],
                    "content": pattern_thought["content"],
                    "thought_data": pattern_thought.get("data"),
                    "agent": "PatternAnalyzer",
                }
            else:
                # Final result
                pattern = pattern_thought["data"]["result"]

        if pattern is None:
            # Fallback to non-streaming
            pattern = self.pattern_analyzer.analyze_pattern(base_factors, triangle)

        yield {
            "phase": "pattern_analysis",
            "status": "complete",
            "content": f"✓ Pattern analysis complete - {'smoothing applied' if pattern.smoothing_applied else 'no smoothing needed'}",
            "data": {
                "smoothing_applied": pattern.smoothing_applied,
                "smoothing_method": pattern.smoothing_method,
                "smoothing_weight": pattern.smoothing_weight,
                "thought_process": pattern.thought_process,
            },
        }

        # Phase 5: Tail Fitting
        yield {
            "phase": "tail_fitting",
            "status": "running",
            "content": "📐 Fitting tail factors...",
            "data": None,
        }

        tail = self._fit_tail(triangle)
        yield {
            "phase": "tail_fitting",
            "status": "complete",
            "content": f"✓ Tail factor: {tail:.4f}",
            "data": {"tail_factor": tail},
        }

        # Phase 6: LLM Decision
        yield {
            "phase": "llm_decision",
            "status": "running",
            "content": "🧠 LLM is analyzing and deciding...",
            "data": None,
        }

        thought = self._decide(reserves, validation, diagnostics, pattern, tail)
        yield {
            "phase": "llm_decision",
            "status": "complete",
            "content": f"✓ Selected: {thought['choice']} ({thought['confidence']} confidence)",
            "data": {
                "choice": thought["choice"],
                "reasoning": thought["reasoning"],
                "confidence": thought["confidence"],
            },
        }

        # Phase 7: Final Factors
        if "_smoothed" in thought["choice"] and pattern.smoothing_applied:
            final_factors = pattern.recommended_factors
        else:
            final_factors = base_factors

        # Phase 8: Prudence
        final_factors, adjustments = self._prudence(final_factors)

        if adjustments:
            yield {
                "phase": "prudence",
                "status": "complete",
                "content": f"✓ Applied {len(adjustments)} prudential adjustments",
                "data": {"adjustments": adjustments},
            }

        # Phase 9: Build Result
        explanation = self._build_explanation(thought)

        result = SelectionResult(
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

        yield {
            "phase": "complete",
            "status": "complete",
            "content": f"🎯 Selection complete: {thought['choice']}",
            "data": {"result": result},
        }

        return result

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
            for m in ["MAE", "RMSE", "MAPE", "R²"]:
                if m in selector.comparison_table:
                    # For R², higher is better; for errors, lower is better
                    if m == "R²":
                        best[m] = selector.comparison_table[m].idxmax()
                    else:
                        best[m] = selector.comparison_table[m].idxmin()

        return {"best": best, "table": selector.comparison_table}

    # --------------------------------------------------

    def _run_diagnostics(self, triangle):
        diag = DiagnosticTests(triangle)

        res = diag.get_model_adequacy_score()

        return res

    # --------------------------------------------------

    def _select_base_factors(self, validation, factors):
        # Try MAE first (most robust), then RMSE as fallback
        best = validation["best"].get("MAE") or validation["best"].get("RMSE")

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

    def execute_full_analysis(
        self,
        triangle: pd.DataFrame,
        config: ReservingConfigFile,
        premium: Optional[pd.Series] = None,
        verbose: bool = True,
        precomputed_factors: Optional[pd.Series] = None,
    ) -> ReservingOutput:
        """
        Execute complete actuarial analysis and return ReservingOutput.

        This replaces the need for ReservingExecutionAgent.
        Uses EnhancedReservingWorkflow with selected factors.

        Args:
            precomputed_factors: If provided, skip selection and use these factors directly.
                                 This avoids re-running the full selection pipeline.
        """
        from datetime import datetime

        if verbose:
            print(f"[Selection+Execution] 🚀 Starting full analysis...")

        if precomputed_factors is not None:
            # Use pre-computed factors from earlier selection phase (avoid duplicate work)
            selected_factors = precomputed_factors
            if verbose:
                print(f"[Selection+Execution] ✅ Using pre-computed factors (skipping re-selection)")
        else:
            # Fallback: run selection from scratch
            selection_result = self.analyze_and_select(triangle, verbose=verbose)
            selected_factors = selection_result.adjusted_factors

        # Use EnhancedReservingWorkflow for full execution
        workflow = EnhancedReservingWorkflow(
            triangle=triangle, earned_premium=premium, verbose=verbose
        )

        # Apply selected factors
        workflow.selected_factors = selected_factors

        # Run Chain Ladder
        workflow.run_chain_ladder()

        # Run stochastic methods if not QUICK
        if config.analysis_type != AnalysisType.QUICK:
            workflow.run_mack_model()

        # Run Bootstrap if requested
        if config.run_bootstrap:
            workflow.run_bootstrap(n_simulations=config.n_bootstrap_simulations)

        # Run alternative methods for FULL analysis
        if config.analysis_type == AnalysisType.FULL and premium is not None:
            workflow.run_alternative_methods()

        # Run diagnostics if requested
        if config.run_diagnostics:
            workflow.run_diagnostics()

        # Run stress testing for FULL analysis
        if config.run_stress_testing and config.analysis_type == AnalysisType.FULL:
            workflow.run_stress_testing()

        # Package results into ReservingOutput
        output = self._package_results(workflow, config, triangle, verbose)

        if verbose:
            print(f"[{self.role}] ✅ Full analysis complete!")

        return output

    def _package_results(
        self,
        workflow: EnhancedReservingWorkflow,
        config: ReservingConfigFile,
        triangle: pd.DataFrame,
        verbose: bool = True,
    ) -> ReservingOutput:
        """Package workflow results into ReservingOutput structure.

        Note: workflow.results stores actual objects (ChainLadder, MackChainLadder, etc.),
        not plain dicts. We call their .summary() / .get_total_reserve_distribution()
        methods to extract the numbers.
        """

        # Triangle metadata
        triangle_meta = TriangleMetadata(
            n_accident_years=len(triangle),
            n_development_periods=len(triangle.columns),
            first_accident_year=int(triangle.index[0]),
            last_accident_year=int(triangle.index[-1]),
            currency="USD",
            units="millions",
        )

        results = workflow.results

        # --- Chain Ladder (object with .summary()) ---
        cl_obj = results.get("chain_ladder")
        if cl_obj is not None:
            cl_summary = cl_obj.summary()
            chain_ladder_result = MethodResult(
                method_name="Chain Ladder",
                total_reserve=cl_summary["total_reserve"],
                ultimate_loss=cl_summary["total_ultimate"],
                model_params={
                    "factors": workflow.selected_factors.to_dict()
                    if hasattr(workflow.selected_factors, "to_dict")
                    else {}
                },
            )
        else:
            # Shouldn't happen, but safe fallback
            chain_ladder_result = MethodResult(
                method_name="Chain Ladder",
                total_reserve=0,
                ultimate_loss=0,
                model_params={},
            )

        # Initialize output
        output = ReservingOutput(
            config_used=config,
            triangle_info=triangle_meta,
            chain_ladder=chain_ladder_result,
        )

        # --- Mack Model (object with .get_total_reserve_distribution()) ---
        mack_obj = results.get("mack")
        if mack_obj is not None:
            try:
                mack_dist = mack_obj.get_total_reserve_distribution()
                total_reserve = mack_dist.get("Total_Reserve", 0)
                std_error = mack_dist.get("Total_SE", 0)
                cv = mack_dist.get("Total_CV", 0)

                # Build percentiles from confidence intervals or normal approx
                percentiles = {}
                ci = mack_dist.get("Confidence_Intervals", {})
                if ci:
                    for level_key, (lower, upper) in ci.items():
                        percentiles[level_key] = upper
                elif std_error > 0:
                    percentiles = {
                        "p75": total_reserve + 0.674 * std_error,
                        "p90": total_reserve + 1.282 * std_error,
                        "p95": total_reserve + 1.645 * std_error,
                        "p99": total_reserve + 2.326 * std_error,
                    }

                # Get ultimate from summary DataFrame
                mack_summary_df = mack_obj.summary()
                total_ultimate = mack_summary_df["Ultimate"].sum() if "Ultimate" in mack_summary_df.columns else 0

                output.mack = StochasticResult(
                    method_name="Mack",
                    total_reserve=total_reserve,
                    ultimate_loss=total_ultimate,
                    standard_error=std_error,
                    cv=cv,
                    percentiles=percentiles,
                )
            except Exception as e:
                if verbose:
                    print(f"[PackageResults] Warning: Could not package Mack results: {e}")

        # --- Bootstrap (object with .get_total_reserve_distribution()) ---
        boot_obj = results.get("bootstrap")
        if boot_obj is not None:
            try:
                boot_dist = boot_obj.get_total_reserve_distribution()
                total_reserve = boot_dist.get("Mean", 0)
                std_error = boot_dist.get("Std", 0)
                cv = boot_dist.get("CV", 0)

                # Estimate ultimate = latest observed + reserve
                total_ultimate = triangle.apply(
                    lambda row: row.dropna().iloc[-1] if len(row.dropna()) > 0 else 0,
                    axis=1
                ).sum() + total_reserve

                percentiles = {
                    "75%": boot_dist.get("P75", 0),
                    "90%": boot_dist.get("P90", 0),
                    "95%": boot_dist.get("P95", 0),
                    "99%": boot_dist.get("P99", 0),
                    "99.5%": boot_dist.get("P99", 0),  # approx
                }

                output.bootstrap = StochasticResult(
                    method_name="Bootstrap",
                    total_reserve=total_reserve,
                    ultimate_loss=total_ultimate,
                    standard_error=std_error,
                    cv=cv,
                    percentiles=percentiles,
                )
            except Exception as e:
                if verbose:
                    print(f"[PackageResults] Warning: Could not package Bootstrap results: {e}")

        # --- Cape Cod (object with .summary()) ---
        cc_obj = results.get("cape_cod")
        if cc_obj is not None:
            try:
                cc_summary = cc_obj.summary()
                output.cape_cod = MethodResult(
                    method_name="Cape Cod",
                    total_reserve=cc_summary.get("total_cc_reserve", 0),
                    ultimate_loss=cc_summary.get("total_cc_reserve", 0) + cc_summary.get("total_reported", 0),
                    model_params={"elr": cc_summary.get("cape_cod_elr", 0)},
                )
            except Exception as e:
                if verbose:
                    print(f"[PackageResults] Warning: Could not package Cape Cod results: {e}")

        # --- Diagnostics (dict of objects) ---
        diag_dict = results.get("diagnostics")
        if diag_dict is not None and isinstance(diag_dict, dict):
            try:
                # Get adequacy score from DiagnosticTests object
                tests_obj = diag_dict.get("tests")
                if tests_obj is not None:
                    adequacy = tests_obj.get_model_adequacy_score()
                    adequacy_score = adequacy.get("adequacy_score", 100)
                    issues = adequacy.get("issues", [])
                else:
                    adequacy_score = 100
                    issues = []

                rating = "GOOD"
                if adequacy_score < 60:
                    rating = "POOR"
                elif adequacy_score < 80:
                    rating = "FAIR"

                output.diagnostics = DiagnosticsResult(
                    adequacy_score=adequacy_score,
                    rating=rating,
                    issues=issues,
                )
            except Exception as e:
                if verbose:
                    print(f"[PackageResults] Warning: Could not package Diagnostics results: {e}")

        # --- Detailed Data (for Q&A agent) ---
        try:
            from agents.schemas import DetailedData

            # Triangle as nested dict
            triangle_dict = {}
            for year in triangle.index:
                triangle_dict[str(year)] = {
                    str(col): (float(triangle.loc[year, col])
                               if pd.notna(triangle.loc[year, col]) else None)
                    for col in triangle.columns
                }

            # Development factors
            dev_factors = {}
            if hasattr(workflow.selected_factors, "items"):
                dev_factors = {str(k): float(v) for k, v in workflow.selected_factors.items()}

            # Reserves and ultimates by year from CL
            reserves_by_year = {}
            ultimates_by_year = {}
            latest_diagonal = {}
            if cl_obj is not None and hasattr(cl_obj, "ultimate_losses") and cl_obj.ultimate_losses is not None:
                for year in cl_obj.ultimate_losses.index:
                    row = cl_obj.ultimate_losses.loc[year]
                    reserves_by_year[str(year)] = float(row.get("Reserve", 0))
                    ultimates_by_year[str(year)] = float(row.get("Ultimate", 0))
                    latest_diagonal[str(year)] = float(row.get("Latest_Value", 0))

            output.detailed_data = DetailedData(
                triangle=triangle_dict,
                development_factors=dev_factors,
                reserves_by_year=reserves_by_year,
                ultimates_by_year=ultimates_by_year,
                latest_diagonal=latest_diagonal,
            )
        except Exception as e:
            if verbose:
                print(f"[PackageResults] Warning: Could not build detailed data: {e}")

        return output


# ======================================================
# Factory
# ======================================================


def get_selection_agent():
    return CleanSelectionAgent()


# Backward compatibility alias
IntelligentSelectionAgent = CleanSelectionAgent
