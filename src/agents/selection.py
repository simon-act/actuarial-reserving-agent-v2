"""
Selection Agent - Comprehensive LLM-Driven Method Selection

Calculates ALL available metrics and passes them to LLM for intelligent selection:
- 7 Factor Estimators with full validation
- 13 Error Metrics (MAE, RMSE, MAPE, etc.)
- Diagnostic Tests (calendar effect, independence, proportionality)
- Tail Fitting (7 curve methods)
- Maturity Analysis
- Cape Cod Comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import hashlib
import pickle
from datetime import datetime, timedelta

from agents.schemas import AgentRole, AgentLog
from agents.llm_utils import LLMClient

# Cache settings
CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / "cache"
CACHE_TTL_HOURS = 24  # Cache valid for 24 hours

# Import ALL calculation tools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.factor_estimators import get_all_estimators, FactorEstimator
from model_selection.model_selector import ModelSelector
from model_selection.error_metrics import calculate_all_metrics
from model_selection.validation_framework import create_validator
from diagnostics.diagnostic_tests import DiagnosticTests
from diagnostics.volatility_analysis import VolatilityAnalyzer
from tail_fitting.tail_estimator import TailEstimator
from alternative_methods.cape_cod import CapeCod
from chain_ladder import ChainLadder


class SelectionAgent:
    """
    COMPREHENSIVE SELECTION AGENT

    Calculates EVERYTHING and gives LLM full context:
    1. All 7 factor estimators → reserves + factors
    2. Validation metrics (MAE, RMSE, MAPE, R², etc.)
    3. Diagnostic tests (5 Mack tests)
    4. Volatility analysis
    5. Tail fitting (7 methods)
    6. Cape Cod comparison
    7. Maturity by year
    """

    def __init__(self, use_cache: bool = True):
        self.role = AgentRole.METHODOLOGY
        self.llm = LLMClient()
        self.use_cache = use_cache
        self._cache = {}  # In-memory cache

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_triangle_hash(self, triangle: pd.DataFrame) -> str:
        """Generate hash of triangle for cache key."""
        # Use shape + sum + first/last values as hash input
        hash_input = f"{triangle.shape}_{triangle.sum().sum():.2f}_{triangle.iloc[0,0]}_{triangle.iloc[-1,-1]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache."""
        return CACHE_DIR / f"selection_{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached results if valid."""
        # Check in-memory first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                print(f"[{self.role}] ⚡ Using in-memory cache (key: {cache_key})")
                return cached['data']

        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                    print(f"[{self.role}] 💾 Using file cache (key: {cache_key})")
                    # Also store in memory
                    self._cache[cache_key] = cached
                    return cached['data']
            except Exception as e:
                print(f"[{self.role}] Cache read error: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save results to cache."""
        cached = {
            'timestamp': datetime.now(),
            'data': data
        }

        # Save to memory
        self._cache[cache_key] = cached

        # Save to file
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
            print(f"[{self.role}] 💾 Saved to cache (key: {cache_key})")
        except Exception as e:
            print(f"[{self.role}] Cache write error: {e}")

    def analyze_and_select(
        self,
        triangle: pd.DataFrame,
        premium: Optional[pd.Series] = None,
        force_recalculate: bool = False
    ) -> Tuple[Dict, AgentLog]:
        """Run COMPREHENSIVE analysis and let LLM select."""

        print(f"[{self.role}] 🔍 Starting COMPREHENSIVE method selection...")

        # Check cache first
        cache_key = self._get_triangle_hash(triangle)

        if self.use_cache and not force_recalculate:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                log = AgentLog(
                    agent=self.role,
                    action="Comprehensive Method Selection (CACHED)",
                    details=f"Loaded from cache: {cached_data.get('selected_estimator', 'Unknown')}"
                )
                return cached_data, log

        all_data = {}

        # 1. Calculate all 7 estimators
        all_data["estimators"] = self._calculate_all_estimators(triangle)

        # 2. Run validation with ALL error metrics
        all_data["validation"] = self._run_full_validation(triangle)

        # 3. Run diagnostic tests
        all_data["diagnostics"] = self._run_diagnostics(triangle)

        # 4. Volatility analysis
        all_data["volatility"] = self._analyze_volatility(triangle)

        # 5. Tail fitting
        all_data["tail_fitting"] = self._fit_tail(triangle)

        # 6. Calculate maturity
        all_data["maturity"] = self._calculate_maturity(triangle)

        # 7. BF and Cape Cod if premium available
        if premium is not None:
            all_data["bf_comparison"] = self._calculate_bf_comparison(triangle, premium)
            all_data["cape_cod"] = self._calculate_cape_cod(triangle, premium)

        # 8. Prepare context and ask LLM
        context = self._prepare_comprehensive_context(all_data)
        selection = self._llm_select(context, all_data)

        # Enrich with all data
        selection["all_estimators"] = {
            name: data.get("total_reserve", 0)
            for name, data in all_data["estimators"].items()
            if isinstance(data, dict) and "total_reserve" in data
        }
        selection["validation_metrics"] = all_data.get("validation", {}).get("metrics", {})
        selection["maturity_by_year"] = {
            str(year): mat.get("maturity_pct", 0)
            for year, mat in all_data.get("maturity", {}).items()
        }
        selection["diagnostics_summary"] = all_data.get("diagnostics", {}).get("summary", {})
        selection["tail_factor"] = all_data.get("tail_fitting", {}).get("best_tail_factor", 1.0)

        # Save to cache for future use
        if self.use_cache:
            self._save_to_cache(cache_key, selection)

        log = AgentLog(
            agent=self.role,
            action="Comprehensive Method Selection",
            details=f"Selected: {selection.get('selected_estimator', 'Unknown')}"
        )

        return selection, log

    def _calculate_all_estimators(self, triangle: pd.DataFrame) -> Dict:
        """Calculate reserves using all 7 factor estimators."""
        print(f"[{self.role}] 📊 Calculating all 7 factor estimators...")

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

                results[est.name] = {
                    'total_reserve': round(total_reserve, 2),
                    'reserves_by_year': {
                        str(k): round(v, 2)
                        for k, v in cl.ultimate_losses['Reserve'].to_dict().items()
                    },
                    'factors': {
                        str(k): round(float(v), 4)
                        for k, v in factors.items()
                    },
                    'cum_factors': {
                        str(k): round(float(v), 4)
                        for k, v in cl.cum_factors.items()
                    }
                }
                print(f"    ✓ {est.name}: ${total_reserve:,.0f}")

            except Exception as e:
                print(f"    ✗ {est.name}: {e}")
                results[est.name] = {'error': str(e)}

        return results

    def _run_full_validation(self, triangle: pd.DataFrame) -> Dict:
        """Run validation with ALL 13 error metrics."""
        print(f"[{self.role}] 🧪 Running full validation (13 error metrics)...")

        try:
            selector = ModelSelector(triangle, verbose=False)
            selector.run_validation()

            metrics = {}
            best_by_metric = {}

            if selector.comparison_table is not None:
                for idx, row in selector.comparison_table.iterrows():
                    metrics[idx] = {}
                    for col in selector.comparison_table.columns:
                        val = row[col]
                        if pd.notna(val):
                            metrics[idx][col] = round(float(val), 6)

                # Find best by each metric
                for col in selector.comparison_table.columns:
                    if col in ['MSE', 'MAE', 'RMSE', 'MAPE']:  # Lower is better
                        best_by_metric[col] = selector.comparison_table[col].idxmin()
                    elif col == 'R2':  # Higher is better
                        best_by_metric[col] = selector.comparison_table[col].idxmax()

            # Also run statistical tests
            try:
                selector.conduct_statistical_tests(['dm', 't_test'])
                pairwise_pvalues = {}
                if selector.statistical_tests:
                    for test_name, pvals in selector.statistical_tests.items():
                        pairwise_pvalues[test_name] = pvals.to_dict()
            except:
                pairwise_pvalues = {}

            print(f"    Best by MSE: {best_by_metric.get('MSE', 'N/A')}")
            print(f"    Best by MAE: {best_by_metric.get('MAE', 'N/A')}")

            return {
                "metrics": metrics,
                "best_by_metric": best_by_metric,
                "pairwise_tests": pairwise_pvalues
            }

        except Exception as e:
            print(f"    ✗ Validation failed: {e}")
            return {"error": str(e)}

    def _run_diagnostics(self, triangle: pd.DataFrame) -> Dict:
        """Run all 5 Mack diagnostic tests."""
        print(f"[{self.role}] 🔬 Running diagnostic tests...")

        try:
            diag = DiagnosticTests(triangle)
            all_results = diag.run_all_tests()
            adequacy = diag.get_model_adequacy_score()

            summary = {
                "adequacy_score": adequacy['adequacy_score'],
                "rating": adequacy['rating'],
                "issues": adequacy['issues'],
                "calendar_effect": all_results['calendar_year']['calendar_effect_detected'],
                "anomalous_years": all_results['accident_year'].get('anomalous_years', []),
                "independence_violated": all_results['independence']['independence_assumption_violated'],
                "proportionality_holds": all_results['proportionality']['proportionality_holds'],
                "variance_assumption_holds": all_results['variance']['variance_assumption_holds']
            }

            print(f"    Adequacy Score: {adequacy['adequacy_score']}/100 ({adequacy['rating']})")

            return {
                "summary": summary,
                "details": {
                    k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, pd.Series))}
                    for k, v in all_results.items()
                }
            }

        except Exception as e:
            print(f"    ✗ Diagnostics failed: {e}")
            return {"error": str(e)}

    def _analyze_volatility(self, triangle: pd.DataFrame) -> Dict:
        """Analyze factor volatility."""
        print(f"[{self.role}] 📈 Analyzing volatility...")

        try:
            vol = VolatilityAnalyzer(triangle)
            summary = vol.summary()
            high_vol = vol.get_high_volatility_periods(threshold=0.15)

            print(f"    High volatility periods: {len(high_vol)}")

            return {
                "high_volatility_periods": [
                    {"period": str(p['period']), "cv": round(p['cv'], 4)}
                    for p in high_vol[:5]  # Top 5
                ],
                "n_significant_trends": summary.get('n_significant_trends', 0),
                "n_potential_breaks": summary.get('n_potential_breaks', 0)
            }

        except Exception as e:
            print(f"    ✗ Volatility analysis failed: {e}")
            return {"error": str(e)}

    def _fit_tail(self, triangle: pd.DataFrame) -> Dict:
        """Fit tail using all 7 methods."""
        print(f"[{self.role}] 📐 Fitting tail factors...")

        try:
            tail = TailEstimator(triangle)
            results = tail.fit()

            best_method = tail.best_method
            best_factor = tail.tail_factor

            method_comparison = {}
            for name, res in results.items():
                if hasattr(res, 'tail_factor'):
                    method_comparison[name] = {
                        "tail_factor": round(res.tail_factor, 4),
                        "r_squared": round(res.r_squared, 4) if hasattr(res, 'r_squared') and res.r_squared else None
                    }

            print(f"    Best tail method: {best_method} (factor: {best_factor:.4f})")

            return {
                "best_method": best_method,
                "best_tail_factor": round(best_factor, 4),
                "all_methods": method_comparison
            }

        except Exception as e:
            print(f"    ✗ Tail fitting failed: {e}")
            return {"error": str(e)}

    def _calculate_maturity(self, triangle: pd.DataFrame) -> Dict:
        """Calculate maturity percentage for each year."""
        print(f"[{self.role}] 📊 Calculating maturity...")

        cl = ChainLadder(triangle)
        cl.calculate_age_to_age_factors()
        cl.select_development_factors()
        cl.calculate_cumulative_factors()

        maturity = {}
        for year in triangle.index:
            row = triangle.loc[year]
            last_col = row.dropna().index[-1]
            col_idx = list(triangle.columns).index(last_col)

            if last_col in cl.cum_factors.index:
                cum_factor = cl.cum_factors[last_col]
                mat_pct = 1.0 / cum_factor if cum_factor > 0 else 1.0
            else:
                mat_pct = 1.0

            maturity[year] = {
                'development_age': col_idx + 1,
                'maturity_pct': round(mat_pct * 100, 1),
                'is_mature': mat_pct >= 0.9,
                'is_immature': mat_pct < 0.7  # Less than 70% developed
            }

        immature = [str(y) for y, m in maturity.items() if m['is_immature']]
        print(f"    Immature years (<70%): {immature if immature else 'None'}")

        return maturity

    def _calculate_cape_cod(self, triangle: pd.DataFrame, premium: pd.Series) -> Dict:
        """Calculate Cape Cod."""
        print(f"[{self.role}] 🏖️ Calculating Cape Cod...")

        try:
            cc = CapeCod(triangle, premium)
            cc.fit()
            summary = cc.summary()

            return {
                "cape_cod_elr": round(summary['cape_cod_elr'] * 100, 2),
                "total_cc_reserve": round(summary['total_cc_reserve'], 2),
                "total_cl_reserve": round(summary['total_cl_reserve'], 2),
                "difference_pct": round(summary['difference_percent'], 2)
            }

        except Exception as e:
            print(f"    ✗ Cape Cod failed: {e}")
            return {"error": str(e)}

    def _prepare_comprehensive_context(self, all_data: Dict) -> str:
        """Prepare comprehensive context for LLM."""

        context = "=" * 60 + "\n"
        context += "COMPREHENSIVE RESERVING ANALYSIS\n"
        context += "=" * 60 + "\n\n"

        # Estimators
        context += "1. FACTOR ESTIMATOR COMPARISON:\n"
        context += "-" * 40 + "\n"
        for name, data in all_data.get("estimators", {}).items():
            if isinstance(data, dict) and "total_reserve" in data:
                context += f"   {name}: ${data['total_reserve']:,.0f}\n"

        # Validation metrics
        context += "\n2. VALIDATION METRICS (out-of-sample):\n"
        context += "-" * 40 + "\n"
        val = all_data.get("validation", {})
        if "best_by_metric" in val:
            for metric, best in val["best_by_metric"].items():
                context += f"   Best by {metric}: {best}\n"
        if "metrics" in val:
            context += "\n   Full metrics:\n"
            for name, metrics in list(val["metrics"].items())[:3]:  # Top 3
                mse = metrics.get('MSE', 'N/A')
                mae = metrics.get('MAE', 'N/A')
                context += f"   - {name}: MSE={mse}, MAE={mae}\n"

        # Diagnostics
        context += "\n3. DIAGNOSTIC TESTS:\n"
        context += "-" * 40 + "\n"
        diag = all_data.get("diagnostics", {}).get("summary", {})
        context += f"   Adequacy Score: {diag.get('adequacy_score', 'N/A')}/100\n"
        context += f"   Rating: {diag.get('rating', 'N/A')}\n"
        context += f"   Calendar Effect: {'Yes' if diag.get('calendar_effect') else 'No'}\n"
        context += f"   Independence OK: {'No' if diag.get('independence_violated') else 'Yes'}\n"
        if diag.get("issues"):
            context += f"   Issues: {', '.join(diag['issues'][:3])}\n"

        # Volatility
        context += "\n4. VOLATILITY:\n"
        context += "-" * 40 + "\n"
        vol = all_data.get("volatility", {})
        high_vol = vol.get("high_volatility_periods", [])
        if high_vol:
            context += f"   High volatility periods: {[p['period'] for p in high_vol]}\n"
        else:
            context += "   No high volatility periods detected\n"

        # Tail
        context += "\n5. TAIL FITTING:\n"
        context += "-" * 40 + "\n"
        tail = all_data.get("tail_fitting", {})
        context += f"   Best method: {tail.get('best_method', 'N/A')}\n"
        context += f"   Tail factor: {tail.get('best_tail_factor', 'N/A')}\n"

        # Maturity
        context += "\n6. MATURITY BY YEAR:\n"
        context += "-" * 40 + "\n"
        maturity = all_data.get("maturity", {})
        immature_years = [str(y) for y, m in maturity.items() if m.get('is_immature')]
        mature_years = [str(y) for y, m in maturity.items() if m.get('is_mature')]
        context += f"   Mature years (>90%): {len(mature_years)}\n"
        context += f"   Immature years (<70%): {immature_years if immature_years else 'None'}\n"

        # Cape Cod
        cc = all_data.get("cape_cod", {})
        if cc and "error" not in cc:
            context += "\n7. CAPE COD:\n"
            context += "-" * 40 + "\n"
            context += f"   Cape Cod ELR: {cc.get('cape_cod_elr', 0)}%\n"
            context += f"   CC Reserve: ${cc.get('total_cc_reserve', 0):,.0f}\n"

        return context

    def _llm_select(self, context: str, all_data: Dict) -> Dict:
        """Ask LLM to make selection based on comprehensive data."""
        print(f"[{self.role}] 🧠 Asking LLM for selection...")

        if not self.llm.is_available():
            return self._fallback_selection(all_data)

        system_prompt = """You are an expert actuarial advisor analyzing comprehensive reserving data.

Based on the validation metrics, diagnostic tests, and maturity analysis, decide:
1. Which factor estimator to use as BASE method
2. Which years need extra scrutiny (typically <70% maturity)
3. Whether to apply tail factor adjustment

RESPOND IN THIS EXACT JSON FORMAT:
{
    "selected_estimator": "Name of best estimator based on MSE/MAE",
    "estimator_reason": "Specific reason citing metrics",
    "immature_years": ["2022", "2023"],
    "immature_reason": "Why these years need extra attention (cite maturity %)",
    "apply_tail": true,
    "tail_reason": "Why/why not apply tail factor",
    "summary": "Executive summary of selection rationale"
}

SELECTION CRITERIA:
- Primary: Lowest MSE in out-of-sample validation
- Secondary: MAE, diagnostic adequacy score
- Flag years with <70% maturity for extra scrutiny
- Consider diagnostic issues (calendar effects, independence)
"""

        try:
            response = self.llm.get_completion(system_prompt, context)

            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                selection = json.loads(response[start:end])
                print(f"    ✓ LLM selected: {selection.get('selected_estimator')}")
                return selection
            else:
                return self._fallback_selection(all_data)

        except Exception as e:
            print(f"    ✗ LLM failed: {e}")
            return self._fallback_selection(all_data)

    def _fallback_selection(self, all_data: Dict) -> Dict:
        """Fallback based on metrics."""
        print(f"[{self.role}] Using fallback rules...")

        # Best by MSE
        best = all_data.get("validation", {}).get("best_by_metric", {}).get("MSE", "Volume-Weighted")

        # Immature years
        immature_years = [
            str(y) for y, m in all_data.get("maturity", {}).items()
            if m.get("is_immature")
        ]

        return {
            "selected_estimator": best,
            "estimator_reason": "Lowest MSE in out-of-sample validation",
            "immature_years": immature_years,
            "immature_reason": "Years with <70% maturity need extra scrutiny",
            "apply_tail": True,
            "tail_reason": "Tail factor recommended for ultimate projection",
            "summary": f"Selected {best} based on validation. Immature years: {immature_years or 'none'}."
        }


def get_selection_agent() -> SelectionAgent:
    return SelectionAgent()
