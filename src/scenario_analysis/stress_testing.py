"""
Stress Testing Framework for Actuarial Reserves

Provides comprehensive stress testing capabilities including:
- Factor shocks (+/- X% on development factors)
- Uniform and period-specific shocks
- Tail factor stress tests
- Combined multi-factor scenarios

References:
- IAA Paper on Stochastic Reserving
- Solvency II SCR calculations for reserve risk
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StressScenario:
    """
    Definition of a stress scenario.

    Attributes:
        name: Scenario name
        description: Scenario description
        factor_shocks: Dict mapping period to shock multiplier
        tail_shock: Shock to apply to tail factor
        loss_ratio_shock: Shock to expected loss ratios
    """
    name: str
    description: str
    factor_shocks: Optional[Dict[str, float]] = None
    uniform_factor_shock: Optional[float] = None
    tail_shock: Optional[float] = None
    loss_ratio_shock: Optional[float] = None


@dataclass
class StressResult:
    """
    Results from a stress scenario.

    Attributes:
        scenario: The applied scenario
        base_reserve: Reserve before stress
        stressed_reserve: Reserve after stress
        impact: Absolute change in reserve
        impact_pct: Percentage change in reserve
        reserve_by_year: Detailed reserve by accident year
    """
    scenario: StressScenario
    base_reserve: float
    stressed_reserve: float
    impact: float
    impact_pct: float
    reserve_by_year: pd.DataFrame


class StressTestFramework:
    """
    Comprehensive stress testing framework for chain-ladder reserves.

    Applies various stress scenarios to development factors and
    calculates reserve impact.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        base_factors: pd.Series,
        tail_factor: float = 1.0
    ):
        """
        Initialize stress test framework.

        Args:
            triangle: Cumulative loss development triangle
            base_factors: Base development factors
            tail_factor: Factor for development beyond observed periods
        """
        self.triangle = triangle.copy()
        self.base_factors = base_factors.copy()
        self.tail_factor = tail_factor

        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Calculate base reserves
        self.base_reserves = self._calculate_reserves(base_factors, tail_factor)
        self.total_base_reserve = self.base_reserves['Reserve'].sum()

        # Store stress results
        self.stress_results = {}

    def _calculate_reserves(
        self,
        factors: pd.Series,
        tail_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculate reserves using given factors.

        Args:
            factors: Development factors to use
            tail_factor: Tail factor for ultimate

        Returns:
            DataFrame with reserves by accident year
        """
        results = []

        # Calculate cumulative factors
        cum_factors = pd.Series(index=factors.index, dtype=float)
        cum_factor = tail_factor

        for period in reversed(factors.index):
            cum_factor *= factors[period]
            cum_factors[period] = cum_factor

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_age = row.dropna().index[-1]

            # Get cumulative factor
            if latest_age in cum_factors.index:
                cdf = cum_factors[latest_age]
            else:
                cdf = tail_factor

            ultimate = latest * cdf
            reserve = ultimate - latest

            results.append({
                'Accident_Year': year,
                'Latest': latest,
                'CDF': cdf,
                'Ultimate': ultimate,
                'Reserve': reserve
            })

        return pd.DataFrame(results).set_index('Accident_Year')

    def apply_uniform_shock(
        self,
        shock_pct: float,
        scenario_name: Optional[str] = None
    ) -> StressResult:
        """
        Apply uniform shock to all development factors.

        Args:
            shock_pct: Shock percentage (e.g., 0.10 for +10%)
            scenario_name: Name for the scenario

        Returns:
            StressResult object
        """
        if scenario_name is None:
            scenario_name = f"Uniform {shock_pct:+.0%} shock"

        scenario = StressScenario(
            name=scenario_name,
            description=f"All factors increased/decreased by {shock_pct:.0%}",
            uniform_factor_shock=shock_pct
        )

        # Apply shock
        stressed_factors = self.base_factors * (1 + shock_pct)

        # Calculate stressed reserves
        stressed_reserves = self._calculate_reserves(stressed_factors, self.tail_factor)

        total_stressed = stressed_reserves['Reserve'].sum()
        impact = total_stressed - self.total_base_reserve
        impact_pct = impact / self.total_base_reserve if self.total_base_reserve != 0 else 0

        result = StressResult(
            scenario=scenario,
            base_reserve=self.total_base_reserve,
            stressed_reserve=total_stressed,
            impact=impact,
            impact_pct=impact_pct,
            reserve_by_year=stressed_reserves
        )

        self.stress_results[scenario_name] = result
        return result

    def apply_period_specific_shock(
        self,
        period_shocks: Dict[str, float],
        scenario_name: str
    ) -> StressResult:
        """
        Apply shocks to specific development periods.

        Args:
            period_shocks: Dict mapping period to shock (e.g., {'24': 0.15})
            scenario_name: Name for the scenario

        Returns:
            StressResult object
        """
        scenario = StressScenario(
            name=scenario_name,
            description=f"Period-specific shocks: {period_shocks}",
            factor_shocks=period_shocks
        )

        # Apply shocks
        stressed_factors = self.base_factors.copy()
        for period, shock in period_shocks.items():
            if period in stressed_factors.index:
                stressed_factors[period] *= (1 + shock)

        # Calculate stressed reserves
        stressed_reserves = self._calculate_reserves(stressed_factors, self.tail_factor)

        total_stressed = stressed_reserves['Reserve'].sum()
        impact = total_stressed - self.total_base_reserve
        impact_pct = impact / self.total_base_reserve if self.total_base_reserve != 0 else 0

        result = StressResult(
            scenario=scenario,
            base_reserve=self.total_base_reserve,
            stressed_reserve=total_stressed,
            impact=impact,
            impact_pct=impact_pct,
            reserve_by_year=stressed_reserves
        )

        self.stress_results[scenario_name] = result
        return result

    def apply_tail_shock(
        self,
        tail_shock_pct: float,
        scenario_name: Optional[str] = None
    ) -> StressResult:
        """
        Apply shock to tail factor only.

        Args:
            tail_shock_pct: Shock to tail factor
            scenario_name: Name for the scenario

        Returns:
            StressResult object
        """
        if scenario_name is None:
            scenario_name = f"Tail {tail_shock_pct:+.0%} shock"

        scenario = StressScenario(
            name=scenario_name,
            description=f"Tail factor shocked by {tail_shock_pct:.0%}",
            tail_shock=tail_shock_pct
        )

        stressed_tail = self.tail_factor * (1 + tail_shock_pct)

        # Calculate stressed reserves
        stressed_reserves = self._calculate_reserves(self.base_factors, stressed_tail)

        total_stressed = stressed_reserves['Reserve'].sum()
        impact = total_stressed - self.total_base_reserve
        impact_pct = impact / self.total_base_reserve if self.total_base_reserve != 0 else 0

        result = StressResult(
            scenario=scenario,
            base_reserve=self.total_base_reserve,
            stressed_reserve=total_stressed,
            impact=impact,
            impact_pct=impact_pct,
            reserve_by_year=stressed_reserves
        )

        self.stress_results[scenario_name] = result
        return result

    def apply_early_period_shock(
        self,
        n_periods: int,
        shock_pct: float,
        scenario_name: Optional[str] = None
    ) -> StressResult:
        """
        Apply shock to early development periods only.

        Useful for testing sensitivity to changes in initial development.

        Args:
            n_periods: Number of early periods to shock
            shock_pct: Shock percentage
            scenario_name: Name for the scenario

        Returns:
            StressResult object
        """
        if scenario_name is None:
            scenario_name = f"Early {n_periods} periods {shock_pct:+.0%}"

        early_periods = list(self.base_factors.index[:n_periods])
        period_shocks = {p: shock_pct for p in early_periods}

        return self.apply_period_specific_shock(period_shocks, scenario_name)

    def apply_late_period_shock(
        self,
        n_periods: int,
        shock_pct: float,
        scenario_name: Optional[str] = None
    ) -> StressResult:
        """
        Apply shock to late development periods only.

        Useful for testing sensitivity to tail development.

        Args:
            n_periods: Number of late periods to shock
            shock_pct: Shock percentage
            scenario_name: Name for the scenario

        Returns:
            StressResult object
        """
        if scenario_name is None:
            scenario_name = f"Late {n_periods} periods {shock_pct:+.0%}"

        late_periods = list(self.base_factors.index[-n_periods:])
        period_shocks = {p: shock_pct for p in late_periods}

        return self.apply_period_specific_shock(period_shocks, scenario_name)

    def run_standard_scenarios(self) -> Dict[str, StressResult]:
        """
        Run a standard set of stress scenarios.

        Returns:
            Dictionary of scenario results
        """
        scenarios = {
            # Uniform shocks
            'Uniform +5%': self.apply_uniform_shock(0.05),
            'Uniform +10%': self.apply_uniform_shock(0.10),
            'Uniform +20%': self.apply_uniform_shock(0.20),
            'Uniform -5%': self.apply_uniform_shock(-0.05),
            'Uniform -10%': self.apply_uniform_shock(-0.10),

            # Tail shocks
            'Tail +10%': self.apply_tail_shock(0.10),
            'Tail +25%': self.apply_tail_shock(0.25),
            'Tail +50%': self.apply_tail_shock(0.50),

            # Early vs late shocks
            'Early 3 periods +10%': self.apply_early_period_shock(3, 0.10),
            'Late 3 periods +10%': self.apply_late_period_shock(3, 0.10),
            'Early 3 periods +20%': self.apply_early_period_shock(3, 0.20),
            'Late 3 periods +20%': self.apply_late_period_shock(3, 0.20),
        }

        return scenarios

    def run_regulatory_scenarios(self) -> Dict[str, StressResult]:
        """
        Run regulatory-style stress scenarios.

        Based on Solvency II and other regulatory approaches.

        Returns:
            Dictionary of scenario results
        """
        # Solvency II-style: 99.5% VaR approximation
        # Reserve risk: ~10-15% shock for short-tail, ~15-25% for long-tail

        scenarios = {
            '1-in-200 Adverse (15%)': self.apply_uniform_shock(0.15),
            '1-in-200 Adverse (20%)': self.apply_uniform_shock(0.20),
            '1-in-200 Adverse (25%)': self.apply_uniform_shock(0.25),

            # Severe but plausible
            'Severe Adverse (30%)': self.apply_uniform_shock(0.30),
            'Extreme Adverse (50%)': self.apply_uniform_shock(0.50),

            # Favorable scenarios
            '1-in-200 Favorable (-15%)': self.apply_uniform_shock(-0.15),

            # Combined tail + body shock
            'Combined Tail + Body': self._run_combined_shock(0.10, 0.20),
        }

        return scenarios

    def _run_combined_shock(
        self,
        body_shock: float,
        tail_shock: float
    ) -> StressResult:
        """Run combined body and tail shock."""
        scenario_name = f"Combined body {body_shock:+.0%} tail {tail_shock:+.0%}"

        scenario = StressScenario(
            name=scenario_name,
            description=f"Body shocked by {body_shock:.0%}, tail by {tail_shock:.0%}",
            uniform_factor_shock=body_shock,
            tail_shock=tail_shock
        )

        stressed_factors = self.base_factors * (1 + body_shock)
        stressed_tail = self.tail_factor * (1 + tail_shock)

        stressed_reserves = self._calculate_reserves(stressed_factors, stressed_tail)

        total_stressed = stressed_reserves['Reserve'].sum()
        impact = total_stressed - self.total_base_reserve
        impact_pct = impact / self.total_base_reserve if self.total_base_reserve != 0 else 0

        result = StressResult(
            scenario=scenario,
            base_reserve=self.total_base_reserve,
            stressed_reserve=total_stressed,
            impact=impact,
            impact_pct=impact_pct,
            reserve_by_year=stressed_reserves
        )

        self.stress_results[scenario_name] = result
        return result

    def sensitivity_analysis(
        self,
        shock_range: List[float] = None
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis across a range of shocks.

        Args:
            shock_range: List of shock percentages to test

        Returns:
            DataFrame with sensitivity results
        """
        if shock_range is None:
            shock_range = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        results = []

        for shock in shock_range:
            if shock == 0:
                reserve = self.total_base_reserve
                impact = 0
                impact_pct = 0
            else:
                stressed = self.apply_uniform_shock(shock, scenario_name=f"sens_{shock}")
                reserve = stressed.stressed_reserve
                impact = stressed.impact
                impact_pct = stressed.impact_pct

            results.append({
                'Shock_Pct': shock,
                'Reserve': reserve,
                'Impact': impact,
                'Impact_Pct': impact_pct
            })

        return pd.DataFrame(results)

    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of all stress results.

        Returns:
            DataFrame summarizing all scenarios
        """
        rows = []

        for name, result in self.stress_results.items():
            rows.append({
                'Scenario': name,
                'Base_Reserve': result.base_reserve,
                'Stressed_Reserve': result.stressed_reserve,
                'Impact': result.impact,
                'Impact_Pct': result.impact_pct
            })

        return pd.DataFrame(rows).sort_values('Impact', ascending=False)

    def print_summary(self):
        """Print formatted summary of stress test results."""
        print("\n" + "="*80)
        print("STRESS TEST RESULTS SUMMARY")
        print("="*80)

        print(f"\n📊 BASE CASE:")
        print("-"*60)
        print(f"Total Reserve: ${self.total_base_reserve:>15,.0f}")

        if self.stress_results:
            print("\n\n📊 STRESS SCENARIOS:")
            print("-"*60)
            print(f"{'Scenario':<35} {'Reserve':>15} {'Impact':>15} {'%':>10}")
            print("-"*75)

            summary = self.get_summary_table()
            for _, row in summary.iterrows():
                print(f"{row['Scenario']:<35} ${row['Stressed_Reserve']:>14,.0f} "
                      f"${row['Impact']:>14,.0f} {row['Impact_Pct']:>9.1%}")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save stress test results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        self.get_summary_table().to_csv(output_dir / "stress_test_summary.csv", index=False)

        # Save base reserves
        self.base_reserves.to_csv(output_dir / "base_reserves.csv")

        # Save sensitivity analysis
        self.sensitivity_analysis().to_csv(output_dir / "sensitivity_analysis.csv", index=False)

        print(f"✅ Stress test results saved to {output_dir}/")
