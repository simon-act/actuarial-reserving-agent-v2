"""
Economic Scenario Generator for Reserve Analysis
=================================================

Simulates economic scenarios and their impact on insurance reserves.
Considers factors like inflation, interest rates, and economic cycles.

Key components:
1. Inflation scenarios - Impact on claim costs
2. Interest rate scenarios - Impact on discounting
3. Economic cycle scenarios - Impact on claim frequency/severity
4. Correlation modeling - Dependencies between factors

Reference:
- Society of Actuaries (2016). "Economic Scenario Generators: A Practical Guide"
- Wilkie, A.D. (1995). "More on a Stochastic Asset Model for Actuarial Use"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Types of economic scenarios."""
    BASE = "base"
    MILD_INFLATION = "mild_inflation"
    HIGH_INFLATION = "high_inflation"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"
    RECESSION = "recession"
    BOOM = "boom"
    HARD_MARKET = "hard_market"
    SOFT_MARKET = "soft_market"
    CATASTROPHE = "catastrophe"
    CUSTOM = "custom"


@dataclass
class EconomicScenario:
    """Economic scenario parameters."""
    name: str
    scenario_type: ScenarioType
    inflation_rate: float  # Annual inflation rate
    interest_rate: float  # Risk-free rate for discounting
    claim_frequency_factor: float  # Multiplier on claim frequency
    claim_severity_factor: float  # Multiplier on claim severity
    development_speed_factor: float  # Factor for development pattern changes
    correlation_shock: float  # Shock to correlations between lines
    description: str
    probability: float  # Probability of scenario (for weighted analysis)


@dataclass
class ScenarioResult:
    """Result of scenario analysis."""
    scenario: EconomicScenario
    base_reserve: float
    adjusted_reserve: float
    discounted_reserve: float
    reserve_change: float
    reserve_change_pct: float
    by_year: pd.DataFrame


class EconomicScenarioGenerator:
    """
    Generator for economic scenarios and their impact on reserves.

    Provides:
    - Pre-defined scenario templates
    - Custom scenario creation
    - Monte Carlo simulation
    - Reserve impact analysis
    - Discounting calculations
    """

    # Pre-defined scenario templates
    SCENARIO_TEMPLATES = {
        ScenarioType.BASE: {
            'inflation_rate': 0.025,
            'interest_rate': 0.04,
            'claim_frequency_factor': 1.0,
            'claim_severity_factor': 1.0,
            'development_speed_factor': 1.0,
            'correlation_shock': 0.0,
            'probability': 0.50,
            'description': 'Base case with normal economic conditions'
        },
        ScenarioType.MILD_INFLATION: {
            'inflation_rate': 0.04,
            'interest_rate': 0.05,
            'claim_frequency_factor': 1.0,
            'claim_severity_factor': 1.05,
            'development_speed_factor': 1.0,
            'correlation_shock': 0.0,
            'probability': 0.15,
            'description': 'Moderate inflation with proportional rate increase'
        },
        ScenarioType.HIGH_INFLATION: {
            'inflation_rate': 0.08,
            'interest_rate': 0.07,
            'claim_frequency_factor': 1.0,
            'claim_severity_factor': 1.15,
            'development_speed_factor': 0.95,
            'correlation_shock': 0.1,
            'probability': 0.10,
            'description': 'High inflation with claim cost escalation'
        },
        ScenarioType.STAGFLATION: {
            'inflation_rate': 0.06,
            'interest_rate': 0.03,
            'claim_frequency_factor': 1.05,
            'claim_severity_factor': 1.12,
            'development_speed_factor': 0.90,
            'correlation_shock': 0.15,
            'probability': 0.05,
            'description': 'Stagflation with slow growth and inflation'
        },
        ScenarioType.DEFLATION: {
            'inflation_rate': -0.01,
            'interest_rate': 0.01,
            'claim_frequency_factor': 0.95,
            'claim_severity_factor': 0.95,
            'development_speed_factor': 1.05,
            'correlation_shock': 0.05,
            'probability': 0.05,
            'description': 'Deflationary environment with falling prices'
        },
        ScenarioType.RECESSION: {
            'inflation_rate': 0.01,
            'interest_rate': 0.02,
            'claim_frequency_factor': 1.10,
            'claim_severity_factor': 0.98,
            'development_speed_factor': 0.85,
            'correlation_shock': 0.20,
            'probability': 0.08,
            'description': 'Economic recession with increased claims and slower settlement'
        },
        ScenarioType.BOOM: {
            'inflation_rate': 0.03,
            'interest_rate': 0.05,
            'claim_frequency_factor': 1.05,
            'claim_severity_factor': 1.08,
            'development_speed_factor': 1.10,
            'correlation_shock': -0.05,
            'probability': 0.07,
            'description': 'Economic boom with increased activity'
        },
        ScenarioType.HARD_MARKET: {
            'inflation_rate': 0.03,
            'interest_rate': 0.045,
            'claim_frequency_factor': 1.0,
            'claim_severity_factor': 1.02,
            'development_speed_factor': 1.05,
            'correlation_shock': 0.0,
            'probability': 0.10,
            'description': 'Hard insurance market with better pricing'
        },
        ScenarioType.SOFT_MARKET: {
            'inflation_rate': 0.025,
            'interest_rate': 0.04,
            'claim_frequency_factor': 1.02,
            'claim_severity_factor': 1.05,
            'development_speed_factor': 0.98,
            'correlation_shock': 0.05,
            'probability': 0.10,
            'description': 'Soft market with competitive pricing pressure'
        },
        ScenarioType.CATASTROPHE: {
            'inflation_rate': 0.05,
            'interest_rate': 0.04,
            'claim_frequency_factor': 1.50,
            'claim_severity_factor': 1.30,
            'development_speed_factor': 0.75,
            'correlation_shock': 0.40,
            'probability': 0.02,
            'description': 'Major catastrophe event with surge in claims'
        }
    }

    def __init__(
        self,
        triangle: pd.DataFrame,
        base_reserve: float = None,
        payment_pattern: np.ndarray = None,
        random_state: int = 42
    ):
        """
        Initialize scenario generator.

        Args:
            triangle: Loss development triangle
            base_reserve: Base case reserve (if None, calculated from triangle)
            payment_pattern: Expected payment pattern (if None, derived from triangle)
            random_state: Random seed for simulations
        """
        self.triangle = triangle.copy()
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)
        self.random_state = random_state

        np.random.seed(random_state)

        # Calculate base reserve if not provided
        if base_reserve is None:
            self.base_reserve = self._calculate_chain_ladder_reserve()
        else:
            self.base_reserve = base_reserve

        # Derive payment pattern if not provided
        if payment_pattern is None:
            self.payment_pattern = self._derive_payment_pattern()
        else:
            self.payment_pattern = payment_pattern

        # Store results
        self.scenarios: List[EconomicScenario] = []
        self.results: List[ScenarioResult] = []

    def _calculate_chain_ladder_reserve(self) -> float:
        """Calculate base reserve using chain-ladder."""
        # Calculate development factors
        factors = []
        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            curr = self.triangle[col_curr].dropna()
            next_val = self.triangle[col_next].dropna()

            common = curr.index.intersection(next_val.index)
            if len(common) > 0:
                factor = next_val.loc[common].sum() / curr.loc[common].sum()
            else:
                factor = 1.0
            factors.append(factor)

        # Calculate reserves
        total_reserve = 0
        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_idx = len(row.dropna()) - 1

            # Apply remaining factors
            ultimate = latest
            for j in range(latest_idx, self.n_periods - 1):
                ultimate *= factors[j]

            reserve = ultimate - latest
            total_reserve += reserve

        return total_reserve

    def _derive_payment_pattern(self) -> np.ndarray:
        """Derive payment pattern from triangle."""
        # Calculate average incremental development
        incremental = self.triangle.diff(axis=1)
        incremental.iloc[:, 0] = self.triangle.iloc[:, 0]

        # Average pattern (normalized)
        pattern = incremental.mean() / incremental.mean().sum()
        pattern = pattern.fillna(0).values

        # Ensure pattern sums to 1
        if pattern.sum() > 0:
            pattern = pattern / pattern.sum()

        return pattern

    def create_scenario(
        self,
        scenario_type: ScenarioType,
        name: str = None,
        **overrides
    ) -> EconomicScenario:
        """
        Create an economic scenario from template or custom parameters.

        Args:
            scenario_type: Type of scenario
            name: Custom name (optional)
            **overrides: Override default parameters

        Returns:
            EconomicScenario object
        """
        if scenario_type in self.SCENARIO_TEMPLATES:
            params = self.SCENARIO_TEMPLATES[scenario_type].copy()
        else:
            params = self.SCENARIO_TEMPLATES[ScenarioType.BASE].copy()

        # Apply overrides
        params.update(overrides)

        scenario = EconomicScenario(
            name=name or scenario_type.value,
            scenario_type=scenario_type,
            inflation_rate=params['inflation_rate'],
            interest_rate=params['interest_rate'],
            claim_frequency_factor=params['claim_frequency_factor'],
            claim_severity_factor=params['claim_severity_factor'],
            development_speed_factor=params['development_speed_factor'],
            correlation_shock=params['correlation_shock'],
            description=params['description'],
            probability=params['probability']
        )

        self.scenarios.append(scenario)
        return scenario

    def get_all_predefined_scenarios(self) -> List[EconomicScenario]:
        """Get all predefined scenario templates."""
        scenarios = []
        for scenario_type in self.SCENARIO_TEMPLATES.keys():
            scenario = self.create_scenario(scenario_type)
            scenarios.append(scenario)
        return scenarios

    def apply_scenario(self, scenario: EconomicScenario) -> ScenarioResult:
        """
        Apply economic scenario to reserves.

        Args:
            scenario: Economic scenario to apply

        Returns:
            ScenarioResult with impact analysis
        """
        # Calculate adjusted reserve
        # Impact factors:
        # 1. Claim severity: directly multiplies reserve
        # 2. Claim frequency: affects total claims (simplified - multiply reserve)
        # 3. Development speed: affects timing but not total (for non-discounted)

        severity_impact = scenario.claim_severity_factor
        frequency_impact = scenario.claim_frequency_factor

        # Apply inflation to future payments
        # Simplified: assume reserves are paid out over payment pattern
        inflation_impact = self._calculate_inflation_impact(scenario.inflation_rate)

        # Total adjustment
        adjusted_reserve = self.base_reserve * severity_impact * frequency_impact * inflation_impact

        # Calculate discounted reserve
        discounted_reserve = self._calculate_discounted_reserve(
            adjusted_reserve,
            scenario.interest_rate,
            scenario.development_speed_factor
        )

        # Calculate by year
        by_year_df = self._calculate_by_year_impact(scenario)

        result = ScenarioResult(
            scenario=scenario,
            base_reserve=self.base_reserve,
            adjusted_reserve=adjusted_reserve,
            discounted_reserve=discounted_reserve,
            reserve_change=adjusted_reserve - self.base_reserve,
            reserve_change_pct=(adjusted_reserve - self.base_reserve) / self.base_reserve * 100,
            by_year=by_year_df
        )

        self.results.append(result)
        return result

    def _calculate_inflation_impact(self, inflation_rate: float) -> float:
        """
        Calculate inflation impact on reserves.

        Assumes payments are spread according to payment pattern.
        """
        impact = 0
        for t, pct in enumerate(self.payment_pattern):
            # Compound inflation for each payment period
            impact += pct * (1 + inflation_rate) ** t

        return impact

    def _calculate_discounted_reserve(
        self,
        reserve: float,
        interest_rate: float,
        speed_factor: float
    ) -> float:
        """
        Calculate discounted reserve value.

        Args:
            reserve: Undiscounted reserve
            interest_rate: Discount rate
            speed_factor: Factor affecting payment timing

        Returns:
            Present value of reserve
        """
        # Adjust payment pattern for speed
        adjusted_pattern = self._adjust_pattern_for_speed(speed_factor)

        # Discount each payment
        pv = 0
        for t, pct in enumerate(adjusted_pattern):
            discount_factor = 1 / (1 + interest_rate) ** t
            pv += reserve * pct * discount_factor

        return pv

    def _adjust_pattern_for_speed(self, speed_factor: float) -> np.ndarray:
        """
        Adjust payment pattern for development speed changes.

        speed_factor > 1: faster development (payments earlier)
        speed_factor < 1: slower development (payments later)
        """
        n = len(self.payment_pattern)
        adjusted = np.zeros(n)

        for t in range(n):
            # Shift timing
            new_t = t / speed_factor

            # Distribute to adjacent periods
            lower_t = int(new_t)
            upper_t = min(lower_t + 1, n - 1)
            weight = new_t - lower_t

            if lower_t < n:
                adjusted[lower_t] += self.payment_pattern[t] * (1 - weight)
            if upper_t < n:
                adjusted[upper_t] += self.payment_pattern[t] * weight

        # Normalize
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()

        return adjusted

    def _calculate_by_year_impact(self, scenario: EconomicScenario) -> pd.DataFrame:
        """Calculate scenario impact by accident year."""
        results = []

        # Calculate base reserves by year
        factors = []
        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            curr = self.triangle[col_curr].dropna()
            next_val = self.triangle[col_next].dropna()

            common = curr.index.intersection(next_val.index)
            if len(common) > 0:
                factor = next_val.loc[common].sum() / curr.loc[common].sum()
            else:
                factor = 1.0
            factors.append(factor)

        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_idx = len(row.dropna()) - 1

            # Base ultimate and reserve
            base_ultimate = latest
            for j in range(latest_idx, self.n_periods - 1):
                base_ultimate *= factors[j]
            base_reserve = base_ultimate - latest

            # Apply scenario factors
            adjusted_reserve = (base_reserve *
                              scenario.claim_severity_factor *
                              scenario.claim_frequency_factor)

            # Remaining development periods affect inflation impact
            remaining_periods = self.n_periods - 1 - latest_idx
            inflation_factor = (1 + scenario.inflation_rate) ** (remaining_periods / 2)
            adjusted_reserve *= inflation_factor

            results.append({
                'Accident_Year': year,
                'Base_Reserve': base_reserve,
                'Adjusted_Reserve': adjusted_reserve,
                'Change': adjusted_reserve - base_reserve,
                'Change_Pct': (adjusted_reserve - base_reserve) / base_reserve * 100 if base_reserve > 0 else 0
            })

        return pd.DataFrame(results)

    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all predefined scenarios and return comparison.

        Returns:
            DataFrame comparing all scenarios
        """
        self.scenarios = []
        self.results = []

        for scenario_type in self.SCENARIO_TEMPLATES.keys():
            scenario = self.create_scenario(scenario_type)
            self.apply_scenario(scenario)

        return self.get_comparison_table()

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison of all scenario results."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'Scenario': r.scenario.name,
                'Type': r.scenario.scenario_type.value,
                'Probability': r.scenario.probability,
                'Base_Reserve': r.base_reserve,
                'Adjusted_Reserve': r.adjusted_reserve,
                'Discounted_Reserve': r.discounted_reserve,
                'Change': r.reserve_change,
                'Change_Pct': r.reserve_change_pct,
                'Inflation': r.scenario.inflation_rate,
                'Interest_Rate': r.scenario.interest_rate
            }
            for r in self.results
        ])

    def monte_carlo_simulation(
        self,
        n_simulations: int = 10000,
        inflation_vol: float = 0.02,
        rate_vol: float = 0.01,
        severity_vol: float = 0.05
    ) -> Dict:
        """
        Run Monte Carlo simulation of economic scenarios.

        Args:
            n_simulations: Number of simulations
            inflation_vol: Volatility of inflation rate
            rate_vol: Volatility of interest rate
            severity_vol: Volatility of claim severity

        Returns:
            Dictionary with simulation results
        """
        np.random.seed(self.random_state)

        # Base parameters
        base_inflation = 0.025
        base_rate = 0.04

        reserve_dist = []

        for _ in range(n_simulations):
            # Simulate economic factors
            inflation = np.random.normal(base_inflation, inflation_vol)
            rate = np.random.normal(base_rate, rate_vol)
            severity_factor = np.random.lognormal(0, severity_vol)

            # Calculate inflation impact
            inflation_impact = self._calculate_inflation_impact(inflation)

            # Adjusted reserve
            adjusted = self.base_reserve * severity_factor * inflation_impact

            # Discount
            discounted = self._calculate_discounted_reserve(adjusted, max(0.001, rate), 1.0)

            reserve_dist.append({
                'adjusted': adjusted,
                'discounted': discounted,
                'inflation': inflation,
                'rate': rate,
                'severity_factor': severity_factor
            })

        df = pd.DataFrame(reserve_dist)

        return {
            'n_simulations': n_simulations,
            'mean_adjusted': df['adjusted'].mean(),
            'std_adjusted': df['adjusted'].std(),
            'mean_discounted': df['discounted'].mean(),
            'std_discounted': df['discounted'].std(),
            'percentiles_adjusted': {
                'P10': df['adjusted'].quantile(0.10),
                'P25': df['adjusted'].quantile(0.25),
                'P50': df['adjusted'].quantile(0.50),
                'P75': df['adjusted'].quantile(0.75),
                'P90': df['adjusted'].quantile(0.90),
                'P95': df['adjusted'].quantile(0.95),
                'P99': df['adjusted'].quantile(0.99)
            },
            'percentiles_discounted': {
                'P10': df['discounted'].quantile(0.10),
                'P25': df['discounted'].quantile(0.25),
                'P50': df['discounted'].quantile(0.50),
                'P75': df['discounted'].quantile(0.75),
                'P90': df['discounted'].quantile(0.90),
                'P95': df['discounted'].quantile(0.95),
                'P99': df['discounted'].quantile(0.99)
            },
            'distribution': df
        }

    def calculate_weighted_reserve(self) -> Dict:
        """
        Calculate probability-weighted reserve across scenarios.

        Returns:
            Dictionary with weighted statistics
        """
        if not self.results:
            self.run_all_scenarios()

        total_prob = sum(r.scenario.probability for r in self.results)

        weighted_reserve = sum(
            r.adjusted_reserve * r.scenario.probability / total_prob
            for r in self.results
        )

        weighted_discounted = sum(
            r.discounted_reserve * r.scenario.probability / total_prob
            for r in self.results
        )

        # Variance
        variance = sum(
            r.scenario.probability / total_prob * (r.adjusted_reserve - weighted_reserve) ** 2
            for r in self.results
        )

        return {
            'weighted_reserve': weighted_reserve,
            'weighted_discounted': weighted_discounted,
            'weighted_std': np.sqrt(variance),
            'weighted_cv': np.sqrt(variance) / weighted_reserve if weighted_reserve > 0 else 0,
            'range': (
                min(r.adjusted_reserve for r in self.results),
                max(r.adjusted_reserve for r in self.results)
            )
        }

    def summary(self) -> Dict:
        """Get comprehensive summary."""
        if not self.results:
            self.run_all_scenarios()

        weighted = self.calculate_weighted_reserve()

        return {
            'base_reserve': self.base_reserve,
            'n_scenarios': len(self.results),
            'weighted_reserve': weighted['weighted_reserve'],
            'weighted_discounted': weighted['weighted_discounted'],
            'weighted_std': weighted['weighted_std'],
            'reserve_range': weighted['range'],
            'worst_case': max(r.adjusted_reserve for r in self.results),
            'best_case': min(r.adjusted_reserve for r in self.results),
            'worst_scenario': max(self.results, key=lambda r: r.adjusted_reserve).scenario.name,
            'best_scenario': min(self.results, key=lambda r: r.adjusted_reserve).scenario.name
        }

    def print_report(self) -> None:
        """Print comprehensive scenario analysis report."""
        summary = self.summary()
        comparison = self.get_comparison_table()

        print("=" * 70)
        print("ECONOMIC SCENARIO ANALYSIS REPORT")
        print("=" * 70)

        print(f"\nBase Reserve: ${summary['base_reserve']:,.0f}")
        print(f"Number of Scenarios: {summary['n_scenarios']}")

        print(f"\nProbability-Weighted Results:")
        print(f"  - Expected Reserve: ${summary['weighted_reserve']:,.0f}")
        print(f"  - Expected Discounted: ${summary['weighted_discounted']:,.0f}")
        print(f"  - Standard Deviation: ${summary['weighted_std']:,.0f}")

        print(f"\nScenario Range:")
        print(f"  - Best Case ({summary['best_scenario']}): ${summary['best_case']:,.0f}")
        print(f"  - Worst Case ({summary['worst_scenario']}): ${summary['worst_case']:,.0f}")

        print(f"\nScenario Comparison:")
        print("-" * 70)
        print(comparison[['Scenario', 'Probability', 'Adjusted_Reserve', 'Change_Pct']].to_string(index=False))
