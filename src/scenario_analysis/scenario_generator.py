"""
Scenario Generator for Reserve Analysis

Generates comprehensive scenarios for reserve stress testing including:
- Deterministic scenarios (specific factor changes)
- Stochastic scenarios (probability-weighted)
- Historical scenarios (based on observed patterns)
- Regulatory scenarios (Solvency II, etc.)

References:
- Solvency II Technical Specifications
- IAIS Insurance Capital Standard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scenario:
    """
    Complete scenario definition.

    Attributes:
        name: Scenario identifier
        description: Detailed description
        probability: Probability weight (for stochastic scenarios)
        factor_multipliers: Dict of period -> multiplier
        loss_ratio_multiplier: Multiplier for expected loss ratios
        severity_multiplier: Multiplier for claim severity
        frequency_multiplier: Multiplier for claim frequency
    """
    name: str
    description: str
    probability: float = 1.0
    factor_multipliers: Dict[str, float] = field(default_factory=dict)
    loss_ratio_multiplier: float = 1.0
    severity_multiplier: float = 1.0
    frequency_multiplier: float = 1.0
    immature_year_multiplier: float = 1.0


class ScenarioGenerator:
    """
    Generate comprehensive scenarios for reserve analysis.

    Provides pre-built scenario sets and tools for custom scenario creation.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        base_factors: pd.Series,
        earned_premium: Optional[pd.Series] = None
    ):
        """
        Initialize scenario generator.

        Args:
            triangle: Loss development triangle
            base_factors: Base development factors
            earned_premium: Earned premium by year (for LR scenarios)
        """
        self.triangle = triangle.copy()
        self.base_factors = base_factors.copy()
        self.earned_premium = earned_premium

        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Classify maturity
        self._classify_years()

    def _classify_years(self):
        """Classify years by maturity level."""
        self.year_maturity = {}

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            n_observed = row.dropna().shape[0]
            pct_complete = n_observed / self.n_periods

            if pct_complete >= 0.80:
                maturity = 'mature'
            elif pct_complete >= 0.50:
                maturity = 'developing'
            else:
                maturity = 'immature'

            self.year_maturity[year] = {
                'pct_complete': pct_complete,
                'maturity': maturity,
                'n_observed': n_observed
            }

        self.immature_years = [
            y for y, m in self.year_maturity.items() if m['maturity'] == 'immature'
        ]
        self.developing_years = [
            y for y, m in self.year_maturity.items() if m['maturity'] == 'developing'
        ]
        self.mature_years = [
            y for y, m in self.year_maturity.items() if m['maturity'] == 'mature'
        ]

    def generate_base_scenario(self) -> Scenario:
        """Generate base (no stress) scenario."""
        return Scenario(
            name='Base',
            description='Base case with no stress applied',
            factor_multipliers={p: 1.0 for p in self.base_factors.index}
        )

    def generate_adverse_scenarios(self) -> List[Scenario]:
        """
        Generate a set of adverse scenarios.

        Returns:
            List of adverse scenarios
        """
        scenarios = []

        # Uniform adverse
        for shock in [0.05, 0.10, 0.15, 0.20, 0.25]:
            scenarios.append(Scenario(
                name=f'Adverse_{int(shock*100)}pct',
                description=f'Uniform {shock:.0%} increase in all factors',
                factor_multipliers={p: 1 + shock for p in self.base_factors.index}
            ))

        # Early development adverse
        early_periods = list(self.base_factors.index[:3])
        for shock in [0.10, 0.20]:
            scenarios.append(Scenario(
                name=f'EarlyAdverse_{int(shock*100)}pct',
                description=f'{shock:.0%} increase in first 3 development periods',
                factor_multipliers={
                    p: (1 + shock if p in early_periods else 1.0)
                    for p in self.base_factors.index
                }
            ))

        # Immature year adverse
        scenarios.append(Scenario(
            name='ImmatureYearAdverse_20pct',
            description='20% additional development for immature years',
            immature_year_multiplier=1.20
        ))

        scenarios.append(Scenario(
            name='ImmatureYearAdverse_30pct',
            description='30% additional development for immature years',
            immature_year_multiplier=1.30
        ))

        return scenarios

    def generate_favorable_scenarios(self) -> List[Scenario]:
        """
        Generate favorable scenarios.

        Returns:
            List of favorable scenarios
        """
        scenarios = []

        for shock in [-0.05, -0.10, -0.15]:
            scenarios.append(Scenario(
                name=f'Favorable_{int(abs(shock)*100)}pct',
                description=f'Uniform {abs(shock):.0%} decrease in all factors',
                factor_multipliers={p: 1 + shock for p in self.base_factors.index}
            ))

        return scenarios

    def generate_regulatory_scenarios(self) -> List[Scenario]:
        """
        Generate regulatory-style scenarios.

        Based on Solvency II, IFRS 17, and other standards.

        Returns:
            List of regulatory scenarios
        """
        scenarios = []

        # Solvency II Reserve Risk (approx 99.5% VaR)
        # Property: ~10-15%, Liability: ~15-25%
        scenarios.append(Scenario(
            name='SII_Reserve_Risk_Property',
            description='Solvency II 1-in-200 reserve risk (property)',
            probability=0.005,
            factor_multipliers={p: 1.12 for p in self.base_factors.index}
        ))

        scenarios.append(Scenario(
            name='SII_Reserve_Risk_Casualty',
            description='Solvency II 1-in-200 reserve risk (casualty)',
            probability=0.005,
            factor_multipliers={p: 1.20 for p in self.base_factors.index}
        ))

        # IFRS 17 Risk Adjustment scenarios
        scenarios.append(Scenario(
            name='IFRS17_Risk_Adj_High',
            description='IFRS 17 high confidence level (85%)',
            probability=0.15,
            factor_multipliers={p: 1.08 for p in self.base_factors.index}
        ))

        # Extreme but plausible
        scenarios.append(Scenario(
            name='Extreme_Adverse',
            description='Extreme adverse - 1-in-500 event',
            probability=0.002,
            factor_multipliers={p: 1.30 for p in self.base_factors.index}
        ))

        return scenarios

    def generate_historical_scenarios(self) -> List[Scenario]:
        """
        Generate scenarios based on historical factor volatility.

        Uses observed factor ranges to create realistic scenarios.

        Returns:
            List of historically-calibrated scenarios
        """
        scenarios = []

        # Calculate historical factor statistics
        factor_stats = {}
        for col_idx, col in enumerate(self.base_factors.index):
            col_curr = self.triangle.columns[col_idx]
            col_next = self.triangle.columns[col_idx + 1]

            factors = []
            for year in self.triangle.index:
                curr = self.triangle.loc[year, col_curr]
                next_val = self.triangle.loc[year, col_next]
                if pd.notna(curr) and pd.notna(next_val) and curr > 0:
                    factors.append(next_val / curr)

            if factors:
                factor_stats[col] = {
                    'mean': np.mean(factors),
                    'std': np.std(factors),
                    'max': np.max(factors),
                    'min': np.min(factors),
                    'p90': np.percentile(factors, 90),
                    'p10': np.percentile(factors, 10)
                }

        # Historical worst case (all factors at max observed)
        if factor_stats:
            scenarios.append(Scenario(
                name='Historical_Max',
                description='All factors at historically observed maximum',
                factor_multipliers={
                    p: factor_stats[p]['max'] / self.base_factors[p]
                    for p in factor_stats.keys()
                }
            ))

            # Historical 90th percentile
            scenarios.append(Scenario(
                name='Historical_P90',
                description='All factors at 90th percentile of historical',
                factor_multipliers={
                    p: factor_stats[p]['p90'] / self.base_factors[p]
                    for p in factor_stats.keys()
                }
            ))

            # Historical best case (for context)
            scenarios.append(Scenario(
                name='Historical_Min',
                description='All factors at historically observed minimum',
                factor_multipliers={
                    p: factor_stats[p]['min'] / self.base_factors[p]
                    for p in factor_stats.keys()
                }
            ))

        return scenarios

    def generate_inflation_scenarios(self) -> List[Scenario]:
        """
        Generate inflation shock scenarios.

        Assumes inflation impacts future development more than past.

        Returns:
            List of inflation scenarios
        """
        scenarios = []

        # Moderate inflation shock
        # Later periods get larger shocks (compounding effect)
        n_periods = len(self.base_factors)
        moderate_multipliers = {}
        severe_multipliers = {}

        for i, period in enumerate(self.base_factors.index):
            # Moderate: 2% per period cumulative
            moderate_multipliers[period] = 1 + 0.02 * (i + 1)
            # Severe: 5% per period cumulative
            severe_multipliers[period] = 1 + 0.05 * (i + 1)

        scenarios.append(Scenario(
            name='Inflation_Moderate',
            description='Moderate inflation: 2% cumulative per period',
            factor_multipliers=moderate_multipliers
        ))

        scenarios.append(Scenario(
            name='Inflation_Severe',
            description='Severe inflation: 5% cumulative per period',
            factor_multipliers=severe_multipliers
        ))

        return scenarios

    def generate_all_scenarios(self) -> List[Scenario]:
        """
        Generate comprehensive set of all scenarios.

        Returns:
            List of all scenarios
        """
        all_scenarios = [self.generate_base_scenario()]
        all_scenarios.extend(self.generate_adverse_scenarios())
        all_scenarios.extend(self.generate_favorable_scenarios())
        all_scenarios.extend(self.generate_regulatory_scenarios())
        all_scenarios.extend(self.generate_historical_scenarios())
        all_scenarios.extend(self.generate_inflation_scenarios())

        return all_scenarios

    def apply_scenario(
        self,
        scenario: Scenario
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply a scenario to get stressed factors and reserves.

        Args:
            scenario: Scenario to apply

        Returns:
            Tuple of (stressed_factors, reserve_DataFrame)
        """
        # Apply factor multipliers
        stressed_factors = self.base_factors.copy()
        for period, mult in scenario.factor_multipliers.items():
            if period in stressed_factors.index:
                stressed_factors[period] *= mult

        # Calculate reserves
        reserves = self._calculate_reserves(
            stressed_factors,
            scenario.immature_year_multiplier
        )

        return stressed_factors, reserves

    def _calculate_reserves(
        self,
        factors: pd.Series,
        immature_mult: float = 1.0
    ) -> pd.DataFrame:
        """Calculate reserves with given factors."""
        results = []

        # Cumulative factors
        cum_factors = pd.Series(index=factors.index, dtype=float)
        cum = 1.0
        for period in reversed(factors.index):
            cum *= factors[period]
            cum_factors[period] = cum

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_age = row.dropna().index[-1]

            if latest_age in cum_factors.index:
                cdf = cum_factors[latest_age]
            else:
                cdf = 1.0

            # Apply immature year multiplier if applicable
            if self.year_maturity[year]['maturity'] == 'immature':
                cdf *= immature_mult

            ultimate = latest * cdf
            reserve = ultimate - latest

            results.append({
                'Year': year,
                'Latest': latest,
                'CDF': cdf,
                'Ultimate': ultimate,
                'Reserve': reserve,
                'Maturity': self.year_maturity[year]['maturity']
            })

        return pd.DataFrame(results).set_index('Year')

    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all scenarios and summarize results.

        Returns:
            DataFrame with scenario results
        """
        scenarios = self.generate_all_scenarios()
        results = []

        base_factors, base_reserves = self.apply_scenario(scenarios[0])
        base_total = base_reserves['Reserve'].sum()

        for scenario in scenarios:
            _, reserves = self.apply_scenario(scenario)
            total = reserves['Reserve'].sum()

            results.append({
                'Scenario': scenario.name,
                'Description': scenario.description,
                'Probability': scenario.probability,
                'Total_Reserve': total,
                'Impact': total - base_total,
                'Impact_Pct': (total - base_total) / base_total if base_total != 0 else 0
            })

        return pd.DataFrame(results)

    def print_summary(self):
        """Print scenario summary."""
        print("\n" + "="*80)
        print("SCENARIO ANALYSIS SUMMARY")
        print("="*80)

        results = self.run_all_scenarios()

        print("\n📊 YEAR MATURITY CLASSIFICATION:")
        print("-"*60)
        print(f"Mature years:     {len(self.mature_years)}")
        print(f"Developing years: {len(self.developing_years)}")
        print(f"Immature years:   {len(self.immature_years)}")

        print("\n\n📊 SCENARIO RESULTS:")
        print("-"*80)
        print(f"{'Scenario':<30} {'Reserve':>15} {'Impact':>15} {'%':>10}")
        print("-"*70)

        for _, row in results.iterrows():
            print(f"{row['Scenario']:<30} ${row['Total_Reserve']:>14,.0f} "
                  f"${row['Impact']:>14,.0f} {row['Impact_Pct']:>9.1%}")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save scenario results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.run_all_scenarios().to_csv(
            output_dir / "scenario_results.csv", index=False
        )

        print(f"✅ Scenario results saved to {output_dir}/")
