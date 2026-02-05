"""
Tail Risk Analysis for Immature Accident Years

Focuses on the risk associated with immature accident years where
reserve estimates are most uncertain.

Key analyses:
- Maturity assessment
- Tail development uncertainty
- Sensitivity to assumptions
- Risk measures for immature cohorts

References:
- Wüthrich, M.V. (2018). "Machine Learning in Insurance"
- Taylor, G. (2000). "Loss Reserving: An Actuarial Perspective"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TailRiskMetrics:
    """
    Tail risk metrics for an accident year.

    Attributes:
        accident_year: The year being analyzed
        maturity_pct: Percentage of development complete
        remaining_development: Estimated additional development
        tail_uncertainty: Coefficient of variation of tail estimate
        risk_contribution: Contribution to total reserve risk
    """
    accident_year: int
    maturity_pct: float
    remaining_development: float
    tail_uncertainty: float
    risk_contribution: float
    reserve_range: Tuple[float, float]


class TailRiskAnalyzer:
    """
    Analyze tail risk for immature accident years.

    Provides detailed analysis of where reserve uncertainty
    is concentrated and what drives tail risk.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        development_factors: pd.Series,
        factor_volatility: Optional[pd.Series] = None
    ):
        """
        Initialize tail risk analyzer.

        Args:
            triangle: Loss development triangle
            development_factors: Selected development factors
            factor_volatility: CV of factors by period (optional)
        """
        self.triangle = triangle.copy()
        self.development_factors = development_factors.copy()

        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Calculate or use provided factor volatility
        if factor_volatility is not None:
            self.factor_cv = factor_volatility
        else:
            self._estimate_factor_volatility()

        # Results
        self.maturity_analysis = None
        self.tail_risk_metrics = None
        self.risk_decomposition = None

    def _estimate_factor_volatility(self):
        """Estimate factor volatility from triangle."""
        cvs = {}

        for col_idx in range(self.n_periods - 1):
            col_curr = self.triangle.columns[col_idx]
            col_next = self.triangle.columns[col_idx + 1]

            factors = []
            for year in self.triangle.index:
                curr = self.triangle.loc[year, col_curr]
                next_val = self.triangle.loc[year, col_next]
                if pd.notna(curr) and pd.notna(next_val) and curr > 0:
                    factors.append(next_val / curr)

            if len(factors) >= 2:
                cvs[col_curr] = np.std(factors) / np.mean(factors)
            else:
                cvs[col_curr] = 0.05  # Default CV

        self.factor_cv = pd.Series(cvs)

    def fit(self) -> 'TailRiskAnalyzer':
        """
        Run tail risk analysis.

        Returns:
            self for method chaining
        """
        self._analyze_maturity()
        self._calculate_tail_risk_metrics()
        self._decompose_risk()

        return self

    def _analyze_maturity(self):
        """Analyze maturity level of each accident year."""
        maturity = []

        # Calculate cumulative factors
        cum_factors = pd.Series(index=self.development_factors.index, dtype=float)
        cum = 1.0
        for period in reversed(self.development_factors.index):
            cum *= self.development_factors[period]
            cum_factors[period] = cum

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_age = row.dropna().index[-1]
            n_observed = row.dropna().shape[0]

            # Maturity as % of periods observed
            pct_periods = n_observed / self.n_periods

            # Maturity as % of ultimate (using CDF)
            if latest_age in cum_factors.index:
                cdf = cum_factors[latest_age]
                pct_ultimate = 1 / cdf
            else:
                pct_ultimate = 1.0

            # Remaining development
            remaining_dev = cdf - 1 if latest_age in cum_factors.index else 0

            maturity.append({
                'Year': year,
                'Latest_Value': latest,
                'Latest_Age': latest_age,
                'N_Observed': n_observed,
                'Pct_Periods': pct_periods,
                'Pct_Ultimate': pct_ultimate,
                'CDF': cdf if latest_age in cum_factors.index else 1.0,
                'Remaining_Dev_Factor': remaining_dev,
                'Maturity_Category': (
                    'Mature' if pct_ultimate >= 0.95 else
                    'Developing' if pct_ultimate >= 0.75 else
                    'Immature' if pct_ultimate >= 0.50 else
                    'Very_Immature'
                )
            })

        self.maturity_analysis = pd.DataFrame(maturity).set_index('Year')

    def _calculate_tail_risk_metrics(self):
        """Calculate tail risk metrics for each year."""
        metrics = []

        for year in self.triangle.index:
            mat = self.maturity_analysis.loc[year]
            latest = mat['Latest_Value']
            latest_age = mat['Latest_Age']
            cdf = mat['CDF']

            # Calculate uncertainty in remaining development
            # Var(Reserve) ≈ Ultimate² × Σ(σ²_j / f²_j)
            remaining_var = 0
            latest_age_idx = list(self.triangle.columns).index(latest_age)

            for j in range(latest_age_idx, self.n_periods - 1):
                period = self.development_factors.index[j]
                f_j = self.development_factors[period]
                cv_j = self.factor_cv.get(period, 0.05)

                remaining_var += cv_j ** 2

            # Estimate ultimate
            ultimate = latest * cdf
            reserve = ultimate - latest

            # Standard error of reserve
            se_reserve = ultimate * np.sqrt(remaining_var) if remaining_var > 0 else 0

            # Coefficient of variation
            cv_reserve = se_reserve / reserve if reserve > 0 else 0

            # Reserve range (approximate 95% CI)
            if reserve > 0 and cv_reserve > 0:
                sigma = np.sqrt(np.log(1 + cv_reserve**2))
                mu = np.log(reserve) - 0.5 * sigma**2
                lower = np.exp(mu - 1.96 * sigma)
                upper = np.exp(mu + 1.96 * sigma)
            else:
                lower, upper = reserve, reserve

            metrics.append(TailRiskMetrics(
                accident_year=year,
                maturity_pct=mat['Pct_Ultimate'],
                remaining_development=mat['Remaining_Dev_Factor'],
                tail_uncertainty=cv_reserve,
                risk_contribution=0,  # Will be calculated in decomposition
                reserve_range=(lower, upper)
            ))

        self.tail_risk_metrics = metrics

        # Convert to DataFrame for easier access
        self.tail_risk_df = pd.DataFrame([{
            'Year': m.accident_year,
            'Maturity_Pct': m.maturity_pct,
            'Remaining_Dev': m.remaining_development,
            'Uncertainty_CV': m.tail_uncertainty,
            'Reserve_Lower': m.reserve_range[0],
            'Reserve_Upper': m.reserve_range[1],
            'Reserve_Width': m.reserve_range[1] - m.reserve_range[0]
        } for m in metrics]).set_index('Year')

    def _decompose_risk(self):
        """Decompose total reserve risk by accident year."""
        # Calculate variance contribution from each year
        total_var = 0
        year_vars = {}

        for year in self.triangle.index:
            mat = self.maturity_analysis.loc[year]
            latest = mat['Latest_Value']
            cdf = mat['CDF']
            ultimate = latest * cdf

            # Approximate variance
            cv = self.tail_risk_df.loc[year, 'Uncertainty_CV']
            var = (ultimate * cv) ** 2

            year_vars[year] = var
            total_var += var

        # Calculate risk contributions
        self.risk_decomposition = pd.DataFrame({
            'Year': list(year_vars.keys()),
            'Variance': list(year_vars.values()),
            'Std_Dev': [np.sqrt(v) for v in year_vars.values()],
            'Risk_Contribution': [v / total_var if total_var > 0 else 0 for v in year_vars.values()],
            'Maturity': [self.maturity_analysis.loc[y, 'Maturity_Category'] for y in year_vars.keys()]
        }).set_index('Year')

        # Update metrics with risk contributions
        for i, metric in enumerate(self.tail_risk_metrics):
            year = metric.accident_year
            metric.risk_contribution = self.risk_decomposition.loc[year, 'Risk_Contribution']

    def get_immature_year_analysis(self) -> pd.DataFrame:
        """
        Get detailed analysis of immature years.

        Returns:
            DataFrame with immature year details
        """
        immature = self.maturity_analysis[
            self.maturity_analysis['Maturity_Category'].isin(['Immature', 'Very_Immature'])
        ].copy()

        if len(immature) == 0:
            return pd.DataFrame()

        # Add risk metrics
        immature['Uncertainty_CV'] = self.tail_risk_df.loc[immature.index, 'Uncertainty_CV']
        immature['Risk_Contribution'] = self.risk_decomposition.loc[immature.index, 'Risk_Contribution']
        immature['Reserve_Width'] = self.tail_risk_df.loc[immature.index, 'Reserve_Width']

        return immature

    def get_tail_sensitivity(
        self,
        shock_range: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of reserves to tail factor changes.

        Args:
            shock_range: Range of shocks to test

        Returns:
            DataFrame with sensitivity results
        """
        if shock_range is None:
            shock_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        results = []

        for shock in shock_range:
            # Apply shock to remaining development factors
            shocked_total = 0

            for year in self.triangle.index:
                mat = self.maturity_analysis.loc[year]
                latest = mat['Latest_Value']
                cdf = mat['CDF']

                # Shock applies to remaining development only
                remaining = cdf - 1
                shocked_cdf = 1 + remaining * (1 + shock)

                ultimate = latest * shocked_cdf
                reserve = ultimate - latest
                shocked_total += reserve

            results.append({
                'Shock_Pct': shock,
                'Total_Reserve': shocked_total
            })

        df = pd.DataFrame(results)

        # Calculate impacts relative to base
        base = df.loc[df['Shock_Pct'] == 0, 'Total_Reserve'].iloc[0]
        df['Impact'] = df['Total_Reserve'] - base
        df['Impact_Pct'] = df['Impact'] / base if base != 0 else 0

        return df

    def get_risk_concentration(self) -> Dict:
        """
        Analyze where risk is concentrated.

        Returns:
            Dictionary with risk concentration metrics
        """
        # Risk by maturity category
        by_maturity = self.risk_decomposition.groupby('Maturity')['Risk_Contribution'].sum()

        # Top risk years
        top_risk = self.risk_decomposition.nlargest(3, 'Risk_Contribution')

        # Concentration metrics
        total_risk_top3 = top_risk['Risk_Contribution'].sum()
        total_risk_immature = by_maturity.get('Immature', 0) + by_maturity.get('Very_Immature', 0)

        return {
            'risk_by_maturity': by_maturity.to_dict(),
            'top_3_risk_years': top_risk.index.tolist(),
            'top_3_risk_contribution': total_risk_top3,
            'immature_risk_contribution': total_risk_immature,
            'risk_concentrated_in_immature': total_risk_immature > 0.5
        }

    def summary(self) -> Dict:
        """Get summary of tail risk analysis."""
        concentration = self.get_risk_concentration()

        return {
            'n_years': self.n_years,
            'n_immature': len(self.get_immature_year_analysis()),
            'risk_concentration': concentration,
            'max_uncertainty_cv': self.tail_risk_df['Uncertainty_CV'].max(),
            'avg_uncertainty_cv_immature': (
                self.tail_risk_df.loc[
                    self.maturity_analysis['Maturity_Category'].isin(['Immature', 'Very_Immature'])
                ]['Uncertainty_CV'].mean()
                if len(self.get_immature_year_analysis()) > 0 else 0
            )
        }

    def print_summary(self):
        """Print formatted tail risk summary."""
        print("\n" + "="*80)
        print("TAIL RISK ANALYSIS")
        print("="*80)

        # Maturity overview
        print("\n📊 MATURITY OVERVIEW:")
        print("-"*60)
        maturity_counts = self.maturity_analysis['Maturity_Category'].value_counts()
        for cat, count in maturity_counts.items():
            print(f"  {cat}: {count} years")

        # Immature year details
        print("\n\n📊 IMMATURE YEAR ANALYSIS:")
        print("-"*60)
        immature = self.get_immature_year_analysis()
        if len(immature) > 0:
            print(immature[['Pct_Ultimate', 'Remaining_Dev_Factor', 'Uncertainty_CV', 'Risk_Contribution']].round(3).to_string())
        else:
            print("No immature years identified")

        # Risk concentration
        print("\n\n📊 RISK CONCENTRATION:")
        print("-"*60)
        concentration = self.get_risk_concentration()
        print("By Maturity Category:")
        for cat, risk in concentration['risk_by_maturity'].items():
            print(f"  {cat}: {risk:.1%}")

        print(f"\nTop 3 risk years: {concentration['top_3_risk_years']}")
        print(f"Top 3 contribution: {concentration['top_3_risk_contribution']:.1%}")
        print(f"Immature contribution: {concentration['immature_risk_contribution']:.1%}")

        # Tail sensitivity
        print("\n\n📊 TAIL SENSITIVITY:")
        print("-"*60)
        sensitivity = self.get_tail_sensitivity([0, 0.10, 0.20, 0.30])
        print(f"{'Shock':<10} {'Reserve':>15} {'Impact':>15} {'%':>10}")
        for _, row in sensitivity.iterrows():
            print(f"{row['Shock_Pct']:.0%:<10} ${row['Total_Reserve']:>14,.0f} "
                  f"${row['Impact']:>14,.0f} {row['Impact_Pct']:>9.1%}")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save tail risk results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.maturity_analysis.to_csv(output_dir / "maturity_analysis.csv")
        self.tail_risk_df.to_csv(output_dir / "tail_risk_metrics.csv")
        self.risk_decomposition.to_csv(output_dir / "risk_decomposition.csv")
        self.get_tail_sensitivity().to_csv(output_dir / "tail_sensitivity.csv", index=False)

        print(f"✅ Tail risk analysis saved to {output_dir}/")
