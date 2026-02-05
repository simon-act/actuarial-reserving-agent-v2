"""
Cape Cod Method (Stanard-Bühlmann)

The Cape Cod method estimates reserves using an implicit expected loss ratio
derived from the data itself, rather than requiring external ELR inputs.

Formula:
1. Calculate used-up premium: Used_Premium = Premium * Percent_Reported
2. Estimate overall ELR: ELR = Sum(Reported) / Sum(Used_Premium)
3. Expected Ultimate = Premium * ELR
4. Reserve = Expected_Unreported = Expected_Ultimate * (1 - Percent_Reported)

This method is a blend between Chain-Ladder and Bornhuetter-Ferguson.

References:
- Stanard, J.N. (1985). "A Simulation Test of Prediction Errors of Loss Reserve Estimation Techniques"
- Bühlmann, H. et al. (1980). "Estimation of IBNR Reserves by the Methods Chain Ladder, Cape Cod and Complementary Loss Ratio"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path


class CapeCod:
    """
    Cape Cod (Stanard-Bühlmann) reserving method.

    Key characteristics:
    - Self-calibrating: Estimates ELR from data
    - Uses "used-up premium" concept
    - More stable than pure chain-ladder for immature years
    - Single ELR across all years (unlike BF which can vary by year)

    Variants:
    - Standard: Single ELR for all years
    - Generalized: Different ELRs by year group or trend
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        earned_premium: pd.Series
    ):
        """
        Initialize Cape Cod model.

        Args:
            triangle: Cumulative loss development triangle
            earned_premium: Earned premium by accident year
        """
        self.triangle = triangle.copy()
        self.earned_premium = earned_premium.copy()

        # Align indices
        common_years = self.triangle.index.intersection(self.earned_premium.index)
        self.triangle = self.triangle.loc[common_years]
        self.earned_premium = self.earned_premium.loc[common_years]

        self.n_years = len(self.triangle)
        self.n_periods = len(self.triangle.columns)

        # Results
        self.development_factors = None
        self.cumulative_factors = None
        self.percent_reported = None
        self.used_up_premium = None
        self.cape_cod_elr = None
        self.cc_ultimate = None
        self.cc_reserves = None

    def fit(self) -> 'CapeCod':
        """
        Fit the Cape Cod model.

        Returns:
            self for method chaining
        """
        self._calculate_development_factors()
        self._calculate_cumulative_factors()
        self._calculate_percent_reported()
        self._calculate_used_up_premium()
        self._calculate_cape_cod_elr()
        self._calculate_reserves()

        return self

    def _calculate_development_factors(self):
        """Calculate volume-weighted development factors."""
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

        self.development_factors = pd.Series(
            factors,
            index=self.triangle.columns[:-1],
            name='LDF'
        )

    def _calculate_cumulative_factors(self):
        """Calculate cumulative development factors (CDF)."""
        cum_factors = pd.Series(
            index=self.triangle.columns[:-1],
            dtype=float,
            name='CDF'
        )

        cum_factor = 1.0
        for period in reversed(self.development_factors.index):
            cum_factor *= self.development_factors[period]
            cum_factors[period] = cum_factor

        self.cumulative_factors = cum_factors

    def _calculate_percent_reported(self):
        """Calculate percent reported (1/CDF)."""
        self.percent_reported = 1.0 / self.cumulative_factors
        self.percent_reported.name = 'Pct_Reported'

    def _calculate_used_up_premium(self):
        """
        Calculate "used-up" premium for each year.

        Used-Up Premium = Earned Premium * Percent Reported

        This represents the portion of premium that has been
        "used" to generate the reported losses.
        """
        used_up = {}

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest_age = row.dropna().index[-1]

            if latest_age in self.percent_reported.index:
                pct_reported = self.percent_reported[latest_age]
            else:
                pct_reported = 1.0

            used_up[year] = self.earned_premium[year] * pct_reported

        self.used_up_premium = pd.Series(used_up, name='Used_Up_Premium')

    def _calculate_cape_cod_elr(self):
        """
        Calculate the Cape Cod expected loss ratio.

        ELR = Sum(Reported Losses) / Sum(Used-Up Premium)

        This is the implicit loss ratio that balances across all years.
        """
        total_reported = 0
        for year in self.triangle.index:
            row = self.triangle.loc[year]
            total_reported += row.dropna().iloc[-1]

        total_used_up = self.used_up_premium.sum()

        self.cape_cod_elr = total_reported / total_used_up if total_used_up > 0 else 0

    def _calculate_reserves(self):
        """
        Calculate Cape Cod reserves.

        For each year:
        - Expected Ultimate = Premium * ELR
        - Expected Unreported = Expected Ultimate * (1 - Pct Reported)
        - Reserve = Expected Unreported
        """
        results = []

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_age = row.dropna().index[-1]

            # Get percent reported
            if latest_age in self.percent_reported.index:
                pct_reported = self.percent_reported[latest_age]
            else:
                pct_reported = 1.0

            pct_unreported = 1 - pct_reported
            premium = self.earned_premium[year]

            # Cape Cod calculation
            expected_ultimate = premium * self.cape_cod_elr
            expected_unreported = expected_ultimate * pct_unreported
            cc_ultimate = latest + expected_unreported
            cc_reserve = cc_ultimate - latest

            # Also calculate chain-ladder for comparison
            if latest_age in self.cumulative_factors.index:
                cl_ultimate = latest * self.cumulative_factors[latest_age]
            else:
                cl_ultimate = latest
            cl_reserve = cl_ultimate - latest

            # Implied loss ratio
            implied_elr = cc_ultimate / premium if premium > 0 else 0

            results.append({
                'Accident_Year': year,
                'Reported': latest,
                'Latest_Age': latest_age,
                'Pct_Reported': pct_reported,
                'Premium': premium,
                'Used_Up_Premium': self.used_up_premium[year],
                'Expected_Ultimate': expected_ultimate,
                'Expected_Unreported': expected_unreported,
                'CC_Ultimate': cc_ultimate,
                'CC_Reserve': cc_reserve,
                'CL_Ultimate': cl_ultimate,
                'CL_Reserve': cl_reserve,
                'Implied_ELR': implied_elr,
                'CC_vs_CL_Diff': cc_reserve - cl_reserve
            })

        self.cc_results = pd.DataFrame(results).set_index('Accident_Year')
        self.cc_ultimate = self.cc_results['CC_Ultimate']
        self.cc_reserves = self.cc_results['CC_Reserve']

    def fit_generalized(
        self,
        year_groups: Optional[Dict[str, List[int]]] = None,
        trend: Optional[float] = None
    ) -> 'CapeCod':
        """
        Fit Generalized Cape Cod with different ELRs by group or trend.

        Args:
            year_groups: Dict mapping group names to list of years
                        e.g., {'early': [2008,2009,2010], 'recent': [2020,2021,2022]}
            trend: Annual trend to apply to ELR (e.g., 0.02 for +2%/year)

        Returns:
            self for method chaining
        """
        # First fit standard model to get factors
        self._calculate_development_factors()
        self._calculate_cumulative_factors()
        self._calculate_percent_reported()
        self._calculate_used_up_premium()

        if year_groups:
            # Calculate separate ELRs for each group
            group_elrs = {}
            for group_name, years in year_groups.items():
                group_reported = 0
                group_used_up = 0

                for year in years:
                    if year in self.triangle.index:
                        row = self.triangle.loc[year]
                        group_reported += row.dropna().iloc[-1]
                        group_used_up += self.used_up_premium[year]

                group_elrs[group_name] = (
                    group_reported / group_used_up if group_used_up > 0 else 0
                )

            self.group_elrs = group_elrs

            # Map years to group ELRs
            year_to_elr = {}
            for group_name, years in year_groups.items():
                for year in years:
                    year_to_elr[year] = group_elrs[group_name]

            # Fill any missing years with overall ELR
            self._calculate_cape_cod_elr()
            for year in self.triangle.index:
                if year not in year_to_elr:
                    year_to_elr[year] = self.cape_cod_elr

            self.year_elrs = pd.Series(year_to_elr)

        elif trend:
            # Apply trend to ELR
            self._calculate_cape_cod_elr()
            base_year = self.triangle.index.min()
            year_elrs = {}

            for year in self.triangle.index:
                years_from_base = year - base_year
                year_elrs[year] = self.cape_cod_elr * (1 + trend) ** years_from_base

            self.year_elrs = pd.Series(year_elrs)

        else:
            # Standard Cape Cod
            self._calculate_cape_cod_elr()
            self.year_elrs = pd.Series(
                {year: self.cape_cod_elr for year in self.triangle.index}
            )

        # Calculate reserves with year-specific ELRs
        self._calculate_reserves_generalized()

        return self

    def _calculate_reserves_generalized(self):
        """Calculate reserves using year-specific ELRs."""
        results = []

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            latest = row.dropna().iloc[-1]
            latest_age = row.dropna().index[-1]

            if latest_age in self.percent_reported.index:
                pct_reported = self.percent_reported[latest_age]
            else:
                pct_reported = 1.0

            pct_unreported = 1 - pct_reported
            premium = self.earned_premium[year]
            year_elr = self.year_elrs[year]

            expected_ultimate = premium * year_elr
            expected_unreported = expected_ultimate * pct_unreported
            cc_ultimate = latest + expected_unreported
            cc_reserve = cc_ultimate - latest

            # Chain-ladder comparison
            if latest_age in self.cumulative_factors.index:
                cl_ultimate = latest * self.cumulative_factors[latest_age]
            else:
                cl_ultimate = latest
            cl_reserve = cl_ultimate - latest

            results.append({
                'Accident_Year': year,
                'Reported': latest,
                'Latest_Age': latest_age,
                'Pct_Reported': pct_reported,
                'Premium': premium,
                'Year_ELR': year_elr,
                'Expected_Ultimate': expected_ultimate,
                'CC_Ultimate': cc_ultimate,
                'CC_Reserve': cc_reserve,
                'CL_Ultimate': cl_ultimate,
                'CL_Reserve': cl_reserve,
                'CC_vs_CL_Diff': cc_reserve - cl_reserve
            })

        self.cc_results = pd.DataFrame(results).set_index('Accident_Year')
        self.cc_ultimate = self.cc_results['CC_Ultimate']
        self.cc_reserves = self.cc_results['CC_Reserve']

    def get_comparison(self) -> pd.DataFrame:
        """Get comparison of Cape Cod vs Chain-Ladder."""
        return pd.DataFrame({
            'Reported': self.cc_results['Reported'],
            'CC_Ultimate': self.cc_ultimate,
            'CL_Ultimate': self.cc_results['CL_Ultimate'],
            'CC_Reserve': self.cc_reserves,
            'CL_Reserve': self.cc_results['CL_Reserve'],
            'Difference': self.cc_reserves - self.cc_results['CL_Reserve']
        })

    def summary(self) -> Dict:
        """Get summary statistics."""
        cl_reserves = self.cc_results['CL_Reserve']

        return {
            'cape_cod_elr': self.cape_cod_elr,
            'total_cc_reserve': self.cc_reserves.sum(),
            'total_cl_reserve': cl_reserves.sum(),
            'difference': self.cc_reserves.sum() - cl_reserves.sum(),
            'pct_difference': (
                (self.cc_reserves.sum() - cl_reserves.sum()) /
                cl_reserves.sum() * 100
                if cl_reserves.sum() != 0 else 0
            ),
            'total_premium': self.earned_premium.sum(),
            'total_reported': self.cc_results['Reported'].sum(),
            'total_used_up_premium': self.used_up_premium.sum()
        }

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "="*80)
        print("CAPE COD METHOD RESULTS")
        print("="*80)

        print(f"\n📊 CAPE COD EXPECTED LOSS RATIO: {self.cape_cod_elr:.2%}")
        print("-"*60)
        print(f"Total Reported:     ${self.cc_results['Reported'].sum():>15,.0f}")
        print(f"Total Used-Up Prem: ${self.used_up_premium.sum():>15,.0f}")
        print(f"Implied ELR:        {self.cape_cod_elr:>15.2%}")

        print("\n\n📊 DEVELOPMENT FACTORS:")
        print("-"*60)
        factor_display = pd.DataFrame({
            'LDF': self.development_factors,
            'CDF': self.cumulative_factors,
            'Pct_Reported': self.percent_reported
        })
        print(factor_display.round(4).to_string())

        print("\n\n📊 RESERVES BY ACCIDENT YEAR:")
        print("-"*60)
        display_cols = ['Reported', 'Premium', 'Pct_Reported', 'CC_Reserve', 'CL_Reserve', 'CC_vs_CL_Diff']
        print(self.cc_results[display_cols].round(2).to_string())

        print("\n\n📊 SUMMARY:")
        print("-"*60)
        summary = self.summary()
        print(f"Cape Cod ELR:          {summary['cape_cod_elr']:>14.2%}")
        print(f"Total Cape Cod Reserve: ${summary['total_cc_reserve']:>14,.0f}")
        print(f"Total Chain-Ladder:     ${summary['total_cl_reserve']:>14,.0f}")
        print(f"Difference:             ${summary['difference']:>14,.0f}")
        print(f"Percentage Difference:  {summary['pct_difference']:>13.1f}%")

        print("\n\n📊 INTERPRETATION:")
        print("-"*60)

        if abs(summary['pct_difference']) < 5:
            print("Cape Cod and Chain-Ladder produce similar results.")
            print("→ Development patterns are consistent with premium exposure.")
        elif summary['difference'] > 0:
            print("Cape Cod produces HIGHER reserves than Chain-Ladder.")
            print("→ Overall loss ratio suggests more development than CL projects.")
            print("→ CL may be understating reserves for immature years.")
        else:
            print("Cape Cod produces LOWER reserves than Chain-Ladder.")
            print("→ Overall loss ratio suggests less development than CL projects.")
            print("→ CL may be overstating reserves due to volatile early development.")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.cc_results.to_csv(output_dir / "cape_cod_results.csv")
        self.get_comparison().to_csv(output_dir / "cape_cod_vs_cl_comparison.csv")

        # Save factors
        pd.DataFrame({
            'LDF': self.development_factors,
            'CDF': self.cumulative_factors,
            'Pct_Reported': self.percent_reported
        }).to_csv(output_dir / "cape_cod_factors.csv")

        print(f"✅ Results saved to {output_dir}/")
