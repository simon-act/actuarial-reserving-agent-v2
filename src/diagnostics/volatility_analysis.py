"""
Volatility Analysis for Development Factors

Analyzes the stability and variability of development factors across
accident years and development periods.

Key metrics:
- Factor volatility by development period
- Weighted vs unweighted volatility
- Trend detection in factors
- Stability metrics

References:
- Mack, T. (1999). "The Standard Error of Chain Ladder Reserve Estimates: Recursive Calculation and Inclusion of a Tail Factor"
- Wüthrich, M.V. (2016). "Neural Networks Applied to Chain-Ladder Reserving"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.stats_utils import linregress, norm_cdf, skewness


class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis for development factors.

    Measures how stable development patterns are across accident years
    and identifies periods with high variability.
    """

    def __init__(self, triangle: pd.DataFrame):
        """
        Initialize volatility analyzer.

        Args:
            triangle: Cumulative loss development triangle
        """
        self.triangle = triangle.copy()
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Results
        self.age_to_age_factors = None
        self.factor_statistics = None
        self.weighted_statistics = None
        self.volatility_metrics = None

    def fit(self) -> 'VolatilityAnalyzer':
        """
        Calculate volatility metrics.

        Returns:
            self for method chaining
        """
        self._calculate_age_to_age_factors()
        self._calculate_factor_statistics()
        self._calculate_weighted_statistics()
        self._calculate_volatility_metrics()

        return self

    def _calculate_age_to_age_factors(self):
        """Calculate individual age-to-age factors."""
        self.age_to_age_factors = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[:-1],
            dtype=float
        )

        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            for year in self.triangle.index:
                curr_val = self.triangle.loc[year, col_curr]
                next_val = self.triangle.loc[year, col_next]

                if pd.notna(curr_val) and pd.notna(next_val) and curr_val > 0:
                    self.age_to_age_factors.loc[year, col_curr] = next_val / curr_val

    def _calculate_factor_statistics(self):
        """Calculate basic statistics for each development period."""
        stats_list = []

        for col in self.age_to_age_factors.columns:
            factors = self.age_to_age_factors[col].dropna()

            if len(factors) >= 2:
                stats_dict = {
                    'Period': col,
                    'N': len(factors),
                    'Mean': factors.mean(),
                    'Median': factors.median(),
                    'Std': factors.std(),
                    'CV': factors.std() / factors.mean() if factors.mean() > 0 else np.nan,
                    'Min': factors.min(),
                    'Max': factors.max(),
                    'Range': factors.max() - factors.min(),
                    'IQR': factors.quantile(0.75) - factors.quantile(0.25),
                    'Skewness': skewness(factors) if len(factors) >= 3 else np.nan,
                    'P10': factors.quantile(0.10),
                    'P90': factors.quantile(0.90)
                }
            else:
                stats_dict = {
                    'Period': col,
                    'N': len(factors),
                    'Mean': factors.mean() if len(factors) > 0 else np.nan,
                    'Median': np.nan,
                    'Std': np.nan,
                    'CV': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'Range': np.nan,
                    'IQR': np.nan,
                    'Skewness': np.nan,
                    'P10': np.nan,
                    'P90': np.nan
                }

            stats_list.append(stats_dict)

        self.factor_statistics = pd.DataFrame(stats_list).set_index('Period')

    def _calculate_weighted_statistics(self):
        """Calculate volume-weighted statistics."""
        stats_list = []

        for col_idx, col in enumerate(self.age_to_age_factors.columns):
            factors = self.age_to_age_factors[col].dropna()
            weights = self.triangle[col].loc[factors.index]

            if len(factors) >= 2 and weights.sum() > 0:
                # Weighted mean
                weighted_mean = (factors * weights).sum() / weights.sum()

                # Weighted variance
                weighted_var = ((factors - weighted_mean)**2 * weights).sum() / weights.sum()
                weighted_std = np.sqrt(weighted_var)

                stats_dict = {
                    'Period': col,
                    'Weighted_Mean': weighted_mean,
                    'Weighted_Std': weighted_std,
                    'Weighted_CV': weighted_std / weighted_mean if weighted_mean > 0 else np.nan,
                    'Simple_Mean': factors.mean(),
                    'Mean_Diff': abs(weighted_mean - factors.mean()),
                    'Largest_Weight_Year': weights.idxmax(),
                    'Largest_Weight_Pct': weights.max() / weights.sum() * 100
                }
            else:
                stats_dict = {
                    'Period': col,
                    'Weighted_Mean': np.nan,
                    'Weighted_Std': np.nan,
                    'Weighted_CV': np.nan,
                    'Simple_Mean': np.nan,
                    'Mean_Diff': np.nan,
                    'Largest_Weight_Year': np.nan,
                    'Largest_Weight_Pct': np.nan
                }

            stats_list.append(stats_dict)

        self.weighted_statistics = pd.DataFrame(stats_list).set_index('Period')

    def _calculate_volatility_metrics(self):
        """Calculate overall volatility metrics."""
        self.volatility_metrics = {}

        # Average CV across periods
        cvs = self.factor_statistics['CV'].dropna()
        if len(cvs) > 0:
            self.volatility_metrics['average_cv'] = cvs.mean()
            self.volatility_metrics['max_cv'] = cvs.max()
            self.volatility_metrics['max_cv_period'] = cvs.idxmax()
            self.volatility_metrics['min_cv'] = cvs.min()
            self.volatility_metrics['min_cv_period'] = cvs.idxmin()

        # Volatility trend (do later periods have more or less volatility?)
        periods_numeric = np.array(list(range(len(cvs))))
        if len(periods_numeric) >= 3:
            slope, intercept, r_value, p_value, std_err = linregress(
                periods_numeric, np.array(cvs.values)
            )
            self.volatility_metrics['volatility_trend'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'interpretation': 'increasing' if slope > 0 else 'decreasing'
            }

        # Factor stability score (lower is more stable)
        # Based on average CV and range of factors
        ranges = self.factor_statistics['Range'].dropna()
        if len(ranges) > 0 and len(cvs) > 0:
            normalized_ranges = ranges / self.factor_statistics['Mean']
            stability_score = (cvs.mean() + normalized_ranges.mean()) / 2
            self.volatility_metrics['stability_score'] = stability_score
            self.volatility_metrics['stability_rating'] = (
                'High' if stability_score < 0.05 else
                'Medium' if stability_score < 0.15 else
                'Low'
            )

    def test_factor_trend(self) -> Dict:
        """
        Test for trends in factors over accident years.

        Tests whether more recent years have systematically
        different development patterns than earlier years.

        Returns:
            Dictionary with trend test results by period
        """
        trend_results = {}

        for col in self.age_to_age_factors.columns:
            factors = self.age_to_age_factors[col].dropna()
            years = factors.index.values

            if len(factors) >= 5:
                # Linear regression
                slope, intercept, r_value, p_value, std_err = linregress(
                    years, factors.values
                )

                # Mann-Kendall trend test (non-parametric)
                n = len(factors)
                s = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        s += np.sign(factors.iloc[j] - factors.iloc[i])

                # Variance of S
                var_s = (n * (n - 1) * (2 * n + 5)) / 18

                if s > 0:
                    z_mk = (s - 1) / np.sqrt(var_s)
                elif s < 0:
                    z_mk = (s + 1) / np.sqrt(var_s)
                else:
                    z_mk = 0

                p_mk = 2 * (1 - norm_cdf(abs(z_mk)))

                trend_results[col] = {
                    'linear_slope': slope,
                    'linear_r_squared': r_value**2,
                    'linear_p_value': p_value,
                    'mann_kendall_z': z_mk,
                    'mann_kendall_p': p_mk,
                    'significant_trend': p_value < 0.05 or p_mk < 0.05,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                }

        return trend_results

    def detect_structural_breaks(self) -> Dict:
        """
        Detect potential structural breaks in development patterns.

        Uses CUSUM-type analysis to identify when patterns changed.

        Returns:
            Dictionary with structural break analysis
        """
        breaks = {}

        for col in self.age_to_age_factors.columns:
            factors = self.age_to_age_factors[col].dropna()

            if len(factors) >= 6:
                mean = factors.mean()
                std = factors.std()

                # CUSUM statistic
                cusum = np.cumsum(factors - mean)
                cusum_normalized = cusum / (std * np.sqrt(len(factors)))

                # Find max absolute CUSUM
                max_idx = np.argmax(np.abs(cusum_normalized))
                max_cusum = cusum_normalized[max_idx]

                # Critical value (approximate, from Ploberger & Krämer tables)
                critical_value = 1.36  # For 5% significance level

                breaks[col] = {
                    'cusum_values': cusum_normalized.tolist(),
                    'max_cusum': max_cusum,
                    'max_cusum_year': factors.index[max_idx],
                    'potential_break': abs(max_cusum) > critical_value,
                    'years_analyzed': list(factors.index)
                }

        return breaks

    def get_high_volatility_periods(self, threshold_cv: float = 0.10) -> List:
        """
        Identify development periods with high volatility.

        Args:
            threshold_cv: CV threshold for "high" volatility

        Returns:
            List of high volatility periods with details
        """
        high_vol = []

        for period in self.factor_statistics.index:
            cv = self.factor_statistics.loc[period, 'CV']
            if pd.notna(cv) and cv > threshold_cv:
                high_vol.append({
                    'period': period,
                    'cv': cv,
                    'std': self.factor_statistics.loc[period, 'Std'],
                    'range': self.factor_statistics.loc[period, 'Range'],
                    'n_observations': self.factor_statistics.loc[period, 'N']
                })

        return sorted(high_vol, key=lambda x: x['cv'], reverse=True)

    def get_factor_heatmap_data(self) -> pd.DataFrame:
        """
        Prepare data for factor heatmap visualization.

        Returns:
            DataFrame suitable for heatmap plotting
        """
        # Standardize factors relative to mean for each period
        standardized = self.age_to_age_factors.copy()

        for col in standardized.columns:
            mean = standardized[col].mean()
            std = standardized[col].std()
            if std > 0:
                standardized[col] = (standardized[col] - mean) / std

        return standardized

    def summary(self) -> Dict:
        """Get summary of volatility analysis."""
        trend_tests = self.test_factor_trend()
        breaks = self.detect_structural_breaks()

        return {
            'volatility_metrics': self.volatility_metrics,
            'high_volatility_periods': self.get_high_volatility_periods(),
            'n_periods_with_significant_trend': sum(
                1 for v in trend_tests.values() if v.get('significant_trend', False)
            ),
            'n_periods_with_potential_break': sum(
                1 for v in breaks.values() if v.get('potential_break', False)
            )
        }

    def print_summary(self):
        """Print formatted summary of volatility analysis."""
        print("\n" + "="*80)
        print("VOLATILITY ANALYSIS SUMMARY")
        print("="*80)

        # Basic statistics
        print("\n📊 FACTOR STATISTICS BY DEVELOPMENT PERIOD:")
        print("-"*70)
        display_cols = ['N', 'Mean', 'Std', 'CV', 'Min', 'Max']
        print(self.factor_statistics[display_cols].round(4).to_string())

        # Overall metrics
        print("\n\n📊 OVERALL VOLATILITY METRICS:")
        print("-"*60)
        if self.volatility_metrics:
            print(f"Average CV:           {self.volatility_metrics.get('average_cv', np.nan):.4f}")
            print(f"Max CV Period:        {self.volatility_metrics.get('max_cv_period', 'N/A')} "
                  f"({self.volatility_metrics.get('max_cv', np.nan):.4f})")
            print(f"Min CV Period:        {self.volatility_metrics.get('min_cv_period', 'N/A')} "
                  f"({self.volatility_metrics.get('min_cv', np.nan):.4f})")
            print(f"Stability Rating:     {self.volatility_metrics.get('stability_rating', 'N/A')}")

        # Trend analysis
        print("\n\n📊 TREND ANALYSIS BY PERIOD:")
        print("-"*60)
        trend_tests = self.test_factor_trend()
        for period, results in trend_tests.items():
            if results['significant_trend']:
                print(f"  {period}: {results['trend_direction']} trend "
                      f"(p={results['linear_p_value']:.4f})")

        if not any(r['significant_trend'] for r in trend_tests.values()):
            print("  No significant trends detected")

        # Structural breaks
        print("\n\n📊 STRUCTURAL BREAK ANALYSIS:")
        print("-"*60)
        breaks = self.detect_structural_breaks()
        for period, results in breaks.items():
            if results['potential_break']:
                print(f"  {period}: Potential break around {results['max_cusum_year']} "
                      f"(CUSUM={results['max_cusum']:.2f})")

        if not any(r['potential_break'] for r in breaks.values()):
            print("  No structural breaks detected")

        # High volatility periods
        print("\n\n📊 HIGH VOLATILITY PERIODS (CV > 10%):")
        print("-"*60)
        high_vol = self.get_high_volatility_periods()
        if high_vol:
            for hv in high_vol:
                print(f"  {hv['period']}: CV={hv['cv']:.4f}, "
                      f"Range={hv['range']:.4f}, N={hv['n_observations']}")
        else:
            print("  No high volatility periods detected")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save volatility analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.age_to_age_factors.to_csv(output_dir / "age_to_age_factors.csv")
        self.factor_statistics.to_csv(output_dir / "factor_statistics.csv")
        self.weighted_statistics.to_csv(output_dir / "weighted_statistics.csv")
        self.get_factor_heatmap_data().to_csv(output_dir / "standardized_factors.csv")

        print(f"✅ Volatility analysis saved to {output_dir}/")
