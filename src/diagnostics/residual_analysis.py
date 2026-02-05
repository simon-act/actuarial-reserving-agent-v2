"""
Residual Analysis for Chain-Ladder Models

Provides comprehensive residual diagnostics including:
- Raw residuals
- Pearson residuals (standardized by sqrt of fitted value)
- Weighted residuals
- Deviance residuals

These diagnostics help identify:
- Model misspecification
- Outliers
- Heteroscedasticity
- Trends in development patterns

References:
- England, P.D. & Verrall, R.J. (2002). "Stochastic Claims Reserving in General Insurance"
- Mack, T. (1994). "Which Stochastic Model is Underlying the Chain Ladder Method?"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.stats_utils import (
    skewness, kurtosis, shapiro_wilk, kstest, jarque_bera,
    levene, bartlett, norm_cdf
)


class ResidualAnalyzer:
    """
    Comprehensive residual analysis for chain-ladder models.

    Calculates multiple types of residuals and provides diagnostic
    statistics and visualizations for model validation.
    """

    def __init__(self, triangle: pd.DataFrame, development_factors: pd.Series):
        """
        Initialize residual analyzer.

        Args:
            triangle: Cumulative loss development triangle
            development_factors: Selected development factors
        """
        self.triangle = triangle.copy()
        self.development_factors = development_factors.copy()

        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Results
        self.fitted_triangle = None
        self.raw_residuals = None
        self.pearson_residuals = None
        self.standardized_residuals = None
        self.deviance_residuals = None
        self.residual_statistics = None

    def fit(self) -> 'ResidualAnalyzer':
        """
        Calculate all residual types.

        Returns:
            self for method chaining
        """
        self._calculate_fitted_triangle()
        self._calculate_raw_residuals()
        self._calculate_pearson_residuals()
        self._calculate_standardized_residuals()
        self._calculate_deviance_residuals()
        self._calculate_statistics()

        return self

    def _calculate_fitted_triangle(self):
        """
        Calculate fitted values from chain-ladder model.

        Fitted value at (i,j) = Actual at (i,j-1) * f_{j-1}
        """
        self.fitted_triangle = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns,
            dtype=float
        )

        for year in self.triangle.index:
            row = self.triangle.loc[year]
            observed = row.dropna()

            # First column has no fitted value (it's the starting point)
            self.fitted_triangle.loc[year, observed.index[0]] = observed.iloc[0]

            # Calculate fitted for subsequent columns
            for j in range(1, len(observed)):
                prev_col = observed.index[j - 1]
                curr_col = observed.index[j]

                if prev_col in self.development_factors.index:
                    factor = self.development_factors[prev_col]
                    self.fitted_triangle.loc[year, curr_col] = (
                        self.triangle.loc[year, prev_col] * factor
                    )

    def _calculate_raw_residuals(self):
        """
        Calculate raw residuals: Actual - Fitted
        """
        self.raw_residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[1:],  # No residual for first column
            dtype=float
        )

        for year in self.triangle.index:
            for col in self.triangle.columns[1:]:
                actual = self.triangle.loc[year, col]
                fitted = self.fitted_triangle.loc[year, col]

                if pd.notna(actual) and pd.notna(fitted):
                    self.raw_residuals.loc[year, col] = actual - fitted

    def _calculate_pearson_residuals(self):
        """
        Calculate Pearson residuals: (Actual - Fitted) / sqrt(Fitted)

        Assumes variance proportional to mean (Poisson-like).
        """
        self.pearson_residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[1:],
            dtype=float
        )

        for year in self.triangle.index:
            for col in self.triangle.columns[1:]:
                actual = self.triangle.loc[year, col]
                fitted = self.fitted_triangle.loc[year, col]

                if pd.notna(actual) and pd.notna(fitted) and fitted > 0:
                    self.pearson_residuals.loc[year, col] = (
                        (actual - fitted) / np.sqrt(fitted)
                    )

    def _calculate_standardized_residuals(self):
        """
        Calculate standardized residuals adjusted for leverage.

        Standardized = Pearson / sqrt(1 - leverage)

        For chain-ladder, leverage h_ij ≈ C_{i,j-1} / sum(C_{k,j-1})
        """
        self.standardized_residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[1:],
            dtype=float
        )

        for col_idx, col in enumerate(self.triangle.columns[1:]):
            prev_col = self.triangle.columns[col_idx]

            # Calculate column sum for leverage
            col_sum = self.triangle[prev_col].dropna().sum()

            for year in self.triangle.index:
                pearson = self.pearson_residuals.loc[year, col]
                prev_value = self.triangle.loc[year, prev_col]

                if pd.notna(pearson) and pd.notna(prev_value) and col_sum > 0:
                    leverage = prev_value / col_sum
                    if leverage < 1:
                        self.standardized_residuals.loc[year, col] = (
                            pearson / np.sqrt(1 - leverage)
                        )
                    else:
                        self.standardized_residuals.loc[year, col] = pearson

    def _calculate_deviance_residuals(self):
        """
        Calculate deviance residuals.

        For ODP model:
        d_ij = sign(y - μ) * sqrt(2 * [y*log(y/μ) - (y - μ)])
        """
        self.deviance_residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[1:],
            dtype=float
        )

        for year in self.triangle.index:
            for col in self.triangle.columns[1:]:
                actual = self.triangle.loc[year, col]
                fitted = self.fitted_triangle.loc[year, col]

                if pd.notna(actual) and pd.notna(fitted) and fitted > 0 and actual > 0:
                    # Deviance contribution
                    dev = 2 * (actual * np.log(actual / fitted) - (actual - fitted))
                    if dev >= 0:
                        sign = 1 if actual >= fitted else -1
                        self.deviance_residuals.loc[year, col] = sign * np.sqrt(dev)

    def _calculate_statistics(self):
        """Calculate summary statistics for residuals."""
        self.residual_statistics = {}

        for resid_type, residuals in [
            ('raw', self.raw_residuals),
            ('pearson', self.pearson_residuals),
            ('standardized', self.standardized_residuals),
            ('deviance', self.deviance_residuals)
        ]:
            flat = residuals.values.flatten()
            flat = flat[~np.isnan(flat)]

            if len(flat) > 0:
                self.residual_statistics[resid_type] = {
                    'mean': np.mean(flat),
                    'std': np.std(flat),
                    'min': np.min(flat),
                    'max': np.max(flat),
                    'skewness': skewness(flat),
                    'kurtosis': kurtosis(flat),
                    'n': len(flat)
                }

                # Normality tests
                if len(flat) >= 8:
                    _, p_shapiro = shapiro_wilk(flat[:min(5000, len(flat))])
                    self.residual_statistics[resid_type]['shapiro_p'] = p_shapiro

    def test_normality(self) -> Dict:
        """
        Test residuals for normality.

        Returns:
            Dictionary with normality test results
        """
        results = {}

        for resid_type, residuals in [
            ('pearson', self.pearson_residuals),
            ('standardized', self.standardized_residuals)
        ]:
            flat = residuals.values.flatten()
            flat = flat[~np.isnan(flat)]

            if len(flat) >= 8:
                # Shapiro-Wilk test
                stat_sw, p_sw = shapiro_wilk(flat[:min(5000, len(flat))])

                # Kolmogorov-Smirnov test
                stat_ks, p_ks = kstest(flat, 'norm', args=(np.mean(flat), np.std(flat)))

                # Jarque-Bera test
                stat_jb, p_jb = jarque_bera(flat)

                results[resid_type] = {
                    'shapiro_wilk': {'statistic': stat_sw, 'p_value': p_sw},
                    'kolmogorov_smirnov': {'statistic': stat_ks, 'p_value': p_ks},
                    'jarque_bera': {'statistic': stat_jb, 'p_value': p_jb},
                    'normal_at_5pct': p_sw > 0.05 and p_jb > 0.05
                }

        return results

    def test_homoscedasticity(self) -> Dict:
        """
        Test for constant variance across development periods.

        Uses Levene's test and Bartlett's test.

        Returns:
            Dictionary with homoscedasticity test results
        """
        # Group residuals by development period
        period_residuals = []
        period_names = []

        for col in self.pearson_residuals.columns:
            col_resid = self.pearson_residuals[col].dropna().values
            if len(col_resid) >= 3:
                period_residuals.append(col_resid)
                period_names.append(col)

        results = {}

        if len(period_residuals) >= 2:
            # Levene's test (robust to non-normality)
            stat_lev, p_lev = levene(*period_residuals)

            # Bartlett's test (assumes normality)
            stat_bart, p_bart = bartlett(*period_residuals)

            results = {
                'levene': {'statistic': stat_lev, 'p_value': p_lev},
                'bartlett': {'statistic': stat_bart, 'p_value': p_bart},
                'homoscedastic_at_5pct': p_lev > 0.05,
                'periods_tested': period_names,
                'variance_by_period': {
                    name: np.var(resid) for name, resid in zip(period_names, period_residuals)
                }
            }

        return results

    def test_independence(self) -> Dict:
        """
        Test residuals for independence/autocorrelation.

        Uses Durbin-Watson and runs test.

        Returns:
            Dictionary with independence test results
        """
        results = {}

        # Test by accident year (row)
        row_dw = []
        for year in self.pearson_residuals.index:
            row = self.pearson_residuals.loc[year].dropna().values
            if len(row) >= 3:
                # Durbin-Watson statistic
                diff = np.diff(row)
                dw = np.sum(diff**2) / np.sum(row**2) if np.sum(row**2) > 0 else 2
                row_dw.append(dw)

        # Test by development period (column)
        col_dw = []
        for col in self.pearson_residuals.columns:
            col_data = self.pearson_residuals[col].dropna().values
            if len(col_data) >= 3:
                diff = np.diff(col_data)
                dw = np.sum(diff**2) / np.sum(col_data**2) if np.sum(col_data**2) > 0 else 2
                col_dw.append(dw)

        # Overall residuals - runs test
        flat = self.pearson_residuals.values.flatten()
        flat = flat[~np.isnan(flat)]

        if len(flat) >= 10:
            median = np.median(flat)
            runs = np.sum(np.diff(flat > median) != 0) + 1
            n_above = np.sum(flat > median)
            n_below = len(flat) - n_above

            # Expected runs
            if n_above > 0 and n_below > 0:
                expected_runs = (2 * n_above * n_below) / (n_above + n_below) + 1
                var_runs = (2 * n_above * n_below * (2 * n_above * n_below - n_above - n_below)) / \
                          ((n_above + n_below)**2 * (n_above + n_below - 1))

                if var_runs > 0:
                    z_runs = (runs - expected_runs) / np.sqrt(var_runs)
                    p_runs = 2 * (1 - norm_cdf(abs(z_runs)))
                else:
                    z_runs, p_runs = np.nan, np.nan
            else:
                expected_runs, z_runs, p_runs = np.nan, np.nan, np.nan

            results = {
                'durbin_watson_by_row': {
                    'values': row_dw,
                    'mean': np.mean(row_dw) if row_dw else np.nan,
                    'interpretation': 'DW close to 2 indicates no autocorrelation'
                },
                'durbin_watson_by_col': {
                    'values': col_dw,
                    'mean': np.mean(col_dw) if col_dw else np.nan
                },
                'runs_test': {
                    'runs': runs,
                    'expected': expected_runs,
                    'z_statistic': z_runs,
                    'p_value': p_runs,
                    'independent_at_5pct': p_runs > 0.05 if not np.isnan(p_runs) else None
                }
            }

        return results

    def identify_outliers(self, threshold: float = 2.5) -> pd.DataFrame:
        """
        Identify outlier residuals.

        Args:
            threshold: Number of standard deviations for outlier detection

        Returns:
            DataFrame of outlier observations
        """
        outliers = []

        for year in self.standardized_residuals.index:
            for col in self.standardized_residuals.columns:
                resid = self.standardized_residuals.loc[year, col]
                if pd.notna(resid) and abs(resid) > threshold:
                    outliers.append({
                        'Accident_Year': year,
                        'Development_Age': col,
                        'Standardized_Residual': resid,
                        'Actual': self.triangle.loc[year, col],
                        'Fitted': self.fitted_triangle.loc[year, col],
                        'Raw_Residual': self.raw_residuals.loc[year, col]
                    })

        return pd.DataFrame(outliers)

    def get_residual_by_calendar_year(self) -> pd.DataFrame:
        """
        Aggregate residuals by calendar year.

        Useful for detecting calendar year effects.

        Returns:
            DataFrame of mean residuals by calendar year
        """
        calendar_residuals = {}

        for year in self.triangle.index:
            for col_idx, col in enumerate(self.pearson_residuals.columns):
                calendar_year = year + col_idx + 1  # +1 because residuals start from col 1

                if calendar_year not in calendar_residuals:
                    calendar_residuals[calendar_year] = []

                resid = self.pearson_residuals.loc[year, col]
                if pd.notna(resid):
                    calendar_residuals[calendar_year].append(resid)

        # Calculate statistics
        cal_stats = []
        for cal_year, resids in sorted(calendar_residuals.items()):
            if resids:
                cal_stats.append({
                    'Calendar_Year': cal_year,
                    'Mean_Residual': np.mean(resids),
                    'Std_Residual': np.std(resids) if len(resids) > 1 else 0,
                    'N_Observations': len(resids)
                })

        return pd.DataFrame(cal_stats)

    def summary(self) -> Dict:
        """Get summary of residual analysis."""
        return {
            'statistics': self.residual_statistics,
            'normality': self.test_normality(),
            'homoscedasticity': self.test_homoscedasticity(),
            'independence': self.test_independence(),
            'n_outliers': len(self.identify_outliers())
        }

    def print_summary(self):
        """Print formatted summary of residual analysis."""
        print("\n" + "="*80)
        print("RESIDUAL ANALYSIS SUMMARY")
        print("="*80)

        # Basic statistics
        print("\n📊 RESIDUAL STATISTICS (Pearson):")
        print("-"*60)
        if 'pearson' in self.residual_statistics:
            stats_dict = self.residual_statistics['pearson']
            print(f"Mean:      {stats_dict['mean']:>10.4f}")
            print(f"Std Dev:   {stats_dict['std']:>10.4f}")
            print(f"Min:       {stats_dict['min']:>10.4f}")
            print(f"Max:       {stats_dict['max']:>10.4f}")
            print(f"Skewness:  {stats_dict['skewness']:>10.4f}")
            print(f"Kurtosis:  {stats_dict['kurtosis']:>10.4f}")
            print(f"N:         {stats_dict['n']:>10}")

        # Normality tests
        print("\n\n📊 NORMALITY TESTS:")
        print("-"*60)
        norm_results = self.test_normality()
        if 'standardized' in norm_results:
            nr = norm_results['standardized']
            print(f"Shapiro-Wilk p-value:    {nr['shapiro_wilk']['p_value']:.4f}")
            print(f"Jarque-Bera p-value:     {nr['jarque_bera']['p_value']:.4f}")
            print(f"Normal at 5%:            {'Yes' if nr['normal_at_5pct'] else 'No'}")

        # Homoscedasticity
        print("\n\n📊 HOMOSCEDASTICITY TESTS:")
        print("-"*60)
        homo_results = self.test_homoscedasticity()
        if homo_results:
            print(f"Levene's test p-value:   {homo_results['levene']['p_value']:.4f}")
            print(f"Constant variance at 5%: {'Yes' if homo_results['homoscedastic_at_5pct'] else 'No'}")

        # Independence
        print("\n\n📊 INDEPENDENCE TESTS:")
        print("-"*60)
        ind_results = self.test_independence()
        if ind_results:
            print(f"Mean Durbin-Watson (row): {ind_results['durbin_watson_by_row']['mean']:.4f}")
            print(f"Runs test p-value:        {ind_results['runs_test']['p_value']:.4f}")
            if ind_results['runs_test']['independent_at_5pct'] is not None:
                print(f"Independent at 5%:        {'Yes' if ind_results['runs_test']['independent_at_5pct'] else 'No'}")

        # Outliers
        print("\n\n📊 OUTLIERS (|z| > 2.5):")
        print("-"*60)
        outliers = self.identify_outliers()
        if len(outliers) > 0:
            print(f"Number of outliers: {len(outliers)}")
            print(outliers.to_string())
        else:
            print("No outliers detected")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save residual analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_residuals.to_csv(output_dir / "raw_residuals.csv")
        self.pearson_residuals.to_csv(output_dir / "pearson_residuals.csv")
        self.standardized_residuals.to_csv(output_dir / "standardized_residuals.csv")
        self.identify_outliers().to_csv(output_dir / "outliers.csv", index=False)
        self.get_residual_by_calendar_year().to_csv(
            output_dir / "calendar_year_residuals.csv", index=False
        )

        print(f"✅ Residual analysis saved to {output_dir}/")
