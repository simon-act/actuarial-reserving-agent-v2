"""
Diagnostic Tests for Chain-Ladder and Related Reserving Methods

Comprehensive suite of statistical tests to validate model assumptions
and identify potential issues with reserving models.

Tests include:
- Calendar year effect tests
- Development pattern stability tests
- Model adequacy tests
- Predictive accuracy tests

References:
- Mack, T. (1994). "Measuring the Variability of Chain Ladder Reserve Estimates"
- Venter, G. (1998). "Testing the Assumptions of Age-to-Age Factors"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.stats_utils import f_oneway, linregress, pearsonr, t_cdf


class DiagnosticTests:
    """
    Comprehensive diagnostic tests for reserving models.

    Provides statistical tests to validate chain-ladder assumptions
    and identify potential model misspecification.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        development_factors: Optional[pd.Series] = None
    ):
        """
        Initialize diagnostic tests.

        Args:
            triangle: Cumulative loss development triangle
            development_factors: Optional development factors (if not provided, calculated)
        """
        self.triangle = triangle.copy()
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        if development_factors is not None:
            self.development_factors = development_factors.copy()
        else:
            self._calculate_development_factors()

        # Calculate age-to-age factors matrix
        self._calculate_factor_matrix()

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
            factors, index=self.triangle.columns[:-1]
        )

    def _calculate_factor_matrix(self):
        """Calculate matrix of individual age-to-age factors."""
        self.factor_matrix = pd.DataFrame(
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
                    self.factor_matrix.loc[year, col_curr] = next_val / curr_val

    def test_calendar_year_effect(self) -> Dict:
        """
        Test for calendar year effects in development factors.

        Calendar year effects indicate that development patterns change
        over time for all accident years simultaneously (e.g., due to
        claims handling changes, regulatory changes, etc.).

        Uses analysis of residuals by calendar year diagonal.

        Returns:
            Dictionary with calendar year effect test results
        """
        # Calculate residuals (factor - weighted average factor)
        residuals_by_calendar = {}

        for year in self.triangle.index:
            for col_idx, col in enumerate(self.factor_matrix.columns):
                factor = self.factor_matrix.loc[year, col]
                if pd.notna(factor):
                    calendar_year = year + col_idx + 1
                    expected = self.development_factors[col]
                    residual = factor - expected

                    if calendar_year not in residuals_by_calendar:
                        residuals_by_calendar[calendar_year] = []
                    residuals_by_calendar[calendar_year].append(residual)

        # Calculate mean residual by calendar year
        calendar_means = {
            cy: np.mean(resids) for cy, resids in residuals_by_calendar.items()
            if len(resids) >= 2
        }

        # Test for systematic calendar year effects
        all_residuals = []
        groups = []

        for cy, resids in residuals_by_calendar.items():
            if len(resids) >= 2:
                all_residuals.extend(resids)
                groups.extend([cy] * len(resids))

        results = {
            'calendar_year_means': calendar_means,
            'n_calendar_years': len(calendar_means)
        }

        # ANOVA test
        if len(set(groups)) >= 3:
            unique_groups = sorted(set(groups))
            group_data = [
                [r for r, g in zip(all_residuals, groups) if g == ug]
                for ug in unique_groups
            ]
            group_data = [g for g in group_data if len(g) >= 2]

            if len(group_data) >= 2:
                f_stat, p_value = f_oneway(*group_data)
                results['anova_f_statistic'] = f_stat
                results['anova_p_value'] = p_value
                results['calendar_effect_detected'] = p_value < 0.05

        # Trend test on calendar year means
        if len(calendar_means) >= 5:
            years = sorted(calendar_means.keys())
            means = [calendar_means[y] for y in years]

            slope, intercept, r_value, p_value, std_err = linregress(years, means)
            results['trend_slope'] = slope
            results['trend_p_value'] = p_value
            results['trend_detected'] = p_value < 0.05

        return results

    def test_accident_year_effect(self) -> Dict:
        """
        Test for accident year effects beyond expected development.

        Accident year effects indicate that specific accident years
        develop differently from the average pattern.

        Returns:
            Dictionary with accident year effect test results
        """
        # Calculate standardized residuals by accident year
        year_residuals = {}

        for year in self.triangle.index:
            residuals = []
            for col in self.factor_matrix.columns:
                factor = self.factor_matrix.loc[year, col]
                if pd.notna(factor):
                    expected = self.development_factors[col]
                    # Standardize by expected variance (Mack's formula)
                    residual = factor - expected
                    residuals.append(residual)

            if residuals:
                year_residuals[year] = {
                    'residuals': residuals,
                    'mean': np.mean(residuals),
                    'sum': np.sum(residuals),
                    'n': len(residuals)
                }

        # Identify anomalous years
        all_means = [yr['mean'] for yr in year_residuals.values()]
        mean_of_means = np.mean(all_means)
        std_of_means = np.std(all_means)

        anomalous_years = []
        for year, data in year_residuals.items():
            if std_of_means > 0:
                z_score = (data['mean'] - mean_of_means) / std_of_means
                if abs(z_score) > 2:
                    anomalous_years.append({
                        'year': year,
                        'mean_residual': data['mean'],
                        'z_score': z_score,
                        'interpretation': 'higher' if z_score > 0 else 'lower'
                    })

        return {
            'year_statistics': year_residuals,
            'anomalous_years': anomalous_years,
            'n_anomalous': len(anomalous_years)
        }

    def test_development_period_independence(self) -> Dict:
        """
        Test that development factors are independent across periods.

        Chain-ladder assumes C_{i,j}/C_{i,j-1} is independent of
        C_{i,k}/C_{i,k-1} for j ≠ k.

        Uses correlation analysis between successive factors.

        Returns:
            Dictionary with independence test results
        """
        correlations = {}

        # Test correlation between adjacent development periods
        for col_idx in range(len(self.factor_matrix.columns) - 1):
            col1 = self.factor_matrix.columns[col_idx]
            col2 = self.factor_matrix.columns[col_idx + 1]

            factors1 = self.factor_matrix[col1].dropna()
            factors2 = self.factor_matrix[col2].dropna()

            common = factors1.index.intersection(factors2.index)

            if len(common) >= 5:
                f1 = factors1.loc[common]
                f2 = factors2.loc[common]

                corr, p_value = pearsonr(f1, f2)
                correlations[f'{col1}_vs_{col2}'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n': len(common)
                }

        # Overall assessment
        significant_correlations = sum(
            1 for c in correlations.values() if c.get('significant', False)
        )

        return {
            'pairwise_correlations': correlations,
            'n_significant': significant_correlations,
            'n_tested': len(correlations),
            'independence_assumption_violated': significant_correlations > 0
        }

    def test_proportionality(self) -> Dict:
        """
        Test the proportionality assumption of chain-ladder.

        Chain-ladder assumes E[C_{i,j}|C_{i,j-1}] = f_{j-1} * C_{i,j-1}
        This implies that log(C_{i,j}) = log(f_{j-1}) + log(C_{i,j-1})

        Tests for non-linear relationships.

        Returns:
            Dictionary with proportionality test results
        """
        results = {}

        for col_idx in range(self.n_periods - 1):
            col_curr = self.triangle.columns[col_idx]
            col_next = self.triangle.columns[col_idx + 1]

            curr = self.triangle[col_curr].dropna()
            next_val = self.triangle[col_next].dropna()
            common = curr.index.intersection(next_val.index)

            if len(common) >= 5:
                x = curr.loc[common].values
                y = next_val.loc[common].values

                # Filter out non-positive values for log transformation
                valid = (x > 0) & (y > 0)
                x_valid = x[valid]
                y_valid = y[valid]

                if len(x_valid) >= 5:
                    log_x = np.log(x_valid)
                    log_y = np.log(y_valid)

                    # Linear regression in log space
                    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

                    # Under proportionality, slope should be ~1
                    # Test H0: slope = 1
                    t_stat = (slope - 1) / std_err
                    p_slope_1 = 2 * (1 - t_cdf(abs(t_stat), df=len(x_valid) - 2))

                    results[col_curr] = {
                        'log_slope': slope,
                        'log_intercept': intercept,
                        'r_squared': r_value**2,
                        'slope_se': std_err,
                        'test_slope_1_t': t_stat,
                        'test_slope_1_p': p_slope_1,
                        'proportional': p_slope_1 > 0.05,
                        'n': len(x_valid)
                    }

        # Overall assessment
        non_proportional = [k for k, v in results.items() if not v.get('proportional', True)]

        return {
            'period_results': results,
            'non_proportional_periods': non_proportional,
            'proportionality_holds': len(non_proportional) == 0
        }

    def test_variance_structure(self) -> Dict:
        """
        Test the variance structure assumption.

        Mack's model assumes Var(C_{i,j}|C_{i,j-1}) = sigma^2_j * C_{i,j-1}

        Tests for heteroscedasticity.

        Returns:
            Dictionary with variance structure test results
        """
        results = {}

        for col_idx in range(self.n_periods - 1):
            col_curr = self.triangle.columns[col_idx]
            col_next = self.triangle.columns[col_idx + 1]

            curr = self.triangle[col_curr].dropna()
            next_val = self.triangle[col_next].dropna()
            common = curr.index.intersection(next_val.index)

            if len(common) >= 5:
                x = curr.loc[common].values
                y = next_val.loc[common].values

                # Expected value under chain-ladder
                f = self.development_factors[col_curr]
                expected = x * f

                # Residuals
                residuals = y - expected

                # Under Mack's assumption, squared residuals / x should be constant
                variance_proxy = residuals**2 / x

                # Test for trend in variance with size
                valid = x > 0
                if valid.sum() >= 5:
                    log_x = np.log(x[valid])
                    log_var_proxy = np.log(variance_proxy[valid] + 1e-10)

                    slope, intercept, r_value, p_value, std_err = linregress(
                        log_x, log_var_proxy
                    )

                    # Under Mack's assumption, slope should be 0
                    results[col_curr] = {
                        'variance_slope': slope,
                        'variance_p_value': p_value,
                        'heteroscedastic': p_value < 0.05,
                        'variance_pattern': (
                            'increasing' if slope > 0 else
                            'decreasing' if slope < 0 else
                            'constant'
                        ),
                        'n': valid.sum()
                    }

        # Overall assessment
        heteroscedastic_periods = [
            k for k, v in results.items() if v.get('heteroscedastic', False)
        ]

        return {
            'period_results': results,
            'heteroscedastic_periods': heteroscedastic_periods,
            'variance_assumption_holds': len(heteroscedastic_periods) == 0
        }

    def run_all_tests(self) -> Dict:
        """
        Run all diagnostic tests and return comprehensive results.

        Returns:
            Dictionary with all test results
        """
        return {
            'calendar_year_effect': self.test_calendar_year_effect(),
            'accident_year_effect': self.test_accident_year_effect(),
            'independence': self.test_development_period_independence(),
            'proportionality': self.test_proportionality(),
            'variance_structure': self.test_variance_structure()
        }

    def get_model_adequacy_score(self) -> Dict:
        """
        Calculate an overall model adequacy score.

        Returns:
            Dictionary with model adequacy assessment
        """
        all_tests = self.run_all_tests()

        issues = []

        # Check each test
        if all_tests['calendar_year_effect'].get('calendar_effect_detected', False):
            issues.append('Calendar year effects detected')

        if all_tests['accident_year_effect'].get('n_anomalous', 0) > 0:
            issues.append(f"{all_tests['accident_year_effect']['n_anomalous']} anomalous accident years")

        if all_tests['independence'].get('independence_assumption_violated', False):
            issues.append('Independence assumption violated')

        if not all_tests['proportionality'].get('proportionality_holds', True):
            issues.append('Proportionality assumption violated')

        if not all_tests['variance_structure'].get('variance_assumption_holds', True):
            issues.append('Variance structure assumption violated')

        # Score (5 = all assumptions hold, 0 = all violated)
        n_tests = 5
        n_passed = n_tests - len(issues)
        score = n_passed / n_tests * 100

        return {
            'adequacy_score': score,
            'issues': issues,
            'n_issues': len(issues),
            'rating': (
                'Excellent' if score >= 80 else
                'Good' if score >= 60 else
                'Fair' if score >= 40 else
                'Poor'
            ),
            'recommendation': (
                'Chain-ladder assumptions generally hold' if score >= 80 else
                'Minor violations - consider robustness checks' if score >= 60 else
                'Moderate violations - consider alternative methods' if score >= 40 else
                'Significant violations - alternative methods recommended'
            )
        }

    def print_summary(self):
        """Print formatted summary of all diagnostic tests."""
        print("\n" + "="*80)
        print("DIAGNOSTIC TESTS SUMMARY")
        print("="*80)

        all_tests = self.run_all_tests()
        adequacy = self.get_model_adequacy_score()

        # Calendar year effects
        print("\n📊 CALENDAR YEAR EFFECTS:")
        print("-"*60)
        cye = all_tests['calendar_year_effect']
        if 'anova_p_value' in cye:
            print(f"ANOVA F-test p-value: {cye['anova_p_value']:.4f}")
            print(f"Effect detected: {'Yes' if cye.get('calendar_effect_detected') else 'No'}")
        if 'trend_p_value' in cye:
            print(f"Trend p-value: {cye['trend_p_value']:.4f}")
            print(f"Trend detected: {'Yes' if cye.get('trend_detected') else 'No'}")

        # Accident year effects
        print("\n\n📊 ACCIDENT YEAR EFFECTS:")
        print("-"*60)
        aye = all_tests['accident_year_effect']
        print(f"Anomalous years: {aye['n_anomalous']}")
        if aye['anomalous_years']:
            for ay in aye['anomalous_years']:
                print(f"  {ay['year']}: {ay['interpretation']} development (z={ay['z_score']:.2f})")

        # Independence
        print("\n\n📊 INDEPENDENCE TEST:")
        print("-"*60)
        ind = all_tests['independence']
        print(f"Significant correlations: {ind['n_significant']} of {ind['n_tested']}")
        print(f"Independence holds: {'Yes' if not ind['independence_assumption_violated'] else 'No'}")

        # Proportionality
        print("\n\n📊 PROPORTIONALITY TEST:")
        print("-"*60)
        prop = all_tests['proportionality']
        print(f"Non-proportional periods: {len(prop['non_proportional_periods'])}")
        if prop['non_proportional_periods']:
            print(f"  Periods: {prop['non_proportional_periods']}")
        print(f"Proportionality holds: {'Yes' if prop['proportionality_holds'] else 'No'}")

        # Variance structure
        print("\n\n📊 VARIANCE STRUCTURE TEST:")
        print("-"*60)
        var = all_tests['variance_structure']
        print(f"Heteroscedastic periods: {len(var['heteroscedastic_periods'])}")
        if var['heteroscedastic_periods']:
            print(f"  Periods: {var['heteroscedastic_periods']}")
        print(f"Variance assumption holds: {'Yes' if var['variance_assumption_holds'] else 'No'}")

        # Overall assessment
        print("\n\n📊 OVERALL MODEL ADEQUACY:")
        print("-"*60)
        print(f"Adequacy Score: {adequacy['adequacy_score']:.0f}%")
        print(f"Rating: {adequacy['rating']}")
        print(f"\nRecommendation: {adequacy['recommendation']}")

        if adequacy['issues']:
            print("\nIssues identified:")
            for issue in adequacy['issues']:
                print(f"  • {issue}")

        print("\n" + "="*80)

    def save_results(self, output_dir: Path):
        """Save diagnostic test results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_tests = self.run_all_tests()
        adequacy = self.get_model_adequacy_score()

        # Save summary
        with open(output_dir / "diagnostic_summary.txt", 'w') as f:
            f.write("DIAGNOSTIC TESTS SUMMARY\n")
            f.write("="*60 + "\n\n")

            f.write(f"Adequacy Score: {adequacy['adequacy_score']:.0f}%\n")
            f.write(f"Rating: {adequacy['rating']}\n")
            f.write(f"Recommendation: {adequacy['recommendation']}\n\n")

            if adequacy['issues']:
                f.write("Issues identified:\n")
                for issue in adequacy['issues']:
                    f.write(f"  • {issue}\n")

        # Save factor matrix
        self.factor_matrix.to_csv(output_dir / "factor_matrix.csv")

        print(f"✅ Diagnostic tests saved to {output_dir}/")
