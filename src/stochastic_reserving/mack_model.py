"""
Mack Chain-Ladder Model

Implementation of Thomas Mack's stochastic chain-ladder model for
calculating standard errors and confidence intervals of reserves.

Reference: Mack, T. (1993). "Distribution-free Calculation of the Standard Error
of Chain Ladder Reserve Estimates"
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

# Native implementations to avoid scipy dependency
def norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF using rational approximation."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p == 0.5:
        return 0.0

    # Rational approximation for inverse normal
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


class MackChainLadder:
    """
    Mack's Chain-Ladder Model for stochastic reserve estimation.

    Provides:
    - Point estimates of reserves (same as standard chain-ladder)
    - Standard errors of development factors
    - Standard errors of reserves by accident year
    - Confidence intervals for reserves
    - Process variance and parameter variance decomposition
    """

    def __init__(self, triangle: pd.DataFrame):
        """
        Initialize Mack model.

        Args:
            triangle: Cumulative loss development triangle
                     Rows = accident years, Columns = development periods
        """
        self.triangle = triangle.copy()
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Results
        self.development_factors = None
        self.sigma_squared = None  # Variance parameters
        self.factor_std_errors = None
        self.reserve_std_errors = None
        self.ultimate = None
        self.reserves = None
        self.mse_reserves = None
        self.confidence_intervals = None

    def fit(self) -> 'MackChainLadder':
        """
        Fit the Mack model to the triangle.

        Returns:
            self for method chaining
        """
        self._calculate_development_factors()
        self._calculate_sigma_squared()
        self._calculate_factor_std_errors()
        self._calculate_reserves()
        self._calculate_reserve_std_errors()

        return self

    def _calculate_development_factors(self):
        """Calculate volume-weighted development factors."""
        factors = []

        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            # Get available data (non-NaN pairs)
            curr_values = self.triangle[col_curr].dropna()
            next_values = self.triangle[col_next].dropna()

            # Find common indices
            common_idx = curr_values.index.intersection(next_values.index)

            if len(common_idx) > 0:
                curr = curr_values.loc[common_idx]
                next_val = next_values.loc[common_idx]

                # Volume-weighted factor
                factor = next_val.sum() / curr.sum()
            else:
                factor = 1.0

            factors.append(factor)

        self.development_factors = pd.Series(
            factors,
            index=self.triangle.columns[:-1],
            name='Development_Factor'
        )

    def _calculate_sigma_squared(self):
        """
        Calculate sigma^2 parameters (Mack's variance estimators).

        sigma^2_j estimates the variance of C_{i,j+1}/C_{i,j} around f_j
        """
        sigma_sq = []

        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            # Get available data
            curr_values = self.triangle[col_curr].dropna()
            next_values = self.triangle[col_next].dropna()
            common_idx = curr_values.index.intersection(next_values.index)

            n_obs = len(common_idx)

            if n_obs > 1:
                curr = curr_values.loc[common_idx]
                next_val = next_values.loc[common_idx]
                f_j = self.development_factors.iloc[j]

                # Mack's sigma^2 estimator
                residuals = curr * (next_val / curr - f_j) ** 2
                sigma_sq_j = residuals.sum() / (n_obs - 1)
            elif j == self.n_periods - 2:
                # Last development period - use extrapolation
                # min(sigma^2_{n-3}/sigma^2_{n-4}, sigma^2_{n-3})
                if len(sigma_sq) >= 2:
                    sigma_sq_j = min(
                        sigma_sq[-1]**2 / sigma_sq[-2] if sigma_sq[-2] > 0 else sigma_sq[-1],
                        sigma_sq[-1]
                    )
                elif len(sigma_sq) >= 1:
                    sigma_sq_j = sigma_sq[-1]
                else:
                    sigma_sq_j = 0.0
            else:
                sigma_sq_j = 0.0

            sigma_sq.append(sigma_sq_j)

        self.sigma_squared = pd.Series(
            sigma_sq,
            index=self.triangle.columns[:-1],
            name='Sigma_Squared'
        )

    def _calculate_factor_std_errors(self):
        """Calculate standard errors of development factors."""
        std_errors = []

        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]

            # Sum of weights for factor j
            curr_values = self.triangle[col_curr].dropna()
            sum_weights = curr_values.sum()

            if sum_weights > 0 and self.sigma_squared.iloc[j] > 0:
                se = np.sqrt(self.sigma_squared.iloc[j] / sum_weights)
            else:
                se = 0.0

            std_errors.append(se)

        self.factor_std_errors = pd.Series(
            std_errors,
            index=self.triangle.columns[:-1],
            name='Factor_SE'
        )

    def _calculate_reserves(self):
        """Calculate ultimate losses and reserves."""
        results = []

        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            latest_value = row.dropna().iloc[-1]
            latest_age_idx = row.dropna().index.tolist().index(row.dropna().index[-1])
            latest_age = row.dropna().index[-1]

            # Calculate ultimate by applying remaining factors
            ultimate = latest_value
            for j in range(latest_age_idx, self.n_periods - 1):
                ultimate *= self.development_factors.iloc[j]

            reserve = ultimate - latest_value

            results.append({
                'Accident_Year': year,
                'Latest_Value': latest_value,
                'Latest_Age': latest_age,
                'Latest_Age_Idx': latest_age_idx,
                'Ultimate': ultimate,
                'Reserve': reserve
            })

        self.ultimate = pd.DataFrame(results).set_index('Accident_Year')
        self.reserves = self.ultimate['Reserve']

    def _calculate_reserve_std_errors(self):
        """
        Calculate standard errors of reserves using Mack's formula.

        MSE(R_i) = C_{i,n}^2 * sum_{j=I+1-i}^{n-1} [
            sigma^2_j / f_j^2 * (1/C_{i,j} + 1/sum_{k=1}^{I+1-j} C_{k,j})
        ]

        Where:
        - C_{i,n} is the ultimate loss for year i
        - sigma^2_j is the variance parameter for period j
        - f_j is the development factor for period j
        """
        mse_results = []

        for i, year in enumerate(self.triangle.index):
            latest_age_idx = int(self.ultimate.loc[year, 'Latest_Age_Idx'])
            ultimate = self.ultimate.loc[year, 'Ultimate']

            if latest_age_idx >= self.n_periods - 1:
                # Already at ultimate, no uncertainty
                mse = 0.0
            else:
                mse = 0.0

                # Current projected value at each future period
                projected = self.ultimate.loc[year, 'Latest_Value']

                for j in range(latest_age_idx, self.n_periods - 1):
                    col = self.triangle.columns[j]
                    f_j = self.development_factors.iloc[j]
                    sigma_sq_j = self.sigma_squared.iloc[j]

                    if f_j > 0 and sigma_sq_j > 0:
                        # Process variance term: 1/C_{i,j}
                        process_var = 1.0 / projected if projected > 0 else 0

                        # Parameter variance term: 1/sum(C_{k,j})
                        sum_weights = self.triangle[col].dropna().sum()
                        param_var = 1.0 / sum_weights if sum_weights > 0 else 0

                        # Add to MSE
                        mse += (sigma_sq_j / f_j**2) * (process_var + param_var)

                    # Update projected value
                    projected *= f_j

                # MSE is multiplied by ultimate squared
                mse = ultimate**2 * mse

            mse_results.append({
                'Accident_Year': year,
                'MSE': mse,
                'SE': np.sqrt(max(0, mse)),
                'CV': np.sqrt(max(0, mse)) / self.ultimate.loc[year, 'Reserve']
                      if self.ultimate.loc[year, 'Reserve'] > 0 else 0
            })

        self.mse_reserves = pd.DataFrame(mse_results).set_index('Accident_Year')
        self.reserve_std_errors = self.mse_reserves['SE']

    def get_confidence_intervals(
        self,
        confidence_levels: list = [0.75, 0.90, 0.95, 0.99]
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals for reserves.

        Uses log-normal distribution assumption for reserves.

        Args:
            confidence_levels: List of confidence levels (e.g., [0.75, 0.90, 0.95])

        Returns:
            DataFrame with confidence intervals by accident year
        """
        results = []

        for year in self.triangle.index:
            reserve = self.ultimate.loc[year, 'Reserve']
            se = self.mse_reserves.loc[year, 'SE']

            row = {'Accident_Year': year, 'Reserve': reserve, 'SE': se}

            if reserve > 0 and se > 0:
                # Log-normal parameters
                cv = se / reserve
                sigma = np.sqrt(np.log(1 + cv**2))
                mu = np.log(reserve) - 0.5 * sigma**2

                for level in confidence_levels:
                    alpha = 1 - level
                    z_lower = norm_ppf(alpha / 2)
                    z_upper = norm_ppf(1 - alpha / 2)

                    lower = np.exp(mu + sigma * z_lower)
                    upper = np.exp(mu + sigma * z_upper)

                    row[f'CI_{int(level*100)}_Lower'] = lower
                    row[f'CI_{int(level*100)}_Upper'] = upper
            else:
                for level in confidence_levels:
                    row[f'CI_{int(level*100)}_Lower'] = reserve
                    row[f'CI_{int(level*100)}_Upper'] = reserve

            results.append(row)

        self.confidence_intervals = pd.DataFrame(results).set_index('Accident_Year')
        return self.confidence_intervals

    def get_total_reserve_distribution(
        self,
        confidence_levels: list = [0.75, 0.90, 0.95, 0.99]
    ) -> Dict:
        """
        Calculate total reserve with standard error and confidence intervals.

        For total reserve, we need to account for correlation between years.
        Mack's formula for total MSE includes covariance terms.

        Returns:
            Dictionary with total reserve statistics
        """
        total_reserve = self.reserves.sum()

        # Calculate total MSE including correlations
        # MSE(R) = sum_i MSE(R_i) + 2 * sum_{i<k} Cov(R_i, R_k)

        total_mse = 0.0

        # Add individual MSEs
        for year in self.triangle.index:
            total_mse += self.mse_reserves.loc[year, 'MSE']

        # Add covariance terms (simplified - assumes independence between years)
        # Full Mack covariance formula is more complex
        # For a full implementation, would need:
        # Cov(R_i, R_k) = C_{i,n} * C_{k,n} * sum_j [sigma^2_j / f_j^2 / sum(C_{l,j})]

        # Simplified: add covariance contribution
        for i, year_i in enumerate(self.triangle.index[:-1]):
            for year_k in self.triangle.index[i+1:]:
                ult_i = self.ultimate.loc[year_i, 'Ultimate']
                ult_k = self.ultimate.loc[year_k, 'Ultimate']

                latest_idx_i = int(self.ultimate.loc[year_i, 'Latest_Age_Idx'])
                latest_idx_k = int(self.ultimate.loc[year_k, 'Latest_Age_Idx'])

                # Start from max of both latest indices
                start_j = max(latest_idx_i, latest_idx_k)

                cov = 0.0
                for j in range(start_j, self.n_periods - 1):
                    col = self.triangle.columns[j]
                    f_j = self.development_factors.iloc[j]
                    sigma_sq_j = self.sigma_squared.iloc[j]

                    if f_j > 0:
                        sum_weights = self.triangle[col].dropna().sum()
                        if sum_weights > 0:
                            cov += sigma_sq_j / f_j**2 / sum_weights

                cov = ult_i * ult_k * cov
                total_mse += 2 * cov

        total_se = np.sqrt(max(0, total_mse))
        total_cv = total_se / total_reserve if total_reserve > 0 else 0

        # Confidence intervals (log-normal)
        ci_results = {}
        if total_reserve > 0 and total_se > 0:
            sigma = np.sqrt(np.log(1 + total_cv**2))
            mu = np.log(total_reserve) - 0.5 * sigma**2

            for level in confidence_levels:
                alpha = 1 - level
                z_lower = norm_ppf(alpha / 2)
                z_upper = norm_ppf(1 - alpha / 2)

                ci_results[f'CI_{int(level*100)}'] = (
                    np.exp(mu + sigma * z_lower),
                    np.exp(mu + sigma * z_upper)
                )

        return {
            'Total_Reserve': total_reserve,
            'Total_SE': total_se,
            'Total_CV': total_cv,
            'Confidence_Intervals': ci_results
        }

    def summary(self) -> pd.DataFrame:
        """
        Get summary of reserves with standard errors.

        Returns:
            DataFrame with reserves, SEs, and CVs by accident year
        """
        if self.mse_reserves is None:
            self.fit()

        summary = self.ultimate[['Latest_Value', 'Ultimate', 'Reserve']].copy()
        summary['SE'] = self.mse_reserves['SE']
        summary['CV'] = self.mse_reserves['CV']

        return summary

    def get_factor_summary(self) -> pd.DataFrame:
        """
        Get summary of development factors with standard errors.

        Returns:
            DataFrame with factors, SEs, and CVs
        """
        return pd.DataFrame({
            'Factor': self.development_factors,
            'Sigma_Sq': self.sigma_squared,
            'SE': self.factor_std_errors,
            'CV': self.factor_std_errors / self.development_factors
        })

    def print_summary(self):
        """Print formatted summary of results."""
        print("\n" + "="*80)
        print("MACK CHAIN-LADDER MODEL RESULTS")
        print("="*80)

        print("\n📊 DEVELOPMENT FACTORS:")
        print("-"*60)
        factor_summary = self.get_factor_summary()
        print(factor_summary.round(4).to_string())

        print("\n\n📊 RESERVES BY ACCIDENT YEAR:")
        print("-"*60)
        summary = self.summary()
        print(summary.round(2).to_string())

        print("\n\n📊 CONFIDENCE INTERVALS (95%):")
        print("-"*60)
        ci = self.get_confidence_intervals([0.95])
        ci_display = ci[['Reserve', 'SE', 'CI_95_Lower', 'CI_95_Upper']].round(2)
        print(ci_display.to_string())

        print("\n\n📊 TOTAL RESERVE:")
        print("-"*60)
        total = self.get_total_reserve_distribution()
        print(f"Total Reserve:     ${total['Total_Reserve']:>15,.0f}")
        print(f"Standard Error:    ${total['Total_SE']:>15,.0f}")
        print(f"Coefficient of Var: {total['Total_CV']:>14.2%}")

        if total['Confidence_Intervals']:
            print("\nConfidence Intervals:")
            for level, (lower, upper) in total['Confidence_Intervals'].items():
                print(f"  {level}: ${lower:,.0f} - ${upper:,.0f}")

        print("\n" + "="*80)
