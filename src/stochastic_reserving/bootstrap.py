"""
Bootstrap Methods for Chain-Ladder Reserving

Implements residual bootstrapping and ODP (Over-dispersed Poisson)
bootstrapping for generating reserve distributions.

References:
- England, P.D. & Verrall, R.J. (2002). "Stochastic Claims Reserving in General Insurance"
- Shapland, M.R. (2016). "Using the ODP Bootstrap Model"
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import warnings


class BootstrapChainLadder:
    """
    Residual Bootstrap for Chain-Ladder Reserves.

    Process:
    1. Fit standard chain-ladder to get development factors
    2. Calculate standardized residuals (Pearson or adjusted)
    3. Bootstrap: resample residuals, create pseudo-triangles
    4. For each pseudo-triangle, calculate reserves
    5. Aggregate to get reserve distribution
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        n_simulations: int = 10000,
        random_state: Optional[int] = None
    ):
        """
        Initialize bootstrap model.

        Args:
            triangle: Cumulative loss development triangle
            n_simulations: Number of bootstrap simulations
            random_state: Random seed for reproducibility
        """
        self.triangle = triangle.copy()
        self.n_simulations = n_simulations
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        if random_state is not None:
            np.random.seed(random_state)

        # Results
        self.development_factors = None
        self.fitted_triangle = None
        self.residuals = None
        self.simulated_reserves = None
        self.reserve_distribution = None

    def fit(self) -> 'BootstrapChainLadder':
        """
        Fit the bootstrap model.

        Returns:
            self for method chaining
        """
        self._calculate_development_factors()
        self._calculate_fitted_triangle()
        self._calculate_residuals()
        self._run_bootstrap()

        return self

    def _calculate_development_factors(self):
        """Calculate volume-weighted development factors."""
        factors = []

        for j in range(self.n_periods - 1):
            col_curr = self.triangle.columns[j]
            col_next = self.triangle.columns[j + 1]

            curr_values = self.triangle[col_curr].dropna()
            next_values = self.triangle[col_next].dropna()
            common_idx = curr_values.index.intersection(next_values.index)

            if len(common_idx) > 0:
                curr = curr_values.loc[common_idx]
                next_val = next_values.loc[common_idx]
                factor = next_val.sum() / curr.sum()
            else:
                factor = 1.0

            factors.append(factor)

        self.development_factors = pd.Series(
            factors,
            index=self.triangle.columns[:-1]
        )

    def _calculate_fitted_triangle(self):
        """
        Calculate fitted values using the chain-ladder model.

        For the upper triangle (observed):
        m_{i,j} = C_{i,j-1} * f_{j-1}

        Where C_{i,j-1} is the actual observed value.
        """
        fitted = self.triangle.copy()

        # For observed cells, calculate expected value
        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            observed = row.dropna()

            for j in range(1, len(observed)):
                col_prev = self.triangle.columns[j - 1]
                col_curr = self.triangle.columns[j]

                # Fitted = previous actual * factor
                fitted.loc[year, col_curr] = (
                    self.triangle.loc[year, col_prev] *
                    self.development_factors.iloc[j - 1]
                )

        self.fitted_triangle = fitted

    def _calculate_residuals(self):
        """
        Calculate Pearson residuals.

        r_{i,j} = (C_{i,j} - m_{i,j}) / sqrt(m_{i,j})

        Also calculate adjusted residuals for bias correction.
        """
        # Pearson residuals
        residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[1:],  # No residual for first column
            dtype=float
        )

        residual_list = []  # Flat list of all residuals

        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            observed = row.dropna()

            for j in range(1, len(observed)):
                col = self.triangle.columns[j]

                actual = self.triangle.loc[year, col]
                fitted = self.fitted_triangle.loc[year, col]

                if fitted > 0:
                    residual = (actual - fitted) / np.sqrt(fitted)
                else:
                    residual = 0

                residuals.loc[year, col] = residual
                residual_list.append(residual)

        # Adjust for degrees of freedom (bias correction)
        n_residuals = len(residual_list)
        n_params = self.n_periods - 1  # Number of factors estimated
        if n_residuals > n_params:
            scale_factor = np.sqrt(n_residuals / (n_residuals - n_params))
            self.adjusted_residuals = [r * scale_factor for r in residual_list]
        else:
            self.adjusted_residuals = residual_list

        self.residuals = residuals
        self.residual_list = residual_list

    def _run_bootstrap(self):
        """
        Run bootstrap simulations.

        For each simulation:
        1. Resample residuals with replacement
        2. Create pseudo-triangle
        3. Re-estimate factors
        4. Project to ultimate
        5. Calculate reserves
        """
        simulated_reserves = []
        simulated_ultimates = []

        for sim in range(self.n_simulations):
            # Resample residuals
            resampled_residuals = np.random.choice(
                self.adjusted_residuals,
                size=len(self.adjusted_residuals),
                replace=True
            )

            # Create pseudo-triangle
            pseudo_triangle = self._create_pseudo_triangle(resampled_residuals)

            # Re-estimate factors on pseudo-triangle
            pseudo_factors = self._estimate_factors(pseudo_triangle)

            # Project to ultimate (including process variance)
            reserves, ultimates = self._project_reserves(
                pseudo_triangle, pseudo_factors, include_process_variance=True
            )

            simulated_reserves.append(reserves)
            simulated_ultimates.append(ultimates)

        self.simulated_reserves = pd.DataFrame(
            simulated_reserves,
            columns=self.triangle.index
        )
        self.simulated_ultimates = pd.DataFrame(
            simulated_ultimates,
            columns=self.triangle.index
        )

    def _create_pseudo_triangle(self, resampled_residuals: np.ndarray) -> pd.DataFrame:
        """
        Create a pseudo-triangle using resampled residuals.

        C*_{i,j} = m_{i,j} + r*_{i,j} * sqrt(m_{i,j})
        """
        pseudo = self.triangle.copy()
        resid_idx = 0

        for i, year in enumerate(self.triangle.index):
            row = self.triangle.loc[year]
            observed = row.dropna()

            for j in range(1, len(observed)):
                col = self.triangle.columns[j]
                fitted = self.fitted_triangle.loc[year, col]

                if fitted > 0:
                    # Reconstruct using resampled residual
                    pseudo.loc[year, col] = max(
                        0.01,  # Floor to avoid negative/zero values
                        fitted + resampled_residuals[resid_idx] * np.sqrt(fitted)
                    )
                resid_idx += 1

        return pseudo

    def _estimate_factors(self, triangle: pd.DataFrame) -> pd.Series:
        """Estimate development factors from a triangle."""
        factors = []

        for j in range(self.n_periods - 1):
            col_curr = triangle.columns[j]
            col_next = triangle.columns[j + 1]

            curr_values = triangle[col_curr].dropna()
            next_values = triangle[col_next].dropna()
            common_idx = curr_values.index.intersection(next_values.index)

            if len(common_idx) > 0:
                curr = curr_values.loc[common_idx]
                next_val = next_values.loc[common_idx]
                factor = next_val.sum() / curr.sum() if curr.sum() > 0 else 1.0
            else:
                factor = 1.0

            factors.append(factor)

        return pd.Series(factors, index=triangle.columns[:-1])

    def _project_reserves(
        self,
        triangle: pd.DataFrame,
        factors: pd.Series,
        include_process_variance: bool = True
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Project reserves from a triangle using given factors.

        Args:
            triangle: The triangle to project from
            factors: Development factors to use
            include_process_variance: Add gamma-distributed process variance

        Returns:
            Tuple of (reserves, ultimates) as Series
        """
        reserves = {}
        ultimates = {}

        for year in triangle.index:
            row = triangle.loc[year]
            latest_value = row.dropna().iloc[-1]
            latest_idx = list(triangle.columns).index(row.dropna().index[-1])

            # Project to ultimate
            projected = latest_value
            for j in range(latest_idx, self.n_periods - 1):
                projected *= factors.iloc[j]

                # Add process variance (gamma distributed)
                if include_process_variance and projected > 0:
                    # Use gamma to maintain mean while adding variance
                    cv = 0.05  # Small coefficient of variation for process error
                    shape = 1 / cv**2
                    scale = projected * cv**2
                    projected = np.random.gamma(shape, scale)

            ultimates[year] = projected
            reserves[year] = projected - latest_value

        return pd.Series(reserves), pd.Series(ultimates)

    def get_reserve_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics of simulated reserves.

        Returns:
            DataFrame with mean, std, percentiles by accident year
        """
        stats_df = pd.DataFrame(index=self.triangle.index)

        stats_df['Mean'] = self.simulated_reserves.mean()
        stats_df['Std'] = self.simulated_reserves.std()
        stats_df['CV'] = stats_df['Std'] / stats_df['Mean']
        stats_df['P10'] = self.simulated_reserves.quantile(0.10)
        stats_df['P25'] = self.simulated_reserves.quantile(0.25)
        stats_df['P50'] = self.simulated_reserves.quantile(0.50)
        stats_df['P75'] = self.simulated_reserves.quantile(0.75)
        stats_df['P90'] = self.simulated_reserves.quantile(0.90)
        stats_df['P95'] = self.simulated_reserves.quantile(0.95)
        stats_df['P99'] = self.simulated_reserves.quantile(0.99)

        return stats_df

    def get_total_reserve_distribution(self) -> Dict:
        """
        Get distribution of total reserves across all years.

        Returns:
            Dictionary with statistics of total reserve
        """
        total_reserves = self.simulated_reserves.sum(axis=1)

        return {
            'Mean': total_reserves.mean(),
            'Std': total_reserves.std(),
            'CV': total_reserves.std() / total_reserves.mean(),
            'P10': total_reserves.quantile(0.10),
            'P25': total_reserves.quantile(0.25),
            'P50': total_reserves.quantile(0.50),
            'P75': total_reserves.quantile(0.75),
            'P90': total_reserves.quantile(0.90),
            'P95': total_reserves.quantile(0.95),
            'P99': total_reserves.quantile(0.99),
            'Min': total_reserves.min(),
            'Max': total_reserves.max()
        }

    def get_var_tvar(
        self,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate Value-at-Risk and Tail Value-at-Risk for reserves.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with VaR and TVaR for each year and total
        """
        results = {'by_year': {}, 'total': {}}

        # By accident year
        for year in self.triangle.index:
            reserves = self.simulated_reserves[year]
            var = reserves.quantile(confidence_level)
            tvar = reserves[reserves >= var].mean()

            results['by_year'][year] = {
                'VaR': var,
                'TVaR': tvar
            }

        # Total
        total_reserves = self.simulated_reserves.sum(axis=1)
        var_total = total_reserves.quantile(confidence_level)
        tvar_total = total_reserves[total_reserves >= var_total].mean()

        results['total'] = {
            'VaR': var_total,
            'TVaR': tvar_total
        }

        return results

    def print_summary(self):
        """Print formatted summary of bootstrap results."""
        print("\n" + "="*80)
        print("BOOTSTRAP CHAIN-LADDER RESULTS")
        print(f"({self.n_simulations:,} simulations)")
        print("="*80)

        print("\n📊 RESERVE DISTRIBUTION BY ACCIDENT YEAR:")
        print("-"*70)
        stats = self.get_reserve_statistics()
        display_cols = ['Mean', 'Std', 'CV', 'P50', 'P75', 'P95']
        print(stats[display_cols].round(2).to_string())

        print("\n\n📊 TOTAL RESERVE DISTRIBUTION:")
        print("-"*70)
        total = self.get_total_reserve_distribution()
        print(f"Mean:              ${total['Mean']:>15,.0f}")
        print(f"Std Dev:           ${total['Std']:>15,.0f}")
        print(f"Coefficient of Var: {total['CV']:>14.2%}")
        print(f"\nPercentiles:")
        print(f"  10th:            ${total['P10']:>15,.0f}")
        print(f"  25th:            ${total['P25']:>15,.0f}")
        print(f"  50th (Median):   ${total['P50']:>15,.0f}")
        print(f"  75th:            ${total['P75']:>15,.0f}")
        print(f"  90th:            ${total['P90']:>15,.0f}")
        print(f"  95th:            ${total['P95']:>15,.0f}")
        print(f"  99th:            ${total['P99']:>15,.0f}")

        print("\n\n📊 RISK MEASURES (95% Confidence):")
        print("-"*70)
        risk = self.get_var_tvar(0.95)
        print(f"Value-at-Risk (VaR 95%):      ${risk['total']['VaR']:>12,.0f}")
        print(f"Tail VaR (TVaR 95%):          ${risk['total']['TVaR']:>12,.0f}")

        print("\n" + "="*80)


class ODPBootstrap:
    """
    Over-Dispersed Poisson (ODP) Bootstrap.

    Assumes incremental losses follow an ODP distribution.
    Uses GLM framework for parameter estimation.

    This is the method recommended by the CAS Working Party on
    Quantifying Variability in Reserve Estimates.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        n_simulations: int = 10000,
        random_state: Optional[int] = None
    ):
        """
        Initialize ODP bootstrap model.

        Args:
            triangle: Cumulative loss development triangle
            n_simulations: Number of bootstrap simulations
            random_state: Random seed for reproducibility
        """
        self.triangle = triangle.copy()
        self.n_simulations = n_simulations
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        if random_state is not None:
            np.random.seed(random_state)

        # Convert to incremental
        self.incremental = self._to_incremental(triangle)

        # Results
        self.alpha = None  # Row (accident year) parameters
        self.beta = None   # Column (development) parameters
        self.phi = None    # Dispersion parameter
        self.fitted_incremental = None
        self.pearson_residuals = None
        self.simulated_reserves = None

    def _to_incremental(self, cumulative: pd.DataFrame) -> pd.DataFrame:
        """Convert cumulative triangle to incremental."""
        incremental = cumulative.copy()

        for j in range(len(cumulative.columns) - 1, 0, -1):
            col_curr = cumulative.columns[j]
            col_prev = cumulative.columns[j - 1]
            incremental[col_curr] = cumulative[col_curr] - cumulative[col_prev]

        return incremental

    def _to_cumulative(self, incremental: pd.DataFrame) -> pd.DataFrame:
        """Convert incremental triangle to cumulative."""
        cumulative = incremental.copy()

        for j in range(1, len(incremental.columns)):
            col_curr = incremental.columns[j]
            col_prev = incremental.columns[j - 1]
            cumulative[col_curr] = cumulative[col_prev] + incremental[col_curr]

        return cumulative

    def fit(self) -> 'ODPBootstrap':
        """
        Fit the ODP model and run bootstrap.

        Returns:
            self for method chaining
        """
        self._fit_odp_model()
        self._calculate_residuals()
        self._run_bootstrap()

        return self

    def _fit_odp_model(self):
        """
        Fit the ODP model using the chain-ladder structure.

        The ODP model assumes:
        E[I_{i,j}] = alpha_i * beta_j

        Where alpha captures row effects and beta captures column effects.
        """
        # Estimate using chain-ladder relationship
        # beta_j = proportion of total at development j
        # alpha_i = ultimate for row i

        # Get column sums (for beta)
        col_sums = self.incremental.sum()
        total = col_sums.sum()

        self.beta = col_sums / total

        # Get row ultimates (using CL projection for incomplete years)
        # For simplicity, use the cumulative latest values adjusted by completion
        completion = self.triangle.iloc[0].dropna().iloc[-1] / self.triangle.iloc[0].iloc[0]

        self.alpha = {}
        for year in self.triangle.index:
            latest = self.triangle.loc[year].dropna().iloc[-1]
            # Crude completion estimate
            n_developed = self.triangle.loc[year].dropna().shape[0]
            completion_factor = col_sums.iloc[:n_developed].sum() / total
            self.alpha[year] = latest / completion_factor if completion_factor > 0 else latest

        self.alpha = pd.Series(self.alpha)

        # Calculate fitted values
        self.fitted_incremental = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns,
            dtype=float
        )

        for year in self.triangle.index:
            for col in self.triangle.columns:
                if pd.notna(self.incremental.loc[year, col]):
                    self.fitted_incremental.loc[year, col] = (
                        self.alpha[year] * self.beta[col]
                    )

        # Estimate dispersion parameter
        residuals = []
        for year in self.triangle.index:
            for col in self.triangle.columns:
                actual = self.incremental.loc[year, col]
                fitted = self.fitted_incremental.loc[year, col]

                if pd.notna(actual) and pd.notna(fitted) and fitted > 0:
                    pearson_resid = (actual - fitted) / np.sqrt(fitted)
                    residuals.append(pearson_resid ** 2)

        n_obs = len(residuals)
        n_params = self.n_years + self.n_periods - 1  # alpha's + beta's - 1 (constraint)

        if n_obs > n_params:
            self.phi = sum(residuals) / (n_obs - n_params)
        else:
            self.phi = 1.0

    def _calculate_residuals(self):
        """Calculate scaled Pearson residuals."""
        self.pearson_residuals = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns,
            dtype=float
        )

        residual_list = []

        scale = np.sqrt(self.phi) if self.phi else 1.0

        for year in self.triangle.index:
            for col in self.triangle.columns:
                actual = self.incremental.loc[year, col]
                fitted = self.fitted_incremental.loc[year, col]

                if pd.notna(actual) and pd.notna(fitted) and fitted > 0:
                    residual = (actual - fitted) / (np.sqrt(fitted) * scale)
                    self.pearson_residuals.loc[year, col] = residual
                    residual_list.append(residual)

        # Adjust for degrees of freedom
        n_obs = len(residual_list)
        n_params = self.n_years + self.n_periods - 1

        if n_obs > n_params:
            adj_factor = np.sqrt(n_obs / (n_obs - n_params))
            self.adjusted_residuals = [r * adj_factor for r in residual_list]
        else:
            self.adjusted_residuals = residual_list

    def _run_bootstrap(self):
        """Run ODP bootstrap simulations."""
        simulated_reserves = []

        for sim in range(self.n_simulations):
            # Resample residuals
            resampled = np.random.choice(
                self.adjusted_residuals,
                size=len(self.adjusted_residuals),
                replace=True
            )

            # Create pseudo-incremental triangle
            pseudo_incr = self._create_pseudo_incremental(resampled)

            # Convert to cumulative
            pseudo_cum = self._to_cumulative(pseudo_incr)

            # Re-estimate model and project
            reserves = self._project_with_process_variance(pseudo_cum, pseudo_incr)

            simulated_reserves.append(reserves)

        self.simulated_reserves = pd.DataFrame(
            simulated_reserves,
            columns=self.triangle.index
        )

    def _create_pseudo_incremental(self, resampled_residuals: np.ndarray) -> pd.DataFrame:
        """Create pseudo-incremental triangle from resampled residuals."""
        pseudo = self.incremental.copy()
        resid_idx = 0

        scale = np.sqrt(self.phi) if self.phi else 1.0

        for year in self.triangle.index:
            for col in self.triangle.columns:
                fitted = self.fitted_incremental.loc[year, col]

                if pd.notna(fitted) and fitted > 0:
                    # Reconstruct: actual = fitted + residual * sqrt(fitted) * scale
                    pseudo.loc[year, col] = max(
                        0.01,
                        fitted + resampled_residuals[resid_idx] * np.sqrt(fitted) * scale
                    )
                    resid_idx += 1

        return pseudo

    def _project_with_process_variance(
        self,
        pseudo_cum: pd.DataFrame,
        pseudo_incr: pd.DataFrame
    ) -> Dict:
        """
        Project reserves including process variance.

        Uses gamma distribution for process variance in the ODP framework.
        """
        # Re-estimate factors from pseudo-triangle
        factors = []
        for j in range(self.n_periods - 1):
            col_curr = pseudo_cum.columns[j]
            col_next = pseudo_cum.columns[j + 1]

            curr = pseudo_cum[col_curr].dropna()
            next_val = pseudo_cum[col_next].dropna()
            common = curr.index.intersection(next_val.index)

            if len(common) > 0:
                factor = next_val.loc[common].sum() / curr.loc[common].sum()
            else:
                factor = self.development_factors.iloc[j] if hasattr(self, 'development_factors') else 1.0

            factors.append(factor)

        factors = pd.Series(factors, index=pseudo_cum.columns[:-1])

        # Project each year
        reserves = {}

        for year in pseudo_cum.index:
            row = pseudo_cum.loc[year]
            latest = row.dropna().iloc[-1]
            latest_idx = list(pseudo_cum.columns).index(row.dropna().index[-1])

            # Project future incremental with process variance
            projected = latest
            for j in range(latest_idx, self.n_periods - 1):
                expected_incr = projected * (factors.iloc[j] - 1)

                if expected_incr > 0:
                    # ODP process variance: Var = phi * mean
                    # Use gamma: shape = mean/phi, scale = phi
                    shape = expected_incr / self.phi if self.phi > 0 else expected_incr
                    scale = self.phi if self.phi > 0 else 1.0

                    if shape > 0:
                        actual_incr = np.random.gamma(shape, scale)
                    else:
                        actual_incr = expected_incr
                else:
                    actual_incr = 0

                projected += actual_incr

            reserves[year] = projected - latest

        return reserves

    def get_reserve_statistics(self) -> pd.DataFrame:
        """Get summary statistics of simulated reserves."""
        stats_df = pd.DataFrame(index=self.triangle.index)

        stats_df['Mean'] = self.simulated_reserves.mean()
        stats_df['Std'] = self.simulated_reserves.std()
        stats_df['CV'] = stats_df['Std'] / stats_df['Mean'].replace(0, np.nan)
        stats_df['P10'] = self.simulated_reserves.quantile(0.10)
        stats_df['P25'] = self.simulated_reserves.quantile(0.25)
        stats_df['P50'] = self.simulated_reserves.quantile(0.50)
        stats_df['P75'] = self.simulated_reserves.quantile(0.75)
        stats_df['P90'] = self.simulated_reserves.quantile(0.90)
        stats_df['P95'] = self.simulated_reserves.quantile(0.95)
        stats_df['P99'] = self.simulated_reserves.quantile(0.99)

        return stats_df

    def get_total_reserve_distribution(self) -> Dict:
        """Get distribution of total reserves."""
        total = self.simulated_reserves.sum(axis=1)

        return {
            'Mean': total.mean(),
            'Std': total.std(),
            'CV': total.std() / total.mean() if total.mean() > 0 else 0,
            'P10': total.quantile(0.10),
            'P25': total.quantile(0.25),
            'P50': total.quantile(0.50),
            'P75': total.quantile(0.75),
            'P90': total.quantile(0.90),
            'P95': total.quantile(0.95),
            'P99': total.quantile(0.99)
        }

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "="*80)
        print("ODP BOOTSTRAP RESULTS")
        print(f"({self.n_simulations:,} simulations)")
        print("="*80)

        print(f"\nDispersion Parameter (phi): {self.phi:.4f}")

        print("\n📊 RESERVE DISTRIBUTION BY ACCIDENT YEAR:")
        print("-"*70)
        stats = self.get_reserve_statistics()
        display_cols = ['Mean', 'Std', 'CV', 'P50', 'P75', 'P95']
        print(stats[display_cols].round(2).to_string())

        print("\n\n📊 TOTAL RESERVE DISTRIBUTION:")
        print("-"*70)
        total = self.get_total_reserve_distribution()
        print(f"Mean:              ${total['Mean']:>15,.0f}")
        print(f"Std Dev:           ${total['Std']:>15,.0f}")
        print(f"Coefficient of Var: {total['CV']:>14.2%}")
        print(f"\nPercentiles:")
        print(f"  50th (Median):   ${total['P50']:>15,.0f}")
        print(f"  75th:            ${total['P75']:>15,.0f}")
        print(f"  90th:            ${total['P90']:>15,.0f}")
        print(f"  95th:            ${total['P95']:>15,.0f}")
        print(f"  99th:            ${total['P99']:>15,.0f}")

        print("\n" + "="*80)
