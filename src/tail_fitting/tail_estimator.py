"""
Tail Factor Estimation Module
=============================

Automatic tail factor estimation using multiple curve fitting methods.

Methods implemented:
1. Exponential decay: f(k) = 1 + a * exp(-b * k)
2. Inverse power: f(k) = 1 + a / k^b
3. Weibull: f(k) = 1 + a * exp(-(k/b)^c)
4. Sherman curve: f(k) = 1 + a / (b + k)
5. Bondy method: extrapolation based on last factors
6. Simple extrapolation: linear/log-linear decay

References:
- Sherman, R. (1984). "Extrapolating, Smoothing and Interpolating Development Factors"
- Bardis, E., Majidi, A., Murphy, D. (2012). "A Family of Chain-Ladder Factor Models"
- England, P., Verrall, R. (2002). "Stochastic Claims Reserving in General Insurance"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class TailFittingResult:
    """
    Container for tail fitting results.

    Attributes:
        method: Name of the fitting method
        tail_factor: Estimated tail factor (cumulative from last observed to ultimate)
        extrapolated_factors: Age-to-age factors beyond observed data
        parameters: Fitted model parameters
        r_squared: Goodness of fit (R²)
        aic: Akaike Information Criterion
        rmse: Root Mean Square Error
        years_to_ultimate: Estimated years to reach ultimate (99.5% developed)
    """
    method: str
    tail_factor: float
    extrapolated_factors: pd.Series
    parameters: Dict[str, float]
    r_squared: float
    aic: float = None
    rmse: float = None
    years_to_ultimate: int = None
    convergence: bool = True
    message: str = ""


class TailEstimator:
    """
    Automatic tail factor estimation using curve fitting.

    Fits multiple parametric curves to observed development factors
    and extrapolates to estimate the tail factor.

    Example:
        >>> estimator = TailEstimator(triangle)
        >>> estimator.fit()
        >>> print(estimator.best_method)
        >>> print(estimator.tail_factor)
        >>> estimator.print_comparison()
    """

    # Available fitting methods
    METHODS = [
        'exponential',
        'inverse_power',
        'weibull',
        'sherman',
        'bondy',
        'linear_decay',
        'log_linear'
    ]

    def __init__(
        self,
        triangle: pd.DataFrame,
        development_factors: pd.Series = None,
        min_periods_for_fit: int = 5,
        extrapolation_periods: int = 100,
        convergence_threshold: float = 0.9995
    ):
        """
        Initialize tail estimator.

        Args:
            triangle: Loss development triangle
            development_factors: Pre-calculated age-to-age factors (optional)
            min_periods_for_fit: Minimum periods needed for curve fitting
            extrapolation_periods: Maximum periods to extrapolate
            convergence_threshold: % developed considered "ultimate" (default 99.95%)
        """
        self.triangle = triangle
        self.n_periods = len(triangle.columns)
        self.min_periods_for_fit = min_periods_for_fit
        self.extrapolation_periods = extrapolation_periods
        self.convergence_threshold = convergence_threshold

        # Calculate development factors if not provided
        if development_factors is not None:
            self.development_factors = development_factors
        else:
            self.development_factors = self._calculate_development_factors()

        # Results storage
        self.results: Dict[str, TailFittingResult] = {}
        self.best_method: str = None
        self.tail_factor: float = None
        self.extrapolated_factors: pd.Series = None

    def _calculate_development_factors(self) -> pd.Series:
        """Calculate age-to-age factors from triangle using volume-weighted average."""
        factors = {}

        for i, col in enumerate(self.triangle.columns[:-1]):
            next_col = self.triangle.columns[i + 1]

            current = self.triangle[col].dropna()
            next_vals = self.triangle[next_col].dropna()

            # Align indices
            common_idx = current.index.intersection(next_vals.index)

            if len(common_idx) > 0:
                c = current.loc[common_idx]
                n = next_vals.loc[common_idx]

                # Volume-weighted average
                factor = n.sum() / c.sum() if c.sum() != 0 else 1.0
                factors[col] = factor

        return pd.Series(factors)

    def fit(self, methods: List[str] = None) -> 'TailEstimator':
        """
        Fit all specified methods and select the best one.

        Args:
            methods: List of methods to fit (default: all available)

        Returns:
            self for method chaining
        """
        if methods is None:
            methods = self.METHODS

        # Prepare data for fitting
        # Use factors > 1 (still developing) and exclude very early volatile periods
        factors_for_fit = self.development_factors.copy()

        # Convert index to numeric periods
        x_data = np.arange(1, len(factors_for_fit) + 1)
        y_data = factors_for_fit.values

        # Only fit to factors > 1 (still developing)
        mask = y_data > 1.0001
        if mask.sum() < self.min_periods_for_fit:
            # Not enough developing periods, use simple approach
            warnings.warn(f"Only {mask.sum()} developing periods. Using simple tail estimation.")
            methods = ['bondy', 'linear_decay']

        # Fit each method
        for method in methods:
            try:
                result = self._fit_method(method, x_data, y_data)
                self.results[method] = result
            except Exception as e:
                warnings.warn(f"Method {method} failed: {str(e)}")
                self.results[method] = TailFittingResult(
                    method=method,
                    tail_factor=np.nan,
                    extrapolated_factors=pd.Series(),
                    parameters={},
                    r_squared=0,
                    convergence=False,
                    message=str(e)
                )

        # Select best method based on AIC/R²
        self._select_best_method()

        return self

    def _fit_method(self, method: str, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """Fit a specific method to the data."""

        if method == 'exponential':
            return self._fit_exponential(x, y)
        elif method == 'inverse_power':
            return self._fit_inverse_power(x, y)
        elif method == 'weibull':
            return self._fit_weibull(x, y)
        elif method == 'sherman':
            return self._fit_sherman(x, y)
        elif method == 'bondy':
            return self._fit_bondy(x, y)
        elif method == 'linear_decay':
            return self._fit_linear_decay(x, y)
        elif method == 'log_linear':
            return self._fit_log_linear(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fit_exponential(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Exponential decay model: f(k) = 1 + a * exp(-b * k)

        Linearization: log(f - 1) = log(a) - b * k
        """
        # Only use factors > 1
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 3:
            return self._fallback_result('exponential', 'Insufficient data')

        # Linearize
        y_log = np.log(y_fit - 1)

        # Linear regression
        n = len(x_fit)
        x_mean = np.mean(x_fit)
        y_mean = np.mean(y_log)

        ss_xy = np.sum((x_fit - x_mean) * (y_log - y_mean))
        ss_xx = np.sum((x_fit - x_mean) ** 2)

        if ss_xx == 0:
            return self._fallback_result('exponential', 'Zero variance in x')

        b = -ss_xy / ss_xx  # Note: negative because we want decay
        log_a = y_mean + b * x_mean
        a = np.exp(log_a)

        # Ensure b > 0 (decay)
        if b <= 0:
            b = 0.1  # Default small decay

        # Calculate fit statistics
        y_pred = 1 + a * np.exp(-b * x)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 2)

        # Extrapolate
        extrapolated, tail_factor, years_to_ult = self._extrapolate(
            lambda k: 1 + a * np.exp(-b * k),
            len(x)
        )

        return TailFittingResult(
            method='exponential',
            tail_factor=tail_factor,
            extrapolated_factors=extrapolated,
            parameters={'a': a, 'b': b},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=years_to_ult
        )

    def _fit_inverse_power(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Inverse power model: f(k) = 1 + a / k^b

        Linearization: log(f - 1) = log(a) - b * log(k)
        """
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 3:
            return self._fallback_result('inverse_power', 'Insufficient data')

        # Linearize
        x_log = np.log(x_fit)
        y_log = np.log(y_fit - 1)

        # Linear regression
        n = len(x_fit)
        x_mean = np.mean(x_log)
        y_mean = np.mean(y_log)

        ss_xy = np.sum((x_log - x_mean) * (y_log - y_mean))
        ss_xx = np.sum((x_log - x_mean) ** 2)

        if ss_xx == 0:
            return self._fallback_result('inverse_power', 'Zero variance in x')

        neg_b = ss_xy / ss_xx
        b = -neg_b  # We want f(k) = 1 + a/k^b with b > 0
        log_a = y_mean - neg_b * x_mean
        a = np.exp(log_a)

        if b <= 0:
            b = 1.0  # Default

        # Calculate fit statistics
        y_pred = 1 + a / (x ** b)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 2)

        # Extrapolate
        extrapolated, tail_factor, years_to_ult = self._extrapolate(
            lambda k: 1 + a / (k ** b),
            len(x)
        )

        return TailFittingResult(
            method='inverse_power',
            tail_factor=tail_factor,
            extrapolated_factors=extrapolated,
            parameters={'a': a, 'b': b},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=years_to_ult
        )

    def _fit_weibull(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Weibull model: f(k) = 1 + a * exp(-(k/b)^c)

        Simplified: fix c=2 and fit a, b
        """
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 3:
            return self._fallback_result('weibull', 'Insufficient data')

        c = 2.0  # Fixed shape parameter

        # Linearize: log(f - 1) = log(a) - (k/b)^c
        # log(-log((f-1)/a)) = c*log(k) - c*log(b)
        # Approximate: assume a ≈ y[0] - 1

        a_init = max(y_fit[0] - 1, 0.1)

        # Grid search for b
        best_b = 1.0
        best_sse = np.inf

        for b in np.linspace(0.5, 20, 50):
            y_pred = 1 + a_init * np.exp(-(x_fit / b) ** c)
            sse = np.sum((y_fit - y_pred) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_b = b

        b = best_b

        # Refine a
        y_pred_shape = np.exp(-(x_fit / b) ** c)
        a = np.sum((y_fit - 1) * y_pred_shape) / np.sum(y_pred_shape ** 2)
        a = max(a, 0.01)

        # Calculate fit statistics
        y_pred = 1 + a * np.exp(-(x / b) ** c)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 3)

        # Extrapolate
        extrapolated, tail_factor, years_to_ult = self._extrapolate(
            lambda k: 1 + a * np.exp(-(k / b) ** c),
            len(x)
        )

        return TailFittingResult(
            method='weibull',
            tail_factor=tail_factor,
            extrapolated_factors=extrapolated,
            parameters={'a': a, 'b': b, 'c': c},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=years_to_ult
        )

    def _fit_sherman(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Sherman curve: f(k) = 1 + a / (b + k)

        Linearization: 1/(f-1) = b/a + k/a
        """
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 3:
            return self._fallback_result('sherman', 'Insufficient data')

        # Linearize
        y_inv = 1 / (y_fit - 1)

        # Linear regression: y_inv = b/a + (1/a) * x
        n = len(x_fit)
        x_mean = np.mean(x_fit)
        y_mean = np.mean(y_inv)

        ss_xy = np.sum((x_fit - x_mean) * (y_inv - y_mean))
        ss_xx = np.sum((x_fit - x_mean) ** 2)

        if ss_xx == 0:
            return self._fallback_result('sherman', 'Zero variance in x')

        slope = ss_xy / ss_xx  # 1/a
        intercept = y_mean - slope * x_mean  # b/a

        if slope <= 0:
            return self._fallback_result('sherman', 'Invalid parameters (slope <= 0)')

        a = 1 / slope
        b = intercept * a

        if b < -min(x):
            b = 0.1  # Ensure b + k > 0

        # Calculate fit statistics
        y_pred = 1 + a / (b + x)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 2)

        # Extrapolate
        extrapolated, tail_factor, years_to_ult = self._extrapolate(
            lambda k: 1 + a / (b + k),
            len(x)
        )

        return TailFittingResult(
            method='sherman',
            tail_factor=tail_factor,
            extrapolated_factors=extrapolated,
            parameters={'a': a, 'b': b},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=years_to_ult
        )

    def _fit_bondy(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Bondy method: Tail = (last_factor)^(last_factor / (last_factor - 1))

        Simple extrapolation based on the last few observed factors.
        """
        # Use last 3 factors that are > 1
        mask = y > 1.0001
        y_dev = y[mask]

        if len(y_dev) < 1:
            return self._fallback_result('bondy', 'No developing factors')

        last_factor = y_dev[-1]

        if last_factor <= 1.0001:
            tail_factor = 1.0
        else:
            # Bondy formula
            exponent = last_factor / (last_factor - 1)
            tail_factor = last_factor ** exponent

        # Cap at reasonable value
        tail_factor = min(tail_factor, 2.0)

        # Generate extrapolated factors (geometric decay)
        decay_rate = (last_factor - 1) * 0.5 if last_factor > 1 else 0
        extrapolated = {}

        cum_factor = 1.0
        for i in range(1, self.extrapolation_periods + 1):
            factor = 1 + decay_rate * (0.5 ** i)
            if factor < 1.0001:
                factor = 1.0
            extrapolated[len(x) + i] = factor
            cum_factor *= factor

            if cum_factor >= tail_factor * 0.999:
                break

        return TailFittingResult(
            method='bondy',
            tail_factor=tail_factor,
            extrapolated_factors=pd.Series(extrapolated),
            parameters={'last_factor': last_factor},
            r_squared=np.nan,  # Not applicable
            aic=np.nan,
            rmse=np.nan,
            years_to_ultimate=len(extrapolated)
        )

    def _fit_linear_decay(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Linear decay in (f-1): factor decreases linearly to 1.
        """
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 2:
            return self._fallback_result('linear_decay', 'Insufficient data')

        # Linear regression on (f - 1)
        y_minus_1 = y_fit - 1

        n = len(x_fit)
        x_mean = np.mean(x_fit)
        y_mean = np.mean(y_minus_1)

        ss_xy = np.sum((x_fit - x_mean) * (y_minus_1 - y_mean))
        ss_xx = np.sum((x_fit - x_mean) ** 2)

        if ss_xx == 0:
            slope = -0.01
        else:
            slope = ss_xy / ss_xx

        intercept = y_mean - slope * x_mean

        # Ensure decreasing
        if slope >= 0:
            slope = -np.mean(y_minus_1) / np.mean(x_fit)

        # Calculate fit statistics
        y_pred = 1 + np.maximum(0, intercept + slope * x)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 2)

        # Extrapolate until factor = 1
        extrapolated = {}
        cum_factor = 1.0

        for i in range(1, self.extrapolation_periods + 1):
            k = len(x) + i
            factor = 1 + max(0, intercept + slope * k)
            extrapolated[k] = factor
            cum_factor *= factor

            if factor <= 1.0001:
                break

        tail_factor = cum_factor

        return TailFittingResult(
            method='linear_decay',
            tail_factor=tail_factor,
            extrapolated_factors=pd.Series(extrapolated),
            parameters={'intercept': intercept, 'slope': slope},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=len(extrapolated)
        )

    def _fit_log_linear(self, x: np.ndarray, y: np.ndarray) -> TailFittingResult:
        """
        Log-linear decay: log(f-1) decreases linearly.
        Equivalent to exponential decay but fitted differently.
        """
        mask = y > 1.0001
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 2:
            return self._fallback_result('log_linear', 'Insufficient data')

        # Log transform
        y_log = np.log(y_fit - 1)

        # Linear regression
        n = len(x_fit)
        x_mean = np.mean(x_fit)
        y_mean = np.mean(y_log)

        ss_xy = np.sum((x_fit - x_mean) * (y_log - y_mean))
        ss_xx = np.sum((x_fit - x_mean) ** 2)

        if ss_xx == 0:
            slope = -0.1
        else:
            slope = ss_xy / ss_xx

        intercept = y_mean - slope * x_mean

        # Ensure decreasing
        if slope >= 0:
            slope = -0.1

        # Calculate fit statistics
        y_pred = 1 + np.exp(intercept + slope * x)
        r_squared = self._calculate_r_squared(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = self._calculate_aic(y, y_pred, 2)

        # Extrapolate
        extrapolated, tail_factor, years_to_ult = self._extrapolate(
            lambda k: 1 + np.exp(intercept + slope * k),
            len(x)
        )

        return TailFittingResult(
            method='log_linear',
            tail_factor=tail_factor,
            extrapolated_factors=extrapolated,
            parameters={'intercept': intercept, 'slope': slope},
            r_squared=r_squared,
            aic=aic,
            rmse=rmse,
            years_to_ultimate=years_to_ult
        )

    def _extrapolate(
        self,
        factor_func,
        last_period: int
    ) -> Tuple[pd.Series, float, int]:
        """Extrapolate factors and calculate tail."""
        extrapolated = {}
        cum_factor = 1.0

        for i in range(1, self.extrapolation_periods + 1):
            k = last_period + i
            factor = factor_func(k)

            # Floor at 1.0
            if factor < 1.0:
                factor = 1.0

            extrapolated[k] = factor
            cum_factor *= factor

            # Check convergence (99.95% developed)
            if factor < 1 + (1 - self.convergence_threshold):
                break

        return pd.Series(extrapolated), cum_factor, len(extrapolated)

    def _fallback_result(self, method: str, message: str) -> TailFittingResult:
        """Return a fallback result when fitting fails."""
        return TailFittingResult(
            method=method,
            tail_factor=1.0,
            extrapolated_factors=pd.Series(),
            parameters={},
            r_squared=0,
            convergence=False,
            message=message
        )

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot

    def _calculate_aic(self, y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
        """Calculate Akaike Information Criterion."""
        n = len(y_true)
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)

        if ss_res <= 0:
            return np.inf

        # AIC = n * log(RSS/n) + 2k
        aic = n * np.log(ss_res / n) + 2 * n_params

        return aic

    def _select_best_method(self):
        """Select best method based on AIC (lower is better) and R²."""
        valid_results = {
            k: v for k, v in self.results.items()
            if v.convergence and not np.isnan(v.tail_factor) and v.tail_factor > 0
        }

        if not valid_results:
            # Fallback to bondy
            if 'bondy' in self.results:
                self.best_method = 'bondy'
            else:
                self.best_method = list(self.results.keys())[0]
        else:
            # Prefer methods with valid AIC
            methods_with_aic = {
                k: v for k, v in valid_results.items()
                if v.aic is not None and not np.isnan(v.aic)
            }

            if methods_with_aic:
                # Select by lowest AIC
                self.best_method = min(methods_with_aic.keys(), key=lambda k: methods_with_aic[k].aic)
            else:
                # Select by highest R²
                self.best_method = max(valid_results.keys(), key=lambda k: valid_results[k].r_squared or 0)

        best_result = self.results[self.best_method]
        self.tail_factor = best_result.tail_factor
        self.extrapolated_factors = best_result.extrapolated_factors

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table of all fitted methods."""
        data = []

        for method, result in self.results.items():
            data.append({
                'Method': method,
                'Tail Factor': result.tail_factor,
                'R²': result.r_squared,
                'AIC': result.aic,
                'RMSE': result.rmse,
                'Years to Ultimate': result.years_to_ultimate,
                'Converged': result.convergence,
                'Selected': '✓' if method == self.best_method else ''
            })

        return pd.DataFrame(data)

    def get_full_development_factors(self) -> pd.Series:
        """Get observed factors + extrapolated tail factors."""
        observed = self.development_factors.copy()

        # Append extrapolated factors
        full_factors = pd.concat([observed, self.extrapolated_factors])

        return full_factors

    def get_full_cumulative_factors(self) -> pd.Series:
        """Get cumulative factors including tail."""
        ata_factors = self.get_full_development_factors()

        # Calculate cumulative (from end to beginning)
        cum_factors = []
        cum = 1.0

        for f in reversed(ata_factors.values):
            cum *= f
            cum_factors.append(cum)

        cum_factors = list(reversed(cum_factors))

        return pd.Series(cum_factors, index=ata_factors.index)

    def print_summary(self):
        """Print summary of tail fitting results."""
        print("\n" + "=" * 70)
        print("TAIL FACTOR ESTIMATION SUMMARY")
        print("=" * 70)

        print(f"\nSelected Method: {self.best_method}")
        print(f"Estimated Tail Factor: {self.tail_factor:.6f}")

        best_result = self.results[self.best_method]

        if best_result.years_to_ultimate:
            print(f"Years to Ultimate: {best_result.years_to_ultimate}")

        print(f"\nParameters: {best_result.parameters}")

        if best_result.r_squared and not np.isnan(best_result.r_squared):
            print(f"R²: {best_result.r_squared:.4f}")

        print("\n" + "-" * 70)
        print("METHOD COMPARISON")
        print("-" * 70)
        print(self.get_comparison_table().to_string(index=False))
        print("=" * 70)

    def save_results(self, output_dir: str):
        """Save results to files."""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        self.get_comparison_table().to_csv(output_path / "tail_fitting_comparison.csv", index=False)

        # Save full factors
        self.get_full_development_factors().to_csv(output_path / "full_development_factors.csv")
        self.get_full_cumulative_factors().to_csv(output_path / "full_cumulative_factors.csv")

        # Save extrapolated factors for best method
        self.extrapolated_factors.to_csv(output_path / "extrapolated_factors.csv")

        print(f"✅ Tail fitting results saved to {output_path}")
