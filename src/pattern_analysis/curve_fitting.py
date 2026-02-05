"""
Curve Fitting Methods for Development Factors
==============================================

Mathematical tools for fitting curves to development patterns.
These are TOOLS that the intelligent agent can use - the agent
decides IF and HOW to use them based on its analysis.

References:
- Sherman (1984) - "Extrapolating, Smoothing, and Interpolating Development Factors"
- England & Verrall (2002) - "Stochastic Claims Reserving"
- Mack (1999) - "The Standard Error of Chain Ladder Reserve Estimates"
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import warnings


@dataclass
class FitResult:
    """Result of curve fitting."""
    method: str                      # Name of fitting method
    fitted_factors: pd.Series        # Fitted factor values
    original_factors: pd.Series      # Original factor values
    parameters: Dict                 # Fitted parameters
    r_squared: float                 # Goodness of fit
    residuals: pd.Series             # Residuals (original - fitted)
    rmse: float                      # Root mean squared error
    description: str                 # Human-readable description


class CurveFitter:
    """
    Fits various curves to development factor patterns.

    This class provides mathematical tools - the intelligent agent
    decides which method to use and whether to apply it.
    """

    def __init__(self, factors: pd.Series):
        """
        Initialize with development factors.

        Args:
            factors: Series of development factors indexed by period
        """
        self.original_factors = factors.copy()
        self.n_periods = len(factors)

        # Clean data (remove NaN, ensure numeric)
        self.factors = factors.dropna().astype(float)
        self.periods = np.arange(1, len(self.factors) + 1)

    def fit_all(self) -> Dict[str, FitResult]:
        """Fit all available methods and return results."""
        results = {}

        methods = [
            ('exponential_decay', self.fit_exponential_decay),
            ('inverse_power', self.fit_inverse_power),
            ('weibull', self.fit_weibull),
            ('linear_decay', self.fit_linear_decay),
            ('monotonic_spline', self.fit_monotonic_spline),
        ]

        for name, method in methods:
            try:
                result = method()
                if result is not None:
                    results[name] = result
            except Exception as e:
                print(f"  Warning: {name} fitting failed: {e}")

        return results

    def fit_exponential_decay(self) -> Optional[FitResult]:
        """
        Fit exponential decay: f(k) = 1 + a * exp(-b * k)

        This assumes factors decay exponentially towards 1.0.
        Good for patterns with rapid early development.
        """
        def exp_decay(k, a, b):
            return 1 + a * np.exp(-b * k)

        try:
            # Initial guesses
            a0 = float(self.factors.iloc[0] - 1)
            b0 = 0.5

            popt, pcov = curve_fit(
                exp_decay, self.periods, self.factors.values,
                p0=[a0, b0], bounds=([0, 0], [10, 5]),
                maxfev=5000
            )

            fitted = pd.Series(
                exp_decay(self.periods, *popt),
                index=self.factors.index
            )

            return self._create_result(
                method="exponential_decay",
                fitted=fitted,
                params={"a": popt[0], "b": popt[1]},
                description=f"f(k) = 1 + {popt[0]:.4f} * exp(-{popt[1]:.4f} * k)"
            )

        except Exception as e:
            return None

    def fit_inverse_power(self) -> Optional[FitResult]:
        """
        Fit inverse power: f(k) = 1 + a / k^b

        Good for slower, more gradual decay patterns.
        """
        def inv_power(k, a, b):
            return 1 + a / np.power(k, b)

        try:
            a0 = float((self.factors.iloc[0] - 1) * 1)
            b0 = 1.0

            popt, pcov = curve_fit(
                inv_power, self.periods, self.factors.values,
                p0=[a0, b0], bounds=([0, 0.1], [10, 5]),
                maxfev=5000
            )

            fitted = pd.Series(
                inv_power(self.periods, *popt),
                index=self.factors.index
            )

            return self._create_result(
                method="inverse_power",
                fitted=fitted,
                params={"a": popt[0], "b": popt[1]},
                description=f"f(k) = 1 + {popt[0]:.4f} / k^{popt[1]:.4f}"
            )

        except Exception as e:
            return None

    def fit_weibull(self) -> Optional[FitResult]:
        """
        Fit Weibull-based decay: f(k) = 1 + a * exp(-(k/b)^c)

        More flexible than simple exponential, can model various decay shapes.
        """
        def weibull_decay(k, a, b, c):
            return 1 + a * np.exp(-np.power(k / b, c))

        try:
            a0 = float(self.factors.iloc[0] - 1)
            b0 = 2.0
            c0 = 1.0

            popt, pcov = curve_fit(
                weibull_decay, self.periods, self.factors.values,
                p0=[a0, b0, c0], bounds=([0, 0.1, 0.1], [10, 20, 5]),
                maxfev=5000
            )

            fitted = pd.Series(
                weibull_decay(self.periods, *popt),
                index=self.factors.index
            )

            return self._create_result(
                method="weibull",
                fitted=fitted,
                params={"a": popt[0], "b": popt[1], "c": popt[2]},
                description=f"f(k) = 1 + {popt[0]:.4f} * exp(-(k/{popt[1]:.4f})^{popt[2]:.4f})"
            )

        except Exception as e:
            return None

    def fit_linear_decay(self) -> Optional[FitResult]:
        """
        Fit linear decay: f(k) = max(1, a - b*k)

        Simple linear model, often too simplistic but useful as baseline.
        """
        try:
            # Simple linear regression
            x = self.periods
            y = self.factors.values

            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x ** 2)

            b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            a = (sum_y - b * sum_x) / n

            fitted_raw = a + b * self.periods
            fitted = pd.Series(
                np.maximum(fitted_raw, 1.0),  # Floor at 1.0
                index=self.factors.index
            )

            return self._create_result(
                method="linear_decay",
                fitted=fitted,
                params={"intercept": a, "slope": b},
                description=f"f(k) = max(1, {a:.4f} + {b:.4f} * k)"
            )

        except Exception as e:
            return None

    def fit_monotonic_spline(self) -> Optional[FitResult]:
        """
        Fit monotonically decreasing spline.

        Enforces that factors don't increase, while smoothing.
        Uses isotonic regression principle.
        """
        try:
            # First fit a regular spline
            k = min(3, len(self.factors) - 1)  # Spline degree
            s = len(self.factors) * 0.5  # Smoothing factor

            spline = UnivariateSpline(
                self.periods, self.factors.values,
                k=k, s=s
            )

            fitted_raw = spline(self.periods)

            # Enforce monotonicity (decreasing)
            fitted_monotonic = self._make_monotonic_decreasing(fitted_raw)

            # Floor at 1.0
            fitted_monotonic = np.maximum(fitted_monotonic, 1.0)

            fitted = pd.Series(fitted_monotonic, index=self.factors.index)

            return self._create_result(
                method="monotonic_spline",
                fitted=fitted,
                params={"smoothing": s, "degree": k},
                description=f"Monotonic spline (degree={k}, smoothing={s:.1f})"
            )

        except Exception as e:
            return None

    def _make_monotonic_decreasing(self, values: np.ndarray) -> np.ndarray:
        """Enforce monotonically decreasing constraint."""
        result = values.copy()
        for i in range(1, len(result)):
            if result[i] > result[i-1]:
                result[i] = result[i-1]
        return result

    def _create_result(self, method: str, fitted: pd.Series,
                       params: Dict, description: str) -> FitResult:
        """Create a FitResult with computed metrics."""
        residuals = self.factors - fitted

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.factors - self.factors.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        return FitResult(
            method=method,
            fitted_factors=fitted,
            original_factors=self.factors,
            parameters=params,
            r_squared=r_squared,
            residuals=residuals,
            rmse=rmse,
            description=description
        )

    def blend_with_original(self, fitted: pd.Series, weight: float = 0.7) -> pd.Series:
        """
        Blend fitted factors with original.

        Args:
            fitted: Fitted factor values
            weight: Weight for fitted (0-1). Higher = more smoothing.

        Returns:
            Blended factors
        """
        return weight * fitted + (1 - weight) * self.factors

    def get_comparison_table(self, results: Dict[str, FitResult]) -> pd.DataFrame:
        """Create comparison table of all fit methods."""
        rows = []
        for name, result in results.items():
            rows.append({
                'Method': name,
                'R²': result.r_squared,
                'RMSE': result.rmse,
                'Description': result.description
            })

        return pd.DataFrame(rows).sort_values('R²', ascending=False)

    def detect_anomalies_basic(self) -> List[Dict]:
        """
        Basic anomaly detection (used when LLM not available).

        Returns list of detected anomalies with details.
        """
        anomalies = []

        for i in range(1, len(self.factors)):
            prev_factor = self.factors.iloc[i-1]
            curr_factor = self.factors.iloc[i]
            period = self.factors.index[i]

            # Check for non-monotonicity (increase when should decrease)
            if curr_factor > prev_factor:
                anomalies.append({
                    'type': 'non_monotonic',
                    'period': period,
                    'value': curr_factor,
                    'expected_below': prev_factor,
                    'deviation': curr_factor - prev_factor,
                    'message': f"Factor at period {period} ({curr_factor:.4f}) > previous ({prev_factor:.4f})"
                })

        # Check for outliers using IQR
        q1 = self.factors.quantile(0.25)
        q3 = self.factors.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for period, factor in self.factors.items():
            if factor < lower or factor > upper:
                anomalies.append({
                    'type': 'outlier',
                    'period': period,
                    'value': factor,
                    'expected_range': (lower, upper),
                    'message': f"Factor at period {period} ({factor:.4f}) outside IQR range [{lower:.4f}, {upper:.4f}]"
                })

        return anomalies
