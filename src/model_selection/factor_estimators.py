"""
Factor Estimation Methods for Chain-Ladder Reserving

Implements multiple aggregation strategies for age-to-age development factors,
enabling model selection based on out-of-sample predictive performance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class FactorEstimator(ABC):
    """
    Abstract base class for development factor estimation methods.
    
    Each estimator implements a specific aggregation strategy for computing
    age-to-age development factors from historical loss development data.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.factors_ = None
        
    @abstractmethod
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        """
        Estimate development factors from a loss triangle.
        
        Args:
            triangle: Loss development triangle (n_years × n_periods)
                     Rows = accident years, Columns = development periods
        
        Returns:
            Series of age-to-age development factors (length = n_periods - 1)
        """
        pass
    
    def _calculate_raw_factors(self, triangle: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate raw age-to-age factors for each accident year.
        
        Returns:
            DataFrame of raw factors (n_years × n_periods-1)
        """
        n_periods = len(triangle.columns)
        raw_factors = pd.DataFrame(
            index=triangle.index,
            columns=triangle.columns[:-1]
        )
        
        for i in range(n_periods - 1):
            col_current = triangle.columns[i]
            col_next = triangle.columns[i + 1]
            raw_factors[col_current] = triangle[col_next] / triangle[col_current]
        
        return raw_factors


class SimpleAverageEstimator(FactorEstimator):
    """
    Simple arithmetic average of historical age-to-age factors.
    
    f̂ⱼ = (1/n) Σᵢ fᵢⱼ
    
    Standard chain-ladder approach. Treats all accident years equally.
    """
    
    def __init__(self):
        super().__init__("Simple Average")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        raw_factors = self._calculate_raw_factors(triangle)
        self.factors_ = raw_factors.mean(axis=0)
        return self.factors_.fillna(1.0)


class VolumeWeightedEstimator(FactorEstimator):
    """
    Volume-weighted average of development factors.
    
    f̂ⱼ = Σᵢ (Cᵢⱼ₊₁) / Σᵢ (Cᵢⱼ)
    
    Weights factors by exposure (loss volume). Gives more influence to 
    larger accident years, reducing impact of volatile small exposures.
    """
    
    def __init__(self):
        super().__init__("Volume-Weighted")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        n_periods = len(triangle.columns)
        factors = pd.Series(index=triangle.columns[:-1], dtype=float)
        
        for i in range(n_periods - 1):
            col_current = triangle.columns[i]
            col_next = triangle.columns[i + 1]
            
            # Remove NaN values
            mask = triangle[col_current].notna() & triangle[col_next].notna()
            
            numerator = triangle.loc[mask, col_next].sum()
            denominator = triangle.loc[mask, col_current].sum()
            
            factors[col_current] = numerator / denominator if denominator > 0 else 1.0
        
        self.factors_ = factors
        return factors


class GeometricAverageEstimator(FactorEstimator):
    """
    Geometric mean of development factors.
    
    f̂ⱼ = (∏ᵢ fᵢⱼ)^(1/n)
    
    Less sensitive to extreme values than arithmetic mean. Appropriate when
    factors represent multiplicative processes or when outliers are present.
    """
    
    def __init__(self):
        super().__init__("Geometric Average")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        raw_factors = self._calculate_raw_factors(triangle)
        
        # Geometric mean: exp(mean(log(x)))
        # Handle potential zeros/negatives
        factors = raw_factors.apply(
            lambda col: np.exp(np.log(col[col > 0]).mean()) if (col > 0).any() else 1.0
        )
        
        self.factors_ = factors
        return factors.fillna(1.0)


class MedianEstimator(FactorEstimator):
    """
    Median of development factors.
    
    f̂ⱼ = median(f₁ⱼ, f₂ⱼ, ..., fₙⱼ)
    
    Robust to outliers. Appropriate when development patterns contain
    extreme observations that should not dominate the estimate.
    """
    
    def __init__(self):
        super().__init__("Median")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        raw_factors = self._calculate_raw_factors(triangle)
        self.factors_ = raw_factors.median(axis=0)
        return self.factors_.fillna(1.0)


class LeverageWeightedEstimator(FactorEstimator):
    """
    Leverage-weighted average using regression diagnostics.
    
    Downweights observations with high leverage (extreme values in predictor space).
    Based on Hat matrix diagonal: hᵢ = xᵢ(X'X)⁻¹xᵢ'
    
    Weight: wᵢ = 1 - hᵢ (observations with high leverage get lower weight)
    
    Reduces influence of accident years with atypical loss patterns.
    """
    
    def __init__(self):
        super().__init__("Leverage-Weighted")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        raw_factors = self._calculate_raw_factors(triangle)
        n_periods = len(triangle.columns) - 1
        factors = pd.Series(index=triangle.columns[:-1], dtype=float)
        
        for j, col in enumerate(triangle.columns[:-1]):
            # Get current column values (predictors)
            X = triangle.iloc[:, j].values
            X = X[~np.isnan(X)].reshape(-1, 1)
            
            if len(X) == 0:
                factors[col] = 1.0
                continue
            
            # Calculate leverage (hat values)
            X_mean = X.mean()
            X_centered = X - X_mean
            
            # Leverage for each observation
            n = len(X)
            ss_x = np.sum(X_centered ** 2)
            
            if ss_x > 0:
                leverage = 1/n + (X_centered ** 2) / ss_x
            else:
                leverage = np.ones(n) / n
            
            # Weight = 1 - leverage (downweight high-leverage points)
            weights = 1 - leverage.flatten()
            weights = np.maximum(weights, 0.01)  # Minimum weight 0.01
            
            # Normalize weights to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            # Get corresponding factors
            factors_col = raw_factors[col].values[:len(weights)]
            valid = ~np.isnan(factors_col)
            
            if valid.sum() > 0:
                weighted_avg = np.average(
                    factors_col[valid], 
                    weights=weights[valid]
                )
                factors[col] = weighted_avg
            else:
                factors[col] = 1.0
        
        self.factors_ = factors
        return factors.fillna(1.0)


class ExcludeHighLowEstimator(FactorEstimator):
    """
    Exclude highest and lowest factors, then average.
    
    Trimmed mean approach. Removes extreme values before averaging.
    Similar to Olympic scoring in figure skating.
    
    Args:
        n_exclude: Number of factors to exclude from each tail (default=1)
    """
    
    def __init__(self, n_exclude: int = 1):
        super().__init__(f"Exclude High/Low (n={n_exclude})")
        self.n_exclude = n_exclude
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        raw_factors = self._calculate_raw_factors(triangle)
        
        def trimmed_mean(col):
            valid = col.dropna().values
            if len(valid) <= 2 * self.n_exclude:
                return col.mean()  # Not enough data to trim
            
            sorted_vals = np.sort(valid)
            trimmed = sorted_vals[self.n_exclude:-self.n_exclude]
            return trimmed.mean()
        
        self.factors_ = raw_factors.apply(trimmed_mean)
        return self.factors_.fillna(1.0)


class MackAdjustedEstimator(FactorEstimator):
    """
    Mack's regression-adjusted factors.
    
    Adjusts volume-weighted factors to reduce bias from correlation between
    numerator and denominator in the volume-weighted formula.
    
    Adjustment based on linear regression: C_{i,j+1} = α + β·C_{i,j}
    
    Reference: Mack, T. (1993). "Distribution-free calculation of the standard
    error of chain ladder reserve estimates." ASTIN Bulletin.
    """
    
    def __init__(self):
        super().__init__("Mack Adjusted")
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        n_periods = len(triangle.columns)
        factors = pd.Series(index=triangle.columns[:-1], dtype=float)
        
        for i in range(n_periods - 1):
            col_current = triangle.columns[i]
            col_next = triangle.columns[i + 1]
            
            # Get valid pairs
            mask = triangle[col_current].notna() & triangle[col_next].notna()
            X = triangle.loc[mask, col_current].values
            Y = triangle.loc[mask, col_next].values
            
            if len(X) < 2:
                factors[col_current] = 1.0
                continue
            
            # Fit regression: Y = α + β·X
            X_mean = X.mean()
            Y_mean = Y.mean()
            
            # β = Σ(X - X̄)(Y - Ȳ) / Σ(X - X̄)²
            numerator = np.sum((X - X_mean) * (Y - Y_mean))
            denominator = np.sum((X - X_mean) ** 2)
            
            if denominator > 0:
                beta = numerator / denominator
                alpha = Y_mean - beta * X_mean
                
                # Adjusted factor based on mean of X
                factors[col_current] = (alpha + beta * X_mean) / X_mean if X_mean > 0 else 1.0
            else:
                factors[col_current] = Y.sum() / X.sum() if X.sum() > 0 else 1.0
        
        self.factors_ = factors
        return factors.fillna(1.0)


# Factory function to get all estimators
def get_all_estimators() -> list[FactorEstimator]:
    """
    Returns a list of all available factor estimators.
    
    Returns:
        List of instantiated estimator objects
    """
    return [
        SimpleAverageEstimator(),
        VolumeWeightedEstimator(),
        GeometricAverageEstimator(),
        MedianEstimator(),
        LeverageWeightedEstimator(),
        ExcludeHighLowEstimator(n_exclude=1),
        MackAdjustedEstimator()
    ]


def get_estimator_by_name(name: str) -> Optional[FactorEstimator]:
    """
    Get a specific estimator by name.
    
    Args:
        name: Name of the estimator
    
    Returns:
        Estimator instance or None if not found
    """
    estimators = {est.name: est for est in get_all_estimators()}
    return estimators.get(name)