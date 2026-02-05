"""
Windowed Factor Estimators

Extends factor estimation methods to operate on any contiguous subset
of development years, generating a combinatorial grid of candidate models.

Each candidate model is defined by:
1. Base aggregation rule (simple average, volume-weighted, etc.)
2. Time window (contiguous subset of accident years)

This enables systematic exploration of the optimal lookback period
for factor estimation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from itertools import combinations
from .factor_estimators import (
    FactorEstimator,
    SimpleAverageEstimator,
    VolumeWeightedEstimator,
    GeometricAverageEstimator,
    MedianEstimator,
    LeverageWeightedEstimator,
    ExcludeHighLowEstimator,
    MackAdjustedEstimator
)


class WindowedEstimator(FactorEstimator):
    """
    Wrapper that applies a base estimator to a specific time window.
    
    This allows systematic exploration of different lookback periods
    for factor estimation.
    
    Attributes:
        base_estimator: Underlying aggregation method
        window_start: Start index of window (inclusive)
        window_end: End index of window (exclusive)
        window_years: Actual accident years in window (set after estimation)
    """
    
    def __init__(
        self,
        base_estimator: FactorEstimator,
        window_start: int,
        window_end: int
    ):
        """
        Initialize windowed estimator.
        
        Args:
            base_estimator: Base aggregation method
            window_start: Start index (0 = oldest year)
            window_end: End index (exclusive)
        """
        self.base_estimator = base_estimator
        self.window_start = window_start
        self.window_end = window_end
        self.window_years = None
        
        # Create descriptive name
        window_length = window_end - window_start
        name = f"{base_estimator.name} [window={window_length}, pos={window_start}]"
        super().__init__(name)
    
    def estimate(self, triangle: pd.DataFrame) -> pd.Series:
        """
        Estimate factors using only the specified window of years.
        
        Args:
            triangle: Full loss development triangle
            
        Returns:
            Series of development factors
        """
        # Extract windowed triangle
        windowed_triangle = triangle.iloc[self.window_start:self.window_end, :]
        
        # Store which years were actually used
        self.window_years = windowed_triangle.index.tolist()
        
        # Apply base estimator to windowed data
        factors = self.base_estimator.estimate(windowed_triangle)
        
        self.factors_ = factors
        return factors
    
    def __repr__(self):
        years_str = f"{self.window_years}" if self.window_years else "not estimated"
        return f"WindowedEstimator({self.base_estimator.name}, years={years_str})"


def generate_all_windows(n_years: int, min_window: int = 2) -> List[Tuple[int, int]]:
    """
    Generate all possible contiguous time windows.
    
    For n_years accident years, generates all contiguous subsets
    of length >= min_window.
    
    Args:
        n_years: Total number of accident years available
        min_window: Minimum window length (default: 2)
        
    Returns:
        List of (start, end) tuples representing windows
    """
    windows = []
    
    # For each possible window length
    for length in range(min_window, n_years + 1):
        # For each possible starting position
        for start in range(n_years - length + 1):
            end = start + length
            windows.append((start, end))
    
    return windows


def generate_windowed_estimators(
    triangle: pd.DataFrame,
    base_estimators: List[FactorEstimator] = None,
    min_window: int = 3,
    max_window: int = 10,
    recent_only: bool = False
) -> List[WindowedEstimator]:
    """
    Generate all combinations of estimators and time windows.
    
    Creates a combinatorial grid of:
    - Base aggregation methods
    - Time windows (contiguous subsets of years)
    
    Args:
        triangle: Loss development triangle
        base_estimators: List of base estimation methods (default: all)
        min_window: Minimum window length (default: 3)
        max_window: Maximum window length (default: 10)
        recent_only: If True, only use windows ending at most recent year
        
    Returns:
        List of WindowedEstimator objects
    """
    if base_estimators is None:
        base_estimators = [
            SimpleAverageEstimator(),
            VolumeWeightedEstimator(),
            GeometricAverageEstimator(),
            MedianEstimator(),
            LeverageWeightedEstimator(),
            ExcludeHighLowEstimator(n_exclude=1),
            MackAdjustedEstimator()
        ]
    
    n_years = len(triangle)
    
    if max_window is None:
        max_window = n_years
    
    # Generate all windows
    all_windows = []
    
    for length in range(min_window, min(max_window, n_years) + 1):
        if recent_only:
            # Only windows ending at most recent year
            start = n_years - length
            end = n_years
            all_windows.append((start, end))
        else:
            # All possible positions
            for start in range(n_years - length + 1):
                end = start + length
                all_windows.append((start, end))
    
    # Create windowed estimators for all combinations
    windowed_estimators = []
    
    for base_est in base_estimators:
        for start, end in all_windows:
            windowed_est = WindowedEstimator(base_est, start, end)
            windowed_estimators.append(windowed_est)
    
    return windowed_estimators


def summarize_window_grid(
    triangle: pd.DataFrame,
    min_window: int = 3,
    max_window: int = 10,
    n_base_estimators: int = 7
) -> Dict:
    """
    Summarize the size of the window grid.
    
    Args:
        triangle: Loss development triangle
        min_window: Minimum window length
        max_window: Maximum window length
        n_base_estimators: Number of base estimation methods
        
    Returns:
        Dictionary with grid statistics
    """
    n_years = len(triangle)
    
    if max_window is None:
        max_window = n_years
    
    # Count windows
    n_windows = 0
    windows_by_length = {}
    
    for length in range(min_window, min(max_window, n_years) + 1):
        n_positions = n_years - length + 1
        n_windows += n_positions
        windows_by_length[length] = n_positions
    
    total_candidates = n_windows * n_base_estimators
    
    return {
        'n_years': n_years,
        'min_window': min_window,
        'max_window': min(max_window, n_years),
        'n_windows': n_windows,
        'n_base_estimators': n_base_estimators,
        'total_candidates': total_candidates,
        'windows_by_length': windows_by_length,
        'years_range': (triangle.index.min(), triangle.index.max())
    }


def get_optimal_window_by_method(
    validation_results: Dict[str, 'ValidationResult'],
    criterion: str = 'RMSE'
) -> pd.DataFrame:
    """
    Identify optimal window for each base estimation method.
    
    Args:
        validation_results: Dictionary of validation results
        criterion: Metric to optimize
        
    Returns:
        DataFrame with best window for each method
    """
    # Parse estimator names to extract method and window info
    results_data = []
    
    for name, result in validation_results.items():
        # Parse name: "Method [window=N, pos=M]"
        if '[window=' in name:
            parts = name.split('[')
            method = parts[0].strip()
            
            # Extract window info
            window_info = parts[1].rstrip(']')
            window_parts = window_info.split(', ')
            
            window_length = int(window_parts[0].split('=')[1])
            window_pos = int(window_parts[1].split('=')[1])
            
            error = result.errors.get(criterion, np.nan)
            
            results_data.append({
                'Method': method,
                'Window_Length': window_length,
                'Window_Position': window_pos,
                'Error': error,
                'Full_Name': name
            })
        else:
            # Non-windowed estimator
            method = name
            error = result.errors.get(criterion, np.nan)
            
            results_data.append({
                'Method': method,
                'Window_Length': np.nan,
                'Window_Position': np.nan,
                'Error': error,
                'Full_Name': name
            })
    
    df = pd.DataFrame(results_data)
    
    # Find best window for each method
    best_by_method = df.loc[df.groupby('Method')['Error'].idxmin()]
    
    return best_by_method.sort_values('Error')


def analyze_window_sensitivity(
    validation_results: Dict[str, 'ValidationResult'],
    criterion: str = 'RMSE'
) -> pd.DataFrame:
    """
    Analyze how performance varies with window length.
    
    Args:
        validation_results: Dictionary of validation results
        criterion: Metric to analyze
        
    Returns:
        DataFrame with performance by window length
    """
    results_data = []
    
    for name, result in validation_results.items():
        if '[window=' in name:
            parts = name.split('[')
            method = parts[0].strip()
            
            window_info = parts[1].rstrip(']')
            window_parts = window_info.split(', ')
            
            window_length = int(window_parts[0].split('=')[1])
            error = result.errors.get(criterion, np.nan)
            
            results_data.append({
                'Method': method,
                'Window_Length': window_length,
                'Error': error
            })
    
    df = pd.DataFrame(results_data)
    
    # Aggregate by method and window length
    summary = df.groupby(['Method', 'Window_Length'])['Error'].agg(['mean', 'std', 'min', 'max'])
    
    return summary


def filter_recent_windows_only(
    estimators: List[WindowedEstimator],
    n_years: int
) -> List[WindowedEstimator]:
    """
    Filter to keep only windows ending at the most recent year.
    
    This is a common constraint: only use the most recent N years.
    
    Args:
        estimators: List of windowed estimators
        n_years: Total number of years in triangle
        
    Returns:
        Filtered list of estimators
    """
    return [
        est for est in estimators
        if est.window_end == n_years
    ]