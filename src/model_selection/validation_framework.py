"""
Validation Framework for Reserving Model Selection

Implements out-of-sample validation schemes for loss development triangles,
including holdout validation, rolling-origin validation, and k-fold approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
from .factor_estimators import FactorEstimator


@dataclass
class ValidationResult:
    """
    Container for validation results from a single estimator.
    
    Attributes:
        estimator_name: Name of the factor estimation method
        predictions: Predicted values for holdout diagonal
        actuals: Actual observed values for holdout diagonal
        factors: Estimated development factors
        errors: Dictionary of error metrics
    """
    estimator_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    factors: pd.Series
    errors: Dict[str, float]
    
    def __repr__(self):
        return f"ValidationResult(estimator={self.estimator_name}, errors={self.errors})"


class TriangleValidator:
    """
    Base class for triangle validation schemes.
    
    Provides infrastructure for out-of-sample validation of factor estimation
    methods on loss development triangles.
    """
    
    def __init__(self, triangle: pd.DataFrame, verbose: bool = True):
        """
        Initialize validator with a loss triangle.
        
        Args:
            triangle: Full loss development triangle
            verbose: Whether to print progress messages
        """
        self.triangle = triangle.copy()
        self.verbose = verbose
        self.validation_results = {}
        
    def _remove_last_diagonal(self, triangle: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove the last observed diagonal from a triangle.
        
        This creates a pseudo out-of-sample set by removing the most recent
        observations across all accident years.
        
        Args:
            triangle: Full triangle
            
        Returns:
            Tuple of (reduced_triangle, holdout_diagonal)
        """
        reduced = triangle.copy()
        holdout = pd.DataFrame(
            index=triangle.index,
            columns=['holdout_value', 'holdout_age'],
            dtype=float
        )
        
        # For each accident year, find last non-NaN value and remove it
        for year in triangle.index:
            row = triangle.loc[year]
            non_nan = row.dropna()
            
            if len(non_nan) > 0:
                last_age = non_nan.index[-1]
                last_value = non_nan.iloc[-1]
                
                # Store holdout value
                holdout.loc[year, 'holdout_value'] = last_value
                holdout.loc[year, 'holdout_age'] = last_age
                
                # Remove from reduced triangle
                reduced.loc[year, last_age] = np.nan
        
        return reduced, holdout
    
    def _predict_next_diagonal(
        self, 
        reduced_triangle: pd.DataFrame, 
        holdout: pd.DataFrame,
        estimator: FactorEstimator
    ) -> np.ndarray:
        """
        Predict the next diagonal using estimated factors.
        
        Args:
            reduced_triangle: Triangle with last diagonal removed
            holdout: DataFrame with holdout positions
            estimator: Factor estimation method
            
        Returns:
            Array of predictions for holdout diagonal
        """
        # Estimate factors on reduced triangle
        factors = estimator.estimate(reduced_triangle)
        
        predictions = []
        
        for year in reduced_triangle.index:
            # Find the last available value in reduced triangle
            row = reduced_triangle.loc[year]
            non_nan = row.dropna()
            
            if len(non_nan) == 0:
                predictions.append(np.nan)
                continue
            
            last_age = non_nan.index[-1]
            last_value = non_nan.iloc[-1]
            
            # Get the factor to apply
            if last_age in factors.index:
                factor = factors[last_age]
                prediction = last_value * factor
            else:
                # No factor available (shouldn't happen in normal cases)
                prediction = np.nan
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def validate_estimator(
        self, 
        estimator: FactorEstimator,
        error_metrics: Dict[str, Callable] = None
    ) -> ValidationResult:
        """
        Validate a single estimator using holdout validation.
        
        Args:
            estimator: Factor estimation method to validate
            error_metrics: Dictionary of error metric functions
            
        Returns:
            ValidationResult object
        """
        if self.verbose:
            print(f"  Validating: {estimator.name}")
        
        # Remove last diagonal
        reduced, holdout = self._remove_last_diagonal(self.triangle)
        
        # Predict holdout values
        predictions = self._predict_next_diagonal(reduced, holdout, estimator)
        actuals = holdout['holdout_value'].values
        
        # Calculate errors
        if error_metrics is None:
            from .error_metrics import get_default_metrics
            error_metrics = get_default_metrics()
        
        errors = {}
        for metric_name, metric_func in error_metrics.items():
            errors[metric_name] = metric_func(actuals, predictions)
        
        # Store factors used
        factors = estimator.estimate(reduced)
        
        result = ValidationResult(
            estimator_name=estimator.name,
            predictions=predictions,
            actuals=actuals,
            factors=factors,
            errors=errors
        )
        
        self.validation_results[estimator.name] = result
        
        return result
    
    def validate_multiple(
        self, 
        estimators: List[FactorEstimator],
        error_metrics: Dict[str, Callable] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple estimators and compare performance.
        
        Args:
            estimators: List of factor estimation methods
            error_metrics: Dictionary of error metric functions
            
        Returns:
            Dictionary mapping estimator names to ValidationResult objects
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"HOLDOUT VALIDATION")
            print(f"{'='*80}")
            print(f"Triangle shape: {self.triangle.shape}")
            print(f"Number of estimators: {len(estimators)}")
            print(f"{'='*80}\n")
        
        results = {}
        
        for estimator in estimators:
            result = self.validate_estimator(estimator, error_metrics)
            results[estimator.name] = result
        
        self.validation_results = results
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("VALIDATION COMPLETE")
            print(f"{'='*80}\n")
        
        return results
    
    def get_best_estimator(self, metric: str = 'RMSE') -> Tuple[str, ValidationResult]:
        """
        Get the best performing estimator based on a specific metric.
        
        Args:
            metric: Error metric to use for comparison (lower is better)
            
        Returns:
            Tuple of (best_estimator_name, ValidationResult)
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_multiple first.")
        
        best_name = None
        best_error = float('inf')
        
        for name, result in self.validation_results.items():
            if metric in result.errors:
                error = result.errors[metric]
                if error < best_error:
                    best_error = error
                    best_name = name
        
        return best_name, self.validation_results[best_name]
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all validation results.
        
        Returns:
            DataFrame with estimators as rows and metrics as columns
        """
        if not self.validation_results:
            raise ValueError("No validation results available.")
        
        # Get all unique metrics
        all_metrics = set()
        for result in self.validation_results.values():
            all_metrics.update(result.errors.keys())
        
        all_metrics = sorted(all_metrics)
        
        # Build summary table
        summary = pd.DataFrame(
            index=list(self.validation_results.keys()),
            columns=all_metrics
        )
        
        for name, result in self.validation_results.items():
            for metric in all_metrics:
                if metric in result.errors:
                    summary.loc[name, metric] = result.errors[metric]
        
        # Sort by first metric (typically RMSE or MAE)
        summary = summary.sort_values(by=all_metrics[0])
        
        return summary


class RollingOriginValidator(TriangleValidator):
    """
    Rolling-origin validation for loss triangles.
    
    Removes multiple diagonals sequentially, trains on progressively larger
    training sets, and validates on each removed diagonal. Provides more
    robust performance estimates than single holdout.
    
    This is analogous to time-series cross-validation adapted for triangular data.
    """
    
    def __init__(self, triangle: pd.DataFrame, n_holdouts: int = 3, verbose: bool = True):
        """
        Initialize rolling-origin validator.
        
        Args:
            triangle: Full loss development triangle
            n_holdouts: Number of diagonals to hold out sequentially
            verbose: Whether to print progress
        """
        super().__init__(triangle, verbose)
        self.n_holdouts = n_holdouts
        
    def _remove_n_diagonals(self, triangle: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Remove n diagonals from the triangle.
        
        Args:
            triangle: Full triangle
            n: Number of diagonals to remove
            
        Returns:
            Tuple of (reduced_triangle, list_of_holdout_diagonals)
        """
        current = triangle.copy()
        holdouts = []
        
        for i in range(n):
            reduced, holdout = self._remove_last_diagonal(current)
            holdouts.append(holdout)
            current = reduced
        
        # Reverse holdouts so they're in chronological order
        holdouts.reverse()
        
        return current, holdouts
    
    def validate_estimator(
        self, 
        estimator: FactorEstimator,
        error_metrics: Dict[str, Callable] = None
    ) -> ValidationResult:
        """
        Validate estimator using rolling-origin approach.
        
        Args:
            estimator: Factor estimation method
            error_metrics: Dictionary of error metric functions
            
        Returns:
            ValidationResult with aggregated errors across all holdouts
        """
        if self.verbose:
            print(f"  Validating: {estimator.name} (rolling origin, n={self.n_holdouts})")
        
        # Get error metrics
        if error_metrics is None:
            from .error_metrics import get_default_metrics
            error_metrics = get_default_metrics()
        
        all_predictions = []
        all_actuals = []
        
        # For each holdout window
        for i in range(self.n_holdouts):
            # Remove i+1 diagonals
            reduced, holdouts = self._remove_n_diagonals(self.triangle, i + 1)
            
            # The most recent holdout is our test set
            test_holdout = holdouts[-1]
            
            # Predict on it
            predictions = self._predict_next_diagonal(reduced, test_holdout, estimator)
            actuals = test_holdout['holdout_value'].values
            
            all_predictions.extend(predictions[~np.isnan(predictions)])
            all_actuals.extend(actuals[~np.isnan(predictions)])
        
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # Calculate aggregated errors
        errors = {}
        for metric_name, metric_func in error_metrics.items():
            errors[metric_name] = metric_func(all_actuals, all_predictions)
        
        # Get factors from full reduced triangle (last iteration)
        reduced, _ = self._remove_n_diagonals(self.triangle, self.n_holdouts)
        factors = estimator.estimate(reduced)
        
        result = ValidationResult(
            estimator_name=estimator.name,
            predictions=all_predictions,
            actuals=all_actuals,
            factors=factors,
            errors=errors
        )
        
        self.validation_results[estimator.name] = result
        
        return result
    
    def validate_multiple(
        self, 
        estimators: List[FactorEstimator],
        error_metrics: Dict[str, Callable] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple estimators using rolling-origin approach.
        
        Args:
            estimators: List of factor estimation methods
            error_metrics: Dictionary of error metric functions
            
        Returns:
            Dictionary of validation results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ROLLING-ORIGIN VALIDATION")
            print(f"{'='*80}")
            print(f"Triangle shape: {self.triangle.shape}")
            print(f"Number of holdouts: {self.n_holdouts}")
            print(f"Number of estimators: {len(estimators)}")
            print(f"{'='*80}\n")
        
        results = {}
        
        for estimator in estimators:
            result = self.validate_estimator(estimator, error_metrics)
            results[estimator.name] = result
        
        self.validation_results = results
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("ROLLING-ORIGIN VALIDATION COMPLETE")
            print(f"{'='*80}\n")
        
        return results


def create_validator(
    triangle: pd.DataFrame,
    method: str = 'holdout',
    **kwargs
) -> TriangleValidator:
    """
    Factory function to create appropriate validator.
    
    Args:
        triangle: Loss development triangle
        method: Validation method ('holdout' or 'rolling_origin')
        **kwargs: Additional arguments for specific validator
        
    Returns:
        Validator instance
    """
    if method == 'holdout':
        return TriangleValidator(triangle, **kwargs)
    elif method == 'rolling_origin':
        return RollingOriginValidator(triangle, **kwargs)
    else:
        raise ValueError(f"Unknown validation method: {method}")