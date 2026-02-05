"""
K-Fold Cross-Validation for Loss Development Triangles

Implements k-fold cross-validation adapted for triangular data structures.
Unlike traditional k-fold which randomly partitions data, this uses
structured folds appropriate for loss reserving:

1. Leave-One-Year-Out (LOYO): Each accident year is held out once
2. Grouped K-Fold: Years grouped into k contiguous folds
3. Time-Series K-Fold: Expanding window with k test periods
4. Diagonal K-Fold: Different diagonals used as test sets

References:
- Shi, P. (2017). "Insurance Analytics"
- Wüthrich, M.V. & Merz, M. (2008). "Stochastic Claims Reserving Methods in Insurance"
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional, Generator
from dataclasses import dataclass
from .factor_estimators import FactorEstimator
from .validation_framework import ValidationResult
from .error_metrics import get_default_metrics


@dataclass
class KFoldResult:
    """
    Container for k-fold cross-validation results.

    Attributes:
        estimator_name: Name of the factor estimation method
        fold_results: List of ValidationResults for each fold
        aggregated_errors: Aggregated error metrics across all folds
        error_std: Standard deviation of errors across folds
    """
    estimator_name: str
    fold_results: List[ValidationResult]
    aggregated_errors: Dict[str, float]
    error_std: Dict[str, float]
    n_folds: int

    def __repr__(self):
        return (f"KFoldResult(estimator={self.estimator_name}, "
                f"n_folds={self.n_folds}, errors={self.aggregated_errors})")


class KFoldTriangleValidator:
    """
    K-Fold Cross-Validation for loss triangles.

    Provides multiple strategies for creating folds from triangular data.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        n_folds: int = 5,
        fold_strategy: str = 'diagonal',
        verbose: bool = True
    ):
        """
        Initialize k-fold validator.

        Args:
            triangle: Full loss development triangle
            n_folds: Number of folds for cross-validation
            fold_strategy: Strategy for creating folds:
                - 'loyo': Leave-one-year-out
                - 'grouped': Contiguous year groups
                - 'diagonal': Different diagonals as test sets
                - 'timeseries': Expanding window
            verbose: Print progress messages
        """
        self.triangle = triangle.copy()
        self.n_folds = n_folds
        self.fold_strategy = fold_strategy
        self.verbose = verbose

        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        self.validation_results = {}

        # Validate n_folds
        if fold_strategy == 'loyo':
            self.n_folds = self.n_years
        elif fold_strategy == 'diagonal':
            max_diagonals = min(self.n_years, self.n_periods) - 2  # Keep at least 2 for training
            self.n_folds = min(n_folds, max_diagonals)
        elif fold_strategy == 'grouped':
            self.n_folds = min(n_folds, self.n_years - 1)  # Need at least 1 year for training per fold

    def _generate_folds(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate train/test splits based on fold strategy.

        Yields:
            Tuple of (training_triangle, test_data)
        """
        if self.fold_strategy == 'loyo':
            yield from self._loyo_folds()
        elif self.fold_strategy == 'grouped':
            yield from self._grouped_folds()
        elif self.fold_strategy == 'diagonal':
            yield from self._diagonal_folds()
        elif self.fold_strategy == 'timeseries':
            yield from self._timeseries_folds()
        else:
            raise ValueError(f"Unknown fold strategy: {self.fold_strategy}")

    def _loyo_folds(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Leave-One-Year-Out cross-validation.

        Each accident year is held out in turn.
        """
        for hold_year in self.triangle.index:
            # Training: all years except hold_year
            train_triangle = self.triangle.drop(hold_year)

            # Test: the held out year's diagonal
            test_data = pd.DataFrame({
                'year': [hold_year],
                'actual': [self.triangle.loc[hold_year].dropna().iloc[-1]],
                'age': [self.triangle.loc[hold_year].dropna().index[-1]],
                'prev_value': [self.triangle.loc[hold_year].dropna().iloc[-2]
                              if len(self.triangle.loc[hold_year].dropna()) > 1 else np.nan],
                'prev_age': [self.triangle.loc[hold_year].dropna().index[-2]
                            if len(self.triangle.loc[hold_year].dropna()) > 1 else None]
            })

            yield train_triangle, test_data

    def _grouped_folds(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Grouped K-Fold with contiguous year groups.

        Years are split into k contiguous groups.
        """
        years = list(self.triangle.index)
        fold_size = len(years) // self.n_folds

        for fold in range(self.n_folds):
            start_idx = fold * fold_size
            if fold == self.n_folds - 1:
                # Last fold gets remaining years
                test_years = years[start_idx:]
            else:
                test_years = years[start_idx:start_idx + fold_size]

            train_years = [y for y in years if y not in test_years]

            # Training triangle
            train_triangle = self.triangle.loc[train_years]

            # Test data: last observed value for each test year
            test_data = pd.DataFrame([{
                'year': year,
                'actual': self.triangle.loc[year].dropna().iloc[-1],
                'age': self.triangle.loc[year].dropna().index[-1],
                'prev_value': (self.triangle.loc[year].dropna().iloc[-2]
                              if len(self.triangle.loc[year].dropna()) > 1 else np.nan),
                'prev_age': (self.triangle.loc[year].dropna().index[-2]
                            if len(self.triangle.loc[year].dropna()) > 1 else None)
            } for year in test_years])

            yield train_triangle, test_data

    def _diagonal_folds(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Diagonal K-Fold: Different diagonals as test sets.

        Removes k different diagonals as test sets, trains on remaining.
        """
        for fold in range(self.n_folds):
            # Remove (fold + 1) diagonals
            train_triangle = self.triangle.copy()
            test_records = []

            for diag in range(fold + 1):
                # Remove one diagonal
                for i, year in enumerate(train_triangle.index):
                    row = train_triangle.loc[year]
                    non_nan = row.dropna()

                    if len(non_nan) > 1:
                        # Record test value
                        last_age = non_nan.index[-1]
                        last_value = non_nan.iloc[-1]
                        prev_value = non_nan.iloc[-2]
                        prev_age = non_nan.index[-2]

                        if diag == fold:  # Only record the outermost diagonal
                            test_records.append({
                                'year': year,
                                'actual': last_value,
                                'age': last_age,
                                'prev_value': prev_value,
                                'prev_age': prev_age
                            })

                        # Remove from training
                        train_triangle.loc[year, last_age] = np.nan

            test_data = pd.DataFrame(test_records)
            yield train_triangle, test_data

    def _timeseries_folds(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Time-Series K-Fold with expanding window.

        Initial training set expands, test set moves forward.
        """
        years = list(self.triangle.index)
        min_train_size = max(3, len(years) - self.n_folds)

        for fold in range(self.n_folds):
            train_end_idx = min_train_size + fold
            if train_end_idx >= len(years):
                break

            train_years = years[:train_end_idx]
            test_year = years[train_end_idx]

            train_triangle = self.triangle.loc[train_years]

            test_data = pd.DataFrame([{
                'year': test_year,
                'actual': self.triangle.loc[test_year].dropna().iloc[-1],
                'age': self.triangle.loc[test_year].dropna().index[-1],
                'prev_value': (self.triangle.loc[test_year].dropna().iloc[-2]
                              if len(self.triangle.loc[test_year].dropna()) > 1 else np.nan),
                'prev_age': (self.triangle.loc[test_year].dropna().index[-2]
                            if len(self.triangle.loc[test_year].dropna()) > 1 else None)
            }])

            yield train_triangle, test_data

    def _predict_from_factors(
        self,
        train_triangle: pd.DataFrame,
        test_data: pd.DataFrame,
        estimator: FactorEstimator
    ) -> np.ndarray:
        """
        Predict test values using estimated factors.

        Args:
            train_triangle: Training triangle
            test_data: Test data with prev_value and prev_age
            estimator: Factor estimation method

        Returns:
            Array of predictions
        """
        # Estimate factors from training triangle
        factors = estimator.estimate(train_triangle)

        predictions = []
        for _, row in test_data.iterrows():
            prev_age = row['prev_age']
            prev_value = row['prev_value']

            if pd.isna(prev_value) or prev_age is None:
                predictions.append(np.nan)
            elif prev_age in factors.index:
                predictions.append(prev_value * factors[prev_age])
            else:
                predictions.append(np.nan)

        return np.array(predictions)

    def validate_estimator(
        self,
        estimator: FactorEstimator,
        error_metrics: Dict[str, Callable] = None
    ) -> KFoldResult:
        """
        Validate a single estimator using k-fold cross-validation.

        Args:
            estimator: Factor estimation method
            error_metrics: Dictionary of error metric functions

        Returns:
            KFoldResult object with fold-wise and aggregated results
        """
        if self.verbose:
            print(f"  Validating: {estimator.name} ({self.n_folds}-fold, {self.fold_strategy})")

        if error_metrics is None:
            error_metrics = get_default_metrics()

        fold_results = []
        all_predictions = []
        all_actuals = []
        fold_errors = {metric: [] for metric in error_metrics}

        for fold_idx, (train_triangle, test_data) in enumerate(self._generate_folds()):
            if test_data.empty:
                continue

            # Get predictions
            predictions = self._predict_from_factors(train_triangle, test_data, estimator)
            actuals = test_data['actual'].values

            # Filter out NaN
            valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
            if not valid_mask.any():
                continue

            fold_preds = predictions[valid_mask]
            fold_actuals = actuals[valid_mask]

            all_predictions.extend(fold_preds)
            all_actuals.extend(fold_actuals)

            # Calculate fold errors
            fold_error = {}
            for metric_name, metric_func in error_metrics.items():
                error = metric_func(fold_actuals, fold_preds)
                fold_error[metric_name] = error
                fold_errors[metric_name].append(error)

            # Create fold result
            factors = estimator.estimate(train_triangle)
            fold_result = ValidationResult(
                estimator_name=f"{estimator.name}_fold{fold_idx}",
                predictions=fold_preds,
                actuals=fold_actuals,
                factors=factors,
                errors=fold_error
            )
            fold_results.append(fold_result)

        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        aggregated_errors = {}
        error_std = {}

        for metric_name, metric_func in error_metrics.items():
            if len(all_predictions) > 0:
                aggregated_errors[metric_name] = metric_func(all_actuals, all_predictions)
            else:
                aggregated_errors[metric_name] = np.nan

            if fold_errors[metric_name]:
                error_std[metric_name] = np.std(fold_errors[metric_name])
            else:
                error_std[metric_name] = np.nan

        result = KFoldResult(
            estimator_name=estimator.name,
            fold_results=fold_results,
            aggregated_errors=aggregated_errors,
            error_std=error_std,
            n_folds=len(fold_results)
        )

        self.validation_results[estimator.name] = result
        return result

    def validate_multiple(
        self,
        estimators: List[FactorEstimator],
        error_metrics: Dict[str, Callable] = None
    ) -> Dict[str, KFoldResult]:
        """
        Validate multiple estimators using k-fold cross-validation.

        Args:
            estimators: List of factor estimation methods
            error_metrics: Dictionary of error metric functions

        Returns:
            Dictionary mapping estimator names to KFoldResult objects
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"K-FOLD CROSS-VALIDATION")
            print(f"{'='*80}")
            print(f"Triangle shape: {self.triangle.shape}")
            print(f"Fold strategy: {self.fold_strategy}")
            print(f"Number of folds: {self.n_folds}")
            print(f"Number of estimators: {len(estimators)}")
            print(f"{'='*80}\n")

        results = {}

        for estimator in estimators:
            result = self.validate_estimator(estimator, error_metrics)
            results[estimator.name] = result

        self.validation_results = results

        if self.verbose:
            print(f"\n{'='*80}")
            print("K-FOLD VALIDATION COMPLETE")
            print(f"{'='*80}\n")

        return results

    def get_best_estimator(self, metric: str = 'RMSE') -> Tuple[str, KFoldResult]:
        """
        Get the best performing estimator based on a specific metric.

        Args:
            metric: Error metric to use for comparison

        Returns:
            Tuple of (best_estimator_name, KFoldResult)
        """
        if not self.validation_results:
            raise ValueError("No validation results available.")

        best_name = None
        best_error = float('inf')

        for name, result in self.validation_results.items():
            if metric in result.aggregated_errors:
                error = result.aggregated_errors[metric]
                if not np.isnan(error) and error < best_error:
                    best_error = error
                    best_name = name

        return best_name, self.validation_results[best_name]

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all validation results.

        Returns:
            DataFrame with estimators as rows, metrics + std as columns
        """
        if not self.validation_results:
            raise ValueError("No validation results available.")

        # Get all metrics
        all_metrics = set()
        for result in self.validation_results.values():
            all_metrics.update(result.aggregated_errors.keys())
        all_metrics = sorted(all_metrics)

        # Build summary
        rows = []
        for name, result in self.validation_results.items():
            row = {'Estimator': name, 'N_Folds': result.n_folds}
            for metric in all_metrics:
                row[metric] = result.aggregated_errors.get(metric, np.nan)
                row[f'{metric}_std'] = result.error_std.get(metric, np.nan)
            rows.append(row)

        summary = pd.DataFrame(rows)
        summary = summary.set_index('Estimator')
        summary = summary.sort_values(by=all_metrics[0])

        return summary


class NestedCrossValidation:
    """
    Nested Cross-Validation for hyperparameter tuning and model selection.

    Outer loop: Model evaluation
    Inner loop: Hyperparameter tuning (window size, etc.)

    This prevents overfitting to the validation set during model selection.
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        outer_folds: int = 5,
        inner_folds: int = 3,
        outer_strategy: str = 'diagonal',
        inner_strategy: str = 'diagonal',
        verbose: bool = True
    ):
        """
        Initialize nested cross-validation.

        Args:
            triangle: Full loss development triangle
            outer_folds: Number of outer CV folds
            inner_folds: Number of inner CV folds
            outer_strategy: Fold strategy for outer loop
            inner_strategy: Fold strategy for inner loop
            verbose: Print progress
        """
        self.triangle = triangle.copy()
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.outer_strategy = outer_strategy
        self.inner_strategy = inner_strategy
        self.verbose = verbose

        self.results = {}

    def run(
        self,
        estimators: List[FactorEstimator],
        error_metrics: Dict[str, Callable] = None,
        selection_metric: str = 'RMSE'
    ) -> Dict:
        """
        Run nested cross-validation.

        Args:
            estimators: List of estimators to evaluate
            error_metrics: Error metric functions
            selection_metric: Metric for inner loop selection

        Returns:
            Dictionary with nested CV results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("NESTED CROSS-VALIDATION")
            print(f"{'='*80}")
            print(f"Outer folds: {self.outer_folds} ({self.outer_strategy})")
            print(f"Inner folds: {self.inner_folds} ({self.inner_strategy})")
            print(f"Selection metric: {selection_metric}")
            print(f"{'='*80}\n")

        outer_validator = KFoldTriangleValidator(
            self.triangle,
            n_folds=self.outer_folds,
            fold_strategy=self.outer_strategy,
            verbose=False
        )

        outer_results = {est.name: [] for est in estimators}
        best_inner_models = []

        for fold_idx, (train_triangle, test_data) in enumerate(outer_validator._generate_folds()):
            if test_data.empty:
                continue

            if self.verbose:
                print(f"Outer fold {fold_idx + 1}/{self.outer_folds}")

            # Inner CV on training data to select best model
            inner_validator = KFoldTriangleValidator(
                train_triangle,
                n_folds=self.inner_folds,
                fold_strategy=self.inner_strategy,
                verbose=False
            )

            inner_results = inner_validator.validate_multiple(estimators, error_metrics)

            # Select best model from inner CV
            best_name, _ = inner_validator.get_best_estimator(selection_metric)
            best_inner_models.append(best_name)

            if self.verbose:
                print(f"  Best inner model: {best_name}")

            # Evaluate all models on outer test set
            for estimator in estimators:
                predictions = outer_validator._predict_from_factors(
                    train_triangle, test_data, estimator
                )
                actuals = test_data['actual'].values

                valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
                if valid_mask.any():
                    if error_metrics is None:
                        error_metrics = get_default_metrics()

                    errors = {}
                    for metric_name, metric_func in error_metrics.items():
                        errors[metric_name] = metric_func(
                            actuals[valid_mask],
                            predictions[valid_mask]
                        )
                    outer_results[estimator.name].append(errors)

        # Aggregate outer results
        final_results = {}
        for name, fold_errors in outer_results.items():
            if fold_errors:
                metrics = fold_errors[0].keys()
                final_results[name] = {
                    'mean_errors': {
                        m: np.mean([f[m] for f in fold_errors]) for m in metrics
                    },
                    'std_errors': {
                        m: np.std([f[m] for f in fold_errors]) for m in metrics
                    },
                    'n_outer_folds': len(fold_errors)
                }

        self.results = {
            'estimator_results': final_results,
            'best_inner_selections': best_inner_models
        }

        if self.verbose:
            print(f"\n{'='*80}")
            print("NESTED CV COMPLETE")
            print(f"Best model selected in inner loops: {pd.Series(best_inner_models).mode().iloc[0]}")
            print(f"{'='*80}\n")

        return self.results

    def summary(self) -> pd.DataFrame:
        """Create summary table of nested CV results."""
        if not self.results:
            raise ValueError("No results available. Run nested CV first.")

        rows = []
        for name, data in self.results['estimator_results'].items():
            row = {'Estimator': name}
            for metric, value in data['mean_errors'].items():
                row[f'{metric}_mean'] = value
                row[f'{metric}_std'] = data['std_errors'][metric]
            rows.append(row)

        return pd.DataFrame(rows).set_index('Estimator')
