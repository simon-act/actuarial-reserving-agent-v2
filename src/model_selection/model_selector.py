"""
Automated Model Selection for Actuarial Reserving

Orchestrates the complete model selection workflow: estimates factors using
multiple methods, validates performance out-of-sample, conducts statistical
tests, and selects the optimal reserving approach.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

from .factor_estimators import FactorEstimator, get_all_estimators
from .validation_framework import create_validator, ValidationResult
from .error_metrics import get_default_metrics, calculate_all_metrics
from .statistical_tests import (
    diebold_mariano_test,
    paired_t_test,
    wilcoxon_signed_rank_test,
    compute_pairwise_comparisons,
    model_confidence_set,
    TestResult
)


class ModelSelector:
    """
    Automated model selection system for actuarial reserving.
    
    Implements a rigorous framework for selecting the optimal factor estimation
    method based on out-of-sample predictive performance.
    
    Workflow:
    1. Apply multiple factor estimation methods to reduced triangle
    2. Predict holdout diagonal
    3. Compute prediction errors
    4. Compare methods using statistical tests
    5. Select best method or model confidence set
    """
    
    def __init__(
        self,
        triangle: pd.DataFrame,
        estimators: Optional[List[FactorEstimator]] = None,
        validation_method: str = 'holdout',
        verbose: bool = True
    ):
        """
        Initialize the model selector.
        
        Args:
            triangle: Loss development triangle
            estimators: List of factor estimators to compare (default: all)
            validation_method: 'holdout' or 'rolling_origin'
            verbose: Whether to print detailed progress
        """
        self.triangle = triangle.copy()
        self.estimators = estimators if estimators else get_all_estimators()
        self.validation_method = validation_method
        self.verbose = verbose
        
        # Results storage
        self.validation_results = None
        self.comparison_table = None
        self.statistical_tests = None
        self.best_estimator = None
        self.mcs_results = None
        
        if self.verbose:
            print("="*80)
            print("MODEL SELECTOR INITIALIZED")
            print("="*80)
            print(f"Triangle shape: {self.triangle.shape}")
            print(f"Number of estimators: {len(self.estimators)}")
            print(f"Validation method: {self.validation_method}")
            print(f"Estimators: {[e.name for e in self.estimators]}")
            print("="*80 + "\n")
    
    @classmethod
    def create_with_windowed_grid(
        cls,
        triangle: pd.DataFrame,
        min_window: int = 3,
        max_window: int = 10,
        recent_only: bool = False,
        validation_method: str = 'holdout',
        verbose: bool = True
    ):
        """
        Create ModelSelector with full windowed grid of estimators.
        
        Generates all combinations of:
        - Base aggregation methods
        - Time windows (contiguous subsets of years)
        
        Args:
            triangle: Loss development triangle
            min_window: Minimum window length (years)
            max_window: Maximum window length (default: 10 years)
            recent_only: If True, only windows ending at most recent year
            validation_method: 'holdout' or 'rolling_origin'
            verbose: Print progress
            
        Returns:
            ModelSelector instance with windowed estimators
        """
        from .windowed_estimators import (
            generate_windowed_estimators,
            summarize_window_grid
        )
        
        if verbose:
            print("\n" + "="*80)
            print("CREATING WINDOWED GRID MODEL SELECTOR")
            print("="*80 + "\n")
            
            grid_summary = summarize_window_grid(
                triangle,
                min_window=min_window,
                max_window=max_window,
                n_base_estimators=7
            )
            
            print(f"Grid configuration:")
            print(f"  Min window: {grid_summary['min_window']} years")
            print(f"  Max window: {grid_summary['max_window']} years")
            print(f"  Total windows: {grid_summary['n_windows']}")
            print(f"  Base methods: {grid_summary['n_base_estimators']}")
            print(f"  Total models: {grid_summary['total_candidates']}")
            print(f"  Recent only: {recent_only}")
            print()
        
        # Generate windowed estimators
        windowed_estimators = generate_windowed_estimators(
            triangle,
            min_window=min_window,
            max_window=max_window,
            recent_only=recent_only
        )
        
        # Create selector with windowed estimators
        return cls(
            triangle=triangle,
            estimators=windowed_estimators,
            validation_method=validation_method,
            verbose=verbose
        )
    
    def run_validation(self, **kwargs) -> Dict[str, ValidationResult]:
        """
        Run out-of-sample validation for all estimators.
        
        Args:
            **kwargs: Additional arguments for validator (e.g., n_holdouts)
            
        Returns:
            Dictionary of validation results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("STEP 1: OUT-OF-SAMPLE VALIDATION")
            print("="*80 + "\n")
        
        # Create validator
        validator = create_validator(
            self.triangle,
            method=self.validation_method,
            verbose=self.verbose,
            **kwargs
        )
        
        # Run validation
        self.validation_results = validator.validate_multiple(
            self.estimators,
            error_metrics=get_default_metrics()
        )
        
        # Create comparison table
        self.comparison_table = validator.create_summary_table()
        
        if self.verbose:
            print("\nVALIDATION RESULTS SUMMARY:")
            print("-"*80)
            print(self.comparison_table.round(4))
            print("-"*80 + "\n")
        
        return self.validation_results
    
    def conduct_statistical_tests(self, test_types: List[str] = None) -> Dict:
        """
        Conduct statistical tests comparing all methods.
        
        Args:
            test_types: List of tests to run ('dm', 't_test', 'wilcoxon', 'all')
            
        Returns:
            Dictionary of test results
        """
        if self.validation_results is None:
            raise ValueError("Run validation first using run_validation()")
        
        if test_types is None:
            test_types = ['dm', 't_test']
        elif 'all' in test_types:
            test_types = ['dm', 't_test', 'wilcoxon']
        
        if self.verbose:
            print("\n" + "="*80)
            print("STEP 2: STATISTICAL HYPOTHESIS TESTING")
            print("="*80 + "\n")
        
        # Extract predictions and actuals
        predictions_dict = {}
        actuals = None
        
        for name, result in self.validation_results.items():
            predictions_dict[name] = result.predictions
            if actuals is None:
                actuals = result.actuals
        
        # Run tests
        self.statistical_tests = {}
        
        for test_type in test_types:
            if self.verbose:
                print(f"Running {test_type.upper()} test...")
            
            p_values = compute_pairwise_comparisons(
                predictions_dict,
                actuals,
                test_type=test_type
            )
            
            self.statistical_tests[test_type] = p_values
            
            if self.verbose:
                print(f"\nP-values ({test_type.upper()}):")
                print(p_values.round(4))
                print()
        
        return self.statistical_tests
    
    def compute_model_confidence_set(self, alpha: float = 0.10) -> Dict:
        """
        Compute Model Confidence Set (MCS).
        
        Identifies the set of models that cannot be statistically distinguished
        from the best model.
        
        Args:
            alpha: Significance level for MCS procedure
            
        Returns:
            Dictionary with MCS results
        """
        if self.validation_results is None:
            raise ValueError("Run validation first using run_validation()")
        
        if self.verbose:
            print("\n" + "="*80)
            print("STEP 3: MODEL CONFIDENCE SET")
            print("="*80 + "\n")
        
        # Extract predictions and actuals
        predictions_dict = {}
        actuals = None
        
        for name, result in self.validation_results.items():
            predictions_dict[name] = result.predictions
            if actuals is None:
                actuals = result.actuals
        
        # Compute MCS
        self.mcs_results = model_confidence_set(
            predictions_dict,
            actuals,
            alpha=alpha,
            test_type='dm'
        )
        
        if self.verbose:
            print(f"Significance level: {alpha}")
            print(f"\nModel Confidence Set (α={alpha}):")
            for model in self.mcs_results['mcs']:
                avg_loss = self.mcs_results['avg_losses'][model]
                print(f"  ✓ {model:30s} (Avg MSE: {avg_loss:.4f})")
            
            print(f"\nEliminated models:")
            for model, p_val in self.mcs_results['elimination_order']:
                avg_loss = self.mcs_results['avg_losses'][model]
                print(f"  ✗ {model:30s} (p={p_val:.4f}, Avg MSE: {avg_loss:.4f})")
            
            print(f"\nBest model: {self.mcs_results['best_model']}")
            print()
        
        return self.mcs_results
    
    def select_best_model(self, criterion: str = 'RMSE') -> Tuple[str, ValidationResult]:
        """
        Select the single best model based on a criterion.
        
        Args:
            criterion: Metric to use for selection (e.g., 'RMSE', 'MAE')
            
        Returns:
            Tuple of (best_estimator_name, validation_result)
        """
        if self.validation_results is None:
            raise ValueError("Run validation first using run_validation()")
        
        if self.verbose:
            print("\n" + "="*80)
            print("STEP 4: MODEL SELECTION")
            print("="*80 + "\n")
        
        # Find best based on criterion
        best_name = None
        best_value = float('inf')
        
        for name, result in self.validation_results.items():
            if criterion in result.errors:
                value = result.errors[criterion]
                if value < best_value:
                    best_value = value
                    best_name = name
        
        self.best_estimator = best_name
        
        if self.verbose:
            print(f"Selection criterion: {criterion}")
            print(f"Best model: {best_name}")
            print(f"{criterion}: {best_value:.4f}")
            
            # Show comparison with other models
            print(f"\nFull ranking by {criterion}:")
            rankings = sorted(
                [(name, res.errors[criterion]) for name, res in self.validation_results.items()],
                key=lambda x: x[1]
            )
            for i, (name, value) in enumerate(rankings, 1):
                marker = "⭐" if name == best_name else "  "
                print(f"  {marker} {i}. {name:30s} {criterion}={value:.4f}")
            print()
        
        return best_name, self.validation_results[best_name]
    
    def run_full_analysis(
        self,
        validation_kwargs: Dict = None,
        test_types: List[str] = None,
        mcs_alpha: float = 0.10,
        selection_criterion: str = 'RMSE'
    ) -> Dict:
        """
        Run complete model selection analysis.
        
        Executes all steps:
        1. Out-of-sample validation
        2. Statistical hypothesis testing
        3. Model confidence set
        4. Best model selection
        
        Args:
            validation_kwargs: Arguments for validation (e.g., n_holdouts)
            test_types: Types of statistical tests to run
            mcs_alpha: Significance level for MCS
            selection_criterion: Metric for selecting best model
            
        Returns:
            Dictionary with all results
        """
        if validation_kwargs is None:
            validation_kwargs = {}
        
        if self.verbose:
            print("\n" + "="*80)
            print("AUTOMATED MODEL SELECTION FOR ACTUARIAL RESERVING")
            print("="*80)
            print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
        
        # Step 1: Validation
        self.run_validation(**validation_kwargs)
        
        # Step 2: Statistical tests
        self.conduct_statistical_tests(test_types)
        
        # Step 3: Model confidence set
        self.compute_model_confidence_set(alpha=mcs_alpha)
        
        # Step 4: Select best
        best_name, best_result = self.select_best_model(criterion=selection_criterion)
        
        # Compile results
        results = {
            'validation_results': self.validation_results,
            'comparison_table': self.comparison_table,
            'statistical_tests': self.statistical_tests,
            'mcs_results': self.mcs_results,
            'best_model': best_name,
            'best_result': best_result,
            'selection_criterion': selection_criterion
        }
        
        if self.verbose:
            self._print_final_summary(results)
        
        return results
    
    def run_windowed_analysis(
        self,
        validation_kwargs: Dict = None,
        selection_criterion: str = 'RMSE',
        save_sensitivity: bool = True
    ) -> Dict:
        """
        Run analysis specifically for windowed estimators.
        
        Adds window-specific analyses:
        - Optimal window by method
        - Window sensitivity analysis
        
        Args:
            validation_kwargs: Arguments for validation
            selection_criterion: Metric for selection
            save_sensitivity: Whether to compute sensitivity analysis
            
        Returns:
            Dictionary with results including window analysis
        """
        if validation_kwargs is None:
            validation_kwargs = {}
        
        if self.verbose:
            print("\n" + "="*80)
            print("WINDOWED MODEL SELECTION ANALYSIS")
            print("="*80)
            print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total candidate models: {len(self.estimators)}")
            print("="*80 + "\n")
        
        # Step 1: Validation (no statistical tests for large grids)
        self.run_validation(**validation_kwargs)
        
        # Step 2: Select best
        best_name, best_result = self.select_best_model(criterion=selection_criterion)
        
        # Step 3: Window-specific analysis
        from .windowed_estimators import (
            get_optimal_window_by_method,
            analyze_window_sensitivity
        )
        
        if self.verbose:
            print("\n" + "="*80)
            print("WINDOW-SPECIFIC ANALYSIS")
            print("="*80 + "\n")
        
        # Best window by method
        best_by_method = get_optimal_window_by_method(
            self.validation_results,
            criterion=selection_criterion
        )
        
        if self.verbose:
            print("Optimal window for each base method:")
            print("-"*80)
            for _, row in best_by_method.head(10).iterrows():
                if pd.notna(row['Window_Length']):
                    print(f"  {row['Method']:<25} window={int(row['Window_Length']):2d} years, "
                          f"{selection_criterion}={row['Error']:.4f}")
                else:
                    print(f"  {row['Method']:<25} (non-windowed), "
                          f"{selection_criterion}={row['Error']:.4f}")
            print()
        
        # Sensitivity analysis
        if save_sensitivity:
            sensitivity = analyze_window_sensitivity(
                self.validation_results,
                criterion=selection_criterion
            )
        else:
            sensitivity = None
        
        # Compile results
        results = {
            'validation_results': self.validation_results,
            'comparison_table': self.comparison_table,
            'best_model': best_name,
            'best_result': best_result,
            'best_by_method': best_by_method,
            'window_sensitivity': sensitivity,
            'selection_criterion': selection_criterion
        }
        
        if self.verbose:
            self._print_windowed_summary(results)
        
        return results
    
    def _print_windowed_summary(self, results: Dict):
        """Print summary for windowed analysis."""
        print("\n" + "="*80)
        print("WINDOWED ANALYSIS SUMMARY")
        print("="*80 + "\n")
        
        print("BEST OVERALL MODEL:")
        print(f"  → {results['best_model']}")
        print(f"  → {results['selection_criterion']}: "
              f"{results['best_result'].errors[results['selection_criterion']]:.4f}")
        
        print("\nTOP 5 BASE METHODS (with optimal windows):")
        for i, (_, row) in enumerate(results['best_by_method'].head(5).iterrows(), 1):
            method = row['Method']
            if pd.notna(row['Window_Length']):
                window_info = f"{int(row['Window_Length'])} years"
            else:
                window_info = "all data"
            error = row['Error']
            print(f"  {i}. {method:<25} ({window_info:<12}) {results['selection_criterion']}={error:.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def _print_final_summary(self, results: Dict):
        """Print final summary of analysis."""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80 + "\n")
        
        print("RECOMMENDED MODEL:")
        print(f"  → {results['best_model']}")
        print(f"  → {results['selection_criterion']}: {results['best_result'].errors[results['selection_criterion']]:.4f}")
        
        print(f"\nMODEL CONFIDENCE SET (α={self.mcs_results['alpha']}):")
        for model in results['mcs_results']['mcs']:
            marker = "⭐" if model == results['best_model'] else "  "
            print(f"  {marker} {model}")
        
        print("\nKEY INSIGHTS:")
        n_estimators = len(self.estimators)
        n_mcs = len(results['mcs_results']['mcs'])
        
        if n_mcs == 1:
            print(f"  • Clear winner: {results['best_model']} significantly outperforms all others")
        elif n_mcs == n_estimators:
            print(f"  • No statistically significant differences among methods")
            print(f"  • Consider using simple average or ensemble")
        else:
            print(f"  • {n_mcs}/{n_estimators} methods in confidence set")
            print(f"  • {results['best_model']} recommended but alternatives exist")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_dir: Path):
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison table
        if self.comparison_table is not None:
            self.comparison_table.to_csv(
                output_dir / f"comparison_table_{timestamp}.csv"
            )
        
        # Save statistical test results
        if self.statistical_tests is not None:
            for test_type, p_values in self.statistical_tests.items():
                p_values.to_csv(
                    output_dir / f"pvalues_{test_type}_{timestamp}.csv"
                )
        
        # Save MCS results
        if self.mcs_results is not None:
            with open(output_dir / f"mcs_results_{timestamp}.json", 'w') as f:
                # Convert to serializable format
                mcs_serializable = {
                    'mcs': self.mcs_results['mcs'],
                    'eliminated': self.mcs_results['eliminated'],
                    'best_model': self.mcs_results['best_model'],
                    'alpha': self.mcs_results['alpha'],
                    'avg_losses': self.mcs_results['avg_losses']
                }
                json.dump(mcs_serializable, f, indent=2)
        
        # Save predictions and actuals
        if self.validation_results is not None:
            predictions_df = pd.DataFrame({
                name: result.predictions 
                for name, result in self.validation_results.items()
            })
            
            # Add actuals
            first_result = list(self.validation_results.values())[0]
            predictions_df['Actual'] = first_result.actuals
            
            predictions_df.to_csv(
                output_dir / f"predictions_{timestamp}.csv"
            )
        
        # Save summary report
        self._save_text_report(output_dir, timestamp)
        
        if self.verbose:
            print(f"\n✅ Results saved to: {output_dir}/")
    
    def _save_text_report(self, output_dir: Path, timestamp: str):
        """Generate and save text report."""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("MODEL SELECTION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("TRIANGLE INFORMATION:")
        report_lines.append(f"  Shape: {self.triangle.shape}")
        report_lines.append(f"  Accident years: {self.triangle.index.min()} - {self.triangle.index.max()}")
        report_lines.append(f"  Development periods: {len(self.triangle.columns)}")
        report_lines.append("")
        
        report_lines.append("VALIDATION SETUP:")
        report_lines.append(f"  Method: {self.validation_method}")
        report_lines.append(f"  Number of estimators: {len(self.estimators)}")
        report_lines.append(f"  Estimators tested: {', '.join([e.name for e in self.estimators])}")
        report_lines.append("")
        
        if self.comparison_table is not None:
            report_lines.append("PERFORMANCE COMPARISON:")
            report_lines.append(str(self.comparison_table.round(4)))
            report_lines.append("")
        
        if self.best_estimator is not None:
            report_lines.append("RECOMMENDED MODEL:")
            report_lines.append(f"  {self.best_estimator}")
            report_lines.append("")
        
        if self.mcs_results is not None:
            report_lines.append(f"MODEL CONFIDENCE SET (α={self.mcs_results['alpha']}):")
            for model in self.mcs_results['mcs']:
                report_lines.append(f"  • {model}")
            report_lines.append("")
        
        report_lines.append("="*80)
        
        # Write to file
        with open(output_dir / f"report_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))


def quick_select(
    triangle: pd.DataFrame,
    validation_method: str = 'holdout',
    verbose: bool = True
) -> str:
    """
    Quick model selection with default settings.
    
    Convenience function for standard use cases.
    
    Args:
        triangle: Loss development triangle
        validation_method: 'holdout' or 'rolling_origin'
        verbose: Whether to print progress
        
    Returns:
        Name of the best estimator
    """
    selector = ModelSelector(
        triangle=triangle,
        validation_method=validation_method,
        verbose=verbose
    )
    
    results = selector.run_full_analysis()
    
    return results['best_model']


def windowed_quick_select(
    triangle: pd.DataFrame,
    min_window: int = 3,
    max_window: int = 10,
    recent_only: bool = True,
    validation_method: str = 'holdout',
    verbose: bool = True
) -> Tuple[str, pd.DataFrame]:
    """
    Quick windowed model selection with sensible defaults.
    
    Uses recent windows only to reduce computation.
    
    Args:
        triangle: Loss development triangle
        min_window: Minimum window length
        max_window: Maximum window length
        recent_only: Only use windows ending at most recent year
        validation_method: 'holdout' or 'rolling_origin'
        verbose: Print progress
        
    Returns:
        Tuple of (best_model_name, best_by_method_dataframe)
    """
    selector = ModelSelector.create_with_windowed_grid(
        triangle=triangle,
        min_window=min_window,
        max_window=max_window,
        recent_only=recent_only,
        validation_method=validation_method,
        verbose=verbose
    )
    
    results = selector.run_windowed_analysis(
        selection_criterion='RMSE',
        save_sensitivity=True
    )
    
    return results['best_model'], results['best_by_method']