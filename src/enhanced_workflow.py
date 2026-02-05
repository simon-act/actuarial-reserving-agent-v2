"""
Enhanced Reserving Workflow

Complete end-to-end workflow integrating all advanced features:
1. Model selection with windowed estimators
2. Stochastic reserving (Mack, Bootstrap)
3. Alternative methods (Cape Cod)
4. K-Fold cross-validation
5. Comprehensive diagnostics
6. Scenario analysis and stress testing

This builds on the original FinalReservingWorkflow but adds all new capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# Original modules
from model_selection.model_selector import ModelSelector
from model_selection.windowed_estimators import get_optimal_window_by_method
from model_selection.factor_estimators import get_all_estimators, SimpleAverageEstimator
from chain_ladder import ChainLadder

# New modules - Stochastic
from stochastic_reserving.mack_model import MackChainLadder
from stochastic_reserving.bootstrap import BootstrapChainLadder, ODPBootstrap

# New modules - Alternative methods
from alternative_methods.cape_cod import CapeCod

# New modules - Cross-validation
from model_selection.kfold_validation import KFoldTriangleValidator, NestedCrossValidation

# New modules - Diagnostics
from diagnostics.residual_analysis import ResidualAnalyzer
from diagnostics.volatility_analysis import VolatilityAnalyzer
from diagnostics.diagnostic_tests import DiagnosticTests

# New modules - Scenario Analysis
from scenario_analysis.stress_testing import StressTestFramework
from scenario_analysis.scenario_generator import ScenarioGenerator
from scenario_analysis.tail_risk import TailRiskAnalyzer


class EnhancedReservingWorkflow:
    """
    Comprehensive reserving workflow with all advanced features.

    Orchestrates:
    - Model selection
    - Multiple reserving methods
    - Stochastic analysis
    - Diagnostics
    - Stress testing
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        earned_premium: Optional[pd.Series] = None,
        expected_loss_ratios: Optional[pd.Series] = None,
        verbose: bool = True
    ):
        """
        Initialize enhanced workflow.

        Args:
            triangle: Loss development triangle
            earned_premium: Earned premium by year (for Cape Cod)
            expected_loss_ratios: A priori loss ratios (optional)
            verbose: Print progress
        """
        self.triangle = triangle.copy()
        self.earned_premium = earned_premium
        self.expected_loss_ratios = expected_loss_ratios
        self.verbose = verbose

        # Results containers
        self.results = {
            'model_selection': None,
            'chain_ladder': None,
            'mack': None,
            'bootstrap': None,
            'cape_cod': None,
            'cross_validation': None,
            'diagnostics': None,
            'stress_tests': None,
            'scenarios': None,
            'tail_risk': None
        }

        self.selected_factors = None
        self.selected_method = None

    def run_model_selection(
        self,
        min_window: int = 3,
        max_window: int = 10,
        recent_only: bool = True
    ):
        """
        Run windowed model selection.

        Args:
            min_window: Minimum window length
            max_window: Maximum window length
            recent_only: Only recent windows
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 1: MODEL SELECTION")
            print("="*80 + "\n")

        selector = ModelSelector.create_with_windowed_grid(
            triangle=self.triangle,
            min_window=min_window,
            max_window=max_window,
            recent_only=recent_only,
            validation_method='holdout',
            verbose=self.verbose
        )

        self.results['model_selection'] = selector.run_windowed_analysis(
            selection_criterion='RMSE'
        )

        # Extract best method
        best_name = self.results['model_selection']['best_model']
        self.selected_method = best_name

        # Get factors from best method
        self.selected_factors = self.results['model_selection']['best_result'].factors

        if self.verbose:
            print(f"\n✅ Selected: {best_name}")
            print(f"   RMSE: {self.results['model_selection']['best_result'].errors['RMSE']:.4f}")

    def run_cross_validation(
        self,
        n_folds: int = 5,
        strategy: str = 'diagonal'
    ):
        """
        Run k-fold cross-validation.

        Args:
            n_folds: Number of folds
            strategy: Fold strategy
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 2: K-FOLD CROSS-VALIDATION")
            print("="*80 + "\n")

        estimators = get_all_estimators()

        kfold = KFoldTriangleValidator(
            self.triangle,
            n_folds=n_folds,
            fold_strategy=strategy,
            verbose=self.verbose
        )

        self.results['cross_validation'] = kfold.validate_multiple(estimators)

        if self.verbose:
            best_name, best_result = kfold.get_best_estimator()
            print(f"\n✅ Best by K-Fold: {best_name}")
            print(f"   RMSE: {best_result.aggregated_errors['RMSE']:.4f}")

    def run_chain_ladder(self):
        """Run standard chain-ladder with selected factors."""
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 3: CHAIN-LADDER RESERVES")
            print("="*80 + "\n")

        if self.selected_factors is None:
            # Use simple average if no selection done
            estimator = SimpleAverageEstimator()
            self.selected_factors = estimator.estimate(self.triangle)

        cl = ChainLadder(self.triangle)
        cl.calculate_age_to_age_factors()
        cl.selected_factors = self.selected_factors
        cl.calculate_cumulative_factors()
        cl.project_ultimate_losses()

        self.results['chain_ladder'] = cl

        if self.verbose:
            summary = cl.summary()
            print(f"Total Reserve: ${summary['total_reserve']:,.0f}")

    def run_mack_model(self):
        """Run Mack's stochastic chain-ladder."""
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 4: MACK STOCHASTIC MODEL")
            print("="*80 + "\n")

        mack = MackChainLadder(self.triangle)
        mack.fit()

        self.results['mack'] = mack

        if self.verbose:
            total = mack.get_total_reserve_distribution()
            print(f"Total Reserve: ${total['Total_Reserve']:,.0f}")
            print(f"Standard Error: ${total['Total_SE']:,.0f}")
            print(f"CV: {total['Total_CV']:.2%}")

    def run_bootstrap(self, n_simulations: int = 5000):
        """Run bootstrap analysis."""
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 5: BOOTSTRAP SIMULATION")
            print(f"({n_simulations} simulations)")
            print("="*80 + "\n")

        bootstrap = BootstrapChainLadder(
            self.triangle,
            n_simulations=n_simulations,
            random_state=42
        )
        bootstrap.fit()

        self.results['bootstrap'] = bootstrap

        if self.verbose:
            total = bootstrap.get_total_reserve_distribution()
            print(f"Mean Reserve: ${total['Mean']:,.0f}")
            print(f"Std Dev: ${total['Std']:,.0f}")
            print(f"P75: ${total['P75']:,.0f}")
            print(f"P95: ${total['P95']:,.0f}")

    def run_alternative_methods(self):
        """Run Cape Cod method."""
        if self.earned_premium is None:
            if self.verbose:
                print("\n⚠️ Skipping Cape Cod - no premium data provided")
            return

        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 6: ALTERNATIVE METHODS")
            print("="*80 + "\n")

        # Cape Cod
        if self.verbose:
            print("📊 Cape Cod:")

        cc = CapeCod(self.triangle, self.earned_premium)
        cc.fit()
        self.results['cape_cod'] = cc

        if self.verbose:
            summary = cc.summary()
            print(f"   Cape Cod ELR: {summary['cape_cod_elr']:.2%}")
            print(f"   CC Reserve: ${summary['total_cc_reserve']:,.0f}")
            print(f"   CL Reserve: ${summary['total_cl_reserve']:,.0f}")

    def run_diagnostics(self):
        """Run comprehensive diagnostics."""
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 7: DIAGNOSTICS")
            print("="*80 + "\n")

        diagnostics = {}

        # Residual analysis
        if self.verbose:
            print("📊 Residual Analysis:")
        residual_analyzer = ResidualAnalyzer(self.triangle, self.selected_factors)
        residual_analyzer.fit()
        diagnostics['residuals'] = residual_analyzer

        if self.verbose:
            outliers = residual_analyzer.identify_outliers()
            print(f"   Outliers detected: {len(outliers)}")

        # Volatility analysis
        if self.verbose:
            print("\n📊 Volatility Analysis:")
        volatility_analyzer = VolatilityAnalyzer(self.triangle)
        volatility_analyzer.fit()
        diagnostics['volatility'] = volatility_analyzer

        if self.verbose:
            metrics = volatility_analyzer.volatility_metrics
            print(f"   Average CV: {metrics.get('average_cv', 0):.4f}")
            print(f"   Stability: {metrics.get('stability_rating', 'N/A')}")

        # Diagnostic tests
        if self.verbose:
            print("\n📊 Model Diagnostic Tests:")
        diag_tests = DiagnosticTests(self.triangle, self.selected_factors)
        adequacy = diag_tests.get_model_adequacy_score()
        diagnostics['tests'] = diag_tests

        if self.verbose:
            print(f"   Adequacy Score: {adequacy['adequacy_score']:.0f}%")
            print(f"   Rating: {adequacy['rating']}")
            if adequacy['issues']:
                print(f"   Issues: {', '.join(adequacy['issues'][:3])}")

        self.results['diagnostics'] = diagnostics

    def run_stress_testing(self):
        """Run stress tests and scenario analysis."""
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 8: STRESS TESTING")
            print("="*80 + "\n")

        # Stress tests
        stress = StressTestFramework(
            self.triangle,
            self.selected_factors
        )
        stress.run_standard_scenarios()
        self.results['stress_tests'] = stress

        if self.verbose:
            print("📊 Key Stress Results:")
            summary = stress.get_summary_table()
            for _, row in summary.head(5).iterrows():
                print(f"   {row['Scenario']}: ${row['Stressed_Reserve']:,.0f} ({row['Impact_Pct']:+.1%})")

        # Scenario analysis
        if self.verbose:
            print("\n📊 Scenario Analysis:")
        scenarios = ScenarioGenerator(
            self.triangle,
            self.selected_factors,
            self.earned_premium
        )
        self.results['scenarios'] = scenarios

        scenario_results = scenarios.run_all_scenarios()
        if self.verbose:
            adverse = scenario_results[scenario_results['Impact'] > 0].nlargest(3, 'Impact_Pct')
            for _, row in adverse.iterrows():
                print(f"   {row['Scenario']}: {row['Impact_Pct']:+.1%}")

        # Tail risk
        if self.verbose:
            print("\n📊 Tail Risk Analysis:")
        tail_risk = TailRiskAnalyzer(self.triangle, self.selected_factors)
        tail_risk.fit()
        self.results['tail_risk'] = tail_risk

        if self.verbose:
            concentration = tail_risk.get_risk_concentration()
            print(f"   Immature year risk: {concentration['immature_risk_contribution']:.1%}")
            print(f"   Top 3 years: {concentration['top_3_risk_years']}")

    def run_complete_analysis(
        self,
        run_model_selection: bool = True,
        run_cross_validation: bool = True,
        run_bootstrap: bool = True,
        n_bootstrap_simulations: int = 5000,
        run_alternative_methods: bool = True,
        run_diagnostics: bool = True,
        run_stress_testing: bool = True
    ):
        """
        Run complete enhanced analysis.

        Args:
            run_model_selection: Include model selection
            run_cross_validation: Include k-fold CV
            run_bootstrap: Include bootstrap
            n_bootstrap_simulations: Number of bootstrap simulations
            run_alternative_methods: Include BF and Cape Cod
            run_diagnostics: Include diagnostics
            run_stress_testing: Include stress tests
        """
        if self.verbose:
            print("\n" + "="*80)
            print("ENHANCED RESERVING ANALYSIS")
            print("="*80)
            print(f"Triangle: {self.triangle.shape[0]} years × {self.triangle.shape[1]} periods")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

        # Phase 1: Model Selection
        if run_model_selection:
            self.run_model_selection()

        # Phase 2: Cross-validation
        if run_cross_validation:
            self.run_cross_validation()

        # Phase 3: Chain-Ladder
        self.run_chain_ladder()

        # Phase 4: Mack Model
        self.run_mack_model()

        # Phase 5: Bootstrap
        if run_bootstrap:
            self.run_bootstrap(n_bootstrap_simulations)

        # Phase 6: Alternative methods
        if run_alternative_methods:
            self.run_alternative_methods()

        # Phase 7: Diagnostics
        if run_diagnostics:
            self.run_diagnostics()

        # Phase 8: Stress Testing
        if run_stress_testing:
            self.run_stress_testing()

        if self.verbose:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        report = []
        report.append("="*80)
        report.append("ENHANCED RESERVING ANALYSIS - EXECUTIVE SUMMARY")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Reserve estimates comparison
        report.append("RESERVE ESTIMATES COMPARISON")
        report.append("-"*60)

        cl_reserve = self.results['chain_ladder'].summary()['total_reserve']
        report.append(f"Chain-Ladder (Selected):  ${cl_reserve:>15,.0f}")

        if self.results['mack']:
            mack_total = self.results['mack'].get_total_reserve_distribution()
            report.append(f"Mack Model:               ${mack_total['Total_Reserve']:>15,.0f}")
            report.append(f"  - Standard Error:       ${mack_total['Total_SE']:>15,.0f}")

        if self.results['bootstrap']:
            boot_total = self.results['bootstrap'].get_total_reserve_distribution()
            report.append(f"Bootstrap Mean:           ${boot_total['Mean']:>15,.0f}")
            report.append(f"  - P75:                  ${boot_total['P75']:>15,.0f}")
            report.append(f"  - P95:                  ${boot_total['P95']:>15,.0f}")

        if self.results['cape_cod']:
            cc_sum = self.results['cape_cod'].summary()
            report.append(f"Cape Cod:                 ${cc_sum['total_cc_reserve']:>15,.0f}")

        report.append("")

        # Model adequacy
        if self.results['diagnostics']:
            report.append("MODEL ADEQUACY")
            report.append("-"*60)
            adequacy = self.results['diagnostics']['tests'].get_model_adequacy_score()
            report.append(f"Adequacy Score: {adequacy['adequacy_score']:.0f}%")
            report.append(f"Rating: {adequacy['rating']}")
            report.append(f"Recommendation: {adequacy['recommendation']}")
            report.append("")

        # Key risks
        if self.results['stress_tests']:
            report.append("KEY STRESS SCENARIOS")
            report.append("-"*60)
            summary = self.results['stress_tests'].get_summary_table()
            for _, row in summary.head(5).iterrows():
                report.append(f"  {row['Scenario']}: {row['Impact_Pct']:+.1%}")
            report.append("")

        report.append("="*80)

        return "\n".join(report)

    def save_all_results(self, output_dir: Path):
        """Save all results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary report
        with open(output_dir / "EXECUTIVE_SUMMARY.txt", 'w') as f:
            f.write(self.generate_summary_report())

        # Create subdirectories and save detailed results
        if self.results['chain_ladder']:
            cl_dir = output_dir / "chain_ladder"
            cl_dir.mkdir(exist_ok=True)
            self.results['chain_ladder'].ultimate_losses.to_csv(cl_dir / "reserves.csv")
            self.results['chain_ladder'].selected_factors.to_csv(cl_dir / "factors.csv")

        if self.results['mack']:
            mack_dir = output_dir / "mack_model"
            mack_dir.mkdir(exist_ok=True)
            self.results['mack'].summary().to_csv(mack_dir / "summary.csv")
            self.results['mack'].get_confidence_intervals().to_csv(mack_dir / "confidence_intervals.csv")

        if self.results['bootstrap']:
            boot_dir = output_dir / "bootstrap"
            boot_dir.mkdir(exist_ok=True)
            self.results['bootstrap'].get_reserve_statistics().to_csv(boot_dir / "statistics.csv")

        if self.results['cape_cod']:
            self.results['cape_cod'].save_results(output_dir / "cape_cod")

        if self.results['diagnostics']:
            diag_dir = output_dir / "diagnostics"
            diag_dir.mkdir(exist_ok=True)
            self.results['diagnostics']['residuals'].save_results(diag_dir / "residuals")
            self.results['diagnostics']['volatility'].save_results(diag_dir / "volatility")
            self.results['diagnostics']['tests'].save_results(diag_dir / "tests")

        if self.results['stress_tests']:
            self.results['stress_tests'].save_results(output_dir / "stress_tests")

        if self.results['tail_risk']:
            self.results['tail_risk'].save_results(output_dir / "tail_risk")

        print(f"\n✅ All results saved to: {output_dir}/")


def main():
    """Run enhanced workflow demo."""
    # Load data
    triangle_file = Path("data/processed/reported_absolute_losses.csv")
    premium_file = Path("data/processed/earned_premium.csv")

    if not triangle_file.exists():
        print(f"❌ Error: {triangle_file} not found")
        return

    triangle = pd.read_csv(triangle_file, index_col=0)

    earned_premium = None
    if premium_file.exists():
        premium_df = pd.read_csv(premium_file, index_col=0)
        earned_premium = premium_df.iloc[:, 0]

    # Create workflow
    workflow = EnhancedReservingWorkflow(
        triangle=triangle,
        earned_premium=earned_premium,
        verbose=True
    )

    # Run complete analysis
    workflow.run_complete_analysis(
        run_bootstrap=True,
        n_bootstrap_simulations=5000
    )

    # Save results
    workflow.save_all_results(Path("outputs/enhanced_analysis"))

    # Print summary
    print("\n" + workflow.generate_summary_report())


if __name__ == "__main__":
    main()
