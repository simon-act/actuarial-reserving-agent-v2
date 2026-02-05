"""
Benchmark Script
================

Test the reserving framework on multiple standard triangles from the literature.

This script runs all implemented methods on each sample triangle and
compares results with known/expected values where available.

Usage:
    python benchmark.py
    python benchmark.py --triangle mack
    python benchmark.py --methods chain_ladder mack
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import TriangleLoader, TriangleInfo, load_sample
from chain_ladder import ChainLadder
from stochastic_reserving.mack_model import MackChainLadder
from stochastic_reserving.bootstrap import BootstrapChainLadder
from alternative_methods.cape_cod import CapeCod
from tail_fitting import TailEstimator
from diagnostics.diagnostic_tests import DiagnosticTests
from diagnostics.residual_analysis import ResidualAnalyzer


# Known results from literature for validation
KNOWN_RESULTS = {
    'mack': {
        # From Mack (1993) paper
        'total_reserve': 18681,  # Approximate
        'total_se': 4072,  # Approximate standard error
        'source': 'Mack (1993) - Distribution-free calculation of the SE of CL'
    },
    'taylor_ashe': {
        'total_reserve': 18834,  # Last origin year ultimate
        'source': 'Taylor & Ashe - commonly used benchmark'
    }
}


class BenchmarkRunner:
    """
    Run reserving methods on multiple triangles and compare results.
    """

    def __init__(self, output_dir: str = None):
        self.loader = TriangleLoader()
        self.results = {}

        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "outputs" / "benchmark"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_triangle(
        self,
        name: str,
        triangle: pd.DataFrame = None,
        run_bootstrap: bool = True,
        n_bootstrap: int = 1000
    ) -> Dict:
        """
        Run all methods on a single triangle.

        Args:
            name: Triangle name
            triangle: Triangle DataFrame (if None, loads sample)
            run_bootstrap: Whether to run bootstrap (slow)
            n_bootstrap: Number of bootstrap simulations

        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {name.upper()}")
        print(f"{'='*70}")

        # Load triangle if not provided
        if triangle is None:
            triangle = self.loader.load_sample(name)

        # Triangle info
        info = TriangleInfo(triangle)
        info.print_summary()

        results = {
            'name': name,
            'n_years': info.n_years,
            'n_periods': info.n_periods,
            'total_latest': info.total_latest,
            'methods': {}
        }

        # 1. Chain-Ladder
        print("\n📊 Running Chain-Ladder...")
        cl = ChainLadder(triangle)
        cl.run_full_analysis()
        cl_summary = cl.summary()

        results['methods']['chain_ladder'] = {
            'total_ultimate': cl_summary['total_ultimate'],
            'total_reserve': cl_summary['total_reserve'],
            'reserve_ratio': cl_summary['reserve_to_latest_ratio']
        }
        print(f"   Reserve: {cl_summary['total_reserve']:,.0f}")

        # 2. Mack Model
        print("\n📊 Running Mack Model...")
        mack = MackChainLadder(triangle)
        mack.fit()
        mack_df = mack.summary()

        total_se = mack_df['SE'].sum()
        total_reserve = mack_df['Reserve'].sum()
        cv = total_se / total_reserve if total_reserve > 0 else 0

        results['methods']['mack'] = {
            'total_reserve': total_reserve,
            'total_se': total_se,
            'cv': cv
        }
        print(f"   Reserve: {total_reserve:,.0f} ± {total_se:,.0f} (CV: {cv:.1%})")

        # 3. Bootstrap (optional - slow)
        if run_bootstrap:
            print(f"\n📊 Running Bootstrap ({n_bootstrap} simulations)...")
            try:
                boot = BootstrapChainLadder(triangle, n_simulations=n_bootstrap)
                boot.fit()
                boot_stats = boot.get_reserve_statistics()

                results['methods']['bootstrap'] = {
                    'mean': boot_stats['Mean'].sum(),
                    'std': boot_stats['Std'].sum(),
                    'p50': boot_stats['P50'].sum(),
                    'p75': boot_stats['P75'].sum(),
                    'p95': boot_stats['P95'].sum(),
                    'p99': boot_stats['P99'].sum()
                }
                print(f"   P50: {boot_stats['P50'].sum():,.0f}")
                print(f"   P95: {boot_stats['P95'].sum():,.0f}")
            except Exception as e:
                print(f"   ⚠️ Bootstrap failed: {e}")
                results['methods']['bootstrap'] = {'error': str(e)}

        # 4. Tail Fitting
        print("\n📊 Running Tail Fitting...")
        try:
            tail = TailEstimator(triangle)
            tail.fit()

            results['methods']['tail_fitting'] = {
                'best_method': tail.best_method,
                'tail_factor': tail.tail_factor,
                'comparison': tail.get_comparison_table().to_dict('records')
            }
            print(f"   Best method: {tail.best_method}")
            print(f"   Tail factor: {tail.tail_factor:.6f}")
        except Exception as e:
            print(f"   ⚠️ Tail fitting failed: {e}")
            results['methods']['tail_fitting'] = {'error': str(e)}

        # 5. Diagnostics
        print("\n📊 Running Diagnostics...")
        try:
            diag = DiagnosticTests(triangle, cl.selected_factors)
            diag.run_all_tests()
            score = diag.get_model_adequacy_score()

            results['diagnostics'] = {
                'adequacy_score': score['adequacy_score'],
                'rating': score['rating'],
                'n_issues': score['n_issues'],
                'issues': score['issues']
            }
            print(f"   Score: {score['adequacy_score']:.0f}% ({score['rating']})")
            if score['issues']:
                print(f"   Issues: {', '.join(score['issues'][:3])}")
        except Exception as e:
            print(f"   ⚠️ Diagnostics failed: {e}")
            results['diagnostics'] = {'error': str(e)}

        # Compare with known results
        if name in KNOWN_RESULTS:
            known = KNOWN_RESULTS[name]
            print(f"\n📚 Comparison with Literature ({known['source']}):")

            if 'total_reserve' in known:
                diff_pct = (cl_summary['total_reserve'] - known['total_reserve']) / known['total_reserve'] * 100
                print(f"   Known reserve: {known['total_reserve']:,.0f}")
                print(f"   Our reserve:   {cl_summary['total_reserve']:,.0f} ({diff_pct:+.1f}%)")

            if 'total_se' in known:
                diff_pct = (total_se - known['total_se']) / known['total_se'] * 100
                print(f"   Known SE: {known['total_se']:,.0f}")
                print(f"   Our SE:   {total_se:,.0f} ({diff_pct:+.1f}%)")

            results['known_results'] = known

        return results

    def run_all_samples(self, run_bootstrap: bool = False) -> Dict:
        """Run benchmark on all sample triangles."""
        print("\n" + "=" * 70)
        print("RUNNING BENCHMARK ON ALL SAMPLE TRIANGLES")
        print("=" * 70)

        all_results = {}
        samples = ['mack', 'taylor_ashe', 'abc', 'uk_motor']

        for name in samples:
            try:
                results = self.run_single_triangle(name, run_bootstrap=run_bootstrap)
                all_results[name] = results
            except Exception as e:
                print(f"\n❌ Failed on {name}: {e}")
                all_results[name] = {'error': str(e)}

        # Summary table
        self._print_summary_table(all_results)

        # Save results
        self._save_results(all_results)

        return all_results

    def _print_summary_table(self, all_results: Dict):
        """Print summary comparison table."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Build summary DataFrame
        rows = []
        for name, results in all_results.items():
            if 'error' in results:
                continue

            row = {
                'Triangle': name,
                'Years': results.get('n_years', ''),
                'Periods': results.get('n_periods', ''),
                'Total Latest': results.get('total_latest', 0),
            }

            methods = results.get('methods', {})

            if 'chain_ladder' in methods:
                row['CL Reserve'] = methods['chain_ladder'].get('total_reserve', 0)

            if 'mack' in methods:
                row['Mack SE'] = methods['mack'].get('total_se', 0)
                row['CV'] = methods['mack'].get('cv', 0)

            diag = results.get('diagnostics', {})
            row['Diag Score'] = diag.get('adequacy_score', '')

            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))

    def _save_results(self, all_results: Dict):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV summary
        rows = []
        for name, results in all_results.items():
            if 'error' in results:
                continue

            methods = results.get('methods', {})
            row = {
                'triangle': name,
                'n_years': results.get('n_years'),
                'n_periods': results.get('n_periods'),
                'total_latest': results.get('total_latest'),
                'cl_reserve': methods.get('chain_ladder', {}).get('total_reserve'),
                'mack_reserve': methods.get('mack', {}).get('total_reserve'),
                'mack_se': methods.get('mack', {}).get('total_se'),
                'mack_cv': methods.get('mack', {}).get('cv'),
                'tail_factor': methods.get('tail_fitting', {}).get('tail_factor'),
                'tail_method': methods.get('tail_fitting', {}).get('best_method'),
                'diag_score': results.get('diagnostics', {}).get('adequacy_score'),
                'diag_rating': results.get('diagnostics', {}).get('rating')
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        output_file = self.output_dir / f"benchmark_{timestamp}.csv"
        df.to_csv(output_file, index=False)

        print(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run reserving benchmark')
    parser.add_argument('--triangle', '-t', type=str, default=None,
                       help='Specific triangle to test (mack, taylor_ashe, abc, uk_motor)')
    parser.add_argument('--bootstrap', '-b', action='store_true',
                       help='Include bootstrap simulations (slower)')
    parser.add_argument('--n-bootstrap', '-n', type=int, default=1000,
                       help='Number of bootstrap simulations')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available sample triangles')

    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.list:
        runner.loader.list_samples()
        return

    if args.triangle:
        runner.run_single_triangle(
            args.triangle,
            run_bootstrap=args.bootstrap,
            n_bootstrap=args.n_bootstrap
        )
    else:
        runner.run_all_samples(run_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
