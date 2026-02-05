"""
Chain-Ladder Calculator for Loss Reserving
Calculates ultimate losses and reserves from development triangles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


class ChainLadder:
    """Chain-Ladder reserving method implementation"""
    
    def __init__(self, triangle: pd.DataFrame):
        """
        Initialize with a loss development triangle
        
        Args:
            triangle: DataFrame with accident years as rows, development periods as columns
        """
        self.triangle = triangle.copy()
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)
        
        # Results (calculated later)
        self.age_to_age_factors = None
        self.selected_factors = None
        self.cum_factors = None
        self.ultimate_losses = None
        self.reserves = None
        
    def calculate_age_to_age_factors(self) -> pd.DataFrame:
        """
        Calculate age-to-age development factors
        
        Returns:
            DataFrame of development factors
        """
        factors = pd.DataFrame(
            index=self.triangle.index,
            columns=self.triangle.columns[:-1]
        )
        
        for i in range(len(self.triangle.columns) - 1):
            col_current = self.triangle.columns[i]
            col_next = self.triangle.columns[i + 1]
            
            # Calculate factor: next period / current period
            factors[col_current] = self.triangle[col_next] / self.triangle[col_current]
        
        self.age_to_age_factors = factors
        return factors
    
    def select_development_factors(self, method: str = 'simple_average') -> pd.Series:
        """
        Select development factors to use for projection
        
        Args:
            method: Method for selecting factors
                - 'simple_average': arithmetic mean
                - 'volume_weighted': weighted by exposure
                - 'latest_year': use most recent year
        
        Returns:
            Series of selected factors
        """
        if self.age_to_age_factors is None:
            self.calculate_age_to_age_factors()
        
        if method == 'simple_average':
            # Exclude infinite values and calculate mean
            selected = self.age_to_age_factors.replace([np.inf, -np.inf], np.nan).mean()
        
        elif method == 'volume_weighted':
            # Weight by the triangle values
            weights = self.triangle.iloc[:, :-1]
            weighted_sum = (self.age_to_age_factors * weights).sum()
            weight_total = weights.sum()
            selected = weighted_sum / weight_total
        
        elif method == 'latest_year':
            # Use the most recent year's factors
            selected = self.age_to_age_factors.iloc[-1]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Replace NaN with 1.0 (no development)
        selected = selected.fillna(1.0)
        
        self.selected_factors = selected
        return selected
    
    def calculate_cumulative_factors(self) -> pd.Series:
        """
        Calculate cumulative development factors (tail to each age)
        
        Returns:
            Series of cumulative factors
        """
        if self.selected_factors is None:
            self.select_development_factors()
        
        # Start from the end and multiply backwards
        cum_factors = pd.Series(index=self.selected_factors.index, dtype=float)
        
        cum_factor = 1.0
        for period in reversed(self.selected_factors.index):
            cum_factor *= self.selected_factors[period]
            cum_factors[period] = cum_factor
        
        self.cum_factors = cum_factors
        return cum_factors
    
    def project_ultimate_losses(self) -> pd.DataFrame:
        """
        Project ultimate losses for all accident years
        
        Returns:
            DataFrame with latest values, ultimate values, and reserves
        """
        if self.cum_factors is None:
            self.calculate_cumulative_factors()
        
        results = pd.DataFrame(index=self.triangle.index)
        
        # For each accident year, find the latest known value
        latest_values = []
        latest_ages = []
        
        for year in self.triangle.index:
            # Find last non-NaN value in the row
            row = self.triangle.loc[year]
            last_value = row.dropna().iloc[-1]
            last_age = row.dropna().index[-1]
            
            latest_values.append(last_value)
            latest_ages.append(last_age)
        
        results['Latest_Value'] = latest_values
        results['Latest_Age'] = latest_ages
        
        # Calculate ultimate by applying cumulative factor
        ultimate = []
        for i, year in enumerate(self.triangle.index):
            latest_age = latest_ages[i]
            latest_value = latest_values[i]
            
            # Get the cumulative factor from latest age to ultimate
            if latest_age in self.cum_factors.index:
                cum_factor = self.cum_factors[latest_age]
            else:
                # Already at ultimate
                cum_factor = 1.0
            
            ult = latest_value * cum_factor
            ultimate.append(ult)
        
        results['Ultimate'] = ultimate
        results['Reserve'] = results['Ultimate'] - results['Latest_Value']
        
        self.ultimate_losses = results
        return results
    
    def run_full_analysis(self, method: str = 'simple_average') -> dict:
        """
        Run complete chain-ladder analysis
        
        Args:
            method: Method for selecting development factors
        
        Returns:
            Dictionary with all results
        """
        print("🔗 Running Chain-Ladder Analysis...\n")
        
        # Step 1: Age-to-age factors
        print("📊 Step 1: Calculate age-to-age factors")
        self.calculate_age_to_age_factors()
        print(f"   ✅ Calculated factors for {len(self.age_to_age_factors.columns)} development periods\n")
        
        # Step 2: Select factors
        print(f"📊 Step 2: Select development factors (method: {method})")
        self.select_development_factors(method=method)
        print(f"   ✅ Selected {len(self.selected_factors)} factors\n")
        
        # Step 3: Cumulative factors
        print("📊 Step 3: Calculate cumulative development factors")
        self.calculate_cumulative_factors()
        print(f"   ✅ Calculated cumulative factors\n")
        
        # Step 4: Project ultimate
        print("📊 Step 4: Project ultimate losses and calculate reserves")
        self.project_ultimate_losses()
        print(f"   ✅ Projected ultimate for {len(self.ultimate_losses)} accident years\n")
        
        return {
            'age_to_age_factors': self.age_to_age_factors,
            'selected_factors': self.selected_factors,
            'cumulative_factors': self.cum_factors,
            'ultimate_losses': self.ultimate_losses
        }
    
    def summary(self) -> dict:
        """Get summary statistics"""
        if self.ultimate_losses is None:
            self.run_full_analysis()
        
        total_reserve = self.ultimate_losses['Reserve'].sum()
        total_ultimate = self.ultimate_losses['Ultimate'].sum()
        total_latest = self.ultimate_losses['Latest_Value'].sum()
        
        return {
            'total_latest': total_latest,
            'total_ultimate': total_ultimate,
            'total_reserve': total_reserve,
            'reserve_to_latest_ratio': total_reserve / total_latest if total_latest > 0 else 0,
            'n_accident_years': len(self.ultimate_losses)
        }


def display_results(cl: ChainLadder):
    """Pretty print chain-ladder results"""
    
    print("\n" + "="*80)
    print("AGE-TO-AGE DEVELOPMENT FACTORS")
    print("="*80 + "\n")
    print(cl.age_to_age_factors.round(4))
    
    print("\n" + "="*80)
    print("SELECTED DEVELOPMENT FACTORS")
    print("="*80 + "\n")
    print(cl.selected_factors.round(4))
    
    print("\n" + "="*80)
    print("CUMULATIVE DEVELOPMENT FACTORS (to Ultimate)")
    print("="*80 + "\n")
    print(cl.cum_factors.round(4))
    
    print("\n" + "="*80)
    print("ULTIMATE LOSSES AND RESERVES (USDm)")
    print("="*80 + "\n")
    print(cl.ultimate_losses.round(2))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    summary = cl.summary()
    print(f"Total Latest Reported:  ${summary['total_latest']:>15,.0f}m")
    print(f"Total Ultimate:         ${summary['total_ultimate']:>15,.0f}m")
    print(f"Total Reserve Needed:   ${summary['total_reserve']:>15,.0f}m")
    print(f"Reserve/Latest Ratio:   {summary['reserve_to_latest_ratio']:>15.2%}")
    print(f"Number of Years:        {summary['n_accident_years']:>15}")


def save_results(cl: ChainLadder, output_dir: Path):
    """Save all results to CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cl.age_to_age_factors.to_csv(output_dir / 'age_to_age_factors.csv')
    cl.selected_factors.to_csv(output_dir / 'selected_factors.csv')
    cl.cum_factors.to_csv(output_dir / 'cumulative_factors.csv')
    cl.ultimate_losses.to_csv(output_dir / 'ultimate_and_reserves.csv')
    
    print(f"\n💾 Results saved to {output_dir}/")


if __name__ == "__main__":
    print("🚀 Chain-Ladder Reserving Analysis")
    print("="*80 + "\n")
    
    # Load the triangle
    triangle_file = Path("data/processed/reported_absolute_losses.csv")
    
    if not triangle_file.exists():
        print(f"❌ Error: {triangle_file} not found!")
        print("   Run extract_triangle.py first to generate the data.")
        exit(1)
    
    print(f"📂 Loading triangle from: {triangle_file}")
    triangle = pd.read_csv(triangle_file, index_col=0)
    print(f"   ✅ Loaded triangle: {triangle.shape[0]} years × {triangle.shape[1]} periods\n")
    
    # Run chain-ladder
    cl = ChainLadder(triangle)
    results = cl.run_full_analysis(method='simple_average')
    
    # Display results
    display_results(cl)
    
    # Save results
    output_dir = Path("outputs/chain_ladder")
    save_results(cl, output_dir)
    
    print("\n" + "="*80)
    print("✨ Chain-Ladder analysis complete!")
    print("="*80)