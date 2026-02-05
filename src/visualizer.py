"""
Visualization tools for loss triangles and chain-ladder results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_loss_triangle_heatmap(triangle: pd.DataFrame, title: str, save_path: Path = None):
    """
    Plot loss triangle as a heatmap
    
    Args:
        triangle: Loss triangle DataFrame
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(
        triangle,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Loss Amount (USDm)'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Development Period (months)', fontsize=12)
    ax.set_ylabel('Accident Year', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.close()


def plot_development_patterns(triangle: pd.DataFrame, save_path: Path = None):
    """
    Plot development patterns by accident year
    
    Args:
        triangle: Loss triangle DataFrame
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each accident year
    for year in triangle.index:
        values = triangle.loc[year].dropna()
        ax.plot(values.index, values.values, marker='o', label=str(year), linewidth=2)
    
    ax.set_title('Loss Development Patterns by Accident Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Development Period (months)', fontsize=12)
    ax.set_ylabel('Cumulative Loss (USDm)', fontsize=12)
    ax.legend(title='Accident Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.close()


def plot_age_to_age_factors(factors_df: pd.DataFrame, selected_factors: pd.Series, 
                            save_path: Path = None):
    """
    Plot age-to-age factors with selected factors highlighted
    
    Args:
        factors_df: DataFrame of age-to-age factors by year
        selected_factors: Series of selected factors
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot individual year factors
    for year in factors_df.index:
        values = factors_df.loc[year].dropna()
        ax.plot(values.index, values.values, marker='o', alpha=0.3, color='gray', linewidth=1)
    
    # Plot selected factors
    ax.plot(selected_factors.index, selected_factors.values, 
            marker='s', color='red', linewidth=3, markersize=10, 
            label='Selected Factors (Simple Average)')
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title('Age-to-Age Development Factors', fontsize=16, fontweight='bold')
    ax.set_xlabel('Development Period (months)', fontsize=12)
    ax.set_ylabel('Development Factor', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.close()


def plot_reserves_by_year(ultimate_df: pd.DataFrame, save_path: Path = None):
    """
    Plot reserves by accident year
    
    Args:
        ultimate_df: DataFrame with Ultimate and Reserve columns
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of reserves
    years = ultimate_df.index
    reserves = ultimate_df['Reserve']
    
    colors = ['green' if r < 0 else 'red' for r in reserves]
    
    ax1.bar(years, reserves, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Reserves by Accident Year', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Accident Year', fontsize=12)
    ax1.set_ylabel('Reserve (USDm)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linewidth=1)
    
    # Rotate x labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Stacked bar: Latest vs Reserve
    latest = ultimate_df['Latest_Value']
    
    ax2.bar(years, latest, label='Latest Reported', color='steelblue', alpha=0.7)
    ax2.bar(years, reserves, bottom=latest, label='Reserve', color='orange', alpha=0.7)
    
    ax2.set_title('Latest Reported vs Ultimate Losses', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Accident Year', fontsize=12)
    ax2.set_ylabel('Loss Amount (USDm)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.close()


def plot_reserve_analysis_summary(ultimate_df: pd.DataFrame, save_path: Path = None):
    """
    Create a comprehensive summary dashboard
    
    Args:
        ultimate_df: DataFrame with Ultimate and Reserve columns
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Reserve by year (bar)
    ax1 = fig.add_subplot(gs[0, 0])
    years = ultimate_df.index
    reserves = ultimate_df['Reserve']
    colors = ['green' if r < 0 else 'red' for r in reserves]
    ax1.bar(years, reserves, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Reserves by Accident Year', fontweight='bold')
    ax1.set_ylabel('Reserve (USDm)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Ultimate vs Latest (scatter)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ultimate_df['Latest_Value'], ultimate_df['Ultimate'], 
                s=100, alpha=0.6, edgecolor='black')
    
    # Add diagonal line
    max_val = max(ultimate_df['Ultimate'].max(), ultimate_df['Latest_Value'].max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
    
    ax2.set_title('Ultimate vs Latest Reported', fontweight='bold')
    ax2.set_xlabel('Latest Reported (USDm)')
    ax2.set_ylabel('Ultimate (USDm)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reserve percentage by year
    ax3 = fig.add_subplot(gs[1, 0])
    reserve_pct = (ultimate_df['Reserve'] / ultimate_df['Latest_Value'] * 100)
    ax3.bar(years, reserve_pct, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_title('Reserve as % of Latest Reported', fontweight='bold')
    ax3.set_ylabel('Reserve %')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Cumulative reserves
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative_reserves = reserves.sort_index().cumsum()
    ax4.fill_between(cumulative_reserves.index, 0, cumulative_reserves.values, 
                     alpha=0.3, color='orange')
    ax4.plot(cumulative_reserves.index, cumulative_reserves.values, 
            marker='o', linewidth=2, markersize=6, color='darkorange')
    ax4.set_title('Cumulative Reserves by Accident Year', fontweight='bold')
    ax4.set_ylabel('Cumulative Reserve (USDm)')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Summary statistics (text)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    total_latest = ultimate_df['Latest_Value'].sum()
    total_ultimate = ultimate_df['Ultimate'].sum()
    total_reserve = ultimate_df['Reserve'].sum()
    avg_reserve_pct = (total_reserve / total_latest * 100)
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*60}
    
    Total Latest Reported:        ${total_latest:>15,.0f}m
    Total Ultimate Losses:        ${total_ultimate:>15,.0f}m
    Total Reserve Needed:         ${total_reserve:>15,.0f}m
    
    Reserve as % of Latest:       {avg_reserve_pct:>15.2f}%
    Number of Accident Years:     {len(ultimate_df):>15}
    
    Largest Reserve (Year):       {reserves.idxmax()} (${reserves.max():,.0f}m)
    Smallest Reserve (Year):      {reserves.idxmin()} (${reserves.min():,.0f}m)
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    fig.suptitle('Chain-Ladder Reserve Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations for the chain-ladder analysis"""
    
    print("\n🎨 Generating Visualizations...")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading data files...")
    triangle = pd.read_csv("data/processed/reported_absolute_losses.csv", index_col=0)
    age_to_age = pd.read_csv("outputs/chain_ladder/age_to_age_factors.csv", index_col=0)
    selected = pd.read_csv("outputs/chain_ladder/selected_factors.csv", index_col=0).iloc[:, 0]
    ultimate = pd.read_csv("outputs/chain_ladder/ultimate_and_reserves.csv", index_col=0)
    print("   ✅ Data loaded\n")
    
    # Generate plots
    print("📊 Generating plots...\n")
    
    print("1. Loss Triangle Heatmap")
    plot_loss_triangle_heatmap(
        triangle,
        "Swiss Re Property Reinsurance - Reported Losses Triangle",
        output_dir / "01_triangle_heatmap.png"
    )
    
    print("2. Development Patterns")
    plot_development_patterns(
        triangle,
        output_dir / "02_development_patterns.png"
    )
    
    print("3. Age-to-Age Factors")
    plot_age_to_age_factors(
        age_to_age,
        selected,
        output_dir / "03_age_to_age_factors.png"
    )
    
    print("4. Reserves by Year")
    plot_reserves_by_year(
        ultimate,
        output_dir / "04_reserves_by_year.png"
    )
    
    print("5. Reserve Analysis Dashboard")
    plot_reserve_analysis_summary(
        ultimate,
        output_dir / "05_reserve_dashboard.png"
    )
    
    print("\n" + "="*80)
    print(f"✨ All visualizations saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    generate_all_visualizations()