"""
Extract Property Reinsurance REPORTED triangle from Swiss Re data
Converts loss ratios to absolute values using earned premium
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_reported_triangle(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract REPORTED loss triangle from Swiss Re Excel
    
    Returns:
        tuple: (loss_ratios_df, absolute_losses_df)
    """
    # Load the sheet
    df = pd.read_excel(file_path, sheet_name='Property Reinsurance', header=None)
    
    print("🔍 Extracting REPORTED Loss Ratios triangle...\n")
    
    # Extract development months (row 10, starting from column 7)
    dev_months_row = df.iloc[10, 7:].values
    dev_months = [int(x) for x in dev_months_row if pd.notna(x)]
    print(f"📅 Development months found: {dev_months}")
    
    # Extract data rows (rows 11-26)
    data_rows = df.iloc[11:27].copy()
    
    # Extract key columns
    treaty_years = data_rows.iloc[:, 3].astype(int).values
    earned_premium = data_rows.iloc[:, 5].values
    
    # Extract loss ratios (from column 7 onwards, matching dev_months length)
    loss_ratios_data = data_rows.iloc[:, 7:7+len(dev_months)].values
    
    # Create loss ratios DataFrame
    loss_ratios_df = pd.DataFrame(
        loss_ratios_data,
        index=treaty_years,
        columns=dev_months
    )
    loss_ratios_df.index.name = 'Treaty Year'
    
    print(f"\n✅ Loss Ratios Triangle extracted: {loss_ratios_df.shape}")
    print(f"   Years: {treaty_years[0]} - {treaty_years[-1]}")
    print(f"   Development months: {len(dev_months)}")
    
    # Calculate absolute losses (Loss Ratio × Earned Premium)
    absolute_losses_data = loss_ratios_data * earned_premium[:, np.newaxis]
    
    absolute_losses_df = pd.DataFrame(
        absolute_losses_data,
        index=treaty_years,
        columns=dev_months
    )
    absolute_losses_df.index.name = 'Treaty Year'
    
    print(f"\n✅ Absolute Losses Triangle calculated")
    
    # Also return earned premium for reference
    earned_premium_df = pd.DataFrame({
        'Treaty Year': treaty_years,
        'Earned Premium (USDm)': earned_premium
    }).set_index('Treaty Year')
    
    return loss_ratios_df, absolute_losses_df, earned_premium_df


def display_triangle(df: pd.DataFrame, title: str, decimals: int = 4):
    """Pretty print a triangle"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    if decimals == 0:
        print(df.round(0))
    else:
        print(df.round(decimals))


def save_triangles(loss_ratios_df, absolute_losses_df, earned_premium_df):
    """Save triangles to CSV files"""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    loss_ratios_df.to_csv(output_dir / "reported_loss_ratios.csv")
    absolute_losses_df.to_csv(output_dir / "reported_absolute_losses.csv")
    earned_premium_df.to_csv(output_dir / "earned_premium.csv")
    
    print(f"\n💾 Files saved to data/processed/:")
    print(f"   - reported_loss_ratios.csv")
    print(f"   - reported_absolute_losses.csv")
    print(f"   - earned_premium.csv")


if __name__ == "__main__":
    data_file = Path("data/raw/swiss_re_2023_triangles.xlsx")
    
    print("🚀 Swiss Re Property Reinsurance - REPORTED Triangle Extraction\n")
    print("="*80 + "\n")
    
    # Extract triangles
    loss_ratios, absolute_losses, earned_premium = extract_reported_triangle(data_file)
    
    # Display results
    display_triangle(loss_ratios, "REPORTED LOSS RATIOS (as % of Earned Premium)", decimals=4)
    display_triangle(absolute_losses, "REPORTED ABSOLUTE LOSSES (USDm)", decimals=2)
    display_triangle(earned_premium, "EARNED PREMIUM (USDm)", decimals=2)
    
    # Save to CSV
    save_triangles(loss_ratios, absolute_losses, earned_premium)
    
    print("\n" + "="*80)
    print("✨ Extraction complete!")
    print("="*80 + "\n")
    
    # Show some summary stats
    print("📊 Quick Stats:")
    print(f"   Total Earned Premium: ${earned_premium.sum().values[0]:,.0f}m")
    print(f"   Latest Year Ultimate Loss Ratio: {loss_ratios.iloc[-1, -1]:.2%}")
    print(f"   Average Ultimate Loss Ratio (mature years): {loss_ratios.iloc[:10, -1].mean():.2%}")