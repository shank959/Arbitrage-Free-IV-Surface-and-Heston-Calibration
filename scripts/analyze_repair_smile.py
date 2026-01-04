#!/usr/bin/env python3
"""
Analyze Repair Impact on Smile Characteristics

This script investigates why repair makes the smile steeper by analyzing:
1. Repair adjustments by moneyness bucket (OTM, ATM, ITM)
2. IV changes by moneyness
3. Systematic biases in repair algorithm
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.common.iv_computation import implied_vol

# Default paths
DEFAULT_PARQUET = Path("reports/phase4/1545/repaired_options.parquet")
DEFAULT_OUTPUT = Path("reports/diagnostic")


def analyze_repair_by_moneyness(
    parquet_path: Path,
    quote_date: str,
    output_dir: Path,
) -> Dict:
    """
    Analyze repair adjustments and IV changes by moneyness bucket.
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING REPAIR IMPACT BY MONEYNESS - {quote_date}")
    print(f"{'='*70}")
    
    df = pd.read_parquet(parquet_path)
    df = df[df['quote_date'] == quote_date].copy()
    
    if df.empty:
        raise ValueError(f"No data for quote_date {quote_date}")
    
    spot = df['spot'].iloc[0]
    df['moneyness'] = df['strike'] / spot
    df['adjustment'] = df['price_repaired'] - df['mid']
    df['adjustment_pct'] = (df['adjustment'] / df['mid'].abs()) * 100
    df.loc[df['mid'] == 0, 'adjustment_pct'] = np.nan
    
    # Compute IVs for raw and repaired prices
    print("\nComputing IVs for raw and repaired prices...")
    raw_ivs = []
    rep_ivs = []
    
    for _, row in df.iterrows():
        raw_iv = implied_vol(row['mid'], row['spot'], row['strike'], 
                            row['ttm_years'], 0, 0, row['option_type'] == 'call')
        rep_iv = implied_vol(row['price_repaired'], row['spot'], row['strike'],
                            row['ttm_years'], 0, 0, row['option_type'] == 'call')
        raw_ivs.append(raw_iv)
        rep_ivs.append(rep_iv)
    
    df['raw_iv'] = raw_ivs
    df['repaired_iv'] = rep_ivs
    df['iv_change'] = df['repaired_iv'] - df['raw_iv']
    df['iv_change_pct'] = df['iv_change'] / df['raw_iv'] * 100
    
    # Filter out invalid IVs
    df_valid = df[df['raw_iv'].notna() & df['repaired_iv'].notna()].copy()
    
    # Define moneyness buckets
    def get_moneyness_bucket(m):
        if m < 0.90:
            return 'Deep OTM'
        elif m < 0.95:
            return 'OTM'
        elif m < 1.05:
            return 'ATM'
        elif m < 1.10:
            return 'ITM'
        else:
            return 'Deep ITM'
    
    df_valid['moneyness_bucket'] = df_valid['moneyness'].apply(get_moneyness_bucket)
    
    # Analyze by bucket
    print("\n" + "="*70)
    print("REPAIR ADJUSTMENTS BY MONEYNESS BUCKET")
    print("="*70)
    
    bucket_stats = []
    for bucket in ['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM']:
        bucket_df = df_valid[df_valid['moneyness_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue
        
        stats = {
            'bucket': bucket,
            'n_options': len(bucket_df),
            'mean_adjustment': bucket_df['adjustment'].mean(),
            'median_adjustment': bucket_df['adjustment'].median(),
            'mean_adjustment_pct': bucket_df['adjustment_pct'].mean(),
            'mean_iv_change': bucket_df['iv_change'].mean() * 100,  # Convert to %
            'mean_iv_change_pct': bucket_df['iv_change_pct'].mean(),
            'mean_raw_iv': bucket_df['raw_iv'].mean() * 100,
            'mean_repaired_iv': bucket_df['repaired_iv'].mean() * 100,
        }
        bucket_stats.append(stats)
        
        print(f"\n{bucket}:")
        print(f"  Options: {stats['n_options']}")
        print(f"  Mean adjustment: ${stats['mean_adjustment']:.4f} ({stats['mean_adjustment_pct']:.2f}%)")
        print(f"  Mean IV change: {stats['mean_iv_change']:+.3f}% ({stats['mean_iv_change_pct']:+.2f}%)")
        print(f"  Raw IV: {stats['mean_raw_iv']:.2f}% → Repaired IV: {stats['mean_repaired_iv']:.2f}%")
    
    bucket_df = pd.DataFrame(bucket_stats)
    
    # Check if OTM adjustments are systematically larger
    if len(bucket_df) >= 2:
        otm_buckets = bucket_df[bucket_df['bucket'].isin(['Deep OTM', 'OTM'])]
        atm_bucket = bucket_df[bucket_df['bucket'] == 'ATM']
        
        if len(otm_buckets) > 0 and len(atm_bucket) > 0:
            otm_mean_iv_change = otm_buckets['mean_iv_change'].mean()
            atm_mean_iv_change = atm_bucket['mean_iv_change'].iloc[0]
            
            print(f"\n{'='*70}")
            print("SMILE STEEPENING ANALYSIS")
            print(f"{'='*70}")
            print(f"OTM mean IV change: {otm_mean_iv_change:+.3f}%")
            print(f"ATM mean IV change: {atm_mean_iv_change:+.3f}%")
            print(f"Difference (OTM - ATM): {otm_mean_iv_change - atm_mean_iv_change:+.3f}%")
            
            if otm_mean_iv_change > atm_mean_iv_change:
                print("\n⚠️  REPAIR IS INCREASING OTM IVs MORE THAN ATM - This makes smile steeper!")
            else:
                print("\n✓ Repair is not systematically steepening the smile")
    
    # Analyze by expiration
    print(f"\n{'='*70}")
    print("REPAIR ADJUSTMENTS BY EXPIRATION")
    print(f"{'='*70}")
    
    exp_stats = []
    for exp, exp_df in df_valid.groupby('expiration'):
        stats = {
            'expiration': exp,
            'n_options': len(exp_df),
            'mean_adjustment': exp_df['adjustment'].mean(),
            'mean_iv_change': exp_df['iv_change'].mean() * 100,
            'mean_raw_iv': exp_df['raw_iv'].mean() * 100,
            'mean_repaired_iv': exp_df['repaired_iv'].mean() * 100,
        }
        exp_stats.append(stats)
        print(f"{exp}: {stats['n_options']} options, IV change: {stats['mean_iv_change']:+.3f}%")
    
    # Create visualizations
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Adjustment by moneyness
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Adjustment amount by moneyness
    ax = axes[0, 0]
    ax.scatter(df_valid['moneyness'], df_valid['adjustment'], alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Price Adjustment ($)')
    ax.set_title('Price Adjustments by Moneyness')
    ax.grid(True, alpha=0.3)
    
    # IV change by moneyness
    ax = axes[0, 1]
    ax.scatter(df_valid['moneyness'], df_valid['iv_change'] * 100, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('IV Change (%)')
    ax.set_title('IV Changes by Moneyness')
    ax.grid(True, alpha=0.3)
    
    # Box plot by bucket
    ax = axes[1, 0]
    bucket_order = ['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM']
    bucket_data = [df_valid[df_valid['moneyness_bucket'] == b]['iv_change'].values * 100 
                   for b in bucket_order if len(df_valid[df_valid['moneyness_bucket'] == b]) > 0]
    bucket_labels = [b for b in bucket_order if len(df_valid[df_valid['moneyness_bucket'] == b]) > 0]
    ax.boxplot(bucket_data, labels=bucket_labels)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('IV Change (%)')
    ax.set_title('IV Changes by Moneyness Bucket')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Smile comparison
    ax = axes[1, 1]
    # Sample one expiration for clarity
    sample_exp = df_valid['expiration'].iloc[0]
    exp_df = df_valid[df_valid['expiration'] == sample_exp].sort_values('moneyness')
    ax.plot(exp_df['moneyness'], exp_df['raw_iv'] * 100, 'o-', label='Raw', alpha=0.7)
    ax.plot(exp_df['moneyness'], exp_df['repaired_iv'] * 100, 's-', label='Repaired', alpha=0.7)
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('IV (%)')
    ax.set_title(f'Smile Comparison ({sample_exp})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'repair_smile_analysis_{quote_date}.png', dpi=150)
    print(f"\nSaved plot: {output_dir / f'repair_smile_analysis_{quote_date}.png'}")
    plt.close()
    
    # Save detailed results
    bucket_df.to_csv(output_dir / f'repair_by_moneyness_{quote_date}.csv', index=False)
    df_valid[['strike', 'moneyness', 'moneyness_bucket', 'mid', 'price_repaired', 
              'adjustment', 'raw_iv', 'repaired_iv', 'iv_change']].to_csv(
        output_dir / f'repair_details_{quote_date}.csv', index=False)
    
    return {
        'bucket_stats': bucket_stats,
        'n_options': len(df_valid),
        'mean_iv_change': df_valid['iv_change'].mean() * 100,
    }


def main(argv=None):
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze repair impact on smile characteristics",
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default="2025-10-15",
        help="Quote date to analyze (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to repaired_options.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for results",
    )
    
    args = parser.parse_args(argv)
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    analyze_repair_by_moneyness(
        parquet_path=args.input,
        quote_date=args.date,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

