#!/usr/bin/env python3
"""
Full Pipeline Diagnostic Investigation

Implements comprehensive diagnostic checks to understand why:
1. Heston RMSE is 5% instead of 1-3%
2. Repair makes Heston calibration WORSE (-1.8% improvement)

Based on the diagnostic plan in full_pipeline_diagnostic_bf162627.plan.md
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.phase3.smooth import implied_vol as iv_phase3
from pipelines.phase5.prepare_data import implied_vol as iv_phase5, prepare_calibration_data
from pipelines.phase5.heston_pricer import heston_implied_vol
from pipelines.phase5.calibrate import calibrate_heston

# Default paths
DEFAULT_PARQUET = Path("reports/phase4/1545/repaired_options.parquet")
DEFAULT_REPAIR_SUMMARY = Path("reports/phase4/1545/repair_summary.csv")
DEFAULT_OUTPUT = Path("reports/diagnostic")


def phase_a1_check_raw_data_quality(parquet_path: Path, quote_date: str) -> Dict:
    """
    A1: Check Raw CBOE Data Quality
    
    Verify raw prices are reasonable, check bid-ask spreads.
    """
    print("\n" + "="*70)
    print("PHASE A1: Raw Data Quality Check")
    print("="*70)
    
    df = pd.read_parquet(parquet_path)
    df = df[df['quote_date'] == quote_date].copy()
    
    if df.empty:
        raise ValueError(f"No data for quote_date {quote_date}")
    
    # Check raw mid prices
    print("\nRaw mid prices:")
    print(f"  Min: ${df['mid'].min():.4f}")
    print(f"  Max: ${df['mid'].max():.4f}")
    print(f"  Mean: ${df['mid'].mean():.4f}")
    print(f"  % zeros: {(df['mid'] == 0).sum() / len(df) * 100:.2f}%")
    print(f"  % negative: {(df['mid'] < 0).sum() / len(df) * 100:.2f}%")
    print(f"  % NaN: {df['mid'].isna().sum() / len(df) * 100:.2f}%")
    
    # Check bid-ask spreads
    df['spread'] = df['ask'] - df['bid']
    print(f"\nBid-ask spreads:")
    print(f"  Mean: ${df['spread'].mean():.4f}")
    print(f"  Median: ${df['spread'].median():.4f}")
    print(f"  % negative: {(df['spread'] < 0).sum() / len(df) * 100:.2f}%")
    print(f"  % zero: {(df['spread'] == 0).sum() / len(df) * 100:.2f}%")
    
    # Check repaired prices
    print(f"\nRepaired prices:")
    print(f"  Min: ${df['price_repaired'].min():.4f}")
    print(f"  Max: ${df['price_repaired'].max():.4f}")
    print(f"  Mean: ${df['price_repaired'].mean():.4f}")
    print(f"  % zeros: {(df['price_repaired'] == 0).sum() / len(df) * 100:.2f}%")
    print(f"  % negative: {(df['price_repaired'] < 0).sum() / len(df) * 100:.2f}%")
    print(f"  % NaN: {df['price_repaired'].isna().sum() / len(df) * 100:.2f}%")
    
    results = {
        'pct_zero_mid': (df['mid'] == 0).sum() / len(df) * 100,
        'pct_negative_mid': (df['mid'] < 0).sum() / len(df) * 100,
        'pct_negative_spread': (df['spread'] < 0).sum() / len(df) * 100,
        'pct_zero_repaired': (df['price_repaired'] == 0).sum() / len(df) * 100,
        'pct_negative_repaired': (df['price_repaired'] < 0).sum() / len(df) * 100,
    }
    
    # Flag issues
    if results['pct_zero_mid'] > 1 or results['pct_negative_mid'] > 1:
        print("\n⚠️  RED FLAG: Raw data quality issues detected!")
    else:
        print("\n✓ Raw data quality looks good")
    
    return results


def phase_a2_compare_repair_adjustments(parquet_path: Path, quote_date: str) -> Dict:
    """
    A2: Compare Repair Adjustments
    
    Check magnitude of repair adjustments.
    """
    print("\n" + "="*70)
    print("PHASE A2: Repair Adjustment Analysis")
    print("="*70)
    
    df = pd.read_parquet(parquet_path)
    df = df[df['quote_date'] == quote_date].copy()
    
    df['adjustment'] = df['price_repaired'] - df['mid']
    df['adjustment_pct'] = df['adjustment'] / df['mid'].abs() * 100
    df.loc[df['mid'] == 0, 'adjustment_pct'] = np.nan
    
    print("\nRepair adjustments:")
    print(f"  Mean: ${df['adjustment'].mean():.4f}")
    print(f"  Median: ${df['adjustment'].median():.4f}")
    print(f"  Std: ${df['adjustment'].std():.4f}")
    print(f"  Min: ${df['adjustment'].min():.4f}")
    print(f"  Max: ${df['adjustment'].max():.4f}")
    
    print(f"\nAdjustment percentages:")
    print(f"  Mean: {df['adjustment_pct'].mean():.2f}%")
    print(f"  Median: {df['adjustment_pct'].median():.2f}%")
    print(f"  % changed >5%: {(df['adjustment_pct'].abs() > 5).sum() / len(df) * 100:.1f}%")
    print(f"  % changed >10%: {(df['adjustment_pct'].abs() > 10).sum() / len(df) * 100:.1f}%")
    print(f"  % changed >50%: {(df['adjustment_pct'].abs() > 50).sum() / len(df) * 100:.1f}%")
    
    results = {
        'mean_adjustment': df['adjustment'].mean(),
        'median_adjustment': df['adjustment'].median(),
        'pct_changed_gt_5pct': (df['adjustment_pct'].abs() > 5).sum() / len(df) * 100,
        'pct_changed_gt_10pct': (df['adjustment_pct'].abs() > 10).sum() / len(df) * 100,
        'pct_changed_gt_50pct': (df['adjustment_pct'].abs() > 50).sum() / len(df) * 100,
    }
    
    # Flag issues
    if results['pct_changed_gt_10pct'] > 20:
        print("\n⚠️  RED FLAG: >20% of options changed by >10% - repair may be too aggressive!")
    else:
        print("\n✓ Repair adjustments appear reasonable")
    
    return results


def phase_a3_verify_iv_consistency() -> Dict:
    """
    A3: Verify IV Computation Consistency
    
    Compare IV from Phase 3 vs Phase 5 on same inputs.
    """
    print("\n" + "="*70)
    print("PHASE A3: IV Computation Consistency Check")
    print("="*70)
    
    # Test cases
    test_cases = [
        (25.0, 665.0, 665.0, 0.25, 0, 0, True),   # ATM call
        (15.0, 665.0, 700.0, 0.25, 0, 0, True),  # OTM call
        (35.0, 665.0, 630.0, 0.25, 0, 0, True),  # ITM call
        (25.0, 665.0, 665.0, 0.5, 0, 0, True),   # Longer maturity
    ]
    
    differences = []
    
    print("\nTesting IV consistency (Phase 3 vs Phase 5):")
    print(f"{'Price':>8} {'Spot':>8} {'Strike':>8} {'TTM':>6} {'Phase3':>10} {'Phase5':>10} {'Diff':>10}")
    print("-" * 70)
    
    for price, spot, strike, ttm, r, q, is_call in test_cases:
        iv_p3 = iv_phase3(price, spot, strike, ttm, r, q, is_call)
        iv_p5 = iv_phase5(price, spot, strike, ttm, r, q, is_call)
        
        if iv_p3 is not None and iv_p5 is not None:
            diff = abs(iv_p3 - iv_p5) * 100
            differences.append(diff)
            print(f"{price:>8.2f} {spot:>8.1f} {strike:>8.1f} {ttm:>6.2f} "
                  f"{iv_p3*100:>9.2f}% {iv_p5*100:>9.2f}% {diff:>9.4f}%")
        else:
            print(f"{price:>8.2f} {spot:>8.1f} {strike:>8.1f} {ttm:>6.2f} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    if differences:
        max_diff = max(differences)
        mean_diff = np.mean(differences)
        
        print(f"\nMax difference: {max_diff:.4f}%")
        print(f"Mean difference: {mean_diff:.4f}%")
        
        results = {
            'max_iv_difference_pct': max_diff,
            'mean_iv_difference_pct': mean_diff,
        }
        
        if max_diff > 0.1:
            print("\n⚠️  RED FLAG: IV computation difference > 0.1% - functions may be inconsistent!")
        else:
            print("\n✓ IV computations are consistent")
    else:
        results = {'max_iv_difference_pct': None, 'mean_iv_difference_pct': None}
        print("\n⚠️  Could not compute differences (some IVs returned None)")
    
    return results


def phase_b1_pick_representative_option(parquet_path: Path, quote_date: str) -> pd.Series:
    """
    B1: Pick Representative Option
    
    Select ATM option with ~3 months maturity.
    """
    print("\n" + "="*70)
    print("PHASE B1: Selecting Representative Option")
    print("="*70)
    
    df = pd.read_parquet(parquet_path)
    df_day = df[df['quote_date'] == quote_date].copy()
    
    if df_day.empty:
        raise ValueError(f"No data for quote_date {quote_date}")
    
    spot = df_day['spot'].iloc[0]
    df_day['moneyness'] = df_day['strike'] / spot
    
    # Find ATM option with ~3 months maturity
    atm = df_day[abs(df_day['moneyness'] - 1.0) < 0.05].copy()
    atm = atm[(atm['ttm_years'] > 0.2) & (atm['ttm_years'] < 0.3)]
    
    if atm.empty:
        # Fallback: just ATM
        atm = df_day[abs(df_day['moneyness'] - 1.0) < 0.05].copy()
        if atm.empty:
            # Fallback: closest to ATM
            df_day['moneyness_dist'] = abs(df_day['moneyness'] - 1.0)
            atm = df_day.nsmallest(1, 'moneyness_dist')
    
    test_option = atm.iloc[0]
    
    print("\nTest option selected:")
    print(f"  Strike: ${test_option['strike']:.2f}")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Moneyness: {test_option['moneyness']:.4f}")
    print(f"  TTM: {test_option['ttm_years']:.3f} years")
    print(f"  Raw mid: ${test_option['mid']:.4f}")
    print(f"  Repaired: ${test_option['price_repaired']:.4f}")
    print(f"  Adjustment: ${test_option['price_repaired'] - test_option['mid']:.4f}")
    print(f"  Bid: ${test_option['bid']:.4f}, Ask: ${test_option['ask']:.4f}")
    
    return test_option


def phase_b2_compute_ivs_from_both_prices(test_option: pd.Series) -> Dict:
    """
    B2: Compute IVs from Both Prices
    
    Calculate IV from raw mid vs repaired price.
    """
    print("\n" + "="*70)
    print("PHASE B2: IV from Raw vs Repaired Prices")
    print("="*70)
    
    spot = test_option['spot']
    strike = test_option['strike']
    ttm = test_option['ttm_years']
    r, q = 0.0, 0.0
    
    raw_iv = iv_phase5(test_option['mid'], spot, strike, ttm, r, q, True)
    rep_iv = iv_phase5(test_option['price_repaired'], spot, strike, ttm, r, q, True)
    
    print(f"\nRaw price IV: {raw_iv*100:.2f}%" if raw_iv else "Raw price IV: N/A")
    print(f"Repaired price IV: {rep_iv*100:.2f}%" if rep_iv else "Repaired price IV: N/A")
    
    if raw_iv and rep_iv:
        iv_change = (rep_iv - raw_iv) * 100
        print(f"IV change: {iv_change:+.2f}%")
        
        results = {
            'raw_iv': raw_iv,
            'repaired_iv': rep_iv,
            'iv_change_pct': iv_change,
        }
    else:
        results = {
            'raw_iv': raw_iv,
            'repaired_iv': rep_iv,
            'iv_change_pct': None,
        }
        print("\n⚠️  Could not compute IVs (returned None)")
    
    return results


def phase_b3_heston_fit_comparison(
    test_option: pd.Series,
    parquet_path: Path,
    quote_date: str,
    max_options: int = 150,
) -> Dict:
    """
    B3: See How Heston Fits This Option
    
    Compare Heston model IV to raw and repaired market IVs.
    """
    print("\n" + "="*70)
    print("PHASE B3: Heston Model Fit Comparison")
    print("="*70)
    
    spot = test_option['spot']
    strike = test_option['strike']
    ttm = test_option['ttm_years']
    r, q = 0.0, 0.0
    
    # Get calibrated params for this day (raw)
    print("\nCalibrating Heston on RAW prices...")
    raw_data, _ = prepare_calibration_data(parquet_path, quote_date, "mid", r, q, max_options=max_options)
    raw_result = calibrate_heston(raw_data, spot, r, q, n_starts=3, verbose=False)
    
    print(f"  RMSE: {raw_result.rmse*100:.4f}%")
    print(f"  Params: v0={raw_result.v0:.4f}, kappa={raw_result.kappa:.2f}, "
          f"theta={raw_result.theta:.4f}, xi={raw_result.xi:.2f}, rho={raw_result.rho:.2f}")
    
    # Get calibrated params for this day (repaired)
    print("\nCalibrating Heston on REPAIRED prices...")
    rep_data, _ = prepare_calibration_data(parquet_path, quote_date, "price_repaired", r, q, max_options=max_options)
    rep_result = calibrate_heston(rep_data, spot, r, q, n_starts=3, verbose=False)
    
    print(f"  RMSE: {rep_result.rmse*100:.4f}%")
    print(f"  Params: v0={rep_result.v0:.4f}, kappa={rep_result.kappa:.2f}, "
          f"theta={rep_result.theta:.4f}, xi={rep_result.xi:.2f}, rho={rep_result.rho:.2f}")
    
    # Compute Heston IVs for test option
    raw_heston_iv = heston_implied_vol(
        spot, strike, ttm, r, q,
        raw_result.v0, raw_result.kappa, raw_result.theta, raw_result.xi, raw_result.rho, True
    )
    rep_heston_iv = heston_implied_vol(
        spot, strike, ttm, r, q,
        rep_result.v0, rep_result.kappa, rep_result.theta, rep_result.xi, rep_result.rho, True
    )
    
    # Market IVs
    raw_market_iv = iv_phase5(test_option['mid'], spot, strike, ttm, r, q, True)
    rep_market_iv = iv_phase5(test_option['price_repaired'], spot, strike, ttm, r, q, True)
    
    print(f"\nTest option (Strike=${strike:.2f}, TTM={ttm:.3f}):")
    print(f"  Raw market IV: {raw_market_iv*100:.2f}%" if raw_market_iv else "  Raw market IV: N/A")
    print(f"  Raw Heston IV: {raw_heston_iv*100:.2f}%" if raw_heston_iv else "  Raw Heston IV: N/A")
    if raw_market_iv and raw_heston_iv:
        print(f"  Raw error: {abs(raw_heston_iv - raw_market_iv)*100:.2f}%")
    
    print(f"  Repaired market IV: {rep_market_iv*100:.2f}%" if rep_market_iv else "  Repaired market IV: N/A")
    print(f"  Repaired Heston IV: {rep_heston_iv*100:.2f}%" if rep_heston_iv else "  Repaired Heston IV: N/A")
    if rep_market_iv and rep_heston_iv:
        print(f"  Repaired error: {abs(rep_heston_iv - rep_market_iv)*100:.2f}%")
    
    results = {
        'raw_rmse': raw_result.rmse,
        'repaired_rmse': rep_result.rmse,
        'raw_market_iv': raw_market_iv,
        'repaired_market_iv': rep_market_iv,
        'raw_heston_iv': raw_heston_iv,
        'repaired_heston_iv': rep_heston_iv,
    }
    
    if raw_market_iv and raw_heston_iv and rep_market_iv and rep_heston_iv:
        raw_error = abs(raw_heston_iv - raw_market_iv)
        rep_error = abs(rep_heston_iv - rep_market_iv)
        results['raw_error'] = raw_error
        results['repaired_error'] = rep_error
        
        if rep_error > raw_error:
            print("\n⚠️  RED FLAG: Repaired error > Raw error for this option!")
        else:
            print("\n✓ Repaired error < Raw error for this option")
    
    return results


def phase_c1_check_sampling_consistency(
    parquet_path: Path,
    quote_date: str,
    max_options: int = 150,
) -> Dict:
    """
    C1: Check Sampling Consistency
    
    Verify raw and repaired get the same options selected.
    """
    print("\n" + "="*70)
    print("PHASE C1: Sampling Consistency Check")
    print("="*70)
    
    raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", max_options=max_options)
    rep_data, spot_rep = prepare_calibration_data(parquet_path, quote_date, "price_repaired", max_options=max_options)
    
    if abs(spot - spot_rep) > 0.01:
        print(f"⚠️  Warning: Spot prices differ: {spot} vs {spot_rep}")
    
    raw_keys = set((round(d[0], 2), round(d[1], 3)) for d in raw_data)  # (strike, ttm) pairs
    rep_keys = set((round(d[0], 2), round(d[1], 3)) for d in rep_data)
    
    overlap = raw_keys & rep_keys
    raw_only = raw_keys - rep_keys
    rep_only = rep_keys - raw_keys
    
    print(f"\nRaw options: {len(raw_keys)}")
    print(f"Repaired options: {len(rep_keys)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/max(len(raw_keys), len(rep_keys))*100:.1f}%)")
    print(f"Raw only: {len(raw_only)}")
    print(f"Repaired only: {len(rep_only)}")
    
    if len(raw_only) > 0:
        print(f"\nSample raw-only options:")
        for i, (k, ttm) in enumerate(list(raw_only)[:5]):
            print(f"  Strike=${k:.2f}, TTM={ttm:.3f}")
    
    if len(rep_only) > 0:
        print(f"\nSample repaired-only options:")
        for i, (k, ttm) in enumerate(list(rep_only)[:5]):
            print(f"  Strike=${k:.2f}, TTM={ttm:.3f}")
    
    overlap_pct = len(overlap) / max(len(raw_keys), len(rep_keys)) * 100 if max(len(raw_keys), len(rep_keys)) > 0 else 0
    
    results = {
        'n_raw_options': len(raw_keys),
        'n_repaired_options': len(rep_keys),
        'n_overlap': len(overlap),
        'overlap_pct': overlap_pct,
        'n_raw_only': len(raw_only),
        'n_repaired_only': len(rep_only),
    }
    
    if overlap_pct < 80:
        print("\n⚠️  RED FLAG: Overlap < 80% - sampling is inconsistent!")
    else:
        print("\n✓ Sampling is consistent")
    
    return results


def phase_c2_check_iv_distributions(
    parquet_path: Path,
    quote_date: str,
    max_options: int = 150,
) -> Dict:
    """
    C2: Check IV Distributions Match
    
    Compare IV distributions for raw vs repaired (should be highly correlated if same options).
    """
    print("\n" + "="*70)
    print("PHASE C2: IV Distribution Comparison")
    print("="*70)
    
    raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", max_options=max_options)
    rep_data, _ = prepare_calibration_data(parquet_path, quote_date, "price_repaired", max_options=max_options)
    
    raw_ivs = np.array([d[2] for d in raw_data])
    rep_ivs = np.array([d[2] for d in rep_data])
    
    print(f"\nRaw IVs:")
    print(f"  Mean: {np.mean(raw_ivs)*100:.1f}%")
    print(f"  Std: {np.std(raw_ivs)*100:.1f}%")
    print(f"  Min: {np.min(raw_ivs)*100:.1f}%")
    print(f"  Max: {np.max(raw_ivs)*100:.1f}%")
    
    print(f"\nRepaired IVs:")
    print(f"  Mean: {np.mean(rep_ivs)*100:.1f}%")
    print(f"  Std: {np.std(rep_ivs)*100:.1f}%")
    print(f"  Min: {np.min(rep_ivs)*100:.1f}%")
    print(f"  Max: {np.max(rep_ivs)*100:.1f}%")
    
    # Match up same options for correlation
    raw_dict = {(round(d[0], 2), round(d[1], 3)): d[2] for d in raw_data}
    rep_dict = {(round(d[0], 2), round(d[1], 3)): d[2] for d in rep_data}
    
    common_keys = set(raw_dict.keys()) & set(rep_dict.keys())
    
    if len(common_keys) > 0:
        common_raw_ivs = [raw_dict[k] for k in common_keys]
        common_rep_ivs = [rep_dict[k] for k in common_keys]
        
        corr = np.corrcoef(common_raw_ivs, common_rep_ivs)[0, 1]
        
        print(f"\nIV correlation for same options ({len(common_keys)} pairs): {corr:.4f}")
        
        results = {
            'raw_iv_mean': np.mean(raw_ivs),
            'raw_iv_std': np.std(raw_ivs),
            'repaired_iv_mean': np.mean(rep_ivs),
            'repaired_iv_std': np.std(rep_ivs),
            'n_common_options': len(common_keys),
            'iv_correlation': corr,
        }
        
        if corr < 0.95:
            print("\n⚠️  RED FLAG: IV correlation < 0.95 - repair is fundamentally changing IV surface!")
        else:
            print("\n✓ IVs are highly correlated")
    else:
        results = {
            'raw_iv_mean': np.mean(raw_ivs),
            'raw_iv_std': np.std(raw_ivs),
            'repaired_iv_mean': np.mean(rep_ivs),
            'repaired_iv_std': np.std(rep_ivs),
            'n_common_options': 0,
            'iv_correlation': None,
        }
        print("\n⚠️  No common options found for correlation")
    
    return results


def phase_d1_load_repair_summary(repair_summary_path: Path) -> pd.DataFrame:
    """
    D1: Load Repair Summary
    
    Check repair quality metrics.
    """
    print("\n" + "="*70)
    print("PHASE D1: Repair Summary Analysis")
    print("="*70)
    
    if not repair_summary_path.exists():
        print(f"⚠️  Repair summary not found: {repair_summary_path}")
        return pd.DataFrame()
    
    repair_summary = pd.read_csv(repair_summary_path)
    
    print("\nRepair summary statistics:")
    print(repair_summary.describe())
    
    print(f"\nAll days status: {repair_summary['status'].unique()}")
    print(f"Mean adjustment: ${repair_summary['mean_adjustment'].mean():.4f}")
    print(f"Max adjustment: ${repair_summary['max_adjustment'].max():.4f}")
    print(f"% within bid-ask: {repair_summary['pct_within_bidask'].mean():.1f}%")
    
    return repair_summary


def phase_d2_spot_check_arbitrage_violations(
    parquet_path: Path,
    quote_date: str,
    test_strike: float = None,
) -> Dict:
    """
    D2: Spot Check Arbitrage Violations
    
    Check if repair actually removed calendar arbitrage violations.
    """
    print("\n" + "="*70)
    print("PHASE D2: Arbitrage Violation Check")
    print("="*70)
    
    df = pd.read_parquet(parquet_path)
    df_day = df[df['quote_date'] == quote_date].copy()
    
    if df_day.empty:
        raise ValueError(f"No data for quote_date {quote_date}")
    
    spot = df_day['spot'].iloc[0]
    
    # Use provided strike or pick ATM
    if test_strike is None:
        test_strike = spot
    
    # Get options at this strike
    at_strike = df_day[abs(df_day['strike'] - test_strike) < 0.5].copy()
    at_strike = at_strike.sort_values('ttm_years')
    
    print(f"\nChecking calendar arbitrage at strike ${test_strike:.2f}:")
    print(f"Found {len(at_strike)} options at this strike")
    
    # Check raw prices
    raw_violations = []
    print("\nCalendar spread check (raw prices):")
    for i in range(len(at_strike) - 1):
        short_price = at_strike.iloc[i]['mid']
        long_price = at_strike.iloc[i+1]['mid']
        short_ttm = at_strike.iloc[i]['ttm_years']
        long_ttm = at_strike.iloc[i+1]['ttm_years']
        
        if short_price > long_price:
            violation = {
                'short_ttm': short_ttm,
                'long_ttm': long_ttm,
                'short_price': short_price,
                'long_price': long_price,
            }
            raw_violations.append(violation)
            print(f"  VIOLATION: T={short_ttm:.3f} (${short_price:.2f}) > T={long_ttm:.3f} (${long_price:.2f})")
    
    if not raw_violations:
        print("  ✓ No violations found")
    
    # Check repaired prices
    rep_violations = []
    print("\nCalendar spread check (repaired prices):")
    for i in range(len(at_strike) - 1):
        short_price = at_strike.iloc[i]['price_repaired']
        long_price = at_strike.iloc[i+1]['price_repaired']
        short_ttm = at_strike.iloc[i]['ttm_years']
        long_ttm = at_strike.iloc[i+1]['ttm_years']
        
        if short_price > long_price:
            violation = {
                'short_ttm': short_ttm,
                'long_ttm': long_ttm,
                'short_price': short_price,
                'long_price': long_price,
            }
            rep_violations.append(violation)
            print(f"  VIOLATION: T={short_ttm:.3f} (${short_price:.2f}) > T={long_ttm:.3f} (${long_price:.2f})")
    
    if not rep_violations:
        print("  ✓ No violations found")
    
    results = {
        'n_raw_violations': len(raw_violations),
        'n_repaired_violations': len(rep_violations),
        'violations_removed': len(raw_violations) - len(rep_violations),
    }
    
    if len(rep_violations) > len(raw_violations):
        print("\n⚠️  RED FLAG: Repair created MORE violations!")
    elif len(rep_violations) < len(raw_violations):
        print(f"\n✓ Repair removed {len(raw_violations) - len(rep_violations)} violations")
    else:
        print(f"\n→ Same number of violations ({len(raw_violations)})")
    
    return results


def phase_e1_measure_smile_characteristics(
    parquet_path: Path,
    quote_date: str,
    max_options: int = 150,
) -> Dict:
    """
    E1: Measure Smile Characteristics
    
    Compare smile slope (OTM - ATM) for raw vs repaired.
    """
    print("\n" + "="*70)
    print("PHASE E1: Smile Characteristics Analysis")
    print("="*70)
    
    raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", max_options=max_options)
    rep_data, _ = prepare_calibration_data(parquet_path, quote_date, "price_repaired", max_options=max_options)
    
    def compute_smile_slope(data, spot_price):
        # Separate OTM and ATM options
        atm_ivs = [iv for k, ttm, iv, w in data if 0.95 < k/spot_price < 1.05]
        otm_ivs = [iv for k, ttm, iv, w in data if k/spot_price < 0.90]
        
        if atm_ivs and otm_ivs:
            return np.mean(otm_ivs) - np.mean(atm_ivs)
        return None
    
    raw_slope = compute_smile_slope(raw_data, spot)
    rep_slope = compute_smile_slope(rep_data, spot)
    
    print(f"\nSmile slope (OTM - ATM):")
    print(f"  Raw: {raw_slope*100:.1f}%" if raw_slope else "  Raw: N/A")
    print(f"  Repaired: {rep_slope*100:.1f}%" if rep_slope else "  Repaired: N/A")
    
    if raw_slope and rep_slope:
        slope_change = rep_slope - raw_slope
        print(f"  Change: {slope_change*100:+.1f}%")
        
        results = {
            'raw_smile_slope': raw_slope,
            'repaired_smile_slope': rep_slope,
            'slope_change': slope_change,
        }
        
        if rep_slope > raw_slope:
            print("\n⚠️  RED FLAG: Repaired slope is STEEPER - repair may be making surface harder to fit!")
        else:
            print("\n✓ Repaired slope is flatter or same")
    else:
        results = {
            'raw_smile_slope': raw_slope,
            'repaired_smile_slope': rep_slope,
            'slope_change': None,
        }
    
    return results


def phase_e2_check_term_structure(
    parquet_path: Path,
    quote_date: str,
    max_options: int = 150,
) -> Dict:
    """
    E2: Check Term Structure Changes
    
    See if repair changes ATM term structure (ATM IV vs TTM).
    """
    print("\n" + "="*70)
    print("PHASE E2: Term Structure Analysis")
    print("="*70)
    
    raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", max_options=max_options)
    rep_data, _ = prepare_calibration_data(parquet_path, quote_date, "price_repaired", max_options=max_options)
    
    def get_atm_term_structure(data, spot_price):
        # Get ATM IVs for each maturity
        atm_options = [(ttm, iv) for k, ttm, iv, w in data if 0.98 < k/spot_price < 1.02]
        return sorted(atm_options)
    
    raw_ts = get_atm_term_structure(raw_data, spot)
    rep_ts = get_atm_term_structure(rep_data, spot)
    
    print("\nATM term structure:")
    print(f"{'TTM':>8} {'Raw IV':>10} {'Rep IV':>10} {'Change':>10}")
    print("-" * 40)
    
    changes = []
    for (ttm_r, iv_r), (ttm_p, iv_p) in zip(raw_ts[:10], rep_ts[:10]):
        if abs(ttm_r - ttm_p) < 0.01:
            change = (iv_p - iv_r) * 100
            changes.append(change)
            print(f"{ttm_r:>8.3f} {iv_r*100:>9.2f}% {iv_p*100:>9.2f}% {change:>+9.2f}%")
    
    results = {
        'n_atm_options': len(raw_ts),
        'mean_term_structure_change': np.mean(changes) if changes else None,
    }
    
    if changes:
        print(f"\nMean term structure change: {np.mean(changes):+.2f}%")
    
    return results


def run_full_diagnostic(
    parquet_path: Path = DEFAULT_PARQUET,
    repair_summary_path: Path = DEFAULT_REPAIR_SUMMARY,
    quote_date: str = "2025-10-15",
    output_dir: Path = DEFAULT_OUTPUT,
    max_options: int = 150,
) -> Dict:
    """
    Run complete diagnostic pipeline.
    
    Returns dictionary with all results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("FULL PIPELINE DIAGNOSTIC INVESTIGATION")
    print("="*70)
    print(f"Quote Date: {quote_date}")
    print(f"Input: {parquet_path}")
    print(f"Output: {output_dir}")
    
    all_results = {}
    
    # Phase A: Data Integrity
    print("\n\n" + "#"*70)
    print("# PHASE A: DATA INTEGRITY CHECKS")
    print("#"*70)
    
    all_results['A1'] = phase_a1_check_raw_data_quality(parquet_path, quote_date)
    all_results['A2'] = phase_a2_compare_repair_adjustments(parquet_path, quote_date)
    all_results['A3'] = phase_a3_verify_iv_consistency()
    
    # Phase B: Trace Single Option
    print("\n\n" + "#"*70)
    print("# PHASE B: TRACE SINGLE OPTION")
    print("#"*70)
    
    test_option = phase_b1_pick_representative_option(parquet_path, quote_date)
    all_results['B1'] = test_option.to_dict()
    all_results['B2'] = phase_b2_compute_ivs_from_both_prices(test_option)
    all_results['B3'] = phase_b3_heston_fit_comparison(test_option, parquet_path, quote_date, max_options)
    
    # Phase C: Compare Calibration Samples
    print("\n\n" + "#"*70)
    print("# PHASE C: CALIBRATION SAMPLE COMPARISON")
    print("#"*70)
    
    all_results['C1'] = phase_c1_check_sampling_consistency(parquet_path, quote_date, max_options)
    all_results['C2'] = phase_c2_check_iv_distributions(parquet_path, quote_date, max_options)
    
    # Phase D: Repair Quality
    print("\n\n" + "#"*70)
    print("# PHASE D: REPAIR QUALITY METRICS")
    print("#"*70)
    
    repair_summary = phase_d1_load_repair_summary(repair_summary_path)
    all_results['D1'] = repair_summary.to_dict('records') if not repair_summary.empty else {}
    all_results['D2'] = phase_d2_spot_check_arbitrage_violations(parquet_path, quote_date)
    
    # Phase E: Model Mismatch
    print("\n\n" + "#"*70)
    print("# PHASE E: MODEL MISMATCH HYPOTHESIS")
    print("#"*70)
    
    all_results['E1'] = phase_e1_measure_smile_characteristics(parquet_path, quote_date, max_options)
    all_results['E2'] = phase_e2_check_term_structure(parquet_path, quote_date, max_options)
    
    # Save results
    import json
    results_file = output_dir / "diagnostic_results.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Print summary of key findings
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)
    
    if all_results.get('A3', {}).get('max_iv_difference_pct'):
        if all_results['A3']['max_iv_difference_pct'] > 0.1:
            print("⚠️  A3: IV computation inconsistency detected")
    
    if all_results.get('A2', {}).get('pct_changed_gt_10pct', 0) > 20:
        print("⚠️  A2: Repair adjustments may be too aggressive")
    
    if all_results.get('C1', {}).get('overlap_pct', 100) < 80:
        print("⚠️  C1: Sampling inconsistency detected")
    
    if all_results.get('C2', {}).get('iv_correlation'):
        if all_results['C2']['iv_correlation'] < 0.95:
            print("⚠️  C2: IV correlation low - repair changing surface structure")
    
    if all_results.get('E1', {}).get('slope_change'):
        if all_results['E1']['slope_change'] > 0:
            print("⚠️  E1: Repaired smile is steeper - may be harder to fit")
    
    return all_results


def main(argv=None):
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Full Pipeline Diagnostic Investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--repair-summary",
        type=Path,
        default=DEFAULT_REPAIR_SUMMARY,
        help="Path to repair_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for diagnostic results",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=150,
        help="Maximum options for calibration (default: 150)",
    )
    
    args = parser.parse_args(argv)
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    run_full_diagnostic(
        parquet_path=args.input,
        repair_summary_path=args.repair_summary,
        quote_date=args.date,
        output_dir=args.output,
        max_options=args.max_options,
    )


if __name__ == "__main__":
    main()

