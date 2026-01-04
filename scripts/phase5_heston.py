#!/usr/bin/env python3
"""
Phase 5: Heston Model Calibration

CLI entrypoint for calibrating the Heston stochastic volatility model on
both raw and repaired option surfaces.

Usage:
    # Single day calibration
    python scripts/phase5_heston.py --mode single --date 2025-10-15
    
    # All days calibration
    python scripts/phase5_heston.py --mode all
    
    # Both (single day detailed + all days stability)
    python scripts/phase5_heston.py --mode both
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.phase5.calibrate import calibrate_heston, CalibrationResult
from pipelines.phase5.prepare_data import (
    prepare_calibration_data,
    get_available_quote_dates,
    summarize_market_data,
)


# Default paths
DEFAULT_INPUT = Path("reports/phase4/1545/repaired_options.parquet")
DEFAULT_OUTPUT = Path("reports/phase5")


def calibrate_single_day(
    parquet_path: Path,
    quote_date: str,
    output_dir: Path,
    r: float = 0.0,
    q: float = 0.0,
    n_starts: int = 5,
    max_options: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run calibration on a single day, comparing raw vs repaired.
    
    Returns dict with both calibration results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 5: HESTON CALIBRATION - {quote_date}")
        print(f"{'='*70}")
    
    results = {}
    
    # Calibrate on RAW (mid) prices
    if verbose:
        print(f"\n--- Calibrating on RAW prices (mid) ---")
    
    raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", r, q, max_options=max_options)
    raw_summary = summarize_market_data(raw_data)
    
    if verbose:
        print(f"  Options: {raw_summary['n_options']}")
        print(f"  Spot: ${spot:.2f}")
        print(f"  IV range: [{raw_summary['iv_range'][0]*100:.1f}%, {raw_summary['iv_range'][1]*100:.1f}%]")
        print(f"  Calibrating...")
    
    raw_result = calibrate_heston(raw_data, spot, r, q, n_starts=n_starts, verbose=False)
    results["raw"] = raw_result
    
    if verbose:
        print(f"  RMSE: {raw_result.rmse*100:.4f}%")
        print(f"  Feller: {raw_result.feller_satisfied} (ratio: {raw_result.feller_ratio:.2f})")
        print(f"  Params: v0={raw_result.v0:.4f}, kappa={raw_result.kappa:.2f}, "
              f"theta={raw_result.theta:.4f}, xi={raw_result.xi:.2f}, rho={raw_result.rho:.2f}")
    
    # Calibrate on REPAIRED prices
    if verbose:
        print(f"\n--- Calibrating on REPAIRED prices ---")
    
    rep_data, spot = prepare_calibration_data(parquet_path, quote_date, "price_repaired", r, q, max_options=max_options)
    rep_summary = summarize_market_data(rep_data)
    
    if verbose:
        print(f"  Options: {rep_summary['n_options']}")
        print(f"  Calibrating...")
    
    rep_result = calibrate_heston(rep_data, spot, r, q, n_starts=n_starts, verbose=False)
    results["repaired"] = rep_result
    
    if verbose:
        print(f"  RMSE: {rep_result.rmse*100:.4f}%")
        print(f"  Feller: {rep_result.feller_satisfied} (ratio: {rep_result.feller_ratio:.2f})")
        print(f"  Params: v0={rep_result.v0:.4f}, kappa={rep_result.kappa:.2f}, "
              f"theta={rep_result.theta:.4f}, xi={rep_result.xi:.2f}, rho={rep_result.rho:.2f}")
    
    # Summary comparison
    if verbose:
        print(f"\n--- COMPARISON ---")
        rmse_improvement = (raw_result.rmse - rep_result.rmse) / raw_result.rmse * 100
        print(f"  RMSE improvement: {rmse_improvement:.1f}%")
        print(f"  Raw Feller: {raw_result.feller_satisfied} -> Repaired Feller: {rep_result.feller_satisfied}")
    
    # Save single-day comparison CSV
    comparison_df = pd.DataFrame([
        {
            "quote_date": quote_date,
            "data_type": "raw",
            "n_options": raw_summary["n_options"],
            "spot": spot,
            "v0": raw_result.v0,
            "kappa": raw_result.kappa,
            "theta": raw_result.theta,
            "xi": raw_result.xi,
            "rho": raw_result.rho,
            "rmse": raw_result.rmse,
            "mae": raw_result.mae,
            "max_error": raw_result.max_error,
            "feller_satisfied": raw_result.feller_satisfied,
            "feller_ratio": raw_result.feller_ratio,
            "success": raw_result.success,
        },
        {
            "quote_date": quote_date,
            "data_type": "repaired",
            "n_options": rep_summary["n_options"],
            "spot": spot,
            "v0": rep_result.v0,
            "kappa": rep_result.kappa,
            "theta": rep_result.theta,
            "xi": rep_result.xi,
            "rho": rep_result.rho,
            "rmse": rep_result.rmse,
            "mae": rep_result.mae,
            "max_error": rep_result.max_error,
            "feller_satisfied": rep_result.feller_satisfied,
            "feller_ratio": rep_result.feller_ratio,
            "success": rep_result.success,
        },
    ])
    
    comparison_df.to_csv(output_dir / "single_day_comparison.csv", index=False)
    
    if verbose:
        print(f"\nSaved: {output_dir / 'single_day_comparison.csv'}")
    
    return results


def calibrate_all_days(
    parquet_path: Path,
    output_dir: Path,
    r: float = 0.0,
    q: float = 0.0,
    n_starts: int = 3,
    max_options: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run calibration on all available quote dates.
    
    Returns DataFrame with results for each day.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dates = get_available_quote_dates(parquet_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 5: MULTI-DAY CALIBRATION ({len(dates)} days)")
        print(f"{'='*70}")
    
    all_results = []
    
    for i, quote_date in enumerate(dates):
        if verbose:
            print(f"\n[{i+1}/{len(dates)}] {quote_date}...", flush=True)
        
        try:
            # Raw calibration
            if verbose:
                print(f"  RAW:", flush=True)
            raw_data, spot = prepare_calibration_data(parquet_path, quote_date, "mid", r, q, max_options=max_options)
            raw_result = calibrate_heston(raw_data, spot, r, q, n_starts=n_starts, verbose=verbose)
            
            all_results.append({
                "quote_date": quote_date,
                "data_type": "raw",
                "n_options": len(raw_data),
                "spot": spot,
                "v0": raw_result.v0,
                "kappa": raw_result.kappa,
                "theta": raw_result.theta,
                "xi": raw_result.xi,
                "rho": raw_result.rho,
                "rmse": raw_result.rmse,
                "mae": raw_result.mae,
                "feller_satisfied": raw_result.feller_satisfied,
                "feller_ratio": raw_result.feller_ratio,
                "success": raw_result.success,
            })
            
            # Repaired calibration
            if verbose:
                print(f"  REPAIRED:", flush=True)
            rep_data, spot = prepare_calibration_data(parquet_path, quote_date, "price_repaired", r, q, max_options=max_options)
            rep_result = calibrate_heston(rep_data, spot, r, q, n_starts=n_starts, verbose=verbose)
            
            all_results.append({
                "quote_date": quote_date,
                "data_type": "repaired",
                "n_options": len(rep_data),
                "spot": spot,
                "v0": rep_result.v0,
                "kappa": rep_result.kappa,
                "theta": rep_result.theta,
                "xi": rep_result.xi,
                "rho": rep_result.rho,
                "rmse": rep_result.rmse,
                "mae": rep_result.mae,
                "feller_satisfied": rep_result.feller_satisfied,
                "feller_ratio": rep_result.feller_ratio,
                "success": rep_result.success,
            })
            
            if verbose:
                improvement = (raw_result.rmse - rep_result.rmse) / raw_result.rmse * 100 if raw_result.rmse > 0 else 0
                print(f"  → Improvement: {improvement:+.1f}% | "
                      f"Feller: {raw_result.feller_satisfied} → {rep_result.feller_satisfied}", flush=True)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_days_calibration.csv", index=False)
    
    if verbose:
        print(f"\nSaved: {output_dir / 'all_days_calibration.csv'}")
    
    # Compute stability summary
    if verbose:
        print(f"\n{'='*70}")
        print("STABILITY SUMMARY")
        print(f"{'='*70}")
        
        raw_df = results_df[results_df["data_type"] == "raw"]
        rep_df = results_df[results_df["data_type"] == "repaired"]
        
        print("\nParameter Standard Deviations (lower = more stable):")
        print(f"{'Parameter':<10} {'Raw':>12} {'Repaired':>12} {'Improvement':>12}")
        print("-" * 48)
        
        for param in ["v0", "kappa", "theta", "xi", "rho"]:
            raw_std = raw_df[param].std()
            rep_std = rep_df[param].std()
            improvement = (raw_std - rep_std) / raw_std * 100 if raw_std > 0 else 0
            print(f"{param:<10} {raw_std:>12.4f} {rep_std:>12.4f} {improvement:>11.1f}%")
        
        print("\nRMSE Statistics:")
        print(f"  Raw:      mean={raw_df['rmse'].mean()*100:.3f}%, std={raw_df['rmse'].std()*100:.3f}%")
        print(f"  Repaired: mean={rep_df['rmse'].mean()*100:.3f}%, std={rep_df['rmse'].std()*100:.3f}%")
        
        print("\nFeller Condition Satisfaction:")
        print(f"  Raw:      {raw_df['feller_satisfied'].sum()}/{len(raw_df)} ({raw_df['feller_satisfied'].mean()*100:.0f}%)")
        print(f"  Repaired: {rep_df['feller_satisfied'].sum()}/{len(rep_df)} ({rep_df['feller_satisfied'].mean()*100:.0f}%)")
    
    return results_df


def generate_summary(
    results_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate summary statistics from multi-day calibration.
    """
    raw_df = results_df[results_df["data_type"] == "raw"]
    rep_df = results_df[results_df["data_type"] == "repaired"]
    
    summary_rows = []
    
    # RMSE comparison
    raw_rmse_mean = raw_df["rmse"].mean()
    rep_rmse_mean = rep_df["rmse"].mean()
    rmse_improvement = (raw_rmse_mean - rep_rmse_mean) / raw_rmse_mean * 100
    
    summary_rows.append({
        "metric": "rmse_mean",
        "raw": raw_rmse_mean,
        "repaired": rep_rmse_mean,
        "improvement_pct": rmse_improvement,
    })
    
    # Parameter stability
    for param in ["v0", "kappa", "theta", "xi", "rho"]:
        raw_std = raw_df[param].std()
        rep_std = rep_df[param].std()
        improvement = (raw_std - rep_std) / raw_std * 100 if raw_std > 0 else 0
        
        summary_rows.append({
            "metric": f"{param}_std",
            "raw": raw_std,
            "repaired": rep_std,
            "improvement_pct": improvement,
        })
        
        summary_rows.append({
            "metric": f"{param}_mean",
            "raw": raw_df[param].mean(),
            "repaired": rep_df[param].mean(),
            "improvement_pct": np.nan,
        })
    
    # Feller condition
    raw_feller_pct = raw_df["feller_satisfied"].mean() * 100
    rep_feller_pct = rep_df["feller_satisfied"].mean() * 100
    
    summary_rows.append({
        "metric": "feller_satisfaction_pct",
        "raw": raw_feller_pct,
        "repaired": rep_feller_pct,
        "improvement_pct": rep_feller_pct - raw_feller_pct,
    })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    if verbose:
        print(f"\nSaved: {output_dir / 'summary.csv'}")
    
    return summary_df


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 5: Heston Model Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "all", "both"],
        default="both",
        help="Calibration mode: single day, all days, or both",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-10-15",
        help="Quote date for single-day calibration (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to repaired_options.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Risk-free rate",
    )
    parser.add_argument(
        "--dividend",
        type=float,
        default=0.0,
        help="Dividend yield",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=5,
        help="Number of optimization starting points",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=150,
        help="Maximum number of options per day (subsample if exceeded, default: 150)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args(argv)
    
    # Check input exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    verbose = not args.quiet
    
    if args.mode in ["single", "both"]:
        calibrate_single_day(
            args.input,
            args.date,
            args.output,
            r=args.rate,
            q=args.dividend,
            n_starts=args.n_starts,
            max_options=args.max_options,
            verbose=verbose,
        )
    
    if args.mode in ["all", "both"]:
        results_df = calibrate_all_days(
            args.input,
            args.output,
            r=args.rate,
            q=args.dividend,
            n_starts=min(args.n_starts, 3),  # Fewer starts for speed
            max_options=args.max_options,
            verbose=verbose,
        )
        
        generate_summary(results_df, args.output, verbose=verbose)
    
    print(f"\n{'='*70}")
    print("PHASE 5 COMPLETE")
    print(f"{'='*70}")
    print(f"Outputs: {args.output}/")


if __name__ == "__main__":
    main()

