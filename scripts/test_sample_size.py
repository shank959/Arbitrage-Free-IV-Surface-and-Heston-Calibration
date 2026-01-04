#!/usr/bin/env python3
"""
Quick test to determine optimal sample size for Heston calibration.
Tests different sample sizes on one day to measure accuracy vs speed tradeoff.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.phase5.calibrate import calibrate_heston
from pipelines.phase5.prepare_data import prepare_calibration_data

# Test parameters
PARQUET_PATH = Path("reports/phase4/1545/repaired_options.parquet")
TEST_DATE = "2025-10-15"
SAMPLE_SIZES = [50, 100, 150, 200, 250, 300, None]  # None = all options
N_STARTS = 3

print("="*70)
print("SAMPLE SIZE OPTIMIZATION TEST")
print("="*70)
print(f"Date: {TEST_DATE}")
print(f"Starts: {N_STARTS}")
print()

results = []

for sample_size in SAMPLE_SIZES:
    print(f"\n{'='*70}")
    print(f"Testing sample_size={sample_size if sample_size else 'ALL'}")
    print(f"{'='*70}")
    
    # Prepare data
    raw_data, spot = prepare_calibration_data(
        PARQUET_PATH, TEST_DATE, "price_repaired", 
        max_options=sample_size
    )
    
    n_options = len(raw_data)
    print(f"Options: {n_options}")
    
    # Time calibration
    start = time.time()
    result = calibrate_heston(raw_data, spot, n_starts=N_STARTS, verbose=False)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.1f}s")
    print(f"RMSE: {result.rmse*100:.3f}%")
    print(f"Feller: {result.feller_satisfied} (ratio: {result.feller_ratio:.2f})")
    print(f"Params: v0={result.v0:.4f}, kappa={result.kappa:.2f}, "
          f"theta={result.theta:.4f}, xi={result.xi:.2f}, rho={result.rho:.2f}")
    
    results.append({
        "sample_size": sample_size if sample_size else n_options,
        "n_options": n_options,
        "time_sec": elapsed,
        "rmse_pct": result.rmse * 100,
        "feller": result.feller_satisfied,
        "feller_ratio": result.feller_ratio,
    })
    
    # Stop if we exceed 60 seconds (would be >45 min for all 46 calibrations)
    if elapsed > 60:
        print(f"\n⚠️  Exceeded 60s, stopping test (would take >{elapsed*46/60:.0f} min for all days)")
        break

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Size':<8} {'Options':<8} {'Time(s)':<10} {'RMSE(%)':<10} {'Feller':<8} {'Est. Total(min)':<15}")
print("-"*70)

for r in results:
    est_total = r["time_sec"] * 46 / 60  # 46 calibrations (23 days × 2)
    print(f"{r['sample_size']:<8} {r['n_options']:<8} {r['time_sec']:<10.1f} "
          f"{r['rmse_pct']:<10.3f} {str(r['feller']):<8} {est_total:<15.1f}")

print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")

# Find best tradeoff: <30 min total, good RMSE
valid = [r for r in results if r["time_sec"] * 46 / 60 < 30]
if valid:
    # Pick largest sample size that fits in 30 min
    best = max(valid, key=lambda x: x["n_options"])
    print(f"Optimal sample size: {best['sample_size']}")
    print(f"  Options per day: {best['n_options']}")
    print(f"  Time per calibration: {best['time_sec']:.1f}s")
    print(f"  Estimated total time: {best['time_sec']*46/60:.1f} minutes")
    print(f"  RMSE: {best['rmse_pct']:.3f}%")
    print(f"  Feller satisfied: {best['feller']}")
else:
    print("All sample sizes exceed 30-minute target!")
    print(f"Smallest tested: {results[0]['sample_size']} → {results[0]['time_sec']*46/60:.1f} min")

