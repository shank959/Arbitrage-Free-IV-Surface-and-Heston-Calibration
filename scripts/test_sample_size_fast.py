#!/usr/bin/env python3
"""
Quick test with reduced starts to hit 30-minute target.
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
SAMPLE_SIZES = [100, 150, 200, 250]
N_STARTS = 2  # Reduced from 3 for speed

print("="*70)
print("SAMPLE SIZE TEST (N_STARTS=2 for speed)")
print("="*70)
print(f"Date: {TEST_DATE}")
print(f"Starts: {N_STARTS}")
print()

results = []

for sample_size in SAMPLE_SIZES:
    print(f"\nTesting sample_size={sample_size}...", end=" ", flush=True)
    
    # Prepare data
    raw_data, spot = prepare_calibration_data(
        PARQUET_PATH, TEST_DATE, "price_repaired", 
        max_options=sample_size
    )
    
    n_options = len(raw_data)
    
    # Time calibration
    start = time.time()
    result = calibrate_heston(raw_data, spot, n_starts=N_STARTS, verbose=False, max_nfev=300)
    elapsed = time.time() - start
    
    print(f"{n_options} opts, {elapsed:.1f}s, RMSE={result.rmse*100:.3f}%")
    
    results.append({
        "sample_size": sample_size,
        "n_options": n_options,
        "time_sec": elapsed,
        "rmse_pct": result.rmse * 100,
        "feller": result.feller_satisfied,
    })
    
    # Stop if we exceed 40 seconds (would be >30 min for all 46 calibrations)
    if elapsed > 40:
        print(f"⚠️  Exceeded 40s, stopping (would take >{elapsed*46/60:.0f} min for all days)")
        break

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Size':<8} {'Options':<8} {'Time(s)':<10} {'RMSE(%)':<10} {'Est. Total(min)':<15}")
print("-"*70)

for r in results:
    est_total = r["time_sec"] * 46 / 60  # 46 calibrations (23 days × 2)
    print(f"{r['sample_size']:<8} {r['n_options']:<8} {r['time_sec']:<10.1f} "
          f"{r['rmse_pct']:<10.3f} {est_total:<15.1f}")

# Find best for 30 min
valid = [r for r in results if r["time_sec"] * 46 / 60 < 30]
if valid:
    best = max(valid, key=lambda x: x["n_options"])
    print(f"\n✓ RECOMMENDED: sample_size={best['sample_size']} → {best['time_sec']*46/60:.1f} min total")
else:
    print(f"\n⚠️  All exceed 30 min. Closest: {results[0]['sample_size']} → {results[0]['time_sec']*46/60:.1f} min")

