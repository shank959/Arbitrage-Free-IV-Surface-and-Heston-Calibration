"""
Phase 4 Validation: Post-Repair Arbitrage Check

This module re-runs Phase 2 detection logic on repaired prices to verify
that all static arbitrage violations have been eliminated.

Step 5 from the plan:
- Re-run Phase 2 detection on repaired prices
- Assert zero violations (or report residual issues)
- Compute adjustment statistics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Tolerance for floating point comparisons
# Using a larger tolerance (1e-4) to account for solver numerical precision
EPS = 1e-4


@dataclass
class ValidationResult:
    """Container for validation outputs."""
    
    # Summary of violations pre and post repair
    pre_violations: Dict[str, int]
    post_violations: Dict[str, int]
    
    # Detailed violation records (post-repair, should be empty if successful)
    monotonicity_violations: pd.DataFrame
    convexity_violations: pd.DataFrame
    calendar_violations: pd.DataFrame
    
    # Overall pass/fail
    is_valid: bool
    message: str


def _check_monotonicity(
    df: pd.DataFrame,
    quote_date: str,
    expiration: str,
    price_col: str = "price_repaired",
) -> List[Dict]:
    """
    Check monotonicity violations for calls.
    Calls should decrease with strike: C(K1) >= C(K2) for K1 < K2.
    
    Unlike Phase 2, we check on repaired prices directly (not bid-ask aware).
    """
    records = []
    g = df.sort_values("strike")
    strikes = g["strike"].to_numpy()
    prices = g[price_col].to_numpy()
    
    if len(g) < 2:
        return records
    
    for i in range(len(g) - 1):
        k_curr, k_next = strikes[i], strikes[i + 1]
        p_curr, p_next = prices[i], prices[i + 1]
        
        if np.isnan(p_curr) or np.isnan(p_next):
            continue
        
        # Violation: lower strike should have higher or equal price
        if p_next > p_curr + EPS:
            records.append({
                "quote_date": quote_date,
                "expiration": expiration,
                "strike_low": k_curr,
                "strike_high": k_next,
                "price_low": p_curr,
                "price_high": p_next,
                "violation": p_next - p_curr,
                "rule": "monotonicity",
            })
    
    return records


def _check_convexity(
    df: pd.DataFrame,
    quote_date: str,
    expiration: str,
    price_col: str = "price_repaired",
) -> List[Dict]:
    """
    Check convexity violations (discrete second difference).
    For interior strikes: C(K1) - 2*C(K2) + C(K3) >= 0
    """
    records = []
    g = df.sort_values("strike")
    strikes = g["strike"].to_numpy()
    prices = g[price_col].to_numpy()
    
    if len(g) < 3:
        return records
    
    for i in range(len(g) - 2):
        k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
        p1, p2, p3 = prices[i], prices[i + 1], prices[i + 2]
        
        if any(np.isnan(v) for v in (p1, p2, p3)):
            continue
        
        # Convexity check: second difference >= 0
        second_diff = p1 - 2 * p2 + p3
        
        if second_diff < -EPS:
            records.append({
                "quote_date": quote_date,
                "expiration": expiration,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "second_diff": second_diff,
                "rule": "convexity",
            })
    
    return records


def _check_calendar(
    df: pd.DataFrame,
    price_col: str = "price_repaired",
) -> List[Dict]:
    """
    Check calendar spread violations.
    For same strike across expirations: C(K, T1) <= C(K, T2) for T1 < T2.
    """
    records = []
    
    for (quote_date, strike), grp in df.groupby(["quote_date", "strike"]):
        g = grp.sort_values("expiration")
        expirations = g["expiration"].to_numpy()
        prices = g[price_col].to_numpy()
        
        for i in range(len(g) - 1):
            exp_short, exp_long = expirations[i], expirations[i + 1]
            p_short, p_long = prices[i], prices[i + 1]
            
            if np.isnan(p_short) or np.isnan(p_long):
                continue
            
            # Violation: shorter maturity should have lower or equal price
            if p_short > p_long + EPS:
                records.append({
                    "quote_date": quote_date,
                    "strike": strike,
                    "expiration_short": exp_short,
                    "expiration_long": exp_long,
                    "price_short": p_short,
                    "price_long": p_long,
                    "violation": p_short - p_long,
                    "rule": "calendar",
                })
    
    return records


def count_violations_in_df(
    df: pd.DataFrame,
    price_col: str = "mid",
) -> Dict[str, int]:
    """
    Count all violations in a DataFrame using a specific price column.
    
    Args:
        df: DataFrame with option data
        price_col: Column to use for price checks
    
    Returns:
        Dict with counts for each violation type
    """
    mono_records = []
    conv_records = []
    
    for (quote_date, expiration), grp in df.groupby(["quote_date", "expiration"]):
        mono_records.extend(_check_monotonicity(grp, quote_date, expiration, price_col))
        conv_records.extend(_check_convexity(grp, quote_date, expiration, price_col))
    
    cal_records = _check_calendar(df, price_col)
    
    return {
        "monotonicity": len(mono_records),
        "convexity": len(conv_records),
        "calendar": len(cal_records),
        "total": len(mono_records) + len(conv_records) + len(cal_records),
    }


def validate_repaired_surface(
    repaired_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """
    Validate that a repaired surface has no arbitrage violations.
    
    Step 5 from the plan:
    - Re-run Phase 2 detection on repaired prices
    - Assert zero violations (or report residual issues)
    
    Args:
        repaired_df: DataFrame with 'price_repaired' column
        original_df: Optional original DataFrame for pre/post comparison
    
    Returns:
        ValidationResult with detailed violation information
    """
    # =========================================================================
    # Count pre-repair violations (on mid prices)
    # =========================================================================
    if original_df is not None:
        pre_violations = count_violations_in_df(original_df, price_col="mid")
    else:
        # Use the mid column from repaired_df if available
        if "mid" in repaired_df.columns:
            pre_violations = count_violations_in_df(repaired_df, price_col="mid")
        else:
            pre_violations = {"monotonicity": 0, "convexity": 0, "calendar": 0, "total": 0}
    
    # =========================================================================
    # Check post-repair violations
    # =========================================================================
    mono_records = []
    conv_records = []
    
    for (quote_date, expiration), grp in repaired_df.groupby(["quote_date", "expiration"]):
        mono_records.extend(_check_monotonicity(grp, quote_date, expiration, "price_repaired"))
        conv_records.extend(_check_convexity(grp, quote_date, expiration, "price_repaired"))
    
    cal_records = _check_calendar(repaired_df, "price_repaired")
    
    mono_df = pd.DataFrame(mono_records)
    conv_df = pd.DataFrame(conv_records)
    cal_df = pd.DataFrame(cal_records)
    
    post_violations = {
        "monotonicity": len(mono_records),
        "convexity": len(conv_records),
        "calendar": len(cal_records),
        "total": len(mono_records) + len(conv_records) + len(cal_records),
    }
    
    # =========================================================================
    # Determine if repair was successful
    # =========================================================================
    is_valid = post_violations["total"] == 0
    
    if is_valid:
        message = "SUCCESS: All arbitrage violations eliminated."
    else:
        message = (
            f"WARNING: {post_violations['total']} residual violations remain "
            f"(mono={post_violations['monotonicity']}, "
            f"conv={post_violations['convexity']}, "
            f"cal={post_violations['calendar']})"
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Pre-repair violations:  {pre_violations['total']}")
    print(f"  - Monotonicity: {pre_violations['monotonicity']}")
    print(f"  - Convexity:    {pre_violations['convexity']}")
    print(f"  - Calendar:     {pre_violations['calendar']}")
    print(f"\nPost-repair violations: {post_violations['total']}")
    print(f"  - Monotonicity: {post_violations['monotonicity']}")
    print(f"  - Convexity:    {post_violations['convexity']}")
    print(f"  - Calendar:     {post_violations['calendar']}")
    print(f"\n{message}")
    print("=" * 60)
    
    return ValidationResult(
        pre_violations=pre_violations,
        post_violations=post_violations,
        monotonicity_violations=mono_df,
        convexity_violations=conv_df,
        calendar_violations=cal_df,
        is_valid=is_valid,
        message=message,
    )


def generate_validation_report(
    validation_result: ValidationResult,
    output_dir: Path,
) -> None:
    """
    Export validation results to files.
    
    Args:
        validation_result: ValidationResult from validate_repaired_surface
        output_dir: Directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary CSV
    summary = pd.DataFrame([{
        "pre_monotonicity": validation_result.pre_violations["monotonicity"],
        "pre_convexity": validation_result.pre_violations["convexity"],
        "pre_calendar": validation_result.pre_violations["calendar"],
        "pre_total": validation_result.pre_violations["total"],
        "post_monotonicity": validation_result.post_violations["monotonicity"],
        "post_convexity": validation_result.post_violations["convexity"],
        "post_calendar": validation_result.post_violations["calendar"],
        "post_total": validation_result.post_violations["total"],
        "is_valid": validation_result.is_valid,
        "message": validation_result.message,
    }])
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    
    # Detailed violations (if any remain)
    if not validation_result.monotonicity_violations.empty:
        validation_result.monotonicity_violations.to_csv(
            output_dir / "residual_monotonicity.csv", index=False
        )
    
    if not validation_result.convexity_violations.empty:
        validation_result.convexity_violations.to_csv(
            output_dir / "residual_convexity.csv", index=False
        )
    
    if not validation_result.calendar_violations.empty:
        validation_result.calendar_violations.to_csv(
            output_dir / "residual_calendar.csv", index=False
        )

