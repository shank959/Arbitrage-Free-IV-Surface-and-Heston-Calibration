"""
Phase 4b: Calendar Consistency via Isotonic Regression

This module applies a lightweight cross-maturity reconciliation AFTER the per-expiration
repair (Phase 4a) to eliminate calendar arbitrage violations without solving a huge joint QP.

Algorithm:
----------
1. Build a common moneyness grid (x = K/S0) across all expirations
2. Interpolate repaired prices onto the grid
3. For each grid point, apply isotonic regression across maturities to enforce:
   C(K, T_{j+1}) >= C(K, T_j) (calendar monotonicity)
4. Clamp results to bid-ask bounds
5. Map back to original strikes

This is O(num_grid_points * num_maturities) per quote_date - much faster than a joint QP.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# Numerical tolerance
EPS = 1e-8


@dataclass
class CalendarReconcileResult:
    """Results from calendar reconciliation."""
    
    reconciled_df: pd.DataFrame
    calendar_violations_before: int
    calendar_violations_after: int
    mean_adjustment: float
    max_adjustment: float
    pct_within_bidask: float


def isotonic_regression_pav(y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pool Adjacent Violators (PAV) algorithm for isotonic regression.
    
    Fits a non-decreasing sequence to y that minimizes weighted squared error.
    
    Args:
        y: Input values (should be in maturity order)
        weights: Optional weights (default: uniform)
    
    Returns:
        Non-decreasing sequence closest to y
    """
    n = len(y)
    if n == 0:
        return np.array([])
    if n == 1:
        return y.copy()
    
    if weights is None:
        weights = np.ones(n)
    
    # Initialize blocks: each element is its own block
    # block_sum[i] = weighted sum of values in block i
    # block_weight[i] = sum of weights in block i
    # block_value[i] = block_sum[i] / block_weight[i] (the isotonic value)
    
    result = y.copy().astype(float)
    block_sum = y * weights
    block_weight = weights.copy().astype(float)
    
    # Pool adjacent violators
    # We iterate and merge blocks that violate monotonicity
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            # Compute current block values
            val_i = block_sum[i] / block_weight[i] if block_weight[i] > 0 else result[i]
            val_next = block_sum[i + 1] / block_weight[i + 1] if block_weight[i + 1] > 0 else result[i + 1]
            
            if val_i > val_next + EPS:
                # Violation: merge blocks i and i+1
                block_sum[i] = block_sum[i] + block_sum[i + 1]
                block_weight[i] = block_weight[i] + block_weight[i + 1]
                
                # Remove block i+1
                block_sum = np.delete(block_sum, i + 1)
                block_weight = np.delete(block_weight, i + 1)
                result = np.delete(result, i + 1)
                
                changed = True
            else:
                i += 1
    
    # Now expand blocks back to original size
    # We need to track which original indices belong to which block
    # Simpler approach: just re-run with tracking
    
    # Actually, let's use a cleaner implementation
    return _isotonic_pav_clean(y, weights)


def _isotonic_pav_clean(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Clean PAV implementation that correctly handles block merging.
    """
    n = len(y)
    if n <= 1:
        return y.copy()
    
    # Result array
    result = np.zeros(n)
    
    # Block structure: list of (start_idx, end_idx, value)
    blocks = []
    
    for i in range(n):
        # Create new block for element i
        block_start = i
        block_end = i
        block_value = y[i]
        block_weight = weights[i]
        
        # Merge with previous blocks while violating monotonicity
        while blocks and blocks[-1][2] > block_value + EPS:
            prev_start, prev_end, prev_value = blocks.pop()
            prev_weight = np.sum(weights[prev_start:prev_end + 1])
            
            # Merge: compute weighted average
            total_weight = prev_weight + block_weight
            if total_weight > 0:
                block_value = (prev_value * prev_weight + block_value * block_weight) / total_weight
            
            block_start = prev_start
            block_weight = total_weight
        
        blocks.append((block_start, block_end, block_value))
    
    # Expand blocks to result
    for start, end, value in blocks:
        result[start:end + 1] = value
    
    return result


def bounded_isotonic_regression(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iters: int = 10,
) -> np.ndarray:
    """
    Isotonic regression with bounds.
    
    Iteratively applies isotonic regression and clamping until convergence.
    
    Args:
        y: Input values
        lower: Lower bounds (e.g., bid prices)
        upper: Upper bounds (e.g., ask prices)
        weights: Optional weights
        max_iters: Maximum iterations
    
    Returns:
        Non-decreasing sequence within bounds (as close as possible)
    """
    result = y.copy()
    
    for _ in range(max_iters):
        prev = result.copy()
        
        # Apply isotonic regression
        result = _isotonic_pav_clean(result, weights if weights is not None else np.ones(len(y)))
        
        # Clamp to bounds
        result = np.clip(result, lower, upper)
        
        # Check convergence
        if np.allclose(result, prev, atol=EPS):
            break
    
    return result


def build_moneyness_grid(
    df: pd.DataFrame,
    spot: float,
    grid_size: int = 200,
    moneyness_bounds: Tuple[float, float] = (0.6, 1.4),
) -> np.ndarray:
    """
    Build a common moneyness grid from the union of strikes across expirations.
    
    Args:
        df: DataFrame with 'strike' column
        spot: Spot price for the quote_date
        grid_size: Number of grid points
        moneyness_bounds: (min, max) moneyness to include
    
    Returns:
        Sorted array of moneyness values
    """
    # Compute moneyness for all strikes
    all_moneyness = (df["strike"] / spot).unique()
    
    # Filter to bounds
    in_bounds = (all_moneyness >= moneyness_bounds[0]) & (all_moneyness <= moneyness_bounds[1])
    bounded_moneyness = all_moneyness[in_bounds]
    
    if len(bounded_moneyness) <= grid_size:
        # Use all available moneyness values
        return np.sort(bounded_moneyness)
    
    # Use quantiles to reduce to grid_size
    quantiles = np.linspace(0, 1, grid_size)
    grid = np.quantile(bounded_moneyness, quantiles)
    
    return np.unique(grid)


def interpolate_to_grid(
    strikes: np.ndarray,
    values: np.ndarray,
    spot: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate values from original strikes to moneyness grid.
    
    Uses linear interpolation, no extrapolation (returns NaN outside range).
    
    Args:
        strikes: Original strikes (sorted)
        values: Values at original strikes (prices, bids, asks)
        spot: Spot price
        x_grid: Target moneyness grid
    
    Returns:
        Interpolated values on x_grid (NaN where extrapolation would be needed)
    """
    if len(strikes) < 2:
        return np.full(len(x_grid), np.nan)
    
    # Convert strikes to moneyness
    x_orig = strikes / spot
    
    # Sort by moneyness (should already be sorted, but ensure)
    order = np.argsort(x_orig)
    x_orig = x_orig[order]
    values = values[order]
    
    # Remove NaN values
    valid = ~np.isnan(values)
    if np.sum(valid) < 2:
        return np.full(len(x_grid), np.nan)
    
    x_valid = x_orig[valid]
    v_valid = values[valid]
    
    # Linear interpolation, NaN outside bounds
    interp_func = interp1d(
        x_valid, v_valid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    
    return interp_func(x_grid)


def interpolate_from_grid(
    x_grid: np.ndarray,
    grid_values: np.ndarray,
    target_strikes: np.ndarray,
    spot: float,
) -> np.ndarray:
    """
    Interpolate from moneyness grid back to original strikes.
    
    Args:
        x_grid: Moneyness grid
        grid_values: Values on the grid
        target_strikes: Original strikes to interpolate to
        spot: Spot price
    
    Returns:
        Interpolated values at target_strikes
    """
    if len(x_grid) < 2:
        return np.full(len(target_strikes), np.nan)
    
    # Remove NaN from grid
    valid = ~np.isnan(grid_values)
    if np.sum(valid) < 2:
        return np.full(len(target_strikes), np.nan)
    
    x_valid = x_grid[valid]
    v_valid = grid_values[valid]
    
    # Target moneyness
    x_target = target_strikes / spot
    
    # Linear interpolation
    interp_func = interp1d(
        x_valid, v_valid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    
    return interp_func(x_target)


def count_calendar_violations(df: pd.DataFrame, price_col: str = "price_repaired") -> int:
    """
    Count calendar violations in the DataFrame.
    
    Calendar violation: C(K, T_short) > C(K, T_long) for same strike, adjacent expirations.
    """
    violations = 0
    
    for (quote_date, strike), grp in df.groupby(["quote_date", "strike"]):
        g = grp.sort_values("expiration")
        prices = g[price_col].to_numpy()
        
        for i in range(len(prices) - 1):
            if not np.isnan(prices[i]) and not np.isnan(prices[i + 1]):
                if prices[i] > prices[i + 1] + EPS:
                    violations += 1
    
    return violations


def calendar_reconcile_quote_date(
    df_day: pd.DataFrame,
    spot: float,
    grid_size: int = 200,
    moneyness_bounds: Tuple[float, float] = (0.6, 1.4),
) -> pd.DataFrame:
    """
    Apply calendar reconciliation to a single quote_date's repaired prices.
    
    Args:
        df_day: DataFrame for one quote_date with columns:
                [expiration, strike, price_repaired, bid, ask, ttm_years]
        spot: Spot price for the quote_date
        grid_size: Number of moneyness grid points
        moneyness_bounds: (min, max) moneyness to process
    
    Returns:
        DataFrame with updated price_repaired that satisfies calendar monotonicity
    """
    df = df_day.copy()
    
    # Get sorted expirations by TTM
    exp_ttm = df.groupby("expiration")["ttm_years"].first().sort_values()
    expirations = exp_ttm.index.tolist()
    n_exp = len(expirations)
    
    if n_exp < 2:
        # Nothing to reconcile
        return df
    
    # Build common moneyness grid
    x_grid = build_moneyness_grid(df, spot, grid_size, moneyness_bounds)
    n_grid = len(x_grid)
    
    if n_grid < 2:
        return df
    
    # =========================================================================
    # Step 1: Interpolate all expirations onto the grid
    # =========================================================================
    # price_grid[j, i] = price for expiration j at grid point i
    price_grid = np.full((n_exp, n_grid), np.nan)
    bid_grid = np.full((n_exp, n_grid), np.nan)
    ask_grid = np.full((n_exp, n_grid), np.nan)
    weight_grid = np.full((n_exp, n_grid), 1.0)
    
    for j, exp in enumerate(expirations):
        exp_df = df[df["expiration"] == exp].sort_values("strike")
        
        strikes = exp_df["strike"].to_numpy()
        prices = exp_df["price_repaired"].to_numpy()
        bids = exp_df["bid"].to_numpy()
        asks = exp_df["ask"].to_numpy()
        
        # Compute weights from spread
        spreads = np.maximum(asks - bids, 0.01)
        weights = 1.0 / spreads
        
        price_grid[j, :] = interpolate_to_grid(strikes, prices, spot, x_grid)
        bid_grid[j, :] = interpolate_to_grid(strikes, bids, spot, x_grid)
        ask_grid[j, :] = interpolate_to_grid(strikes, asks, spot, x_grid)
        weight_grid[j, :] = interpolate_to_grid(strikes, weights, spot, x_grid)
    
    # =========================================================================
    # Step 2: Apply isotonic regression at each grid point across maturities
    # =========================================================================
    cal_price_grid = price_grid.copy()
    
    for i in range(n_grid):
        # Get column (all expirations at this moneyness)
        prices_col = price_grid[:, i]
        bids_col = bid_grid[:, i]
        asks_col = ask_grid[:, i]
        weights_col = weight_grid[:, i]
        
        # Find valid (non-NaN) entries
        valid = ~np.isnan(prices_col)
        n_valid = np.sum(valid)
        
        if n_valid < 2:
            continue
        
        valid_idx = np.where(valid)[0]
        y = prices_col[valid]
        w = weights_col[valid]
        w = np.nan_to_num(w, nan=1.0)
        
        # Get bounds for valid entries
        lower = bids_col[valid]
        upper = asks_col[valid]
        
        # Replace NaN bounds with reasonable defaults
        lower = np.nan_to_num(lower, nan=0.0)
        upper = np.nan_to_num(upper, nan=np.inf)
        
        # Apply bounded isotonic regression
        y_cal = bounded_isotonic_regression(y, lower, upper, w, max_iters=10)
        
        # Store back
        cal_price_grid[valid_idx, i] = y_cal
    
    # =========================================================================
    # Step 3: Map back from grid to original strikes
    # =========================================================================
    df["price_calendar"] = df["price_repaired"].copy()
    
    for j, exp in enumerate(expirations):
        exp_mask = df["expiration"] == exp
        exp_strikes = df.loc[exp_mask, "strike"].to_numpy()
        
        # Interpolate from grid back to original strikes
        cal_prices = interpolate_from_grid(x_grid, cal_price_grid[j, :], exp_strikes, spot)
        
        # Where we have valid calendar prices, use them; otherwise keep original
        valid_cal = ~np.isnan(cal_prices)
        df.loc[exp_mask, "price_calendar"] = np.where(
            valid_cal,
            cal_prices,
            df.loc[exp_mask, "price_repaired"].to_numpy(),
        )
    
    # =========================================================================
    # Step 4: Final clamp to original bid-ask bounds
    # =========================================================================
    df["price_calendar"] = np.maximum(df["price_calendar"], df["bid"])
    df["price_calendar"] = np.minimum(df["price_calendar"], df["ask"])
    
    # =========================================================================
    # Step 5: Fix monotonicity violations introduced by interpolation
    # Apply a simple sweep within each expiration to ensure C(K) >= C(K+1)
    # =========================================================================
    df["price_repaired"] = df["price_calendar"]
    
    for exp in expirations:
        exp_mask = df["expiration"] == exp
        exp_df = df.loc[exp_mask].sort_values("strike")
        
        if len(exp_df) < 2:
            continue
        
        prices = exp_df["price_repaired"].to_numpy().copy()
        bids = exp_df["bid"].to_numpy()
        asks = exp_df["ask"].to_numpy()
        
        # Forward sweep: ensure monotonically decreasing in strike
        # If price[i] < price[i+1], we need to adjust
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1] - EPS:
                # Try to raise prices[i] to match prices[i+1]
                target = prices[i + 1]
                if target <= asks[i]:
                    prices[i] = target
                else:
                    # Can't raise enough, try lowering prices[i+1]
                    prices[i] = asks[i]
                    if bids[i + 1] <= prices[i]:
                        prices[i + 1] = prices[i]
        
        # Write back
        df.loc[exp_df.index, "price_repaired"] = prices
    
    # Re-clamp to bid-ask after monotonicity fix
    df["price_repaired"] = np.maximum(df["price_repaired"], df["bid"])
    df["price_repaired"] = np.minimum(df["price_repaired"], df["ask"])
    
    df = df.drop(columns=["price_calendar"])
    
    # Recompute adjustment
    df["adjustment"] = df["price_repaired"] - df["mid"]
    df["adjustment_pct"] = (df["adjustment"] / df["mid"].clip(lower=EPS)) * 100
    
    return df


def fix_monotonicity_per_expiration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix monotonicity violations within each expiration.
    
    For calls, prices should decrease with strike: C[K_i] >= C[K_{i+1}].
    Uses isotonic regression to enforce non-increasing prices.
    """
    result = df.copy()
    
    for (qd, exp), grp in result.groupby(["quote_date", "expiration"]):
        if len(grp) < 2:
            continue
        
        g = grp.sort_values("strike")
        indices = g.index.to_numpy()
        prices = g["price_repaired"].to_numpy().copy()
        bids = g["bid"].to_numpy()
        asks = g["ask"].to_numpy()
        
        # Check if already monotone (decreasing)
        is_monotonic = True
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1] - EPS:
                is_monotonic = False
                break
        
        if is_monotonic:
            continue
        
        # For monotonicity FIX: we need NON-INCREASING (decreasing) in strike
        # Flip to make it a non-decreasing problem, then flip back
        prices_flipped = -prices
        lower_flipped = -asks  # Negative of asks becomes lower bound
        upper_flipped = -bids  # Negative of bids becomes upper bound
        
        weights = 1.0 / np.maximum(asks - bids, 0.01)
        
        prices_fixed_flipped = bounded_isotonic_regression(
            prices_flipped, lower_flipped, upper_flipped, weights, max_iters=10
        )
        prices_fixed = -prices_fixed_flipped
        
        # Clamp to bid-ask bounds
        prices_fixed = np.clip(prices_fixed, bids, asks)
        
        # Write back
        result.loc[indices, "price_repaired"] = prices_fixed
    
    # Recompute adjustment
    result["adjustment"] = result["price_repaired"] - result["mid"]
    result["adjustment_pct"] = (result["adjustment"] / result["mid"].clip(lower=EPS)) * 100
    
    return result


def fix_convexity_per_expiration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix convexity violations within each expiration using a CONSERVATIVE approach.
    
    Only fix convexity if it doesn't break monotonicity. This prevents oscillation
    in the three-way iterative cleanup.
    """
    result = df.copy()
    
    for (quote_date, exp), grp in result.groupby(["quote_date", "expiration"]):
        if len(grp) < 3:
            continue
        
        g = grp.sort_values("strike")
        prices = g["price_repaired"].to_numpy().copy()
        bids = g["bid"].to_numpy()
        asks = g["ask"].to_numpy()
        indices = g.index.to_numpy()
        
        # Single pass with conservative fixes
        for i in range(1, len(prices) - 1):
            second_diff = prices[i - 1] - 2 * prices[i] + prices[i + 1]
            if second_diff < -EPS:
                # Target to restore convexity
                target = (prices[i - 1] + prices[i + 1]) / 2
                
                # Conservative: only adjust if it won't break monotonicity
                # Monotonicity requires: prices[i-1] >= prices[i] >= prices[i+1]
                lower_mono = prices[i + 1]  # Can't go below next price
                upper_mono = prices[i - 1]  # Can't go above prev price
                
                # Combined bounds
                lower = max(bids[i], lower_mono)
                upper = min(asks[i], upper_mono)
                
                if lower <= upper:  # Feasible range exists
                    # Clamp target to feasible range
                    new_price = max(lower, min(target, upper))
                    
                    # Only apply if it actually improves convexity
                    new_second_diff = prices[i - 1] - 2 * new_price + prices[i + 1]
                    if new_second_diff > second_diff:
                        prices[i] = new_price
        
        # Write back
        result.loc[indices, "price_repaired"] = prices
    
    # Recompute adjustment
    result["adjustment"] = result["price_repaired"] - result["mid"]
    result["adjustment_pct"] = (result["adjustment"] / result["mid"].clip(lower=EPS)) * 100
    
    return result


def count_mono_violations(df: pd.DataFrame, price_col: str = "price_repaired") -> int:
    """Count monotonicity violations (calls should decrease with strike)."""
    violations = 0
    for (qd, exp), grp in df.groupby(["quote_date", "expiration"]):
        g = grp.sort_values("strike")
        prices = g[price_col].to_numpy()
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1] - EPS:
                violations += 1
    return violations


def count_conv_violations(df: pd.DataFrame, price_col: str = "price_repaired") -> int:
    """Count convexity violations (second difference should be >= 0)."""
    violations = 0
    for (qd, exp), grp in df.groupby(["quote_date", "expiration"]):
        g = grp.sort_values("strike")
        prices = g[price_col].to_numpy()
        for i in range(1, len(prices) - 1):
            second_diff = prices[i - 1] - 2 * prices[i] + prices[i + 1]
            if second_diff < -EPS:
                violations += 1
    return violations


def count_all_violations(df: pd.DataFrame, price_col: str = "price_repaired") -> dict:
    """Count all types of violations quickly."""
    mono = 0
    conv = 0
    cal = 0
    
    # Monotonicity and convexity within expirations
    for (qd, exp), grp in df.groupby(["quote_date", "expiration"]):
        g = grp.sort_values("strike")
        prices = g[price_col].to_numpy()
        n = len(prices)
        
        # Monotonicity: C[i] >= C[i+1] for calls
        for i in range(n - 1):
            if prices[i] < prices[i + 1] - EPS:
                mono += 1
        
        # Convexity: C[i-1] - 2*C[i] + C[i+1] >= 0
        for i in range(1, n - 1):
            if prices[i - 1] - 2 * prices[i] + prices[i + 1] < -EPS:
                conv += 1
    
    # Calendar
    cal = count_calendar_violations(df, price_col)
    
    return {"mono": mono, "conv": conv, "cal": cal, "total": mono + conv + cal}


def direct_calendar_fix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply calendar monotonicity fix directly on original strikes.
    
    For each (quote_date, strike), ensures C[exp_short] <= C[exp_long] for 
    all adjacent expirations using isotonic regression with bid-ask clamping.
    """
    result = df.copy()
    
    for (qd, strike), grp in result.groupby(["quote_date", "strike"]):
        if len(grp) < 2:
            continue
        
        g = grp.sort_values("expiration")
        indices = g.index.to_numpy()
        prices = g["price_repaired"].to_numpy().copy()
        bids = g["bid"].to_numpy()
        asks = g["ask"].to_numpy()
        
        # Check if already monotonic
        is_monotonic = True
        for i in range(len(prices) - 1):
            if prices[i] > prices[i + 1] + EPS:
                is_monotonic = False
                break
        
        if is_monotonic:
            continue
        
        # Apply bounded isotonic regression (increasing in maturity)
        weights = 1.0 / np.maximum(asks - bids, 0.01)
        lower = bids
        upper = asks
        
        # Replace NaN bounds with defaults
        lower = np.nan_to_num(lower, nan=0.0)
        upper = np.nan_to_num(upper, nan=np.inf)
        
        # Calendar fix: C should be non-decreasing in maturity
        prices_fixed = bounded_isotonic_regression(prices, lower, upper, weights, max_iters=10)
        
        # Write back
        result.loc[indices, "price_repaired"] = prices_fixed
    
    # Recompute adjustment
    result["adjustment"] = result["price_repaired"] - result["mid"]
    result["adjustment_pct"] = (result["adjustment"] / result["mid"].clip(lower=EPS)) * 100
    
    return result


def apply_calendar_reconciliation(
    repaired_df: pd.DataFrame,
    grid_size: int = 200,
    moneyness_bounds: Tuple[float, float] = (0.6, 1.4),
    fix_convexity_after: bool = True,
) -> CalendarReconcileResult:
    """
    Apply calendar reconciliation to all quote_dates in the repaired DataFrame.
    
    Two-pass approach:
    1. Grid-based isotonic reconciliation (handles most violations)
    2. Direct strike-based fix (catches remaining violations)
    
    Args:
        repaired_df: DataFrame from Phase 4a with price_repaired column
        grid_size: Number of moneyness grid points
        moneyness_bounds: (min, max) moneyness to process
        fix_convexity_after: If True, run a final convexity cleanup pass
    
    Returns:
        CalendarReconcileResult with reconciled data and metrics
    """
    df = repaired_df.copy()
    
    # Count violations before
    violations_before = count_calendar_violations(df, "price_repaired")
    
    # Store original prices for computing adjustment delta
    df["price_before_cal"] = df["price_repaired"].copy()
    
    # Process each quote_date
    reconciled_dfs = []
    
    for quote_date in df["quote_date"].unique():
        day_mask = df["quote_date"] == quote_date
        df_day = df[day_mask].copy()
        
        # Get spot from the data
        spot = df_day["spot"].iloc[0]
        
        # Apply calendar reconciliation
        df_day_cal = calendar_reconcile_quote_date(
            df_day,
            spot=spot,
            grid_size=grid_size,
            moneyness_bounds=moneyness_bounds,
        )
        
        reconciled_dfs.append(df_day_cal)
    
    # Combine results
    result_df = pd.concat(reconciled_dfs, ignore_index=True)
    
    # Count violations after
    violations_after = count_calendar_violations(result_df, "price_repaired")
    
    # Compute adjustment metrics from Phase 4b
    cal_adjustment = (result_df["price_repaired"] - result_df["price_before_cal"]).abs()
    mean_adjustment = cal_adjustment.mean()
    max_adjustment = cal_adjustment.max()
    
    # Check bid-ask compliance
    within_bidask = (
        (result_df["price_repaired"] >= result_df["bid"] - EPS) &
        (result_df["price_repaired"] <= result_df["ask"] + EPS)
    ).mean() * 100
    
    # Clean up temporary column
    result_df = result_df.drop(columns=["price_before_cal"])
    
    print(f"\n=== Phase 4b: Calendar Reconciliation (Grid Pass) ===")
    print(f"Calendar violations before: {violations_before}")
    print(f"Calendar violations after:  {violations_after}")
    print(f"Reduction: {violations_before - violations_after} ({100*(violations_before - violations_after)/max(violations_before, 1):.1f}%)")
    print(f"Mean additional adjustment: ${mean_adjustment:.4f}")
    print(f"Max additional adjustment:  ${max_adjustment:.4f}")
    print(f"Within bid-ask: {within_bidask:.1f}%")
    
    # =========================================================================
    # Second pass: Direct strike-based calendar fix
    # This catches violations that the grid interpolation missed
    # =========================================================================
    violations_after_grid = violations_after
    result_df = direct_calendar_fix(result_df)
    violations_after_direct = count_calendar_violations(result_df, "price_repaired")
    
    print(f"\n=== Phase 4b: Calendar Reconciliation (Direct Pass) ===")
    print(f"Calendar violations after grid:   {violations_after_grid}")
    print(f"Calendar violations after direct: {violations_after_direct}")
    print(f"Additional reduction: {violations_after_grid - violations_after_direct}")
    
    # Update final violation count
    violations_after = violations_after_direct
    
    # =========================================================================
    # Iterative cleanup: Alternate mono+cal until stable, skip convexity
    # (Convexity in the loop causes oscillation and makes things worse)
    # =========================================================================
    print(f"\n=== Iterative Cleanup (Mono + Cal) ===")
    
    for iter_num in range(10):
        # Fix monotonicity
        result_df = fix_monotonicity_per_expiration(result_df)
        mono = count_mono_violations(result_df)
        
        # Fix calendar
        result_df = direct_calendar_fix(result_df)
        cal = count_calendar_violations(result_df, "price_repaired")
        
        print(f"  Iter {iter_num+1}: mono={mono}, cal={cal}")
        
        if mono == 0 and cal == 0:
            print("  Converged!")
            break
    
    # Final monotonicity pass
    result_df = fix_monotonicity_per_expiration(result_df)
    
    final_mono = count_mono_violations(result_df)
    final_conv = count_conv_violations(result_df)
    final_cal = count_calendar_violations(result_df, "price_repaired")
    print(f"Final: mono={final_mono}, conv={final_conv}, cal={final_cal}")
    
    violations_after = final_cal
    
    # Re-check bid-ask compliance after all fixes
    within_bidask = (
        (result_df["price_repaired"] >= result_df["bid"] - EPS) &
        (result_df["price_repaired"] <= result_df["ask"] + EPS)
    ).mean() * 100
    print(f"Within bid-ask: {within_bidask:.1f}%")
    
    return CalendarReconcileResult(
        reconciled_df=result_df,
        calendar_violations_before=violations_before,
        calendar_violations_after=violations_after,
        mean_adjustment=mean_adjustment,
        max_adjustment=max_adjustment,
        pct_within_bidask=within_bidask,
    )

