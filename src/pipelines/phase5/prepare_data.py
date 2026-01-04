"""
Data Preparation for Heston Calibration

This module extracts implied volatility surfaces from Phase 4 repaired option
data, preparing it for Heston calibration.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Import shared IV computation for consistency
from pipelines.common.iv_computation import implied_vol


def prepare_calibration_data(
    parquet_path: Path,
    quote_date: str,
    price_col: str = "mid",
    r: float = 0.0,
    q: float = 0.0,
    min_ttm: float = 0.01,
    max_ttm: float = 2.0,
    min_moneyness: float = 0.7,
    max_moneyness: float = 1.3,
    max_options: Optional[int] = None,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """
    Extract calibration data from Phase 4 output.
    
    Args:
        parquet_path: Path to repaired_options.parquet
        quote_date: Which day to extract (YYYY-MM-DD)
        price_col: "mid" for raw prices, "price_repaired" for cleaned
        r: Risk-free rate
        q: Dividend yield
        min_ttm: Minimum time to maturity (exclude very short-dated)
        max_ttm: Maximum time to maturity (exclude very long-dated)
        min_moneyness: Minimum K/S ratio
        max_moneyness: Maximum K/S ratio
        max_options: If set, subsample to at most this many options (stratified by expiry)
    
    Returns:
        market_data: List of (strike, ttm, iv, weight) tuples
        spot: Spot price for the day
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["quote_date"] == quote_date].copy()
    
    if df.empty:
        raise ValueError(f"No data for quote_date {quote_date}")
    
    # Get spot price
    spot = df["spot"].iloc[0]
    
    # Filter by TTM
    df = df[(df["ttm_years"] >= min_ttm) & (df["ttm_years"] <= max_ttm)]
    
    # Filter by moneyness
    df["moneyness"] = df["strike"] / spot
    df = df[(df["moneyness"] >= min_moneyness) & (df["moneyness"] <= max_moneyness)]
    
    # Compute implied volatilities
    market_data = []
    for _, row in df.iterrows():
        price = row[price_col]
        
        if pd.isna(price) or price <= 0:
            continue
        
        iv = implied_vol(
            price, spot, row["strike"], row["ttm_years"], r, q, is_call=True
        )
        
        if iv is not None and 0.01 < iv < 0.50:  # Filter extreme IVs that Heston cannot model
            # Weight by inverse bid-ask spread (liquidity)
            # Use sqrt and cap to prevent extreme weights from dominating
            spread = max(row["ask"] - row["bid"], 0.01)
            raw_weight = 1.0 / spread
            weight = np.sqrt(raw_weight)  # Gentler scaling: 100→10 instead of 100→1
            weight = min(weight, 20.0)     # Cap at 20x to prevent dominance
            
            market_data.append((
                row["strike"],
                row["ttm_years"],
                iv,
                weight
            ))
    
    # Subsample if requested with improved 2D stratification
    if max_options is not None and len(market_data) > max_options:
        # Group by expiry
        from collections import defaultdict
        by_expiry = defaultdict(list)
        for item in market_data:
            ttm = item[1]
            by_expiry[ttm].append(item)
        
        n_expiries = len(by_expiry)
        min_per_expiry = 5  # Minimum to capture smile shape
        
        if max_options >= n_expiries * min_per_expiry:
            # Can sample from all expiries with 2D stratification
            # Weight expiries by total liquidity (sum of weights)
            expiry_weights = {ttm: sum(item[3] for item in items) 
                            for ttm, items in by_expiry.items()}
            total_weight = sum(expiry_weights.values())
            
            sampled = []
            for ttm, items in sorted(by_expiry.items()):
                # Allocate samples proportional to expiry liquidity
                n_for_expiry = int(max_options * expiry_weights[ttm] / total_weight)
                n_for_expiry = max(n_for_expiry, 3)  # At least 3 per expiry
                
                # Within expiry: sample across moneyness (strike) range
                # Sort by strike to get moneyness distribution
                items_sorted = sorted(items, key=lambda x: x[0])
                
                if len(items_sorted) <= n_for_expiry:
                    sampled.extend(items_sorted)
                else:
                    # Use linspace to get evenly distributed indices across strikes
                    indices = np.linspace(0, len(items_sorted)-1, 
                                        n_for_expiry, dtype=int)
                    sampled.extend([items_sorted[i] for i in indices])
            
            # Trim to exact limit
            market_data = sampled[:max_options]
        else:
            # Need to reduce number of expiries
            n_expiries_to_use = max(max_options // min_per_expiry, 3)
            target_per_expiry = max_options // n_expiries_to_use
            
            # Select evenly spaced expiries across TTM range
            all_ttms = sorted(by_expiry.keys())
            indices = np.linspace(0, len(all_ttms)-1, n_expiries_to_use, dtype=int)
            selected_ttms = [all_ttms[i] for i in indices]
            
            sampled = []
            for ttm in selected_ttms:
                items = by_expiry[ttm]
                # Sort by strike for moneyness sampling
                items_sorted = sorted(items, key=lambda x: x[0])
                
                if len(items_sorted) <= target_per_expiry:
                    sampled.extend(items_sorted)
                else:
                    indices = np.linspace(0, len(items_sorted)-1, 
                                        target_per_expiry, dtype=int)
                    sampled.extend([items_sorted[i] for i in indices])
            
            market_data = sampled[:max_options]
    
    return market_data, spot


def prepare_all_days(
    parquet_path: Path,
    price_col: str = "mid",
    r: float = 0.0,
    q: float = 0.0,
    **kwargs,
) -> dict:
    """
    Prepare calibration data for all quote dates in the dataset.
    
    Returns:
        Dict mapping quote_date -> (market_data, spot)
    """
    df = pd.read_parquet(parquet_path)
    quote_dates = sorted(df["quote_date"].unique())
    
    results = {}
    for qd in quote_dates:
        try:
            market_data, spot = prepare_calibration_data(
                parquet_path, qd, price_col, r, q, **kwargs
            )
            if len(market_data) >= 10:  # Minimum for meaningful calibration
                results[qd] = (market_data, spot)
        except Exception as e:
            print(f"Warning: Failed to prepare {qd}: {e}")
    
    return results


def get_available_quote_dates(parquet_path: Path) -> List[str]:
    """Get list of available quote dates in the dataset."""
    df = pd.read_parquet(parquet_path)
    return sorted(df["quote_date"].unique())


def summarize_market_data(market_data: List[Tuple[float, float, float, float]]) -> dict:
    """
    Summarize market data for diagnostics.
    """
    if not market_data:
        return {}
    
    strikes = np.array([m[0] for m in market_data])
    ttms = np.array([m[1] for m in market_data])
    ivs = np.array([m[2] for m in market_data])
    
    return {
        "n_options": len(market_data),
        "strike_range": (strikes.min(), strikes.max()),
        "ttm_range": (ttms.min(), ttms.max()),
        "iv_range": (ivs.min(), ivs.max()),
        "iv_mean": ivs.mean(),
        "n_expirations": len(np.unique(ttms)),
    }


# =============================================================================
# Test function
# =============================================================================
def _test_data_prep():
    """Test data preparation on actual Phase 4 output."""
    parquet_path = Path("reports/phase4/1545/repaired_options.parquet")
    
    if not parquet_path.exists():
        print(f"Test parquet not found: {parquet_path}")
        return
    
    dates = get_available_quote_dates(parquet_path)
    print(f"Available dates: {len(dates)}")
    print(f"First 5: {dates[:5]}")
    
    # Test on first date
    test_date = dates[0]
    print(f"\nTesting on {test_date}:")
    
    # Raw data
    raw_data, spot = prepare_calibration_data(parquet_path, test_date, "mid")
    raw_summary = summarize_market_data(raw_data)
    print(f"  Raw: {raw_summary['n_options']} options")
    print(f"    IV range: [{raw_summary['iv_range'][0]*100:.1f}%, {raw_summary['iv_range'][1]*100:.1f}%]")
    
    # Repaired data
    rep_data, spot = prepare_calibration_data(parquet_path, test_date, "price_repaired")
    rep_summary = summarize_market_data(rep_data)
    print(f"  Repaired: {rep_summary['n_options']} options")
    print(f"    IV range: [{rep_summary['iv_range'][0]*100:.1f}%, {rep_summary['iv_range'][1]*100:.1f}%]")
    print(f"  Spot: ${spot:.2f}")
    
    return raw_data, rep_data, spot


if __name__ == "__main__":
    _test_data_prep()

