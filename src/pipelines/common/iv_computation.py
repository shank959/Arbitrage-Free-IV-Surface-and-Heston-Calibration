"""
Shared Implied Volatility Computation

This module provides a standardized IV computation function used across all phases.
Uses discounted intrinsic value (theoretically correct) and brentq root finding.
"""

from typing import Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# Numerical constants
MIN_SIGMA = 1e-4
MAX_SIGMA = 5.0


def _bs_price(
    spot: float,
    strike: float,
    ttm: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
) -> float:
    """Black-Scholes price."""
    if vol < MIN_SIGMA or ttm <= 0 or strike <= 0 or spot <= 0:
        return np.nan
    
    sqrt_t = np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol * vol) * ttm) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    
    df_r = np.exp(-r * ttm)
    df_q = np.exp(-q * ttm)
    
    call = df_q * spot * norm.cdf(d1) - df_r * strike * norm.cdf(d2)
    
    if is_call:
        return call
    return call - df_q * spot + df_r * strike


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    ttm: float,
    r: float = 0.0,
    q: float = 0.0,
    is_call: bool = True,
) -> Optional[float]:
    """
    Compute implied volatility from price via root finding.
    
    Uses discounted intrinsic value (theoretically correct) and brentq for
    robust root finding.
    
    Args:
        price: Option price
        spot: Spot price
        strike: Strike price
        ttm: Time to maturity (years)
        r: Risk-free rate (default: 0.0)
        q: Dividend yield (default: 0.0)
        is_call: True for call, False for put (default: True)
    
    Returns:
        Implied volatility (as decimal, e.g., 0.20 for 20%), or None if cannot compute
    """
    if price <= 0 or strike <= 0 or ttm <= 0 or spot <= 0:
        return None
    
    if any(np.isnan(v) for v in (price, spot, strike, ttm)):
        return None
    
    # Intrinsic check using discounted values (theoretically correct)
    if is_call:
        intrinsic = max(spot * np.exp(-q * ttm) - strike * np.exp(-r * ttm), 0.0)
        upper = spot * np.exp(-q * ttm)  # Upper bound for call
    else:
        intrinsic = max(strike * np.exp(-r * ttm) - spot * np.exp(-q * ttm), 0.0)
        upper = strike * np.exp(-r * ttm)  # Upper bound for put
    
    if price < intrinsic - 1e-6:
        return None
    
    if price > upper + 1e-6:
        return None
    
    def objective(vol: float) -> float:
        return _bs_price(spot, strike, ttm, vol, r, q, is_call) - price
    
    try:
        f_low = objective(MIN_SIGMA)
        f_high = objective(MAX_SIGMA)
        
        if np.sign(f_low) == np.sign(f_high):
            # Expand search if needed
            for high in [6.0, 8.0, 10.0]:
                f_high = objective(high)
                if np.sign(f_low) != np.sign(f_high):
                    return brentq(objective, MIN_SIGMA, high, xtol=1e-6)
            return None
        
        return brentq(objective, MIN_SIGMA, MAX_SIGMA, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None

