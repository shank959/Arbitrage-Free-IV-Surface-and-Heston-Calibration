"""
Heston Model Pricing via QuantLib

This module provides a wrapper around QuantLib's Heston model implementation
for pricing European options and computing Heston-implied volatilities.

The Heston model SDE:
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    dW_t^S dW_t^v = rho dt

Parameters:
    v0: Initial variance
    kappa: Mean reversion speed
    theta: Long-run variance
    xi: Volatility of volatility (vol-of-vol)
    rho: Correlation between asset and variance Brownian motions
"""

from typing import Optional

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq
from scipy.stats import norm


# Numerical constants
MIN_SIGMA = 1e-4
MAX_SIGMA = 5.0


def heston_price(
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    is_call: bool = True,
) -> float:
    """
    Price a European option under the Heston model using QuantLib.
    
    Args:
        spot: Current spot price
        strike: Option strike price
        ttm: Time to maturity in years
        r: Risk-free rate (continuously compounded)
        q: Dividend yield (continuously compounded)
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-run variance
        xi: Volatility of volatility
        rho: Correlation between spot and variance
        is_call: True for call, False for put
    
    Returns:
        Option price
    """
    if ttm <= 0 or strike <= 0 or spot <= 0:
        return np.nan
    
    # Validate Heston parameters
    if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0:
        return np.nan
    if rho < -1 or rho > 1:
        return np.nan
    
    try:
        # Setup dates
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today
        
        # Convert TTM to days (QuantLib needs integer days)
        days = max(1, int(ttm * 365))
        expiry = today + ql.Period(days, ql.Days)
        
        # Build term structures for rates
        day_count = ql.Actual365Fixed()
        r_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, r, day_count)
        )
        q_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, q, day_count)
        )
        
        # Spot quote
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        
        # Build Heston process
        heston_process = ql.HestonProcess(
            r_handle, q_handle, spot_handle,
            v0, kappa, theta, xi, rho
        )
        
        # Build Heston model and engine
        heston_model = ql.HestonModel(heston_process)
        engine = ql.AnalyticHestonEngine(heston_model)
        
        # Build option
        option_type = ql.Option.Call if is_call else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.EuropeanExercise(expiry)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        
        return option.NPV()
    
    except Exception as e:
        # Return NaN for any QuantLib errors
        return np.nan


def _bs_price(
    spot: float,
    strike: float,
    ttm: float,
    vol: float,
    r: float,
    q: float,
    is_call: bool,
) -> float:
    """Black-Scholes price for IV inversion."""
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
    return call - df_q * spot + df_r * strike  # Put via put-call parity


def implied_vol_from_price(
    price: float,
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    q: float,
    is_call: bool,
) -> Optional[float]:
    """
    Compute Black-Scholes implied volatility from a given price.
    
    Uses Brent's method for root finding.
    """
    if price <= 0 or strike <= 0 or ttm <= 0 or spot <= 0:
        return None
    
    # Intrinsic value check
    intrinsic = max(spot * np.exp(-q * ttm) - strike * np.exp(-r * ttm), 0.0) if is_call else \
                max(strike * np.exp(-r * ttm) - spot * np.exp(-q * ttm), 0.0)
    
    if price < intrinsic - 1e-6:
        return None
    
    def objective(vol: float) -> float:
        return _bs_price(spot, strike, ttm, vol, r, q, is_call) - price
    
    try:
        # Try to find a root in [MIN_SIGMA, MAX_SIGMA]
        f_low = objective(MIN_SIGMA)
        f_high = objective(MAX_SIGMA)
        
        # Expand search if needed
        if np.sign(f_low) == np.sign(f_high):
            # Try expanding upper bound
            for high in [6.0, 8.0, 10.0]:
                f_high = objective(high)
                if np.sign(f_low) != np.sign(f_high):
                    return brentq(objective, MIN_SIGMA, high, xtol=1e-6)
            return None
        
        return brentq(objective, MIN_SIGMA, MAX_SIGMA, xtol=1e-6)
    
    except (ValueError, RuntimeError):
        return None


def heston_implied_vol(
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    is_call: bool = True,
) -> Optional[float]:
    """
    Compute the Black-Scholes implied volatility that matches the Heston price.
    
    This is the key function for calibration: we compare model IV to market IV.
    """
    heston_px = heston_price(spot, strike, ttm, r, q, v0, kappa, theta, xi, rho, is_call)
    
    if np.isnan(heston_px) or heston_px <= 0:
        return None
    
    return implied_vol_from_price(heston_px, spot, strike, ttm, r, q, is_call)


def check_feller_condition(kappa: float, theta: float, xi: float) -> bool:
    """
    Check if Heston parameters satisfy the Feller condition.
    
    The Feller condition ensures the variance process v_t stays strictly positive:
        2 * kappa * theta > xi^2
    
    If violated, the variance process can hit zero, which is unphysical.
    """
    return 2 * kappa * theta > xi * xi


# =============================================================================
# Sanity test function
# =============================================================================
def _sanity_test():
    """Quick sanity check that Heston pricing works."""
    # Typical equity parameters
    spot = 100.0
    strike = 100.0
    ttm = 0.5
    r = 0.05
    q = 0.0
    v0 = 0.04       # 20% vol
    kappa = 1.5
    theta = 0.04
    xi = 0.3
    rho = -0.7
    
    price = heston_price(spot, strike, ttm, r, q, v0, kappa, theta, xi, rho)
    iv = heston_implied_vol(spot, strike, ttm, r, q, v0, kappa, theta, xi, rho)
    feller = check_feller_condition(kappa, theta, xi)
    
    print(f"Heston ATM call price: ${price:.4f}")
    print(f"Heston implied vol: {iv*100:.2f}%")
    print(f"Feller condition satisfied: {feller}")
    print(f"Expected: price ~5.5, IV ~20%, Feller=True")
    
    return price, iv, feller


if __name__ == "__main__":
    _sanity_test()

