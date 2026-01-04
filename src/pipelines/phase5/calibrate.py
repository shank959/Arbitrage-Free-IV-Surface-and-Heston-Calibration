"""
Heston Model Calibration Engine

This module implements the calibration routine that finds Heston parameters
to best fit observed market implied volatilities.

The calibration solves:
    min_{v0, kappa, theta, xi, rho} Σ w_i * (σ_heston(K_i, T_i) - σ_market(K_i, T_i))^2

Subject to parameter bounds and ideally satisfying the Feller condition.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

from .heston_pricer import heston_implied_vol, check_feller_condition


@dataclass
class CalibrationResult:
    """Container for Heston calibration outputs."""
    
    # Calibrated parameters
    v0: float           # Initial variance
    kappa: float        # Mean reversion speed
    theta: float        # Long-run variance
    xi: float           # Vol-of-vol
    rho: float          # Correlation
    
    # Fit quality metrics
    rmse: float                 # Root mean squared error in IV
    mae: float                  # Mean absolute error in IV
    max_error: float            # Maximum absolute error
    feller_satisfied: bool      # Whether Feller condition holds
    feller_ratio: float         # 2*kappa*theta / xi^2 (>1 means satisfied)
    
    # Calibration metadata
    n_options: int              # Number of options used
    success: bool               # Optimizer success flag
    message: str                # Optimizer message
    n_func_evals: int           # Number of function evaluations


# =============================================================================
# Parameter bounds (from literature/practice)
# =============================================================================
PARAM_BOUNDS = {
    "v0":    (0.001, 1.0),    # Initial variance: 3% to 100% vol
    "kappa": (0.01, 15.0),    # Mean reversion: slow to fast (increased from 10.0)
    "theta": (0.001, 1.0),    # Long-run variance: 3% to 100% vol
    "xi":    (0.01, 5.0),     # Vol-of-vol: low to high (increased from 2.0 for high-vol data)
    "rho":   (-0.99, 0.99),   # Correlation: nearly -1 to +1
}


# Multiple starting points for global optimization
# ORDERED BY XI (high to low) - use high-xi starts first for this data!
STARTING_POINTS = [
    # [v0, kappa, theta, xi, rho]
    [0.15, 1.5, 0.12, 2.00, -0.55],   # Very high vol regime (PRIORITIZED)
    [0.10, 2.0, 0.10, 1.50, -0.60],   # High vol, high xi (PRIORITIZED)
    [0.09, 1.0, 0.09, 0.50, -0.50],   # Higher vol regime
    [0.04, 1.5, 0.04, 0.40, -0.60],   # Moderate parameters
    [0.04, 2.0, 0.04, 0.30, -0.70],   # Typical equity (SPX-like)
    [0.06, 0.5, 0.06, 0.35, -0.65],   # Slow mean reversion
    [0.02, 3.0, 0.02, 0.20, -0.80],   # Low vol, high mean reversion
]


def calibration_objective(
    params: np.ndarray,
    market_data: List[Tuple[float, float, float, float]],
    spot: float,
    r: float,
    q: float,
) -> np.ndarray:
    """
    Objective function for Heston calibration.
    
    Computes weighted residuals between Heston-implied and market implied vols.
    Includes soft Feller condition penalty to encourage well-defined parameters.
    
    Args:
        params: [v0, kappa, theta, xi, rho]
        market_data: List of (strike, ttm, market_iv, weight) tuples
        spot: Spot price
        r: Risk-free rate
        q: Dividend yield
    
    Returns:
        Array of weighted residuals: (model_iv - market_iv) * weight
    """
    v0, kappa, theta, xi, rho = params
    residuals = []
    
    for strike, ttm, market_iv, weight in market_data:
        try:
            model_iv = heston_implied_vol(
                spot, strike, ttm, r, q,
                v0, kappa, theta, xi, rho,
                is_call=True
            )
            
            if model_iv is not None and np.isfinite(model_iv):
                residuals.append((model_iv - market_iv) * weight)
            else:
                # Penalty for failed pricing (increased from 10% to 20%)
                residuals.append(0.20 * weight)
        except Exception:
            residuals.append(0.20 * weight)
    
    # Add soft Feller condition penalty
    # Feller condition: 2*kappa*theta > xi^2 ensures variance stays positive
    feller_ratio = (2 * kappa * theta) / (xi * xi) if xi > 0 else 0.0
    if feller_ratio < 1.0:
        # Penalize violations proportionally to severity
        # Use average weight to scale penalty appropriately
        avg_weight = np.mean([w for _, _, _, w in market_data]) if market_data else 1.0
        # Increased penalty strength from 0.10 to 0.50 for stronger enforcement
        penalty = (1.0 - feller_ratio) * 0.50 * avg_weight
        residuals.append(penalty)
    
    return np.array(residuals)


def calibrate_heston(
    market_data: List[Tuple[float, float, float, float]],
    spot: float,
    r: float = 0.0,
    q: float = 0.0,
    n_starts: int = 5,
    max_nfev: int = 200,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Calibrate Heston parameters using multi-start least squares optimization.
    
    The calibration minimizes weighted squared IV differences using scipy's
    trust-region reflective algorithm with box constraints.
    
    Args:
        market_data: List of (strike, ttm, market_iv, weight) tuples
        spot: Spot price
        r: Risk-free rate
        q: Dividend yield
        n_starts: Number of starting points to try
        max_nfev: Maximum function evaluations per start
        verbose: Print progress
    
    Returns:
        CalibrationResult with best-fit parameters and diagnostics
    """
    if len(market_data) < 5:
        return CalibrationResult(
            v0=np.nan, kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan,
            rmse=np.nan, mae=np.nan, max_error=np.nan,
            feller_satisfied=False, feller_ratio=np.nan,
            n_options=len(market_data),
            success=False, message="Insufficient data", n_func_evals=0
        )
    
    # Setup bounds for optimizer
    lower_bounds = [PARAM_BOUNDS[p][0] for p in ["v0", "kappa", "theta", "xi", "rho"]]
    upper_bounds = [PARAM_BOUNDS[p][1] for p in ["v0", "kappa", "theta", "xi", "rho"]]
    
    best_result = None
    best_cost = float('inf')
    total_nfev = 0
    
    # Multi-start optimization
    import time
    start_time = time.time()
    
    for i, x0 in enumerate(STARTING_POINTS[:n_starts]):
        if verbose:
            print(f"    Start {i+1}/{n_starts} [{x0[0]:.3f}, {x0[1]:.2f}, {x0[2]:.3f}, {x0[3]:.2f}, {x0[4]:.2f}]...", end=" ", flush=True)
        
        try:
            iter_start = time.time()
            result = least_squares(
                calibration_objective,
                x0,
                args=(market_data, spot, r, q),
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            iter_time = time.time() - iter_start
            
            total_nfev += result.nfev
            cost = np.sum(result.fun ** 2)
            rmse_pct = np.sqrt(cost / len(market_data)) * 100
            
            if verbose:
                print(f"RMSE={rmse_pct:.2f}%, nfev={result.nfev}, {iter_time:.1f}s", flush=True)
            
            if cost < best_cost:
                best_cost = cost
                best_result = result
        
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}", flush=True)
            continue
    
    if best_result is None:
        return CalibrationResult(
            v0=np.nan, kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan,
            rmse=np.nan, mae=np.nan, max_error=np.nan,
            feller_satisfied=False, feller_ratio=np.nan,
            n_options=len(market_data),
            success=False, message="All starts failed", n_func_evals=total_nfev
        )
    
    # Extract best parameters
    v0, kappa, theta, xi, rho = best_result.x
    
    # Compute error metrics
    n = len(market_data)
    residuals = best_result.fun
    
    # Unweight the residuals to get actual IV errors
    # Note: residuals may have one extra element if Feller penalty was added
    weights = np.array([md[3] for md in market_data])
    
    # Only unweight the first n residuals (corresponding to actual market data)
    # Any extra residual is the Feller penalty, which we exclude from error metrics
    residuals_data = residuals[:n]
    iv_errors = residuals_data / weights
    
    rmse = np.sqrt(np.mean(iv_errors ** 2))
    mae = np.mean(np.abs(iv_errors))
    max_error = np.max(np.abs(iv_errors))
    
    # Check Feller condition
    feller = check_feller_condition(kappa, theta, xi)
    feller_ratio = (2 * kappa * theta) / (xi * xi) if xi > 0 else np.inf
    
    if verbose:
        total_time = time.time() - start_time
        print(f"    ✓ Best: RMSE={rmse*100:.3f}%, Feller={feller}, {total_time:.1f}s total", flush=True)
    
    return CalibrationResult(
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        rmse=rmse,
        mae=mae,
        max_error=max_error,
        feller_satisfied=feller,
        feller_ratio=feller_ratio,
        n_options=n,
        success=best_result.success,
        message=best_result.message,
        n_func_evals=total_nfev,
    )


def calibrate_heston_from_iv_surface(
    strikes: np.ndarray,
    ttms: np.ndarray,
    market_ivs: np.ndarray,
    spot: float,
    weights: Optional[np.ndarray] = None,
    r: float = 0.0,
    q: float = 0.0,
    **kwargs,
) -> CalibrationResult:
    """
    Convenience function to calibrate from arrays instead of tuples.
    
    Args:
        strikes: Array of strikes
        ttms: Array of times to maturity
        market_ivs: Array of market implied vols
        spot: Spot price
        weights: Optional array of weights (default: uniform)
        r, q: Rates
        **kwargs: Passed to calibrate_heston
    
    Returns:
        CalibrationResult
    """
    n = len(strikes)
    if weights is None:
        weights = np.ones(n)
    
    market_data = list(zip(strikes, ttms, market_ivs, weights))
    
    return calibrate_heston(market_data, spot, r, q, **kwargs)


# =============================================================================
# Synthetic test
# =============================================================================
def _test_calibration():
    """Test calibration on synthetic Heston data."""
    from .heston_pricer import heston_implied_vol
    
    # True parameters
    true_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
    }
    
    spot = 100.0
    r, q = 0.05, 0.0
    
    # Generate synthetic surface
    strikes = np.linspace(80, 120, 9)
    ttms = np.array([0.25, 0.5, 1.0])
    
    market_data = []
    for ttm in ttms:
        for strike in strikes:
            iv = heston_implied_vol(
                spot, strike, ttm, r, q,
                true_params["v0"], true_params["kappa"],
                true_params["theta"], true_params["xi"], true_params["rho"]
            )
            if iv is not None:
                market_data.append((strike, ttm, iv, 1.0))
    
    print(f"Generated {len(market_data)} synthetic options")
    
    # Calibrate
    result = calibrate_heston(market_data, spot, r, q, n_starts=3, verbose=True)
    
    print("\nCalibration Result:")
    print(f"  v0:    {result.v0:.4f} (true: {true_params['v0']:.4f})")
    print(f"  kappa: {result.kappa:.4f} (true: {true_params['kappa']:.4f})")
    print(f"  theta: {result.theta:.4f} (true: {true_params['theta']:.4f})")
    print(f"  xi:    {result.xi:.4f} (true: {true_params['xi']:.4f})")
    print(f"  rho:   {result.rho:.4f} (true: {true_params['rho']:.4f})")
    print(f"\n  RMSE: {result.rmse*100:.4f}%")
    print(f"  Feller: {result.feller_satisfied} (ratio: {result.feller_ratio:.2f})")
    print(f"  Success: {result.success}")
    
    return result


if __name__ == "__main__":
    _test_calibration()

