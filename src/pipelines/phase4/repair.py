"""
Phase 4: Arbitrage-Free Surface Repair via Convex Optimization

This module implements a Quadratic Program (QP) to repair option prices, projecting
them onto an arbitrage-consistent polytope while respecting bid-ask bounds.

Mathematical Foundation (from vol_fitter.pdf Section 7):
------------------------------------------------------------
Decision Variables:
    C_tilde[i,j] = adjusted call price for strike i, expiration j

Objective Function:
    min  Σ w[i,j] * (C_tilde[i,j] - C_mid[i,j])^2
    where w[i,j] = 1 / spread[i,j] (tighter spreads = more trusted)

Linear Constraints:
    1. Bid-Ask Bounds:     bid[i,j] <= C_tilde[i,j] <= ask[i,j]
    2. Monotonicity:       C_tilde[i,j] >= C_tilde[i+1,j]  (calls decrease with strike)
    3. Convexity:          C_tilde[i-1,j] - 2*C_tilde[i,j] + C_tilde[i+1,j] >= 0
    4. Calendar Spread:    C_tilde[i,j] <= C_tilde[i,j+1]  (longer maturity >= shorter)

Infeasibility Handling:
    When constraints are inconsistent, introduce slack variables s[i,j] >= 0
    and penalize: min ... + λ * Σ s[i,j]
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Numerical constants
EPS = 1e-8  # Small epsilon for numerical stability
MIN_SPREAD = 0.01  # Minimum spread for weight calculation


@dataclass
class RepairResult:
    """Container for repair outputs and diagnostics."""
    
    # Repaired DataFrame with adjusted prices
    repaired_df: pd.DataFrame
    
    # Per (quote_date, expiration) summary statistics
    summary_df: pd.DataFrame
    
    # Detailed adjustments per option
    adjustments_df: pd.DataFrame
    
    # Feasibility report (slack variable usage)
    feasibility_df: pd.DataFrame
    
    # Overall metrics
    total_options: int = 0
    total_adjusted: int = 0
    mean_adjustment: float = 0.0
    max_adjustment: float = 0.0
    pct_within_bidask: float = 0.0
    solver_status: str = ""


@dataclass
class GridSpec:
    """Specification for the strike/expiration grid for a single quote_date."""
    
    quote_date: str
    strikes: np.ndarray  # Sorted unique strikes
    expirations: np.ndarray  # Sorted unique expirations (as strings)
    strike_to_idx: Dict[float, int] = field(default_factory=dict)
    exp_to_idx: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self.strike_to_idx = {k: i for i, k in enumerate(self.strikes)}
        self.exp_to_idx = {e: j for j, e in enumerate(self.expirations)}
    
    @property
    def n_strikes(self) -> int:
        return len(self.strikes)
    
    @property
    def n_expirations(self) -> int:
        return len(self.expirations)


def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    """
    Step 1: Data Preparation
    -------------------------
    Load snapshot parquet, filter to calls only, exclude same-day expiry (TTM=0).
    We only repair calls; puts can be derived via put-call parity.
    """
    df = pd.read_parquet(input_path)
    
    # Filter to calls only (as per PDF Section 2.3)
    df = df[df["option_type"] == "call"].copy()
    
    # Exclude same-day expiry (TTM = 0) as they have no time value to adjust
    df = df[df["ttm_years"] > 0].copy()
    
    # Ensure required columns exist
    required_cols = ["quote_date", "expiration", "strike", "bid", "ask", "mid", "spot"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with missing critical values
    df = df.dropna(subset=["bid", "ask", "mid", "strike", "expiration"])
    
    # Ensure bid <= ask (sanity filter)
    df = df[df["bid"] <= df["ask"] + EPS].copy()
    
    # Convert dates to string format for consistent grouping
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.strftime("%Y-%m-%d")
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")
    
    # Sort for consistent ordering
    df = df.sort_values(["quote_date", "expiration", "strike"]).reset_index(drop=True)
    
    return df


def build_grid_spec(df: pd.DataFrame, quote_date: str) -> GridSpec:
    """
    Build a grid specification for a single quote_date.
    Maps strikes and expirations to matrix indices.
    """
    subset = df[df["quote_date"] == quote_date]
    strikes = np.sort(subset["strike"].unique())
    expirations = np.sort(subset["expiration"].unique())
    
    return GridSpec(
        quote_date=quote_date,
        strikes=strikes,
        expirations=expirations,
    )


def build_price_matrices(
    df: pd.DataFrame,
    grid: GridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 2 (partial): Build price matrices for a single quote_date.
    
    Returns:
        mid_matrix: Mid prices [n_strikes x n_expirations], NaN where missing
        bid_matrix: Bid prices
        ask_matrix: Ask prices
        weight_matrix: Weights = 1/spread (liquidity-aware)
    """
    n_k, n_t = grid.n_strikes, grid.n_expirations
    
    # Initialize with NaN to mark missing data
    mid_matrix = np.full((n_k, n_t), np.nan)
    bid_matrix = np.full((n_k, n_t), np.nan)
    ask_matrix = np.full((n_k, n_t), np.nan)
    weight_matrix = np.zeros((n_k, n_t))
    
    subset = df[df["quote_date"] == grid.quote_date]
    
    for _, row in subset.iterrows():
        k_idx = grid.strike_to_idx.get(row["strike"])
        t_idx = grid.exp_to_idx.get(row["expiration"])
        
        if k_idx is None or t_idx is None:
            continue
        
        mid_matrix[k_idx, t_idx] = row["mid"]
        bid_matrix[k_idx, t_idx] = row["bid"]
        ask_matrix[k_idx, t_idx] = row["ask"]
        
        # Weight = 1 / spread (tighter spreads = more trusted)
        spread = max(row["ask"] - row["bid"], MIN_SPREAD)
        weight_matrix[k_idx, t_idx] = 1.0 / spread
    
    return mid_matrix, bid_matrix, ask_matrix, weight_matrix


# NOTE: Legacy function - not used in current implementation
# Current implementation uses repair_single_expiration() instead
# Kept for reference but not called anywhere
def _construct_qp_problem_legacy(
    mid_matrix: np.ndarray,
    bid_matrix: np.ndarray,
    ask_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    slack_penalty: float = 1000.0,
    use_slack: bool = False,
) -> Tuple[cp.Problem, cp.Variable, Optional[cp.Variable], Optional[cp.Variable], Optional[cp.Variable]]:
    """
    Step 2 & 3: Build QP Variables, Weights, and Constraints
    ----------------------------------------------------------
    
    Constructs the convex optimization problem for surface repair.
    
    Args:
        mid_matrix: Mid prices [n_strikes x n_expirations]
        bid_matrix: Bid prices
        ask_matrix: Ask prices
        weight_matrix: Weights for objective
        slack_penalty: Penalty λ for slack variables
        use_slack: Whether to include slack variables for infeasibility handling
    
    Returns:
        problem: cvxpy Problem object
        C_tilde: Decision variables (adjusted prices)
        slack_mono: Slack for monotonicity (if use_slack)
        slack_conv: Slack for convexity (if use_slack)
        slack_cal: Slack for calendar (if use_slack)
    """
    n_k, n_t = mid_matrix.shape
    
    # =========================================================================
    # Step 2: Build Decision Variables
    # =========================================================================
    # C_tilde[i,j] = adjusted call price for strike i, expiration j
    C_tilde = cp.Variable((n_k, n_t), name="C_tilde")
    
    # Mask for valid (non-NaN) entries
    valid_mask = ~np.isnan(mid_matrix)
    
    # =========================================================================
    # Objective Function: Minimize weighted squared deviations from mid-prices
    # =========================================================================
    # min Σ w[i,j] * (C_tilde[i,j] - C_mid[i,j])^2
    
    # Replace NaN with 0 for computation (masked by weight=0 anyway)
    mid_clean = np.nan_to_num(mid_matrix, nan=0.0)
    weight_clean = np.where(valid_mask, weight_matrix, 0.0)
    
    # Weighted squared error objective
    diff = C_tilde - mid_clean
    objective_terms = cp.multiply(weight_clean, cp.square(diff))
    objective = cp.sum(objective_terms)
    
    # =========================================================================
    # Step 3: Construct Constraints
    # =========================================================================
    constraints = []
    
    # Count constraints for slack variable sizing
    # NOTE: Monotonicity is always hard (no slack) - only convexity and calendar get slack
    n_conv_constraints = 0
    n_cal_constraints = 0
    
    # Pre-count constraints
    for j in range(n_t):
        valid_strikes_j = [i for i in range(n_k) if valid_mask[i, j]]
        n_valid = len(valid_strikes_j)
        n_conv_constraints += max(0, n_valid - 2)
    
    for i in range(n_k):
        valid_exps_i = [j for j in range(n_t) if valid_mask[i, j]]
        n_cal_constraints += max(0, len(valid_exps_i) - 1)
    
    # Initialize slack variables if needed (only for convexity and calendar)
    slack_mono = None  # Always None - monotonicity is always hard
    slack_conv = None
    slack_cal = None
    
    if use_slack:
        if n_conv_constraints > 0:
            slack_conv = cp.Variable(n_conv_constraints, nonneg=True, name="slack_conv")
        if n_cal_constraints > 0:
            slack_cal = cp.Variable(n_cal_constraints, nonneg=True, name="slack_cal")
    
    # -------------------------------------------------------------------------
    # Constraint 0: Non-negativity for all prices (call prices >= 0)
    # This prevents unbounded solutions for invalid entries
    # -------------------------------------------------------------------------
    constraints.append(C_tilde >= 0)
    
    # -------------------------------------------------------------------------
    # Constraint 1: Bid-Ask Bounds for valid entries
    # bid[i,j] <= C_tilde[i,j] <= ask[i,j]
    # -------------------------------------------------------------------------
    for i in range(n_k):
        for j in range(n_t):
            if valid_mask[i, j]:
                # Lower bound: C_tilde >= bid
                constraints.append(C_tilde[i, j] >= bid_matrix[i, j])
                # Upper bound: C_tilde <= ask
                constraints.append(C_tilde[i, j] <= ask_matrix[i, j])
            else:
                # Fix invalid entries to 0 to prevent unboundedness
                constraints.append(C_tilde[i, j] == 0)
    
    # -------------------------------------------------------------------------
    # Constraint 2: Monotonicity (calls decrease with strike) - ALWAYS HARD
    # C_tilde[i,j] >= C_tilde[i+1,j] for all adjacent valid strikes in maturity j
    # NOTE: Monotonicity is a fundamental constraint that should NEVER be relaxed.
    # Unlike convexity/calendar which can be infeasible due to bid-ask bounds,
    # monotonicity can always be satisfied within bid-ask if we adjust correctly.
    # -------------------------------------------------------------------------
    for j in range(n_t):
        # Get valid strike indices for this expiration
        valid_strikes_j = [i for i in range(n_k) if valid_mask[i, j]]
        
        for k in range(len(valid_strikes_j) - 1):
            i_curr = valid_strikes_j[k]
            i_next = valid_strikes_j[k + 1]
            # ALWAYS strict: C_tilde[i,j] >= C_tilde[i+1,j]
            constraints.append(C_tilde[i_curr, j] >= C_tilde[i_next, j])
    
    # -------------------------------------------------------------------------
    # Constraint 3: Convexity (non-negative discrete second difference)
    # C_tilde[i-1,j] - 2*C_tilde[i,j] + C_tilde[i+1,j] >= 0 for interior strikes
    # -------------------------------------------------------------------------
    conv_idx = 0
    for j in range(n_t):
        valid_strikes_j = [i for i in range(n_k) if valid_mask[i, j]]
        
        for k in range(1, len(valid_strikes_j) - 1):
            i_prev = valid_strikes_j[k - 1]
            i_curr = valid_strikes_j[k]
            i_next = valid_strikes_j[k + 1]
            
            # Second difference: C[i-1] - 2*C[i] + C[i+1] >= 0
            second_diff = C_tilde[i_prev, j] - 2 * C_tilde[i_curr, j] + C_tilde[i_next, j]
            
            if use_slack and slack_conv is not None:
                # Relaxed: second_diff + slack >= 0
                constraints.append(second_diff + slack_conv[conv_idx] >= 0)
                conv_idx += 1
            else:
                # Strict: second_diff >= 0
                constraints.append(second_diff >= 0)
    
    # -------------------------------------------------------------------------
    # Constraint 4: Calendar Spread (longer maturity >= shorter maturity)
    # C_tilde[i,j] <= C_tilde[i,j+1] for same strike across adjacent expirations
    # -------------------------------------------------------------------------
    cal_idx = 0
    for i in range(n_k):
        # Get valid expiration indices for this strike
        valid_exps_i = [j for j in range(n_t) if valid_mask[i, j]]
        
        for k in range(len(valid_exps_i) - 1):
            j_short = valid_exps_i[k]
            j_long = valid_exps_i[k + 1]
            
            if use_slack and slack_cal is not None:
                # Relaxed: C_tilde[i,j_short] <= C_tilde[i,j_long] + slack
                constraints.append(C_tilde[i, j_short] <= C_tilde[i, j_long] + slack_cal[cal_idx])
                cal_idx += 1
            else:
                # Strict: C_tilde[i,j_short] <= C_tilde[i,j_long]
                constraints.append(C_tilde[i, j_short] <= C_tilde[i, j_long])
    
    # =========================================================================
    # Add slack penalty to objective if using slack variables
    # NOTE: Only convexity and calendar have slack - monotonicity is always hard
    # =========================================================================
    if use_slack:
        if slack_conv is not None:
            objective = objective + slack_penalty * cp.sum(slack_conv)
        if slack_cal is not None:
            objective = objective + slack_penalty * cp.sum(slack_cal)
    
    # Build and return the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    return problem, C_tilde, slack_mono, slack_conv, slack_cal


# NOTE: Legacy function - not used in current implementation
# Current implementation uses repair_single_expiration() instead
# Kept for reference but not called anywhere
def _solve_repair_qp_legacy(
    mid_matrix: np.ndarray,
    bid_matrix: np.ndarray,
    ask_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    slack_penalty: float = 1000.0,
) -> Tuple[np.ndarray, str, Dict[str, np.ndarray]]:
    """
    Step 4: Solve and Handle Infeasibility
    ----------------------------------------
    
    First attempts strict QP. If infeasible, retries with slack variables.
    
    Returns:
        repaired_prices: Adjusted price matrix
        status: Solver status string
        slack_usage: Dict of slack variable values (if used)
    """
    # First attempt: solve strict QP (no slack)
    problem, C_tilde, _, _, _ = _construct_qp_problem_legacy(
        mid_matrix, bid_matrix, ask_matrix, weight_matrix,
        slack_penalty=slack_penalty,
        use_slack=False,
    )
    
    # Try multiple solvers in order of preference
    def try_solve(prob):
        """Try solving with multiple solvers."""
        # Try Clarabel first (modern, robust QP solver)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return True
        except Exception:
            pass
        
        # Try OSQP (fast for QP)
        try:
            prob.solve(
                solver=cp.OSQP,
                verbose=False,
                max_iter=100000,
                eps_abs=1e-4,
                eps_rel=1e-4,
            )
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return True
        except Exception:
            pass
        
        # Try SCS as fallback (handles large problems)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return True
        except Exception:
            pass
        
        return False
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solved = try_solve(problem)
    
    if solved and problem.status in ["optimal", "optimal_inaccurate"]:
        return C_tilde.value, problem.status, {}
    
    # =========================================================================
    # Infeasibility detected: Retry with slack variables
    # =========================================================================
    problem, C_tilde, slack_mono, slack_conv, slack_cal = _construct_qp_problem_legacy(
        mid_matrix, bid_matrix, ask_matrix, weight_matrix,
        slack_penalty=slack_penalty,
        use_slack=True,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solved = try_solve(problem)
    
    if not solved:
        return mid_matrix.copy(), f"failed: {problem.status}", {}
    
    slack_usage = {}
    if slack_mono is not None and slack_mono.value is not None:
        slack_usage["monotonicity"] = slack_mono.value
    if slack_conv is not None and slack_conv.value is not None:
        slack_usage["convexity"] = slack_conv.value
    if slack_cal is not None and slack_cal.value is not None:
        slack_usage["calendar"] = slack_cal.value
    
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return C_tilde.value, f"{problem.status}_with_slack", slack_usage
    else:
        # Fallback: return mid prices unchanged
        return mid_matrix.copy(), f"failed: {problem.status}", slack_usage


def repair_single_expiration(
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    weights: np.ndarray,
    slack_penalty: float = 1000.0,
    smile_preservation_weight: float = 0.1,
    spot: Optional[float] = None,
    strikes: Optional[np.ndarray] = None,
    ttm: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Repair a single expiration (1D array of strikes).
    
    This solves a much smaller problem: only monotonicity + convexity for one expiration.
    No calendar constraints since we're only looking at one expiration.
    
    Args:
        mid: Mid prices for all strikes [n_strikes]
        bid: Bid prices
        ask: Ask prices  
        weights: Weights for objective
        slack_penalty: Penalty for slack variables
        smile_preservation_weight: Weight for smile preservation penalty (default: 0.1)
        spot: Spot price (optional, for smile preservation)
        strikes: Strike prices (optional, for smile preservation)
        ttm: Time to maturity (optional, for smile preservation)
    
    Returns:
        repaired: Repaired prices
        status: Solver status
    """
    n = len(mid)
    valid = ~np.isnan(mid)
    valid_indices = np.where(valid)[0]
    n_valid = len(valid_indices)
    
    if n_valid < 2:
        return mid.copy(), "insufficient_data"
    
    # Decision variable for valid entries only
    C = cp.Variable(n_valid, name="C")
    
    # Extract valid data
    mid_valid = mid[valid]
    bid_valid = bid[valid]
    ask_valid = ask[valid]
    w_valid = weights[valid]
    
    # Objective: minimize weighted squared deviation
    objective = cp.sum(cp.multiply(w_valid, cp.square(C - mid_valid)))
    
    # Add smile preservation penalty: penalize large relative price adjustments
    # This helps preserve smile shape by discouraging disproportionate adjustments
    # The penalty is proportional to (adjustment/mid)^2, which approximates IV change
    if smile_preservation_weight > 0 and spot is not None and strikes is not None:
        strikes_valid = strikes[valid] if strikes is not None else None
        if strikes_valid is not None and len(strikes_valid) == n_valid:
            # Compute moneyness-based penalty weights
            # OTM options (higher moneyness) get higher penalty to prevent steepening
            moneyness = strikes_valid / spot
            # Penalty weight increases for OTM options to preserve smile
            # Use numpy for pre-computation, then apply in objective
            otm_factor = np.maximum(0, moneyness - 1.0) ** 2
            smile_weights = 1.0 + smile_preservation_weight * otm_factor
            
            # Add penalty term: smile_weights * (relative_adjustment)^2
            # Use element-wise division with small epsilon to avoid division by zero
            mid_valid_safe = np.maximum(mid_valid, 1e-6)
            relative_adjustment = (C - mid_valid) / mid_valid_safe
            smile_penalty = cp.sum(cp.multiply(smile_weights, cp.square(relative_adjustment)))
            objective = objective + smile_preservation_weight * smile_penalty
    
    constraints = []
    
    # Bid-ask bounds (HARD)
    constraints.append(C >= bid_valid)
    constraints.append(C <= ask_valid)
    
    # Monotonicity (HARD): C[i] >= C[i+1]
    for i in range(n_valid - 1):
        constraints.append(C[i] >= C[i + 1])
    
    # Convexity with slack: C[i-1] - 2*C[i] + C[i+1] + slack >= 0
    n_conv = max(0, n_valid - 2)
    if n_conv > 0:
        slack_conv = cp.Variable(n_conv, nonneg=True)
        for i in range(1, n_valid - 1):
            second_diff = C[i - 1] - 2 * C[i] + C[i + 1]
            constraints.append(second_diff + slack_conv[i - 1] >= 0)
        objective = objective + slack_penalty * cp.sum(slack_conv)
    
    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000)
        except Exception:
            return mid.copy(), "solver_failed"
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return mid.copy(), f"failed: {problem.status}"
    
    # Map back to full array
    repaired = mid.copy()
    repaired[valid] = C.value
    
    return repaired, problem.status


def repair_single_quote_date(
    df: pd.DataFrame,
    quote_date: str,
    slack_penalty: float = 1000.0,
) -> Tuple[pd.DataFrame, Dict, Dict[str, np.ndarray]]:
    """
    Repair the surface for a single quote_date.
    
    Uses a per-expiration approach for more reliable results:
    - Each expiration is repaired independently (monotonicity + convexity only)
    - This avoids the complexity of calendar constraints across sparse grids
    - Results in smaller problems that solvers handle accurately
    
    Returns:
        repaired_subset: DataFrame with repaired prices for this quote_date
        metrics: Dict of repair metrics
        slack_usage: Dict of slack variable values
    """
    grid = build_grid_spec(df, quote_date)
    
    if grid.n_strikes < 2 or grid.n_expirations < 1:
        # Not enough data to repair
        subset = df[df["quote_date"] == quote_date].copy()
        subset["price_repaired"] = subset["mid"]
        subset["adjustment"] = 0.0
        subset["adjustment_pct"] = 0.0
        return subset, {"status": "insufficient_data", "n_adjusted": 0}, {}
    
    # Build price matrices
    mid_matrix, bid_matrix, ask_matrix, weight_matrix = build_price_matrices(df, grid)
    
    # Get spot price and TTM for smile preservation
    subset_day = df[df["quote_date"] == quote_date]
    spot = subset_day["spot"].iloc[0] if len(subset_day) > 0 else None
    
    # Repair each expiration independently
    repaired_matrix = np.full_like(mid_matrix, np.nan)
    statuses = []
    
    for j in range(grid.n_expirations):
        # Get TTM for this expiration
        exp = grid.expirations[j]
        exp_data = subset_day[subset_day["expiration"] == exp]
        ttm = exp_data["ttm_years"].iloc[0] if len(exp_data) > 0 else None
        
        repaired_col, status = repair_single_expiration(
            mid_matrix[:, j],
            bid_matrix[:, j],
            ask_matrix[:, j],
            weight_matrix[:, j],
            slack_penalty=slack_penalty,
            smile_preservation_weight=0.1,  # Moderate weight to preserve smile
            spot=spot,
            strikes=grid.strikes,
            ttm=ttm,
        )
        repaired_matrix[:, j] = repaired_col
        statuses.append(status)
    
    # Determine overall status
    if all(s == "optimal" for s in statuses):
        overall_status = "optimal"
    elif all(s in ["optimal", "optimal_inaccurate", "insufficient_data"] for s in statuses):
        overall_status = "optimal_per_expiration"
    else:
        overall_status = "mixed"
    
    # Map repaired prices back to DataFrame
    subset = df[df["quote_date"] == quote_date].copy()
    repaired_prices = []
    
    for _, row in subset.iterrows():
        k_idx = grid.strike_to_idx.get(row["strike"])
        t_idx = grid.exp_to_idx.get(row["expiration"])
        
        if k_idx is not None and t_idx is not None and not np.isnan(repaired_matrix[k_idx, t_idx]):
            repaired_prices.append(repaired_matrix[k_idx, t_idx])
        else:
            repaired_prices.append(row["mid"])
    
    subset["price_repaired"] = repaired_prices
    
    # =========================================================================
    # POST-PROCESS: Fix solver numerical issues
    # The solver may return 'optimal_inaccurate' with extreme constraint violations.
    # We apply minimal clamping to fix only the egregious cases.
    # =========================================================================
    
    # 1. Enforce non-negativity (call prices >= 0)
    subset["price_repaired"] = subset["price_repaired"].clip(lower=0.0)
    
    # 2. For prices WILDLY outside bid-ask (more than 10x spread), clamp to bounds
    # This catches solver failures while preserving legitimate adjustments
    spread = (subset["ask"] - subset["bid"]).clip(lower=0.01)
    deviation_from_mid = (subset["price_repaired"] - subset["mid"]).abs()
    is_extreme = deviation_from_mid > 10.0 * spread
    
    # Only clamp extreme cases to bid-ask
    subset.loc[is_extreme & (subset["price_repaired"] < subset["bid"]), "price_repaired"] = \
        subset.loc[is_extreme & (subset["price_repaired"] < subset["bid"]), "bid"]
    subset.loc[is_extreme & (subset["price_repaired"] > subset["ask"]), "price_repaired"] = \
        subset.loc[is_extreme & (subset["price_repaired"] > subset["ask"]), "ask"]
    
    # 3. For any remaining prices outside bid-ask by more than $1, clamp
    # (Small violations are numerical precision, large ones are solver failures)
    below_bid = subset["price_repaired"] < subset["bid"] - 1.0
    above_ask = subset["price_repaired"] > subset["ask"] + 1.0
    subset.loc[below_bid, "price_repaired"] = subset.loc[below_bid, "bid"]
    subset.loc[above_ask, "price_repaired"] = subset.loc[above_ask, "ask"]
    
    subset["adjustment"] = subset["price_repaired"] - subset["mid"]
    subset["adjustment_pct"] = (subset["adjustment"] / subset["mid"].clip(lower=EPS)) * 100
    
    # Compute metrics
    n_adjusted = (np.abs(subset["adjustment"]) > EPS).sum()
    within_bidask = (
        (subset["price_repaired"] >= subset["bid"] - EPS) &
        (subset["price_repaired"] <= subset["ask"] + EPS)
    ).mean() * 100
    
    metrics = {
        "quote_date": quote_date,
        "status": overall_status,
        "n_options": len(subset),
        "n_adjusted": int(n_adjusted),
        "mean_adjustment": float(subset["adjustment"].abs().mean()),
        "max_adjustment": float(subset["adjustment"].abs().max()),
        "pct_within_bidask": float(within_bidask),
    }
    
    # No slack usage tracking in per-expiration mode (slack is internal to each solve)
    return subset, metrics, {}


def repair_surface(
    input_path: Path,
    output_dir: Path,
    slack_penalty: float = 1000.0,
    apply_calendar_fix: bool = True,
    calendar_grid_size: int = 200,
    calendar_moneyness_bounds: Tuple[float, float] = (0.6, 1.4),
    apply_convexity_cleanup: bool = False,
) -> RepairResult:
    """
    Main entry point: Repair the entire surface across all quote_dates.
    
    Two-phase approach:
    - Phase 4a: Per-expiration repair (monotonicity + convexity within each expiration)
    - Phase 4b: Calendar reconciliation (isotonic regression across maturities)
    
    Args:
        input_path: Path to input parquet file
        output_dir: Directory for output files
        slack_penalty: Penalty λ for slack variables (higher = stricter)
        apply_calendar_fix: Whether to apply Phase 4b calendar reconciliation
        calendar_grid_size: Number of moneyness grid points for calendar fix
        calendar_moneyness_bounds: (min, max) moneyness range for calendar fix
    
    Returns:
        RepairResult with repaired data and diagnostics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Data Preparation
    # =========================================================================
    df = load_and_prepare_data(input_path)
    quote_dates = sorted(df["quote_date"].unique())
    
    print(f"Repairing surface for {len(quote_dates)} quote dates...")
    print(f"Total options: {len(df)}")
    
    # =========================================================================
    # Step 4: Solve for each quote_date
    # =========================================================================
    repaired_dfs = []
    all_metrics = []
    all_slack_usage = []
    
    for qd in quote_dates:
        repaired_subset, metrics, slack_usage = repair_single_quote_date(
            df, qd, slack_penalty=slack_penalty
        )
        repaired_dfs.append(repaired_subset)
        all_metrics.append(metrics)
        
        # Record slack usage for feasibility report
        for constraint_type, values in slack_usage.items():
            if values is not None and np.any(values > EPS):
                all_slack_usage.append({
                    "quote_date": qd,
                    "constraint_type": constraint_type,
                    "slack_sum": float(np.sum(values)),
                    "slack_max": float(np.max(values)),
                    "n_active": int(np.sum(values > EPS)),
                })
    
    # Combine all repaired data (Phase 4a complete)
    repaired_df = pd.concat(repaired_dfs, ignore_index=True)
    
    print(f"\nPhase 4a (per-expiration) complete.")
    
    # =========================================================================
    # Phase 4b: Calendar Reconciliation via Isotonic Regression
    # =========================================================================
    if apply_calendar_fix:
        from .calendar_reconcile import apply_calendar_reconciliation
        
        cal_result = apply_calendar_reconciliation(
            repaired_df,
            grid_size=calendar_grid_size,
            moneyness_bounds=calendar_moneyness_bounds,
        )
        repaired_df = cal_result.reconciled_df
        
        # =====================================================================
        # Phase 4c (Optional): Re-run per-expiration convexity repair
        # This restores convexity but may re-introduce calendar violations.
        # Use apply_convexity_cleanup=True if density extraction is needed.
        # For Heston calibration, skip this step (calendar > convexity).
        # =====================================================================
        if apply_convexity_cleanup:
            print("\n=== Phase 4c: Convexity Cleanup (may increase calendar violations) ===")
            from .calendar_reconcile import count_all_violations
            
            pre_cleanup = count_all_violations(repaired_df)
            print(f"Before: mono={pre_cleanup['mono']}, conv={pre_cleanup['conv']}, cal={pre_cleanup['cal']}")
            
            # Re-run per-expiration repair with focus on convexity
            cleanup_dfs = []
            for qd in repaired_df["quote_date"].unique():
                qd_df = repaired_df[repaired_df["quote_date"] == qd].copy()
                
                for exp in qd_df["expiration"].unique():
                    exp_mask = qd_df["expiration"] == exp
                    exp_df = qd_df[exp_mask].sort_values("strike")
                    
                    if len(exp_df) < 3:
                        continue
                    
                    mid = exp_df["price_repaired"].to_numpy()
                    bid = exp_df["bid"].to_numpy()
                    ask = exp_df["ask"].to_numpy()
                    weights = 1.0 / np.maximum(ask - bid, 0.01)
                    
                    # Light convexity repair (use existing function)
                    repaired, status = repair_single_expiration(mid, bid, ask, weights, slack_penalty)
                    
                    qd_df.loc[exp_df.index, "price_repaired"] = repaired
                
                cleanup_dfs.append(qd_df)
            
            repaired_df = pd.concat(cleanup_dfs, ignore_index=True)
            
            # Update adjustment column
            repaired_df["adjustment"] = repaired_df["price_repaired"] - repaired_df["mid"]
            repaired_df["adjustment_pct"] = (repaired_df["adjustment"] / repaired_df["mid"].clip(lower=EPS)) * 100
            
            post_cleanup = count_all_violations(repaired_df)
            print(f"After:  mono={post_cleanup['mono']}, conv={post_cleanup['conv']}, cal={post_cleanup['cal']}")
    
    # =========================================================================
    # Step 5: Compute summary statistics
    # =========================================================================
    summary_df = pd.DataFrame(all_metrics)
    adjustments_df = repaired_df[["quote_date", "expiration", "strike", "mid", 
                                   "price_repaired", "adjustment", "adjustment_pct", 
                                   "bid", "ask"]].copy()
    feasibility_df = pd.DataFrame(all_slack_usage) if all_slack_usage else pd.DataFrame()
    
    # Overall metrics
    total_options = len(repaired_df)
    total_adjusted = (np.abs(repaired_df["adjustment"]) > EPS).sum()
    mean_adjustment = repaired_df["adjustment"].abs().mean()
    max_adjustment = repaired_df["adjustment"].abs().max()
    pct_within_bidask = (
        (repaired_df["price_repaired"] >= repaired_df["bid"] - EPS) &
        (repaired_df["price_repaired"] <= repaired_df["ask"] + EPS)
    ).mean() * 100
    
    # =========================================================================
    # Step 6: Export outputs
    # =========================================================================
    repaired_df.to_parquet(output_dir / "repaired_options.parquet", index=False)
    summary_df.to_csv(output_dir / "repair_summary.csv", index=False)
    adjustments_df.to_csv(output_dir / "adjustments.csv", index=False)
    if not feasibility_df.empty:
        feasibility_df.to_csv(output_dir / "feasibility_report.csv", index=False)
    
    # Generate adjustment heatmap
    _plot_adjustment_heatmap(adjustments_df, output_dir)
    
    print(f"\nRepair complete:")
    print(f"  Total options: {total_options}")
    print(f"  Adjusted: {total_adjusted} ({100*total_adjusted/total_options:.1f}%)")
    print(f"  Mean adjustment: ${mean_adjustment:.4f}")
    print(f"  Max adjustment: ${max_adjustment:.4f}")
    print(f"  Within bid-ask: {pct_within_bidask:.1f}%")
    
    return RepairResult(
        repaired_df=repaired_df,
        summary_df=summary_df,
        adjustments_df=adjustments_df,
        feasibility_df=feasibility_df,
        total_options=total_options,
        total_adjusted=total_adjusted,
        mean_adjustment=mean_adjustment,
        max_adjustment=max_adjustment,
        pct_within_bidask=pct_within_bidask,
        solver_status="mixed" if len(set(summary_df["status"])) > 1 else summary_df["status"].iloc[0],
    )


def _plot_adjustment_heatmap(adjustments_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Visualize where repairs happened by strike bucket and expiration.
    """
    if adjustments_df.empty:
        return
    
    df = adjustments_df.copy()
    
    # Bucket strikes (10-point buckets)
    bucket_width = 10.0
    df["strike_bucket"] = (df["strike"] / bucket_width).round() * bucket_width
    
    # Aggregate mean absolute adjustment
    pivot = df.groupby(["expiration", "strike_bucket"])["adjustment"].apply(
        lambda x: np.abs(x).mean()
    ).unstack(fill_value=0)
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.3), max(6, pivot.shape[0] * 0.25)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd")
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s:.0f}" for s in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    
    ax.set_xlabel("Strike Bucket")
    ax.set_ylabel("Expiration")
    ax.set_title("Mean Absolute Adjustment by Strike/Expiration")
    
    fig.colorbar(im, ax=ax, label="Mean |Adjustment| ($)")
    fig.tight_layout()
    fig.savefig(output_dir / "adjustment_heatmap.png", dpi=150)
    plt.close(fig)

