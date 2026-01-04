import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort:skip

# Import shared IV computation
from pipelines.common.iv_computation import implied_vol, _bs_price, MIN_SIGMA

EPS = 1e-12


@dataclass
class SviParams:
    a: float # overall variance level
    b: float # slope of the variance curve
    rho: float # skew of the variance curve
    m: float # horizontal shift
    sigma: float # curvature around minimum
    success: bool
    message: str


# _bs_price and implied_vol are now imported from pipelines.common.iv_computation
# This ensures consistency across all phases


def extract_iv(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Compute implied vol, log-moneyness, weights for valid rows."""
    # Focus on calls; puts can be recovered via parity if needed elsewhere
    df = df[df["option_type"] == "call"]

    # Keep only the valid rows
    keep = (
        df["mid"].notna()
        & df["spot"].notna()
        & (df["strike"] > 0)
        & (df["ttm_years"] > 0)
    )
    work = df.loc[keep, ["quote_date", "expiration", "option_type", "strike", "mid", "spot", "ttm_years", "bid", "ask"]].copy()
    work["log_moneyness"] = np.log(work["spot"] / work["strike"])

    spreads = (work["ask"] - work["bid"]).clip(lower=EPS)
    work["weight"] = 1.0 / spreads

    ivs: List[Optional[float]] = []
    for _, row in work.iterrows():
        ivs.append(
            implied_vol(
                price=row["mid"],
                spot=row["spot"],
                strike=row["strike"],
                ttm=row["ttm_years"],
                r=r,
                q=q,
                is_call=row["option_type"] == "call",
            )
        )
    work["iv"] = ivs
    work = work[work["iv"].notna()]
    return work


def svi_total_variance(params: np.ndarray, k: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma * sigma))


def fit_svi(log_m: np.ndarray, iv: np.ndarray, ttm: float, weights: np.ndarray) -> SviParams:
    """Fit SVI total variance; returns parameters and success flag."""
    if len(log_m) < 5:
        return SviParams(0, 0, 0, 0, 0, False, "insufficient points")

    y = (iv ** 2) * ttm
    k = log_m
    w = np.sqrt(weights)

    def resid(p):
        return (svi_total_variance(p, k) - y) * w

    init = np.array([0.01, 0.1, 0.0, 0.0, 0.1])
    bounds = (
        [ -1.0, 0.0, -0.999, -5.0, MIN_SIGMA],
        [  1.0, 5.0,  0.999,  5.0, 5.0],
    )
    res = least_squares(resid, init, bounds=bounds, xtol=1e-8, ftol=1e-8, verbose=0)
    if not res.success:
        return SviParams(*res.x, False, res.message)

    # Basic no-butterfly check: b > 0, |rho|<1, sigma>0
    a, b, rho, m, sigma = res.x
    ok = (b > 0) and (abs(rho) < 1) and (sigma > 0)
    return SviParams(a, b, rho, m, sigma, ok, res.message if ok else "svi params fail basic constraints")


def spline_fit(log_m: np.ndarray, iv: np.ndarray, weights: np.ndarray) -> UnivariateSpline:
    order = np.argsort(log_m)
    return UnivariateSpline(log_m[order], iv[order], w=np.sqrt(weights[order]), s=len(log_m))


def generate_grid(strikes: np.ndarray, spot: float, n: int = 50, pad: float = 0.15) -> np.ndarray:
    k_min, k_max = strikes.min(), strikes.max()
    span = k_max - k_min
    lower = max(1e-6, k_min - pad * span)
    upper = k_max + pad * span
    return np.linspace(lower, upper, n)


def evaluate_smile(
    strikes: np.ndarray,
    spot: float,
    ttm: float,
    svi_params: Optional[SviParams],
    spline: Optional[UnivariateSpline],
) -> Tuple[np.ndarray, np.ndarray]:
    log_m = np.log(spot / strikes)
    fitted_iv = np.full_like(strikes, np.nan, dtype=float)
    if svi_params and svi_params.success:
        a, b, rho, m, sigma = svi_params.a, svi_params.b, svi_params.rho, svi_params.m, svi_params.sigma
        total_var = svi_total_variance(np.array([a, b, rho, m, sigma]), log_m)
        fitted_iv = np.sqrt(np.maximum(total_var / ttm, MIN_SIGMA))
    elif spline is not None:
        fitted_iv = np.maximum(spline(log_m), MIN_SIGMA)
    return log_m, fitted_iv


def density_sign_proxy(strikes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Discrete second-diff sign as a proxy for density non-negativity."""
    if len(prices) < 3:
        return np.ones_like(prices, dtype=bool)
    second = np.diff(prices, n=2)
    # Pad ends as True to align length
    padded = np.concatenate([[True], second >= -1e-6, [True]])
    return padded


def fit_smiles(iv_df: pd.DataFrame, r: float, q: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit smiles per expiry; returns params table and pointwise observed-vs-fit."""
    params_rows: List[Dict] = []
    point_rows: List[Dict] = []

    for (quote_date, expiration), grp in iv_df.groupby(["quote_date", "expiration"]):
        spot = grp["spot"].iloc[0]
        ttm = grp["ttm_years"].iloc[0]
        k = grp["strike"].to_numpy()
        log_m = grp["log_moneyness"].to_numpy()
        iv = grp["iv"].to_numpy()
        w = grp["weight"].to_numpy()

        order = np.argsort(log_m)
        log_m_sorted = log_m[order]
        iv_sorted = iv[order]
        w_sorted = w[order]
        k_sorted = k[order]

        svi = fit_svi(log_m_sorted, iv_sorted, ttm, w_sorted)
        spline = None
        if not svi.success:
            spline = spline_fit(log_m_sorted, iv_sorted, w_sorted)

        # evaluate at observed points
        _, fitted_sorted = evaluate_smile(k_sorted, spot, ttm, svi, spline)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        fitted_at_obs = fitted_sorted[inv]
        rmse = float(np.sqrt(np.mean((fitted_at_obs - iv) ** 2)))

        params_rows.append(
            {
                "quote_date": quote_date,
                "expiration": expiration,
                "ttm_years": ttm,
                "spot": spot,
                "method": "svi" if svi.success else "spline",
                "rmse": rmse,
                "a": svi.a,
                "b": svi.b,
                "rho": svi.rho,
                "m": svi.m,
                "sigma": svi.sigma,
                "svi_success": svi.success,
                "svi_message": svi.message,
            }
        )

        for pos, row in enumerate(grp.itertuples(index=False)):
            point_rows.append(
                {
                    "quote_date": quote_date,
                    "expiration": expiration,
                    "strike": row.strike,
                    "log_moneyness": row.log_moneyness,
                    "option_type": row.option_type,
                    "iv_obs": row.iv,
                    "iv_fit": fitted_at_obs[pos],
                    "weight": row.weight,
                }
            )
    return pd.DataFrame(params_rows), pd.DataFrame(point_rows)


def build_grid_outputs(iv_df: pd.DataFrame, params_df: pd.DataFrame, grid_size: int = 50) -> pd.DataFrame:
    """Evaluate fitted smiles on a dense grid per expiry for later phases."""
    grid_rows: List[Dict] = []

    for (quote_date, expiration), grp in iv_df.groupby(["quote_date", "expiration"]):
        spot = grp["spot"].iloc[0]
        ttm = grp["ttm_years"].iloc[0]
        strikes = generate_grid(grp["strike"].to_numpy(), spot, n=grid_size)

        p_row = params_df[(params_df.quote_date == quote_date) & (params_df.expiration == expiration)].iloc[0]
        svi = SviParams(p_row.a, p_row.b, p_row.rho, p_row.m, p_row.sigma, p_row.svi_success, p_row.svi_message)
        spline = None
        if not svi.success:
            # refit spline quickly
            log_m = grp["log_moneyness"].to_numpy()
            iv = grp["iv"].to_numpy()
            w = grp["weight"].to_numpy()
            spline = spline_fit(log_m, iv, w)

        log_m_grid, iv_fit = evaluate_smile(strikes, spot, ttm, svi, spline)
        # approximate call prices using fitted IVs (call only proxy)
        prices = [
            _bs_price(spot, k, ttm, iv_fit[i], r=0.0, q=0.0, is_call=True) for i, k in enumerate(strikes)
        ]
        density_ok = density_sign_proxy(strikes, np.array(prices))

        for i, k in enumerate(strikes):
            grid_rows.append(
                {
                    "quote_date": quote_date,
                    "expiration": expiration,
                    "strike": k,
                    "log_moneyness": log_m_grid[i],
                    "iv_fit": iv_fit[i],
                    "density_nonnegative": bool(density_ok[i]),
                }
            )
    return pd.DataFrame(grid_rows)


def plot_smile(expiration: str, observed: pd.DataFrame, params: Dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(observed["log_moneyness"], observed["iv"], s=10, alpha=0.6, label="obs")
    ax.plot(observed["log_moneyness"], observed["iv_fit"], color="orange", label="fit")
    ax.set_title(f"Smile {expiration}")
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("implied vol")
    ax.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"smile_{expiration}.png")
    plt.close(fig)


def plot_rmse(params_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    params_df.sort_values("expiration", inplace=False).plot(
        x="expiration", y="rmse", kind="bar", ax=ax, legend=False, color="steelblue"
    )
    ax.set_ylabel("RMSE (vol)")
    ax.set_title("Fit RMSE by expiration")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "rmse_by_expiration.png")
    plt.close(fig)


def run_smoothing(
    input_path: Path,
    output_dir: Path,
    r: float,
    q: float,
    grid_size: int = 50,
    max_plots: int = 8,
) -> None:
    """Run smoothing: extract IVs, fit smiles, write reports, plot diagnostics."""

    # Read the input data
    df = pd.read_parquet(input_path)
    iv_df = extract_iv(df, r=r, q=q)

    # Fit the smiles
    params_df, point_df = fit_smiles(iv_df, r=r, q=q)

    # Build the grid outputs
    grid_df = build_grid_outputs(iv_df, params_df, grid_size=grid_size)

    # Write the outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(output_dir / "fit_params.csv", index=False)
    point_df.to_csv(output_dir / "observed_vs_fit.csv", index=False)
    grid_df.to_csv(output_dir / "grid_fit.csv", index=False)

    # Plot the smiles
    for i, ((qd, exp), grp) in enumerate(iv_df.groupby(["quote_date", "expiration"])):
        if i >= max_plots:
            break
        obs = point_df[(point_df.quote_date == qd) & (point_df.expiration == exp)].copy()
        params_row = params_df[(params_df.quote_date == qd) & (params_df.expiration == exp)].iloc[0]
        plot_smile(exp, obs.assign(iv=obs["iv_obs"]), params_row.to_dict(), output_dir)

    plot_rmse(params_df, output_dir)

    # Log the results
    print("Fit params (head):")
    print(params_df.head().to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


def main(argv: Optional[List[str]] = None) -> None:

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Phase 3: smooth implied vols with SVI/spline fallback.")
    parser.add_argument("--input", required=True, help="Parquet snapshot input path.")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and plots.")
    parser.add_argument("--rate", type=float, default=0.0, help="Risk-free rate (continuously compounded).")
    parser.add_argument("--dividend", type=float, default=0.0, help="Dividend yield (continuously compounded).")
    parser.add_argument("--grid-size", type=int, default=50, help="Points in strike grid for evaluation.")
    parser.add_argument("--max-plots", type=int, default=8, help="Max expirations to plot.")
    args = parser.parse_args(argv)

    # Run the smoothing
    run_smoothing(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        r=args.rate,
        q=args.dividend,
        grid_size=args.grid_size,
        max_plots=args.max_plots,
    )


if __name__ == "__main__":
    main()

