# Project Documentation

This document summarizes the pipeline we have built so far, why each component exists, and how it aligns with the mathematical goals (static no-arbitrage, smoothing, repair, and Heston calibration). Use this as a reference when drafting the LaTeX/math report and implementation write-up.

## Overview

Goal: ingest raw SPY option quotes (bid/ask), detect and summarize static arbitrage violations, fit smooth smiles (SVI/spline) to implied vols, and later repair the surface and calibrate Heston pre/post repair.

Core phases implemented:

- Phase 1: Normalize raw Cboe CSVs into structured Parquet snapshots (15:45 and EOD) with derived fields and basic sanity flags.
- Phase 2: Detect bid-ask–aware static arbitrage violations (monotonicity, convexity, calendar) and produce summaries/heatmaps.
- Phase 3: Extract implied vols, fit smooth smiles per expiry (SVI with spline fallback), and output diagnostics/plots.

Later phases (planned): repair via convex optimization within bid/ask intervals; Heston calibration on raw vs repaired surfaces.

## Phase 1: Normalization & Basic Sanity

Files: `scripts/phase1.sh`, `src/pipelines/phase1/normalize.py`, `src/pipelines/phase1/sanity_report.py`

What it does:

- Load raw daily Cboe CSVs, parse into a typed schema (`CboeRow`).
- Derive snapshot-specific bid/ask, mid, spot (underlying mid), intrinsic, moneyness/log-moneyness, time-to-expiry (days, years), and tag snapshot (`1545` or `eod`).
- Write combined Parquet per snapshot plus optional per-quote_date partitions.
- Sanity flags: invalid spread (bid>ask or negative), missing mid, mid below intrinsic.
- Summaries: per (quote_date, expiration) counts of invalid_spread, mid_missing, intrinsic_violations.

Why:

- Establish clean, typed, and derived features needed for detection, smoothing, and later repair/calibration (spot, TTE, moneyness).
- Basic filters reduce noise before stricter arbitrage checks.

Key equations:

- Mid: \( \text{mid} = (bid + ask)/2 \)
- Spot proxy: mid of underlying bid/ask
- Intrinsic: call \(\max(S-K,0)\), put \(\max(K-S,0)\)
- TTM (years): \( (\text{expiration} - \text{quote_date})/365 \)

Outputs:

- `data/processed/1545/options_1545.parquet`, `data/processed/eod/options_eod.parquet`
- Per-day Parquet partitions (optional)
- `sanity_summary_1545.csv`, `sanity_summary_eod.csv`

## Phase 2: Detection (Static No-Arbitrage)

Files: `scripts/phase2_detect.py`, `src/pipelines/phase2/detect.py`

Rules (bid-ask aware):

- Monotonicity: calls should decrease with strike; puts should increase. Use bid of higher strike vs ask of lower strike (best-case intervals) to flag definitive violations.
- Convexity (butterfly): discrete second-diff using best-case bid/ask slopes across three strikes; flags if left slope + epsilon < right slope.
- Calendar: longer-dated ask should not be below shorter-dated bid for same strike/type.

Flow:

- Load snapshot Parquet; keep calls/puts; sort by date/expiry/strike.
- Compute per-rule violation records; aggregate counts per (quote_date, expiration, option_type).
- Export CSVs: `summary.csv` (counts), `sample_violations.csv` (capped examples).
- Heatmaps (PNG): violation counts by expiration × strike_bucket per rule (aggregated over all quote dates for the dataset).

Why:

- Quantifies where the surface is inconsistent before repair.
- Bid-ask awareness avoids over-flagging due to noise within spreads.
- Heatmaps highlight “hot spots” by maturity/strike.

Notes:

- Current heatmaps aggregate across all quote dates; per-day heatmaps would require looping by quote_date.

## Phase 3: Smoothing (Implied Vol Smiles)

Files: `scripts/phase3_smooth.py`, `src/pipelines/phase3/smooth.py`

IV extraction:

- Use mid prices with guards: require mid/spot/strike>0, TTM>0; discard prices outside intrinsic/upper bound.
- Compute log-moneyness \(k = \log(S/K)\).
- Weight by inverse bid-ask spread (proxy for liquidity).
- Solve implied vol by inverting Black–Scholes call/put pricing with bounded search.

Smile fitting (per expiration):

- Primary: SVI on total variance \( w(k) = \sigma^2 T \):
  \( w(k) = a + b(\rho (k-m) + \sqrt{(k-m)^2 + \sigma^2}) \)
  Fit via weighted least squares; basic constraints \(b>0\), \(|\rho|<1\), \(\sigma>0\); record success/message.
- Fallback: weighted smoothing spline on IV vs log-moneyness if SVI fails.

Diagnostics and grid:

- RMSE per expiry between fitted IV and observed IV.
- Evaluate fitted smile on a dense strike grid (padded wings).
- Density sanity proxy: discrete second-difference on fitted call prices to flag negative density regions.

Outputs:

- CSVs: `fit_params.csv` (SVI params, RMSE, method), `observed_vs_fit.csv` (per-point obs/fit IV, weights), `grid_fit.csv` (dense grid IVs, density flags).
- Plots: scatter+fit per expiry (up to max_plots), RMSE bar plot.
- Written to `reports/phase3/{dataset}/`.

Visualization (Plotly):

- Utility: `plot_iv_surface(grid_csv, params_csv, quote_date=None, title=None, output_html=None)` in `src/pipelines/phase3/vis.py`.
- Uses `grid_fit.csv` plus `fit_params.csv` (for TTM mapping) to render a 3D IV surface (strike × TTM → IV). Optional quote_date filter; can export HTML.
- Example:

  ```python
  from pipelines.phase3.vis import plot_iv_surface
  fig = plot_iv_surface(
      "reports/phase3/1545/grid_fit.csv",
      "reports/phase3/1545/fit_params.csv",
      quote_date="2025-10-01",
      title="IV Surface 15:45",
      output_html="iv_surface_1545.html",
  )
  fig.show()
  ```

Why:

- Provides smooth, interpretable smiles for interpolation/extrapolation and diagnostics before repair.
- Serves as a reference shape for later repair and Heston calibration.
- Density proxy offers a quick check for shape pathologies.

## Pending / Next Phases (per roadmap)

- Phase 4: Repair via convex optimization within bid-ask bounds enforcing monotonicity/convexity/calendar; report adjustments and post-repair zero-violation checks.
- Phase 5: Heston calibration on raw vs repaired surfaces; compare RMSE, parameter plausibility (Feller), and stability.
- Phase 6: Integration/CLI, tests, documentation (LaTeX math report), and archiving of reports.

## Usage (summary)

- Phase 1: `./scripts/phase1.sh`
- Phase 2: `python3 scripts/phase2_detect.py --dataset 1545|eod|both --sample-size 200 --strike-bucket 1`
- Phase 3: `python3 scripts/phase3_smooth.py --dataset 1545|eod|both --grid-size 60 --max-plots 6 --rate 0 --dividend 0`

## Key Mathematical Touchpoints (for LaTeX)

- No-arbitrage inequalities: intrinsic bounds, monotonicity in strike, convexity (second differences), calendar monotonicity.
- Implied volatility inversion: monotonicity of BS price in \(\sigma\); bracketing and root finding.
- SVI parameterization: 5-parameter total variance smile; basic constraints; (optionally cite Gatheral SVI no-arb conditions if tightening).
- Density proxy: discrete convexity of call prices implies nonnegative risk-neutral density (Breeden–Litzenberger intuition).
- Repair rationale (upcoming): projection onto an arbitrage-consistent set within bid-ask intervals via convex/QP.
- Heston calibration rationale (upcoming): fitting parametric stochastic vol to cleaned vs raw targets; Feller condition \(2\kappa\theta > \xi^2\) as plausibility check.
