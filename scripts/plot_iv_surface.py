#!/usr/bin/env python
"""
Render an interactive Plotly IV surface (strike x TTM x IV) from Phase 3 outputs.

Usage:
  python scripts/plot_iv_surface.py \
    --grid reports/phase3/1545/grid_fit.csv \
    --params reports/phase3/1545/fit_params.csv \
    --quote-date 2025-10-01 \
    --title "IV Surface 15:45" \
    --output iv_surface_1545.html \
    --show
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.phase3.vis import plot_iv_surface  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3D IV surface using Plotly.")
    parser.add_argument("--grid", required=True, help="Path to grid_fit.csv.")
    parser.add_argument("--params", required=True, help="Path to fit_params.csv.")
    parser.add_argument("--quote-date", default=None, help="Optional quote_date filter (YYYY-MM-DD).")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--output", default=None, help="Optional HTML output path for the interactive plot.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive figure (may launch a browser). If not set, only writes HTML when --output is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = plot_iv_surface(
        grid_csv=args.grid,
        params_csv=args.params,
        quote_date=args.quote_date,
        title=args.title,
        output_html=args.output,
    )
    if args.show:
        fig.show()
    elif args.output:
        print(f"Wrote interactive plot to {args.output}")
    else:
        print("Plot created (no --show or --output supplied). Use --show to view or --output to save HTML.")


if __name__ == "__main__":
    main()

