#!/usr/bin/env python
"""
Phase 3 smoother CLI: compute IVs and fit SVI/spline smiles.

Usage:
  python scripts/phase3_smooth.py --dataset 1545
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.phase3.smooth import main as smooth_main  # noqa: E402


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 smoothing (SVI/spline) on processed snapshots.")
    parser.add_argument(
        "--dataset",
        choices=["1545", "eod", "both"],
        default="both",
        help="Dataset to run.",
    )
    parser.add_argument(
        "--input-1545",
        default=str(PROJECT_ROOT / "data/processed/1545/options_1545.parquet"),
        help="Parquet path for 15:45 snapshot.",
    )
    parser.add_argument(
        "--input-eod",
        default=str(PROJECT_ROOT / "data/processed/eod/options_eod.parquet"),
        help="Parquet path for end-of-day snapshot.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports/phase3"),
        help="Root directory for smoothing outputs.",
    )
    parser.add_argument("--rate", type=float, default=0.0, help="Risk-free rate (ccy).")
    parser.add_argument("--dividend", type=float, default=0.0, help="Dividend yield (ccy).")
    parser.add_argument("--grid-size", type=int, default=50, help="Strike grid size for evaluation.")
    parser.add_argument("--max-plots", type=int, default=8, help="Max expirations to plot.")
    return parser.parse_args(argv)


def run_for_dataset(name: str, input_path: Path, output_root: Path, rate: float, dividend: float, grid_size: int, max_plots: int) -> None:
    output_dir = output_root / name
    argv = [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--rate",
        str(rate),
        "--dividend",
        str(dividend),
        "--grid-size",
        str(grid_size),
        "--max-plots",
        str(max_plots),
    ]
    print(f">>> Running smoothing for {name} ...")
    smooth_main(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    output_root = Path(args.output_root)

    if args.dataset in ("1545", "both"):
        run_for_dataset("1545", Path(args.input_1545), output_root, args.rate, args.dividend, args.grid_size, args.max_plots)

    if args.dataset in ("eod", "both"):
        run_for_dataset("eod", Path(args.input_eod), output_root, args.rate, args.dividend, args.grid_size, args.max_plots)

    print(">>> Done.")


if __name__ == "__main__":
    main()

