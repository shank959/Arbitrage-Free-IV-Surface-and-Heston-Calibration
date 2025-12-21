#!/usr/bin/env python
"""
Phase 2 detector CLI: run bid-ask–aware arbitrage checks on processed snapshots.

Usage:
  ./scripts/phase2_detect.py --dataset 1545
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.phase2.detect import main as detect_main  # noqa: E402


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 arbitrage detection.")
    parser.add_argument(
        "--dataset",
        choices=["1545", "eod", "both"],
        default="both",
        help="Which dataset to scan.",
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
        default=str(PROJECT_ROOT / "reports/phase2"),
        help="Root directory for reports.",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Max sample rows to export.")
    parser.add_argument("--strike-bucket", type=float, default=1.0, help="Strike bucket size for heatmaps.")
    return parser.parse_args(argv)


def run_for_dataset(name: str, input_path: Path, output_root: Path, sample_size: int, strike_bucket: float) -> None:
    output_dir = output_root / name
    argv = [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--sample-size",
        str(sample_size),
        "--strike-bucket",
        str(strike_bucket),
    ]
    print(f">>> Running detection for {name} ...")
    detect_main(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    output_root = Path(args.output_root)

    if args.dataset in ("1545", "both"):
        run_for_dataset("1545", Path(args.input_1545), output_root, args.sample_size, args.strike_bucket)

    if args.dataset in ("eod", "both"):
        run_for_dataset("eod", Path(args.input_eod), output_root, args.sample_size, args.strike_bucket)

    print(">>> Done.")


if __name__ == "__main__":
    main()

