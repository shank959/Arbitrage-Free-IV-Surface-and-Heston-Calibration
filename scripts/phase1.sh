#!/usr/bin/env bash
# Phase 1 runner: normalize local Cboe SPY EOD CSVs and run sanity checks.
# Usage:
#   ./scripts/phase1.sh
#
# Prereqs:
#   - python -m pip install -r requirements.txt
set -euo pipefail

# Ensure local src is on PYTHONPATH so `python -m ...` works from repo root
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

RAW_DIR="data/raw/cboe"
OUT_PARQUET="data/processed/options.parquet"
OUT_SANITY="data/processed/sanity_summary.csv"

echo ">>> Normalizing Cboe EOD CSVs..."
python -m pipelines.phase1.normalize --raw-dir "${RAW_DIR}" --output "${OUT_PARQUET}"

echo ">>> Running sanity checks..."
python -m pipelines.phase1.sanity_report --input "${OUT_PARQUET}" --output "${OUT_SANITY}"

echo ">>> Done. Outputs:"
echo "    Raw:      ${RAW_DIR}"
echo "    Parquet:  ${OUT_PARQUET}"
echo "    Sanity:   ${OUT_SANITY}"
