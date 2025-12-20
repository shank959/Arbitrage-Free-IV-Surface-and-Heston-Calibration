#!/usr/bin/env bash
# Phase 1 runner: fetch raw Massive (Polygon) SPY options, normalize, sanity report.
# Usage examples:
#   ./scripts/phase1.sh                 # default: last 30 days, SPY
#   ./scripts/phase1.sh 2025-01-01 2025-01-10 SPY
#
# Prereqs:
#   - python -m pip install -r requirements.txt
#   - MASSIVE_API_KEY set in env (and optional MASSIVE_BASE_URL, default https://api.massive.com)
set -euo pipefail

# Ensure local src is on PYTHONPATH so `python -m ...` works from repo root
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

start_date="${1:-}"
end_date="${2:-}"
symbol="${3:-SPY}"

# Build optional args
args=("--symbol" "${symbol}")
if [[ -n "${start_date}" ]]; then
  args+=("--start-date" "${start_date}")
fi
if [[ -n "${end_date}" ]]; then
  args+=("--end-date" "${end_date}")
fi

echo ">>> Fetching raw (${symbol}) from Massive..."
python -m pipelines.phase1.ingest "${args[@]}" --output-dir data/raw/massive

echo ">>> Normalizing to Parquet..."
python -m pipelines.phase1.normalize --raw-dir data/raw/massive --output data/processed/options.parquet

echo ">>> Running sanity checks..."
python -m pipelines.phase1.sanity_report --input data/processed/options.parquet --output data/processed/sanity_summary.csv

echo ">>> Done. Outputs:"
echo "    Raw:      data/raw/massive/"
echo "    Parquet:  data/processed/options.parquet"
echo "    Sanity:   data/processed/sanity_summary.csv"

