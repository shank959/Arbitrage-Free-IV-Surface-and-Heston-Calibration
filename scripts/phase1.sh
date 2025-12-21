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

RAW_DIR="data/raw"
OUT_PARQUET_1545="data/processed/1545/options_1545.parquet"
OUT_PARQUET_EOD="data/processed/eod/options_eod.parquet"
OUT_SANITY_1545="data/processed/1545/sanity_summary_1545.csv"
OUT_SANITY_EOD="data/processed/eod/sanity_summary_eod.csv"

echo ">>> Normalizing Cboe CSVs (separate 1545 and EOD)..."
python -m pipelines.phase1.normalize \
  --raw-dir "${RAW_DIR}" \
  --output-1545 "${OUT_PARQUET_1545}" \
  --output-eod "${OUT_PARQUET_EOD}"

echo ">>> Running sanity checks..."
python -m pipelines.phase1.sanity_report --input "${OUT_PARQUET_1545}" --output "${OUT_SANITY_1545}"
python -m pipelines.phase1.sanity_report --input "${OUT_PARQUET_EOD}" --output "${OUT_SANITY_EOD}"

echo ">>> Done. Outputs:"
echo "    Raw:           ${RAW_DIR}"
echo "    Parquet 1545:  ${OUT_PARQUET_1545}"
echo "    Parquet EOD:   ${OUT_PARQUET_EOD}"
echo "    Sanity 1545:   ${OUT_SANITY_1545}"
echo "    Sanity EOD:    ${OUT_SANITY_EOD}"
