import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .cboe_schema import CboeRow


def _to_float(val) -> Optional[float]:
    """Best-effort float cast; returns None on bad inputs."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_type(val: Optional[str]) -> Optional[str]:
    """Normalize option type strings to 'call'/'put' or None."""
    if not val:
        return None
    v = val.strip().lower()
    if v in ("c", "call"):
        return "call"
    if v in ("p", "put"):
        return "put"
    return None


def compute_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """Return midpoint if bid/ask are valid and non-negative."""
    if bid is None or ask is None:
        return None
    if bid < 0 or ask < 0:
        return None
    return 0.5 * (bid + ask)


def parse_row(raw: dict) -> CboeRow:
    """Parse a raw CSV row dict into a typed CboeRow."""
    return CboeRow(
        underlying_symbol=raw.get("underlying_symbol"),
        quote_date=raw.get("quote_date"),
        expiration=raw.get("expiration"),
        strike=_to_float(raw.get("strike")),
        option_type=_normalize_type(raw.get("option_type")),
        bid_eod=_to_float(raw.get("bid_eod")),
        ask_eod=_to_float(raw.get("ask_eod")),
        bid_size_eod=_to_float(raw.get("bid_size_eod")),
        ask_size_eod=_to_float(raw.get("ask_size_eod")),
        underlying_bid_eod=_to_float(raw.get("underlying_bid_eod")),
        underlying_ask_eod=_to_float(raw.get("underlying_ask_eod")),
        open=_to_float(raw.get("open")),
        high=_to_float(raw.get("high")),
        low=_to_float(raw.get("low")),
        close=_to_float(raw.get("close")),
        vwap=_to_float(raw.get("vwap")),
        trade_volume=_to_float(raw.get("trade_volume")),
        open_interest=_to_float(raw.get("open_interest")),
        bid_1545=_to_float(raw.get("bid_1545")),
        ask_1545=_to_float(raw.get("ask_1545")),
        underlying_bid_1545=_to_float(raw.get("underlying_bid_1545")),
        underlying_ask_1545=_to_float(raw.get("underlying_ask_1545")),
    )


def _derive_intrinsic(row) -> Optional[float]:
    """Compute intrinsic value given spot, strike, and option_type."""
    if row["spot"] is None or row["strike"] is None or row["option_type"] is None:
        return None
    if row["option_type"] == "call":
        return max(row["spot"] - row["strike"], 0.0)
    return max(row["strike"] - row["spot"], 0.0)


def derive_fields(df: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Add snapshot-specific bid/ask fields plus mid/spot/intrinsic features."""
    def col(series_name: str):
        return df[series_name] if series_name in df.columns else pd.Series(np.nan, index=df.index)

    if snapshot == "1545":
        df["bid"] = col("bid_1545")
        df["ask"] = col("ask_1545")
        df["bid_size"] = col("bid_size_1545")
        df["ask_size"] = col("ask_size_1545")
        df["underlying_bid"] = col("underlying_bid_1545")
        df["underlying_ask"] = col("underlying_ask_1545")
    else:
        df["bid"] = col("bid_eod")
        df["ask"] = col("ask_eod")
        df["bid_size"] = col("bid_size_eod")
        df["ask_size"] = col("ask_size_eod")
        df["underlying_bid"] = col("underlying_bid_eod")
        df["underlying_ask"] = col("underlying_ask_eod")

    df["mid"] = df.apply(lambda r: compute_mid(r["bid"], r["ask"]), axis=1)
    df["spot"] = df.apply(lambda r: compute_mid(r["underlying_bid"], r["underlying_ask"]), axis=1)
    df["intrinsic"] = df.apply(_derive_intrinsic, axis=1)
    df["moneyness"] = np.where(
        (df["spot"].notna()) & (df["strike"].notna()) & (df["strike"] != 0),
        df["spot"] / df["strike"],
        np.nan,
    )
    df["log_moneyness"] = np.where(
        (df["spot"].notna()) & (df["strike"].notna()) & (df["spot"] > 0) & (df["strike"] > 0),
        np.log(df["spot"] / df["strike"]),
        np.nan,
    )
    df["expiration_dt"] = pd.to_datetime(df["expiration"], utc=True)
    df["quote_dt"] = pd.to_datetime(df["quote_date"], utc=True)
    df["days_to_expiry"] = (df["expiration_dt"] - df["quote_dt"]).dt.days
    df["ttm_years"] = df["days_to_expiry"] / 365.0
    df["snapshot"] = snapshot
    return df


def load_raw_csvs(raw_dir: Path) -> List[CboeRow]:
    """Load all CSVs in a directory into a list of CboeRow records."""
    rows: List[CboeRow] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, raw in df.iterrows():
            rows.append(parse_row(raw))
    return rows


def write_outputs(df: pd.DataFrame, snapshot: str, output: Path, write_partitions: bool) -> None:
    """Write combined Parquet and per-day partitions for a snapshot."""
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    if write_partitions:
        part_root = output.parent
        for quote_date, group in df.groupby("quote_date"):
            part_path = part_root / f"{quote_date}.parquet"
            group.to_parquet(part_path, index=False)


def main() -> None:
    """Normalize raw Cboe CSVs into 15:45 and EOD Parquet outputs with derived fields."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Normalize Cboe SPY options into separate 1545 and EOD datasets.")
    parser.add_argument("--raw-dir", default="data/raw/cboe", help="Directory with daily CSV files.")
    parser.add_argument("--output-1545", default="data/processed/1545/options_1545.parquet", help="Parquet output for 15:45 snapshot.")
    parser.add_argument("--output-eod", default="data/processed/eod/options_eod.parquet", help="Parquet output for EOD snapshot.")
    parser.add_argument("--no-partitions", action="store_true", help="Skip writing per-day partitioned Parquet files.")
    args = parser.parse_args()

    # Load raw CSVs into a list of CboeRow records
    raw_root = Path(args.raw_dir)
    rows = load_raw_csvs(raw_root)
    if not rows:
        print("No rows parsed; check input directory.")
        return

    # Create base dataframe from rows using the defined CboeRow schema
    base_df = pd.DataFrame([r.__dict__ for r in rows])

    # Derive fields for 15:45 snapshot
    df_1545 = derive_fields(base_df.copy(), snapshot="1545")
    # Derive fields for EOD snapshot using the same base dataframe
    df_eod = derive_fields(base_df.copy(), snapshot="eod")

    # Write derived dataframes to Parquet files
    write_outputs(df_1545, "1545", Path(args.output_1545), write_partitions=not args.no_partitions)
    write_outputs(df_eod, "eod", Path(args.output_eod), write_partitions=not args.no_partitions)

    # Log the number of rows written to each output file
    print(f"Wrote {len(df_1545)} rows to {args.output_1545}")
    print(f"Wrote {len(df_eod)} rows to {args.output_eod}")


if __name__ == "__main__":
    main()

