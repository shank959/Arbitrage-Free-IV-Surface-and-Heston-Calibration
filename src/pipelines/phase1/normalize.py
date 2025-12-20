import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .cboe_schema import CboeRow


def parse_row(raw: dict) -> CboeRow:
    return CboeRow(
        underlying_symbol=raw.get("UnderlyingSymbol"),
        quote_date=raw.get("QuoteDate"),
        expiration=raw.get("Expiration"),
        strike=_to_float(raw.get("Strike")),
        option_type=_normalize_type(raw.get("OptionType")),
        bid_eod=_to_float(raw.get("BidEOD")),
        ask_eod=_to_float(raw.get("AskEOD")),
        bid_size_eod=_to_float(raw.get("BidSizeEOD")),
        ask_size_eod=_to_float(raw.get("AskSizeEOD")),
        underlying_bid_eod=_to_float(raw.get("UnderlyingBidEOD")),
        underlying_ask_eod=_to_float(raw.get("UnderlyingAskEOD")),
        open=_to_float(raw.get("Open")),
        high=_to_float(raw.get("High")),
        low=_to_float(raw.get("Low")),
        close=_to_float(raw.get("Close")),
        vwap=_to_float(raw.get("VWAP")),
        trade_volume=_to_float(raw.get("TradeVolume")),
        open_interest=_to_float(raw.get("OpenInterest")),
        bid_1545=_to_float(raw.get("Bid1545")),
        ask_1545=_to_float(raw.get("Ask1545")),
        underlying_bid_1545=_to_float(raw.get("UnderlyingBid1545")),
        underlying_ask_1545=_to_float(raw.get("UnderlyingAsk1545")),
    )


def _to_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_type(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    v = val.strip().lower()
    if v in ("c", "call"):
        return "call"
    if v in ("p", "put"):
        return "put"
    return None


def compute_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None and bid >= 0 and ask >= 0:
        return 0.5 * (bid + ask)
    return None


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_eod"] = df.apply(lambda r: compute_mid(r["bid_eod"], r["ask_eod"]), axis=1)
    df["mid_1545"] = df.apply(lambda r: compute_mid(r["bid_1545"], r["ask_1545"]), axis=1)
    # spot proxy: midpoint of underlying bid/ask EOD
    df["spot_eod"] = df.apply(
        lambda r: compute_mid(r["underlying_bid_eod"], r["underlying_ask_eod"]), axis=1
    )
    df["intrinsic_eod"] = df.apply(
        lambda r: max((r["spot_eod"] - r["strike"]), 0.0) if r["option_type"] == "call" and r["spot_eod"] is not None and r["strike"] else
        max((r["strike"] - r["spot_eod"]), 0.0) if r["option_type"] == "put" and r["spot_eod"] is not None and r["strike"] else None,
        axis=1,
    )
    df["moneyness"] = df.apply(
        lambda r: (r["spot_eod"] / r["strike"]) if r["spot_eod"] and r["strike"] else None, axis=1
    )
    df["log_moneyness"] = df.apply(
        lambda r: (pd.NA if not (r["spot_eod"] and r["strike"]) else float(np.log(r["spot_eod"] / r["strike"]))),
        axis=1,
    )
    df["expiration_dt"] = pd.to_datetime(df["expiration"], utc=True)
    df["quote_dt"] = pd.to_datetime(df["quote_date"], utc=True)
    df["days_to_expiry"] = (df["expiration_dt"] - df["quote_dt"]).dt.days
    df["ttm_years"] = df["days_to_expiry"] / 365.0
    return df


def load_raw_csvs(raw_dir: Path) -> List[CboeRow]:
    rows: List[CboeRow] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        for _, raw in df.iterrows():
            rows.append(parse_row(raw))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Cboe EOD SPY options data to Parquet.")
    parser.add_argument("--raw-dir", default="data/raw/cboe", help="Directory with daily CSV files.")
    parser.add_argument("--output", default="data/processed/options.parquet", help="Parquet output path.")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    rows = load_raw_csvs(raw_root)
    if not rows:
        print("No rows parsed; check input directory.")
        return

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = derive_fields(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def parse_option_type(ticker: str, metadata: Dict) -> str:
    opt_type = metadata.get("contract_type")
    if opt_type in ("call", "put"):
        return opt_type
    if ticker.endswith("C"):
        return "call"
    if ticker.endswith("P"):
        return "put"
    return "unknown"


def compute_mid(bid: Optional[float], ask: Optional[float], close: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None and bid >= 0 and ask >= 0:
        return 0.5 * (bid + ask)
    return close


def normalize_day(day_dir: Path, rows: List[Dict]) -> None:
    options_path = day_dir / "options_list.json"
    underlying_path = day_dir / "underlying.json"
    options = load_json(options_path) or []
    underlying = load_json(underlying_path) or {}
    spot = underlying.get("close")

    trade_date = datetime.strptime(day_dir.name, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for meta in options:
        ticker = meta.get("ticker")
        if not ticker:
            continue
        quote_path = day_dir / "quotes" / f"{ticker}.json"
        quote_data = load_json(quote_path) or []
        if isinstance(quote_data, dict):
            quote_data = quote_data.get("results", []) or []

        bid = ask = bid_size = ask_size = None
        timestamp = None
        close = None
        if quote_data:
            q = quote_data[0]
            bid = q.get("bid_price") or q.get("b")
            ask = q.get("ask_price") or q.get("a")
            bid_size = q.get("bid_size") or q.get("bs")
            ask_size = q.get("ask_size") or q.get("as")
            timestamp = q.get("sip_timestamp") or q.get("t")
            close = q.get("close") or q.get("c")
        mid = compute_mid(bid, ask, close)

        strike = meta.get("strike_price")
        expiry = meta.get("expiration_date")
        if not strike or not expiry or spot is None:
            continue
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ttm_days = max((expiry_dt - trade_date).days, 0)
        ttm_years = ttm_days / 365.0

        opt_type = parse_option_type(ticker, meta)
        intrinsic = max(spot - strike, 0.0) if opt_type == "call" else max(strike - spot, 0.0)

        rows.append(
            {
                "as_of": trade_date.date().isoformat(),
                "option_symbol": ticker,
                "underlying": meta.get("underlying_ticker", "SPY"),
                "expiration": expiry,
                "days_to_expiry": ttm_days,
                "ttm_years": ttm_years,
                "strike": strike,
                "option_type": opt_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "mark_close": close,
                "spot_close": spot,
                "log_moneyness": math.log(spot / strike) if spot and strike else None,
                "moneyness": spot / strike if spot and strike else None,
                "intrinsic": intrinsic,
                "quote_timestamp": timestamp,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw Massive option data to Parquet.")
    parser.add_argument("--raw-dir", default="data/raw/massive", help="Directory containing raw JSON per day.")
    parser.add_argument("--output", default="data/processed/options.parquet", help="Parquet output path.")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    day_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    rows: List[Dict] = []
    for day_dir in day_dirs:
        normalize_day(day_dir, rows)

    if not rows:
        print("No rows to write; check input data.")
        return

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

