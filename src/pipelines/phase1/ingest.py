import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from clients.massive_client import MassiveClient


def iso_day_bounds(target_date: datetime) -> tuple[str, str]:
    start = datetime.combine(target_date.date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    end = datetime.combine(target_date.date(), datetime.max.time()).replace(tzinfo=timezone.utc)
    return start.isoformat(), end.isoformat()


def daterange(start_date: datetime, end_date: datetime) -> Iterable[datetime]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_day(client: MassiveClient, symbol: str, target_date: datetime, root: Path) -> None:
    day_str = target_date.strftime("%Y-%m-%d")
    day_dir = root / day_str
    ensure_dir(day_dir / "quotes")

    options_path = day_dir / "options_list.json"
    if options_path.exists():
        with options_path.open() as f:
            options = json.load(f)
    else:
        options = client.list_options(underlying=symbol, as_of=day_str)
        options_path.write_text(json.dumps(options, indent=2))

    start_ts, end_ts = iso_day_bounds(target_date)

    for opt in tqdm(options, desc=f"{day_str} quotes", leave=False):
        ticker = opt.get("ticker")
        if not ticker:
            continue
        quote_path = day_dir / "quotes" / f"{ticker}.json"
        if quote_path.exists():
            continue
        quotes = client.list_option_quotes(
            option_ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=1,
            order="desc",
        )
        if not quotes:
            quotes = client.get_option_daily_aggs(option_ticker=ticker, date=day_str)
        quote_path.write_text(json.dumps(quotes, indent=2))

    underlying_path = day_dir / "underlying.json"
    if not underlying_path.exists():
        close_price = client.get_underlying_daily_close(ticker=symbol, date=day_str)
        data = {"close": close_price}
        underlying_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Massive option chain data with caching.")
    parser.add_argument("--symbol", default="SPY", help="Underlying ticker (default: SPY)")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", default="data/raw/massive", help="Directory for cached JSON")
    args = parser.parse_args()

    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end_date
        else datetime.now(timezone.utc)
    )
    start_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start_date
        else end_date - timedelta(days=29)
    )

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    client = MassiveClient()
    for day in tqdm(list(daterange(start_date, end_date)), desc="Days"):
        fetch_day(client, symbol=args.symbol, target_date=day, root=output_root)


if __name__ == "__main__":
    main()

