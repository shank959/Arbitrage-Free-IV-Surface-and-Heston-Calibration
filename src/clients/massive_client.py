import os
from typing import Any, Dict, List, Optional

from polygon import RESTClient


class MassiveClient:
    """
    Thin wrapper around the official Massive (Polygon) Python client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not key:
            raise ValueError("MASSIVE_API_KEY is required.")
        host = base_url or os.environ.get("MASSIVE_BASE_URL")
        self.client = RESTClient(api_key=key, host=host)

    def list_options(self, underlying: str, as_of: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"underlying_ticker": underlying, "limit": limit}
        if as_of:
            kwargs["as_of"] = as_of
        results: List[Dict[str, Any]] = []
        for contract in self.client.list_options_contracts(**kwargs):
            results.append(contract.to_dict() if hasattr(contract, "to_dict") else contract)
        return results

    def list_option_quotes(
        self,
        option_ticker: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 1,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"ticker": option_ticker, "limit": limit, "order": order}
        if start_ts:
            kwargs["timestamp_gte"] = start_ts
        if end_ts:
            kwargs["timestamp_lte"] = end_ts
        quotes: List[Dict[str, Any]] = []
        for q in self.client.list_quotes(**kwargs):
            quotes.append(q.to_dict() if hasattr(q, "to_dict") else q)
            if len(quotes) >= limit:
                break
        return quotes

    def get_option_daily_aggs(
        self,
        option_ticker: str,
        date: str,
        adjusted: bool = True,
    ) -> List[Dict[str, Any]]:
        aggs = self.client.list_aggs(
            ticker=option_ticker,
            multiplier=1,
            timespan="day",
            from_=date,
            to=date,
            adjusted=adjusted,
            limit=1,
        )
        return [a.to_dict() if hasattr(a, "to_dict") else a for a in aggs]

    def get_underlying_daily_close(self, ticker: str, date: str) -> Optional[float]:
        aggs = self.client.list_aggs(
            ticker=ticker, multiplier=1, timespan="day", from_=date, to=date, adjusted=True, limit=1
        )
        last = None
        for a in aggs:
            last = a
        if last is None:
            return None
        data = last.to_dict() if hasattr(last, "to_dict") else last
        return data.get("c")

    def list_dividends(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "ticker": ticker,
            "ex_dividend_date_gte": start_date,
            "ex_dividend_date_lte": end_date,
            "limit": limit,
        }
        results: List[Dict[str, Any]] = []
        for div in self.client.list_dividends(**kwargs):
            results.append(div.to_dict() if hasattr(div, "to_dict") else div)
        return results

