from dataclasses import dataclass
from typing import Optional


@dataclass
class CboeRow:
    underlying_symbol: str
    quote_date: str
    expiration: str
    strike: float
    option_type: str
    bid_eod: Optional[float]
    ask_eod: Optional[float]
    bid_size_eod: Optional[float]
    ask_size_eod: Optional[float]
    underlying_bid_eod: Optional[float]
    underlying_ask_eod: Optional[float]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    vwap: Optional[float]
    trade_volume: Optional[float]
    open_interest: Optional[float]
    bid_1545: Optional[float]
    ask_1545: Optional[float]
    underlying_bid_1545: Optional[float]
    underlying_ask_1545: Optional[float]

