# data_sources.py - HYPERtrends v4.1 - Robust Data Aggregator (Alpaca + yfinance + Status)
import os
import logging
import datetime as dt
import pandas as pd
from typing import List, Dict, Any, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.models import BarSet
from alpaca.data.enums import DataFeed
import yfinance as yf

logger = logging.getLogger(__name__)

# Alpaca config (paper/live mode auto detected from env)
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_USE_PAPER = os.getenv("TRADING_MODE", "paper") == "paper"
ALPACA_BASE_URL = (
    "https://paper-api.alpaca.markets" if ALPACA_USE_PAPER
    else "https://api.alpaca.markets"
)

SUPPORTED_SYMBOLS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]
DEFAULT_INTERVAL = "1d"
INTERVAL_MAP = {
    "1m": (TimeFrame.Minute, 1),
    "5m": (TimeFrame.Minute, 5),
    "15m": (TimeFrame.Minute, 15),
    "1h": (TimeFrame.Hour, 1),
    "4h": (TimeFrame.Hour, 4),
    "1d": (TimeFrame.Day, 1)
}

# Market hours
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

def is_market_open(now: Optional[dt.datetime] = None):
    now = now or dt.datetime.now(dt.timezone.utc)
    # Alpaca and US exchanges are closed on weekends
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def get_market_status():
    if is_market_open():
        return "LIVE"
    else:
        return "STALE"  # "SIMULATED" if you use fake data

class HYPERDataAggregator:
    alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    @staticmethod
    def get_current_price(symbol: str) -> Dict[str, Any]:
        # Try Alpaca first
        try:
            bars = HYPERDataAggregator.alpaca_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=1,
                    feed=DataFeed.IEX
                )
            )
            if bars and bars.data.get(symbol):
                price = bars.data[symbol][0].close
                ts = bars.data[symbol][0].timestamp
                return {
                    "price": price,
                    "timestamp": str(ts),
                    "status": get_market_status()
                }
        except Exception as e:
            logger.warning(f"Alpaca price fetch failed ({symbol}): {e}")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.history(period="1d")["Close"].iloc[-1]
            return {
                "price": float(price),
                "timestamp": str(dt.datetime.now()),
                "status": "STALE"
            }
        except Exception as e:
            logger.error(f"yfinance price fetch failed ({symbol}): {e}")
        return {"price": None, "timestamp": None, "status": "ERROR"}

    @staticmethod
    def get_historical_bars(symbol: str, interval: str, limit: int = 30) -> List[Dict[str, Any]]:
        # Convert interval to Alpaca/YF compatible
        bars = []
        alpaca_tf, mult = INTERVAL_MAP.get(interval, (TimeFrame.Day, 1))
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf,
                limit=limit,
                feed=DataFeed.IEX
            )
            alpaca_bars = HYPERDataAggregator.alpaca_client.get_stock_bars(req)
            if alpaca_bars and alpaca_bars.data.get(symbol):
                for b in alpaca_bars.data[symbol]:
                    bars.append({
                        "t": str(b.timestamp),
                        "o": float(b.open),
                        "h": float(b.high),
                        "l": float(b.low),
                        "c": float(b.close),
                        "v": int(b.volume)
                    })
                return bars
        except Exception as e:
            logger.warning(f"Alpaca historical fetch failed ({symbol}): {e}")

        # Fallback to yfinance
        try:
            yf_interval = interval if interval in ["1m", "5m", "15m", "1h", "1d"] else "1d"
            hist = yf.Ticker(symbol).history(period=f"{limit}d", interval=yf_interval)
            for idx, row in hist.iterrows():
                bars.append({
                    "t": str(idx),
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": int(row["Volume"])
                })
            return bars
        except Exception as e:
            logger.error(f"yfinance historical fetch failed ({symbol}): {e}")

        return bars

    @staticmethod
    def get_all_status() -> Dict[str, Any]:
        status = {}
        for symbol in SUPPORTED_SYMBOLS:
            price = HYPERDataAggregator.get_current_price(symbol)
            status[symbol] = price["status"]
        return status
