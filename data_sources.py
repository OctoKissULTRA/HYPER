import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ========================================
# ALPACA MARKET DATA FETCHER (SANDBOX or LIVE)
# ========================================

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
USE_SANDBOX = os.getenv("USE_SANDBOX", "True") == "True"

BASE_URL = "https://paper-api.alpaca.markets" if USE_SANDBOX else "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": APCA_API_KEY_ID,
    "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
}

class AlpacaMarketDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()

    def get_latest_quote(self) -> Optional[float]:
        try:
            url = f"{DATA_URL}/stocks/{self.symbol}/quotes/latest"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            return data["quote"]["ap"]
        except Exception as e:
            logger.warning(f"Alpaca quote fetch failed for {self.symbol}: {e}")
            return None

    def get_historical_data(self, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        try:
            now = datetime.utcnow()
            start = now - timedelta(days=5)
            url = f"{DATA_URL}/stocks/{self.symbol}/bars"
            params = {
                "start": start.isoformat() + "Z",
                "timeframe": timeframe,
                "limit": limit
            }
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            bars = response.json().get("bars", [])
            df = pd.DataFrame(bars)
            if not df.empty:
                df["t"] = pd.to_datetime(df["t"])
                df.set_index("t", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to get Alpaca historical data for {self.symbol}: {e}")
            return pd.DataFrame()

# ========================================
# DYNAMIC MARKET SIMULATOR (FALLBACK)
# ========================================

class DynamicMarketSimulator:
    def __init__(self, base_price: float = 100.0):
        self.price = base_price

    def get_latest_quote(self) -> float:
        drift = (0.5 - time.time() % 1) * 0.2
        noise = (time.time() % 3) * 0.05
        self.price *= 1 + drift * 0.001 + noise * 0.001
        return round(self.price, 2)

    def get_historical_data(self, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        now = datetime.utcnow()
        times = [now - timedelta(minutes=5 * i) for i in range(limit)][::-1]
        prices = [self.price * (1 + 0.001 * i) for i in range(limit)]
        df = pd.DataFrame({
            "time": times,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000 + i for i in range(limit)]
        })
        df.set_index("time", inplace=True)
        return df