import logging
import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
import requests

from config import ALPACA_CONFIG, TICKERS

logger = logging.getLogger(__name__)

class AlpacaDataClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.secret_key = config["secret_key"]
        self.base_url = config["base_url"]
        self.data_url = config["data_url"]
        self.session = requests.Session()
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        self.session.headers.update(self.headers)
        logger.info(f"Initialized AlpacaDataClient with endpoint: {self.base_url}")

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.data_url}/stocks/{symbol}/bars?timeframe=1Min&limit=1"
        response = self.session.get(endpoint)
        if response.status_code == 200:
            bars = response.json().get('bars', [])
            if bars:
                return bars[0]
        return None

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.data_url}/stocks/{symbol}/quotes/latest"
        response = self.session.get(endpoint)
        if response.status_code == 200:
            return response.json().get('quote', None)
        return None

    def get_historical_bars_fallback(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 390,  # up to 1 trading day of minute bars
        days: int = 365    # up to 1 year
    ) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "start": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": now.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        endpoint = f"{self.data_url}/stocks/{symbol}/bars"
        resp = self.session.get(endpoint, headers=self.headers, params=params)
        if resp.status_code == 200:
            bars = resp.json().get("bars", [])
            if bars:
                return bars
        logger.warning(f"No historical bars for {symbol} using Alpaca fallback!")
        return []

class HYPERDataAggregator:
    def __init__(self, config: Dict[str, Any] = ALPACA_CONFIG):
        self.client = AlpacaDataClient(config)
        self.tickers = TICKERS

    async def initialize(self):
        # No async setup needed for now, but keeps main.py happy
        pass

    async def get_comprehensive_data(self, symbol: str) -> dict:
        """
        Gathers latest quote, historical bars, and placeholder trends for the given symbol.
        Returns a dictionary as expected by signal engine.
        """
        bar = self.client.get_latest_bar(symbol)
        quote = self.client.get_latest_quote(symbol)
        historical = self.client.get_historical_bars_fallback(symbol, timeframe="1Min", limit=390)
        trends = {}  # Placeholder for trends/news; plug in your real logic here.
        return {
            "quote": quote,
            "historical": historical,
            "trends": trends,
        }

    def get_historical_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> List[Dict[str, Any]]:
        bars = self.client.get_historical_bars_fallback(symbol, timeframe=timeframe, limit=limit)
        if bars:
            return bars
        logger.warning(f"No historical bars for {symbol}, falling back to simulation")
        return MockMarketSimulator([symbol]).get_historical_data(symbol, timeframe, limit)

    async def cleanup(self):
        # Optional: if you want to close the session or do other cleanup
        self.client.session.close()

class MockMarketSimulator:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def get_historical_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> List[Dict[str, Any]]:
        bars = []
        base = round(random.uniform(100, 600), 2)
        for i in range(limit):
            bars.append({
                "c": round(base + random.uniform(-2, 2), 2),
                "t": int(time.time()) - i*60
            })
        return bars

__all__ = [
    "AlpacaDataClient",
    "HYPERDataAggregator",
    "MockMarketSimulator"
]
