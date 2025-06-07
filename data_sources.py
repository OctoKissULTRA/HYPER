import logging
import time
import random
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
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        })
        logger.info(f"Initialized AlpacaDataClient with endpoint: {self.base_url}")

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.data_url}/stocks/{symbol}/bars?timeframe=1Min&limit=1"
        response = self.session.get(endpoint)
        if response.status_code == 200:
            bars = response.json().get('bars', [])
            if bars:
                return bars[0]
            else:
                logger.warning(f"No bars found for {symbol}")
        else:
            logger.error(f"Failed to fetch bars for {symbol}: {response.text}")
        return None

    def get_historical_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> List[Dict[str, Any]]:
        endpoint = f"{self.data_url}/stocks/{symbol}/bars?timeframe={timeframe}&limit={limit}"
        response = self.session.get(endpoint)
        if response.status_code == 200:
            return response.json().get('bars', [])
        else:
            logger.error(f"Failed to fetch historical bars for {symbol}: {response.text}")
        return []

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.data_url}/stocks/{symbol}/quotes/latest"
        response = self.session.get(endpoint)
        if response.status_code == 200:
            return response.json().get('quote', None)
        else:
            logger.error(f"Failed to fetch quote for {symbol}: {response.text}")
        return None

class HYPERDataAggregator:
    def __init__(self, config: Dict[str, Any]):
        self.client = AlpacaDataClient(config)
        self.tickers = TICKERS

    def get_realtime_data(self) -> Dict[str, Any]:
        data = {}
        for symbol in self.tickers:
            bar = self.client.get_latest_bar(symbol)
            quote = self.client.get_latest_quote(symbol)
            data[symbol] = {
                "bar": bar,
                "quote": quote
            }
        return data

    def get_historical_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> List[Dict[str, Any]]:
        return self.client.get_historical_bars(symbol, timeframe, limit)

class MockMarketSimulator:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def get_realtime_data(self) -> Dict[str, Any]:
        data = {}
        for symbol in self.tickers:
            price = round(random.uniform(100, 600), 2)
            data[symbol] = {
                "bar": {"c": price, "t": int(time.time())},
                "quote": {"ap": price, "bp": price - 0.05, "sp": price + 0.05}
            }
        return data

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
