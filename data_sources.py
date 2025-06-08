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
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        params = {"timeframe": timeframe, "limit": limit}
        if not start or not end:
            end_dt = datetime.utcnow()
            for days in [3, 7]:
                start_dt = end_dt - timedelta(days=days)
                params["start"] = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                params["end"] = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                endpoint = f"{self.data_url}/stocks/{symbol}/bars"
                resp = self.session.get(endpoint, headers=self.headers, params=params)
                if resp.status_code == 200:
                    bars = resp.json().get("bars", [])
                    if bars:
                        return bars
                logger.warning(f"No historical bars for {symbol} in last {days} days using Alpaca fallback!")
            return []
        else:
            params["start"] = start
            params["end"] = end
            endpoint = f"{self.data_url}/stocks/{symbol}/bars"
            resp = self.session.get(endpoint, headers=self.headers, params=params)
            if resp.status_code == 200:
                bars = resp.json().get("bars", [])
                if bars:
                    return bars
            logger.warning(f"No historical bars for {symbol} using Alpaca fallback!")
            return []

class MockMarketSimulator:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def get_historical_data(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict[str, Any]]:
        bars = []
        base = round(random.uniform(100, 600), 2)
        interval = 86400 if timeframe.endswith("Day") else 60
        for i in range(limit):
            bars.append({
                "c": round(base + random.uniform(-2, 2), 2),
                "t": int(time.time()) - i * interval
            })
        return bars

class HYPERDataAggregator:
    def __init__(self, config: Dict[str, Any] = ALPACA_CONFIG):
        self.alpaca = AlpacaDataClient(config)
        self.simulator = MockMarketSimulator(TICKERS)
        self.tickers = TICKERS

    async def initialize(self):
        pass

    async def get_comprehensive_data(self, symbol: str) -> dict:
        bar = self.alpaca.get_latest_bar(symbol)
        quote = self.alpaca.get_latest_quote(symbol)
        historical = self.alpaca.get_historical_bars_fallback(symbol, timeframe="1Min", limit=390)
        trends = {}
        return {
            "quote": quote or {"price": 100.0, "data_source": "fallback"},
            "historical": historical,
            "trends": trends,
        }

    async def get_historical_data_api(self, symbol, timeframe="1Day", start=None, end=None, limit=100):
        bars = self.alpaca.get_historical_bars_fallback(
            symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit
        )
        if not bars:
            bars = self.simulator.get_historical_data(symbol, timeframe, limit)
        return bars

    def get_historical_data(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict[str, Any]]:
        bars = self.alpaca.get_historical_bars_fallback(symbol, timeframe=timeframe, limit=limit)
        if bars:
            return bars
        logger.warning(f"No historical bars for {symbol}, falling back to simulation")
        return self.simulator.get_historical_data(symbol, timeframe, limit)

    async def cleanup(self):
        self.alpaca.session.close()
