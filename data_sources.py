# data_sources.py

import time
import logging
import random
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import config
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # make sure nltk & vader_lexicon are installed

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Async Alpha Vantage wrapper with rate-limiting & fallback."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_call = 0.0
        calls_per_min = config.RATE_LIMITS["alpha_vantage_calls_per_minute"]
        self.min_interval = 60.0 / calls_per_min

    async def _ensure_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.PERFORMANCE_THRESHOLDS["api_response_max_time"]),
                headers={"User-Agent": "HYPER-Trading-System/2.0"}
            )

    async def _throttle(self):
        since = time.time() - self.last_call
        if since < self.min_interval:
            await asyncio.sleep(self.min_interval - since)

    async def _get(self, params: Dict[str, str]) -> Dict[str, Any]:
        await self._ensure_session()
        await self._throttle()
        self.last_call = time.time()
        params["apikey"] = self.api_key

        try:
            async with self.session.get(self.base_url, params=params) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.error(f"AV HTTP {resp.status}: {text[:200]}")
                    return {}
                data = json.loads(text)
                if any(k in data for k in ("Error Message","Note","Information")):
                    logger.warning(f"AV API notice/error: {data}")
                    return {}
                return data
        except Exception as e:
            logger.error(f"AlphaVantageClient error: {e}")
            return {}

    async def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "datatype": "json"}
        data = await self._get(params)
        quote = data.get("Global Quote", {})
        if not quote:
            return self._fallback_quote(symbol)

        try:
            return {
                "symbol": symbol,
                "price": float(quote.get("05. price", 0.0)),
                "change_percent": float(quote.get("10. change percent", "0%").strip("%")),
                "volume": int(float(quote.get("06. volume", 0))),
                "high": float(quote.get("03. high", 0.0)),
                "low": float(quote.get("04. low", 0.0)),
                "data_source": "alpha_vantage",
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return self._fallback_quote(symbol)

    def _fallback_quote(self, symbol: str) -> Dict[str, Any]:
        logger.warning(f"Fallback quote for {symbol}")
        base = config.FALLBACK_PRICES.get(symbol, 100.0)
        pct = random.uniform(-2, 2) / 100
        price = round(base * (1 + pct), 2)
        return {
            "symbol": symbol,
            "price": price,
            "change_percent": round(pct*100, 2),
            "volume": random.randint(1_000_000, 50_000_000),
            "high": price * (1 + random.uniform(0, 0.02)),
            "low": price * (1 - random.uniform(0, 0.02)),
            "data_source": "fallback",
            "timestamp": datetime.now().isoformat()
        }

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class GoogleTrendsClient:
    """Mock Google Trends (fallback)."""
    def __init__(self):
        logger.info("Mock GoogleTrendsClient init")

    async def get_trends_data(self, keywords: List[str]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        data = {}
        for kw in keywords:
            data[kw] = {
                "momentum": random.uniform(-50, 100),
                "velocity": random.uniform(-30, 30)
            }
        return {"keyword_data": data, "timestamp": now, "data_source": "mock_trends"}


class OptionsChainClient:
    """Fetch options chains & Greeks from Polygon.io"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3"
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

    async def get_options_chain(self, symbol: str) -> Dict[str, Any]:
        await self._ensure_session()
        url = f"{self.base_url}/options/chains"
        params = {"underlying_ticker": symbol, "limit": 100}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                calls = sum(o["open_interest"] for o in data.get("results", []) if o["type"] == "call")
                puts  = sum(o["open_interest"] for o in data.get("results", []) if o["type"] == "put")
                pcr   = puts / calls if calls > 0 else None
                return {
                    "open_interest_calls": calls,
                    "open_interest_puts": puts,
                    "put_call_ratio": pcr,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"OptionsChainClient error: {e}")
            return {"put_call_ratio": None}

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class NewsApiClient:
    """Fetch headlines + Vader NLP sentiment via NewsAPI"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        self.vader = SentimentIntensityAnalyzer()

    async def _ensure_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

    async def get_headlines(self, symbol: str) -> List[str]:
        await self._ensure_session()
        url = f"{self.base_url}/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": self.api_key
        }
        try:
            async with self.session.get(url, params=params) as resp:
                j = await resp.json()
                return [a["title"] for a in j.get("articles", [])]
        except Exception as e:
            logger.error(f"NewsApiClient fetch error: {e}")
            return []

    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        if not texts:
            return {"news_sentiment": 50.0}
        scores = [self.vader.polarity_scores(t)["compound"] for t in texts]
        avg = sum(scores) / len(scores)
        # map [-1..1] → [0..100]
        return {"news_sentiment": round((avg + 1) * 50, 1)}

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class HYPERDataAggregator:
    """Unified aggregator: quotes, trends, options, news & quality scoring."""
    def __init__(self, av_key: str):
        self.av      = AlphaVantageClient(av_key)
        self.trends  = GoogleTrendsClient()
        self.options = OptionsChainClient(config.POLYGON_API_KEY)
        self.news    = NewsApiClient(config.NEWS_API_KEY)
        logger.info("🚀 HYPERDataAggregator initialized")

    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        start = time.time()

        quote   = await self.av.get_global_quote(symbol)
        trends  = await self.trends.get_trends_data([symbol])
        opt     = await self.options.get_options_chain(symbol)
        heads   = await self.news.get_headlines(symbol)
        news_st = self.news.analyze_sentiment(heads)

        payload = {
            "symbol": symbol,
            "quote": quote,
            "trends": trends,
            "options": opt,
            "news": news_st,
            "data_quality": "excellent" if quote.get("data_source") == "alpha_vantage" else "poor",
            "latency": round(time.time() - start, 2)
        }
        logger.debug(f"{symbol} aggregated: {payload}")
        return payload

    async def close(self):
        await self.av.close()
        # trends client is stateless/no session to close
        await self.options.close()
        await self.news.close()
