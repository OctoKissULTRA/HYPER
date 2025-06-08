import logging
import asyncio
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from textblob import TextBlob

from config import SENTIMENT_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    sentiment: float  # -100 to 100
    confidence: float  # 0 to 100
    source: str
    timestamp: str
    text_snippet: str = ""
    signal_strength: str = "NEUTRAL"

@dataclass
class SentimentAnalysis:
    overall_sentiment: float  # -100 to 100
    confidence: float  # 0 to 100
    signals: List[SentimentSignal]
    contrarian_signals: List[str] = None
    sentiment_trend: str = "STABLE"

class NewsSimulator:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = 300

    async def fetch_news(self, symbol: str) -> List[Dict[str, Any]]:
        sentiments = ["positive", "negative", "neutral"]
        return [
            {
                "title": f"Simulated News for {symbol}",
                "sentiment": random.choice(sentiments),
                "timestamp": datetime.now().isoformat(),
                "snippet": f"News about {symbol} is {random.choice(sentiments)}."
            }
        ]

class SocialMediaSimulator:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = 300

    async def fetch_social_media(self, symbol: str) -> List[Dict[str, Any]]:
        sentiments = ["bullish", "bearish", "neutral"]
        return [
            {
                "post": f"Simulated post about {symbol}",
                "sentiment": random.choice(sentiments),
                "timestamp": datetime.now().isoformat()
            }
        ]

class RedditSimulator:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = 300

    async def fetch_reddit(self, symbol: str) -> List[Dict[str, Any]]:
        sentiments = ["positive", "negative", "neutral"]
        return [
            {
                "title": f"Simulated Reddit post for {symbol}",
                "sentiment": random.choice(sentiments),
                "timestamp": datetime.now().isoformat()
            }
        ]

class TwitterSimulator:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = 300

    async def fetch_twitter(self, symbol: str) -> List[Dict[str, Any]]:
        sentiments = ["bullish", "bearish", "neutral"]
        return [
            {
                "tweet": f"Simulated tweet about {symbol}",
                "sentiment": random.choice(sentiments),
                "timestamp": datetime.now().isoformat()
            }
        ]

class InstitutionalFlowSimulator:
    def __init__(self):
        self.flow_cache = {}
        self.cache_duration = 300

    async def fetch_institutional_flow(self, symbol: str) -> List[Dict[str, Any]]:
        flows = ["buy", "sell", "neutral"]
        return [
            {
                "flow": random.choice(flows),
                "volume": random.randint(1000, 100000),
                "timestamp": datetime.now().isoformat()
            }
        ]

class AdvancedSentimentAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.news_simulator = NewsSimulator()
        self.social_simulator = SocialMediaSimulator()
        self.reddit_simulator = RedditSimulator()
        self.twitter_simulator = TwitterSimulator()
        self.institutional_simulator = InstitutionalFlowSimulator()
        self.sentiment_cache = {}
        self.cache_duration = config.get("cache_duration", 300)
        logger.info("Sentiment Analyzer initialized with simulators")

    async def analyze(self, symbol: str, quote_data: Dict[str, Any], trends_data: Optional[Dict] = None) -> SentimentAnalysis:
        cache_key = f"{symbol}_{int(time.time() // self.cache_duration)}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        tasks = [
            self.news_simulator.fetch_news(symbol),
            self.social_simulator.fetch_social_media(symbol),
            self.reddit_simulator.fetch_reddit(symbol),
            self.twitter_simulator.fetch_twitter(symbol),
            self.institutional_simulator.fetch_institutional_flow(symbol)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for result, source in zip(results, ["news", "social", "reddit", "twitter", "institutional"]):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {source} data for {symbol}: {result}")
                continue
            for item in result:
                sentiment_score = self._calculate_sentiment(item, source)
                signals.append(SentimentSignal(
                    sentiment=sentiment_score * 100,
                    confidence=75.0,
                    source=source,
                    timestamp=datetime.now().isoformat(),
                    text_snippet=item.get("snippet", item.get("post", item.get("title", "")))
                ))

        overall_sentiment = self._aggregate_sentiments(signals)
        analysis = SentimentAnalysis(
            overall_sentiment=overall_sentiment,
            confidence=80.0,
            signals=signals,
            contrarian_signals=self._detect_contrarian_signals(signals)
        )
        self.sentiment_cache[cache_key] = analysis
        return analysis

    def _calculate_sentiment(self, item: Dict[str, Any], source: str) -> float:
        text = item.get("snippet", item.get("post", item.get("title", "")))
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if source == "institutional":
            flow = item.get("flow", "neutral")
            flow_scores = {"buy": 0.5, "sell": -0.5, "neutral": 0.0}
            return flow_scores.get(flow, 0.0)
        return polarity

    def _aggregate_sentiments(self, signals: List[SentimentSignal]) -> float:
        if not signals:
            return 0.0
        weights = self.config["sentiment_weights"]
        weighted_sum = 0.0
        total_weight = 0.0
        for signal in signals:
            weight = weights.get(signal.source, 0.1)
            weighted_sum += signal.sentiment * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _detect_contrarian_signals(self, signals: List[SentimentSignal]) -> List[str]:
        contrarian_signals = []
        for signal in signals:
            if abs(signal.sentiment) > 80:
                direction = "Overbought" if signal.sentiment > 0 else "Oversold"
                contrarian_signals.append(f"Extreme {signal.source} sentiment indicates {direction}")
        return contrarian_signals
