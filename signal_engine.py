import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from technical_indicators import AdvancedTechnicalAnalyzer, TechnicalAnalysis
from sentiment_analysis import AdvancedSentimentAnalyzer, SentimentAnalysis
from config import SIGNAL_WEIGHTS, CONFIDENCE_THRESHOLDS, is_feature_enabled

logger = logging.getLogger(__name__)

@dataclass
class HYPERSignal:
    symbol: str
    signal_type: str
    confidence: float
    direction: str
    price: float
    timestamp: str
    technical_score: float
    sentiment_score: float
    momentum_score: float
    ml_score: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data_quality: str = "unknown"
    enhanced_features: Dict[str, Any] = field(default_factory=dict)

class HYPERSignalEngine:
    def __init__(self):
        self.signal_cache = {}
        self.cache_duration = 30
        self.generation_count = 0
        self.last_generation_time = None
        self.technical_analyzer = AdvancedTechnicalAnalyzer(TECHNICAL_PARAMS) if is_feature_enabled('technical_indicators') else None
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(SENTIMENT_CONFIG) if is_feature_enabled('sentiment_analysis') else None
        logger.info("HYPERtrends Signal Engine initialized")

    async def generate_signal(self, symbol: str, quote_data: Dict[str, Any], 
                             trends_data: Optional[Dict] = None,
                             historical_data: Optional[List[Dict]] = None) -> HYPERSignal:
        cache_key = f"{symbol}_{time.time() // self.cache_duration}"
        if cache_key in self.signal_cache:
            return self.signal_cache[cache_key]

        start_time = time.time()
        alpaca_features = self._extract_alpaca_features(quote_data)
        technical_analysis = await self._run_technical_analysis(symbol, quote_data, historical_data) if self.technical_analyzer else None
        sentiment_analysis = await self._run_sentiment_analysis(symbol, quote_data, trends_data) if self.sentiment_analyzer else None
        component_scores = self._extract_component_scores(technical_analysis, sentiment_analysis, alpaca_features)
        overall_signal = self._calculate_weighted_signal(component_scores, quote_data, alpaca_features)
        reasons, warnings, recommendations = self._generate_comprehensive_insights(
            technical_analysis, sentiment_analysis, overall_signal, alpaca_features
        )
        data_quality = self._assess_overall_data_quality(technical_analysis, sentiment_analysis, alpaca_features)

        signal = HYPERSignal(
            symbol=symbol,
            signal_type=overall_signal['signal_type'],
            confidence=overall_signal['confidence'],
            direction=overall_signal['direction'],
            price=float(quote_data.get('price', 0)),
            timestamp=datetime.now().isoformat(),
            technical_score=component_scores['technical'],
            sentiment_score=component_scores['sentiment'],
            momentum_score=component_scores['momentum'],
            ml_score=component_scores['ml'],
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations,
            data_quality=data_quality,
            enhanced_features={
                'generation_time': time.time() - start_time,
                'components_used': self._get_active_components(),
                'alpaca_features': alpaca_features,
            }
        )
        self.signal_cache[cache_key] = signal
        self.generation_count += 1
        self.last_generation_time = datetime.now()
        logger.debug(f"Generated {signal.signal_type} signal for {symbol} ({signal.confidence:.0f}% confidence)")
        return signal

    def _extract_alpaca_features(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_features = quote_data.get('enhanced_features', {})
        return {
            'data_source': quote_data.get('data_source', 'unknown'),
            'data_freshness': enhanced_features.get('data_freshness', 'unknown'),
            'market_hours': enhanced_features.get('market_hours', 'UNKNOWN'),
            'spread_bps': enhanced_features.get('spread_bps', 10.0),
            'is_live_data': quote_data.get('data_source', '').startswith('alpaca'),
            'data_quality_score': self._calculate_alpaca_data_quality(quote_data)
        }

    def _calculate_alpaca_data_quality(self, quote_data: Dict[str, Any]) -> float:
        score = 0.0
        data_source = quote_data.get('data_source', '')
        if data_source.startswith('alpaca'):
            score += 40.0
        if quote_data.get('price', 0) > 0:
            score += 20.0
        if all(quote_data.get(k, 0) > 0 for k in ['open', 'high', 'low']):
            score += 15.0
        if quote_data.get('volume', 0) > 1000:
            score += 10.0
        return min(100.0, score)

    async def generate_all_signals(self, data_aggregator) -> Dict[str, HYPERSignal]:
        signals = {}
        signal_tasks = [self._generate_single_signal_with_data(symbol, data_aggregator) for symbol in TICKERS]
        signal_results = await asyncio.gather(*signal_tasks, return_exceptions=True)
        for i, result in enumerate(signal_results):
            symbol = TICKERS[i]
            if isinstance(result, Exception):
                logger.error(f"Signal generation failed for {symbol}: {result}")
                signals[symbol] = self._generate_fallback_signal(symbol, {'price': 100.0})
            else:
                signals[symbol] = result
        return signals

    async def _generate_single_signal_with_data(self, symbol: str, data_aggregator) -> HYPERSignal:
        try:
            data = await data_aggregator.get_comprehensive_data(symbol)
            return await self.generate_signal(
                symbol=symbol,
                quote_data=data.get('quote', {}),
                trends_data=data.get('trends', {}),
                historical_data=data.get('historical', [])
            )
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, {'price': 100.0})

    async def _run_technical_analysis(self, symbol: str, quote_data: Dict[str, Any], 
                                     historical_data: Optional[List[Dict]]) -> Optional[TechnicalAnalysis]:
        try:
            return await self.technical_analyzer.analyze(symbol, quote_data, historical_data)
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return None

    async def _run_sentiment_analysis(self, symbol: str, quote_data: Dict[str, Any], 
                                     trends_data: Optional[Dict]) -> Optional[SentimentAnalysis]:
        try:
            return await self.sentiment_analyzer.analyze(symbol, quote_data, trends_data)
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return None

    def _extract_component_scores(self, technical_analysis: Optional[TechnicalAnalysis],
                                 sentiment_analysis: Optional[SentimentAnalysis],
                                 alpaca_features: Dict[str, Any]) -> Dict[str, float]:
        scores = {
            'technical': 50.0,
            'sentiment': 50.0,
            'momentum': 50.0,
            'ml': 50.0
        }
        if technical_analysis:
            scores['technical'] = technical_analysis.overall_score
            scores['momentum'] = technical_analysis.momentum_analysis.get('momentum_5d', 0) + 50
        if sentiment_analysis:
            scores['sentiment'] = sentiment_analysis.overall_sentiment + 50
        scores['ml'] = (scores['technical'] + scores['sentiment']) / 2
        return scores

    def _calculate_weighted_signal(self, component_scores: Dict[str, float], 
                                  quote_data: Dict[str, Any],
                                  alpaca_features: Dict[str, Any]) -> Dict[str, Any]:
        weighted_score = (
            component_scores['technical'] * SIGNAL_WEIGHTS['technical'] +
            component_scores['sentiment'] * SIGNAL_WEIGHTS['sentiment'] +
            component_scores['momentum'] * SIGNAL_WEIGHTS['momentum'] +
            component_scores['ml'] * SIGNAL_WEIGHTS['ml_prediction']
        )
        confidence = max(0, min(100, weighted_score))
        if confidence >= CONFIDENCE_THRESHOLDS['HYPER_BUY']:
            signal_type = "HYPER_BUY"
            direction = "UP"
        elif confidence >= CONFIDENCE_THRESHOLDS['SOFT_BUY']:
            signal_type = "SOFT_BUY"
            direction = "UP"
        elif confidence <= (100 - CONFIDENCE_THRESHOLDS['HYPER_SELL']):
            signal_type = "HYPER_SELL"
            direction = "DOWN"
        elif confidence <= (100 - CONFIDENCE_THRESHOLDS['SOFT_SELL']):
            signal_type = "SOFT_SELL"
            direction = "DOWN"
        else:
            signal_type = "HOLD"
            direction = "NEUTRAL"
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'direction': direction
        }

    def _generate_comprehensive_insights(self, technical_analysis: Optional[TechnicalAnalysis],
                                        sentiment_analysis: Optional[SentimentAnalysis],
                                        overall_signal: Dict[str, Any],
                                        alpaca_features: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        reasons = []
        warnings = []
        recommendations = []
        if technical_analysis and technical_analysis.overall_score > 75:
            reasons.append(f"Strong technical setup ({technical_analysis.overall_score:.0f}/100)")
        if sentiment_analysis and abs(sentiment_analysis.overall_sentiment) > 40:
            direction = "bullish" if sentiment_analysis.overall_sentiment > 0 else "bearish"
            reasons.append(f"Strong {direction} sentiment")
        if overall_signal['confidence'] < 40:
            warnings.append(f"Low confidence signal ({overall_signal['confidence']:.0f}%)")
        return reasons[:5], warnings[:3], recommendations[:4]

    def _assess_overall_data_quality(self, technical_analysis: Optional[TechnicalAnalysis],
                                    sentiment_analysis: Optional[SentimentAnalysis],
                                    alpaca_features: Dict[str, Any]) -> str:
        quality_scores = [alpaca_features.get('data_quality_score', 50)]
        if technical_analysis:
            quality_scores.append(75)
        if sentiment_analysis:
            quality_scores.append(65)
        weighted_quality = sum(quality_scores) / len(quality_scores)
        if weighted_quality >= 75:
            return "good"
        elif weighted_quality >= 50:
            return "fair"
        else:
            return "poor"

    def _get_active_components(self) -> List[str]:
        components = []
        if self.technical_analyzer:
            components.append("technical")
        if self.sentiment_analyzer:
            components.append("sentiment")
        return components

    def _generate_fallback_signal(self, symbol: str, quote_data: Dict[str, Any]) -> HYPERSignal:
        return HYPERSignal(
            symbol=symbol,
            signal_type="HOLD",
            confidence=50.0,
            direction="NEUTRAL",
            price=float(quote_data.get('price', 0)),
            timestamp=datetime.now().isoformat(),
            technical_score=50.0,
            sentiment_score=50.0,
            momentum_score=50.0,
            ml_score=50.0,
            reasons=["Fallback signal"],
            warnings=["Limited analysis"],
            data_quality="fallback"
        )

    def get_engine_stats(self) -> Dict[str, Any]:
        return {
            "generation_count": self.generation_count,
            "last_generation_time": str(self.last_generation_time),
            "cache_size": len(self.signal_cache),
            "active_components": self._get_active_components()
        }

    async def warm_up_analyzers(self):
        test_quote = {"price": 100.0, "data_source": "alpaca_test"}
        await self.generate_signal('TEST', test_quote)
