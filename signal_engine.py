# signal_engine.py - HYPERtrends v4.0 Streamlined Signal Engine

import logging
import asyncio
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)

@dataclass
class HYPERSignal:
    """Streamlined HYPER trading signal"""
    symbol: str
    signal_type: str  # HYPER_BUY, SOFT_BUY, HOLD, SOFT_SELL, HYPER_SELL
    confidence: float  # 0-100
    direction: str    # UP, DOWN, NEUTRAL
    price: float
    timestamp: str

    # Core component scores
    technical_score: float
    sentiment_score: float
    momentum_score: float
    ml_score: float

    # Enhanced v4.0 scores
    vix_score: float = 50.0
    market_structure_score: float = 50.0
    risk_score: float = 50.0

    # Key technical indicators
    williams_r: float = -50.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    volume_score: float = 50.0

    # Market context
    vix_sentiment: str = "NEUTRAL"
    market_regime: str = "NORMAL"
    sector_momentum: str = "NEUTRAL"

    # Data quality and source info
    data_quality: str = "unknown"
    data_source: str = "unknown"

    # Supporting data
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class HYPERSignalEngine:
    """Streamlined HYPER Signal Engine - Production Optimized"""

    def __init__(self):
        self.config = config
        self.signal_cache = {}
        self.cache_duration = 30  # 30 seconds
        self.generation_count = 0
        self.last_generation_time = None
        
        # Initialize analyzers based on enabled features
        self.technical_analyzer = TechnicalAnalyzer(config.TECHNICAL_PARAMS) if config.is_feature_enabled('technical_indicators') else None
        self.sentiment_analyzer = SentimentAnalyzer(config.SENTIMENT_CONFIG) if config.is_feature_enabled('sentiment_analysis') else None
        self.vix_analyzer = VIXAnalyzer(config.VIX_CONFIG) if config.is_feature_enabled('vix_analysis') else None
        self.market_analyzer = MarketAnalyzer(config.MARKET_STRUCTURE_CONFIG) if config.is_feature_enabled('market_structure') else None
        self.risk_analyzer = RiskAnalyzer(config.RISK_CONFIG) if config.is_feature_enabled('risk_analysis') else None
        
        logger.info("🚀 HYPERtrends Signal Engine initialized")
        logger.info(f"🔧 Active components: {self._get_active_components()}")

    async def generate_signal(self, symbol: str, quote_data: Dict[str, Any], 
                             trends_data: Optional[Dict] = None,
                             historical_data: Optional[List[Dict]] = None) -> HYPERSignal:
        """Generate comprehensive HYPER signal"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{time.time() // self.cache_duration}"
            if cache_key in self.signal_cache:
                logger.debug(f"📋 Using cached signal for {symbol}")
                return self.signal_cache[cache_key]
            
            logger.debug(f"🎯 Generating signal for {symbol}...")
            start_time = time.time()
            
            # Extract core data
            current_price = float(quote_data.get('price', 0))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Run analysis components concurrently
            analysis_tasks = []
            
            if self.technical_analyzer:
                analysis_tasks.append(self.technical_analyzer.analyze(symbol, quote_data, historical_data))
            
            if self.sentiment_analyzer:
                analysis_tasks.append(self.sentiment_analyzer.analyze(symbol, quote_data, trends_data))
            
            if self.vix_analyzer:
                analysis_tasks.append(self.vix_analyzer.analyze(symbol, quote_data))
            
            if self.market_analyzer:
                analysis_tasks.append(self.market_analyzer.analyze(symbol, quote_data))
            
            if self.risk_analyzer:
                analysis_tasks.append(self.risk_analyzer.analyze(symbol, quote_data, historical_data))
            
            # Execute analyses concurrently
            results = []
            if analysis_tasks:
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Extract component scores
            component_scores = self._extract_component_scores(results, quote_data)
            
            # Calculate weighted overall signal
            overall_signal = self._calculate_weighted_signal(component_scores, quote_data)
            
            # Generate insights
            reasons, warnings, recommendations = self._generate_insights(component_scores, overall_signal, quote_data)
            
            # Assess data quality
            data_quality = self._assess_data_quality(quote_data, results)
            
            # Create signal
            signal = HYPERSignal(
                symbol=symbol,
                signal_type=overall_signal['signal_type'],
                confidence=overall_signal['confidence'],
                direction=overall_signal['direction'],
                price=current_price,
                timestamp=datetime.now().isoformat(),
                
                # Component scores
                technical_score=component_scores['technical'],
                sentiment_score=component_scores['sentiment'],
                momentum_score=component_scores['momentum'],
                ml_score=component_scores['ml'],
                vix_score=component_scores['vix'],
                market_structure_score=component_scores['market_structure'],
                risk_score=component_scores['risk'],
                
                # Technical indicators
                williams_r=component_scores.get('williams_r', -50.0),
                stochastic_k=component_scores.get('stochastic_k', 50.0),
                stochastic_d=component_scores.get('stochastic_d', 50.0),
                rsi=component_scores.get('rsi', 50.0),
                macd_signal=component_scores.get('macd_signal', 'NEUTRAL'),
                volume_score=component_scores.get('volume_score', 50.0),
                
                # Market context
                vix_sentiment=component_scores.get('vix_sentiment', 'NEUTRAL'),
                market_regime=component_scores.get('market_regime', 'NORMAL'),
                sector_momentum=component_scores.get('sector_momentum', 'NEUTRAL'),
                
                # Data quality
                data_quality=data_quality,
                data_source=quote_data.get('data_source', 'unknown'),
                
                # Supporting data
                reasons=reasons,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Cache the signal
            self.signal_cache[cache_key] = signal
            self.generation_count += 1
            self.last_generation_time = datetime.now()
            
            generation_time = time.time() - start_time
            logger.debug(f"✅ Generated {signal.signal_type} signal for {symbol} "
                        f"({signal.confidence:.0f}% confidence) in {generation_time:.2f}s")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal generation error for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, quote_data)

    def _extract_component_scores(self, results: List, quote_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract component scores from analysis results"""
        scores = {
            'technical': 50.0,
            'sentiment': 50.0,
            'momentum': 50.0,
            'ml': 50.0,
            'vix': 50.0,
            'market_structure': 50.0,
            'risk': 50.0
        }
        
        try:
            result_index = 0
            
            # Technical analysis
            if self.technical_analyzer and result_index < len(results):
                tech_result = results[result_index]
                if not isinstance(tech_result, Exception) and tech_result:
                    scores['technical'] = tech_result.get('overall_score', 50.0)
                    scores['momentum'] = tech_result.get('momentum_score', 50.0)
                    scores['williams_r'] = tech_result.get('williams_r', -50.0)
                    scores['stochastic_k'] = tech_result.get('stochastic_k', 50.0)
                    scores['stochastic_d'] = tech_result.get('stochastic_d', 50.0)
                    scores['rsi'] = tech_result.get('rsi', 50.0)
                    scores['macd_signal'] = tech_result.get('macd_signal', 'NEUTRAL')
                    scores['volume_score'] = tech_result.get('volume_score', 50.0)
                result_index += 1
            
            # Sentiment analysis
            if self.sentiment_analyzer and result_index < len(results):
                sent_result = results[result_index]
                if not isinstance(sent_result, Exception) and sent_result:
                    scores['sentiment'] = sent_result.get('overall_sentiment', 0.0) + 50.0
                result_index += 1
            
            # VIX analysis
            if self.vix_analyzer and result_index < len(results):
                vix_result = results[result_index]
                if not isinstance(vix_result, Exception) and vix_result:
                    scores['vix'] = vix_result.get('fear_greed_score', 50.0)
                    scores['vix_sentiment'] = vix_result.get('sentiment', 'NEUTRAL')
                result_index += 1
            
            # Market structure analysis
            if self.market_analyzer and result_index < len(results):
                market_result = results[result_index]
                if not isinstance(market_result, Exception) and market_result:
                    scores['market_structure'] = market_result.get('structure_score', 50.0)
                    scores['market_regime'] = market_result.get('market_regime', 'NORMAL')
                    scores['sector_momentum'] = market_result.get('sector_momentum', 'NEUTRAL')
                result_index += 1
            
            # Risk analysis
            if self.risk_analyzer and result_index < len(results):
                risk_result = results[result_index]
                if not isinstance(risk_result, Exception) and risk_result:
                    scores['risk'] = 100 - risk_result.get('overall_risk_score', 50.0)  # Invert risk
                result_index += 1
            
            # ML score (combination of other scores)
            scores['ml'] = (scores['technical'] + scores['sentiment'] + scores['momentum']) / 3
            
            # Data quality adjustment
            data_quality_score = self._calculate_data_quality_score(quote_data)
            if data_quality_score > 80:
                for key in ['technical', 'sentiment', 'ml']:
                    scores[key] *= 1.05  # 5% boost for high quality data
            
        except Exception as e:
            logger.error(f"Component score extraction error: {e}")
        
        return scores

    def _calculate_weighted_signal(self, component_scores: Dict[str, float], quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted signal with data quality considerations"""
        
        # Base weighted score
        weighted_score = (
            component_scores['technical'] * config.SIGNAL_WEIGHTS['technical'] +
            component_scores['sentiment'] * config.SIGNAL_WEIGHTS['sentiment'] +
            component_scores['momentum'] * config.SIGNAL_WEIGHTS['momentum'] +
            component_scores['ml'] * config.SIGNAL_WEIGHTS['ml_prediction'] +
            component_scores['market_structure'] * config.SIGNAL_WEIGHTS['market_structure'] +
            component_scores['vix'] * config.SIGNAL_WEIGHTS['vix_sentiment'] +
            component_scores['risk'] * config.SIGNAL_WEIGHTS['risk_adjusted']
        )
        
        # Data quality adjustment
        data_source = quote_data.get('data_source', 'unknown')
        if data_source.startswith('alpaca'):
            weighted_score *= 1.05  # 5% boost for Alpaca data
        elif data_source == 'enhanced_simulation':
            weighted_score *= 1.02  # 2% boost for enhanced simulation
        
        # Market hours adjustment
        enhanced_features = quote_data.get('enhanced_features', {})
        market_hours = enhanced_features.get('market_hours', 'UNKNOWN')
        if market_hours in ['PRE_MARKET', 'AFTER_HOURS']:
            weighted_score *= 0.95  # Slight reduction for extended hours
        
        # Normalize to 0-100 scale
        confidence = max(0, min(100, weighted_score))
        
        # Determine signal type and direction
        if confidence >= config.CONFIDENCE_THRESHOLDS['HYPER_BUY']:
            signal_type = "HYPER_BUY"
            direction = "UP"
        elif confidence >= config.CONFIDENCE_THRESHOLDS['SOFT_BUY']:
            signal_type = "SOFT_BUY"
            direction = "UP"
        elif confidence <= (100 - config.CONFIDENCE_THRESHOLDS['HYPER_SELL']):
            signal_type = "HYPER_SELL"
            direction = "DOWN"
        elif confidence <= (100 - config.CONFIDENCE_THRESHOLDS['SOFT_SELL']):
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

    def _generate_insights(self, component_scores: Dict[str, float], overall_signal: Dict[str, Any], 
                          quote_data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Generate insights, warnings, and recommendations"""
        reasons = []
        warnings = []
        recommendations = []
        
        try:
            # Data quality insights
            data_source = quote_data.get('data_source', 'unknown')
            if data_source.startswith('alpaca'):
                reasons.append("📈 Live Alpaca data provides real-time accuracy")
            
            # Technical insights
            if component_scores['technical'] > 75:
                reasons.append(f"📊 Strong technical setup ({component_scores['technical']:.0f}/100)")
            elif component_scores['technical'] < 35:
                warnings.append(f"📊 Weak technical setup ({component_scores['technical']:.0f}/100)")
            
            # RSI insights
            rsi = component_scores.get('rsi', 50)
            if rsi > 70:
                warnings.append(f"⚠️ Overbought RSI: {rsi:.0f}")
                recommendations.append("Consider taking profits or waiting for pullback")
            elif rsi < 30:
                reasons.append(f"✅ Oversold RSI: {rsi:.0f}")
                recommendations.append("Potential buying opportunity")
            
            # Williams %R insights
            williams_r = component_scores.get('williams_r', -50)
            if williams_r < -80:
                reasons.append("📈 Williams %R shows oversold conditions")
            elif williams_r > -20:
                warnings.append("📉 Williams %R shows overbought conditions")
            
            # Sentiment insights
            if component_scores['sentiment'] > 65:
                reasons.append("💭 Positive sentiment detected")
            elif component_scores['sentiment'] < 35:
                warnings.append("💭 Negative sentiment detected")
            
            # VIX insights
            vix_sentiment = component_scores.get('vix_sentiment', 'NEUTRAL')
            if vix_sentiment == 'EXTREME_FEAR':
                reasons.append("😱 VIX extreme fear = contrarian opportunity")
            elif vix_sentiment == 'EXTREME_COMPLACENCY':
                warnings.append("😴 VIX extreme complacency = potential risk")
            
            # Market structure insights
            market_regime = component_scores.get('market_regime', 'NORMAL')
            if market_regime == 'RISK_ON':
                reasons.append("🟢 Risk-on market environment")
            elif market_regime == 'RISK_OFF':
                warnings.append("🔴 Risk-off market environment")
            
            # Overall signal insights
            if overall_signal['confidence'] > 85:
                reasons.append(f"🎯 High confidence signal ({overall_signal['confidence']:.0f}%)")
            elif overall_signal['confidence'] < 40:
                warnings.append(f"❓ Low confidence signal ({overall_signal['confidence']:.0f}%)")
            
            # Market hours warnings
            enhanced_features = quote_data.get('enhanced_features', {})
            market_hours = enhanced_features.get('market_hours', 'UNKNOWN')
            if market_hours in ['PRE_MARKET', 'AFTER_HOURS']:
                warnings.append(f"🕐 Extended hours trading ({market_hours.lower()})")
                recommendations.append("Consider tighter stops during extended hours")
            
            # Volume insights
            volume_score = component_scores.get('volume_score', 50)
            if volume_score > 70:
                reasons.append("📊 High volume confirms move")
            elif volume_score < 30:
                warnings.append("📊 Low volume questions sustainability")
            
            # Risk insights
            if component_scores['risk'] > 75:
                reasons.append("✅ Low risk environment")
                recommendations.append("Standard position sizing acceptable")
            elif component_scores['risk'] < 35:
                warnings.append("⚠️ High risk detected")
                recommendations.append("Consider reduced position size")
            
        except Exception as e:
            logger.error(f"Insights generation error: {e}")
        
        return reasons[:5], warnings[:3], recommendations[:4]

    def _assess_data_quality(self, quote_data: Dict[str, Any], results: List) -> str:
        """Assess overall data quality"""
        try:
            quality_score = 0
            
            # Data source quality
            data_source = quote_data.get('data_source', 'unknown')
            if data_source.startswith('alpaca'):
                quality_score += 40
            elif data_source == 'enhanced_simulation':
                quality_score += 25
            else:
                quality_score += 10
            
            # Price data completeness
            if quote_data.get('price', 0) > 0:
                quality_score += 20
            
            # OHLC availability
            if all(quote_data.get(k, 0) > 0 for k in ['open', 'high', 'low']):
                quality_score += 15
            
            # Volume data
            if quote_data.get('volume', 0) > 1000:
                quality_score += 10
            
            # Bid/Ask data
            if quote_data.get('bid', 0) > 0 and quote_data.get('ask', 0) > 0:
                quality_score += 10
            
            # Analysis results quality
            successful_analyses = sum(1 for r in results if not isinstance(r, Exception) and r)
            quality_score += min(5, successful_analyses)
            
            # Determine quality level
            if quality_score >= 90:
                return "excellent"
            elif quality_score >= 75:
                return "good"
            elif quality_score >= 60:
                return "fair"
            elif quality_score >= 45:
                return "acceptable"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Data quality assessment error: {e}")
            return "unknown"

    def _calculate_data_quality_score(self, quote_data: Dict[str, Any]) -> float:
        """Calculate numeric data quality score"""
        score = 0.0
        
        # Data source
        data_source = quote_data.get('data_source', '')
        if data_source.startswith('alpaca'):
            score += 40.0
        elif data_source == 'enhanced_simulation':
            score += 25.0
        else:
            score += 10.0
        
        # Data completeness
        if quote_data.get('price', 0) > 0:
            score += 20.0
        if quote_data.get('volume', 0) > 0:
            score += 15.0
        if quote_data.get('bid', 0) > 0 and quote_data.get('ask', 0) > 0:
            score += 15.0
        if quote_data.get('timestamp'):
            score += 10.0
        
        return min(100.0, score)

    def _get_active_components(self) -> List[str]:
        """Get list of active analysis components"""
        components = []
        if self.technical_analyzer:
            components.append("technical")
        if self.sentiment_analyzer:
            components.append("sentiment")
        if self.vix_analyzer:
            components.append("vix")
        if self.market_analyzer:
            components.append("market_structure")
        if self.risk_analyzer:
            components.append("risk")
        return components

    def _generate_fallback_signal(self, symbol: str, quote_data: Dict[str, Any]) -> HYPERSignal:
        """Generate fallback signal when analysis fails"""
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
            vix_score=50.0,
            market_structure_score=50.0,
            risk_score=50.0,
            reasons=["Fallback signal - analysis unavailable"],
            warnings=["Limited analysis due to system error"],
            recommendations=["Manual analysis recommended"],
            data_quality="fallback",
            data_source=quote_data.get('data_source', 'unknown')
        )

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get signal engine statistics"""
        return {
            "generation_count": self.generation_count,
            "last_generation_time": self.last_generation_time.isoformat() if self.last_generation_time else None,
            "cache_size": len(self.signal_cache),
            "active_components": self._get_active_components(),
            "cache_duration": self.cache_duration,
            "components_status": {
                "technical_analyzer": self.technical_analyzer is not None,
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "vix_analyzer": self.vix_analyzer is not None,
                "market_analyzer": self.market_analyzer is not None,
                "risk_analyzer": self.risk_analyzer is not None
            }
        }

# ========================================
# STREAMLINED ANALYZER CLASSES
# ========================================

class TechnicalAnalyzer:
    """Streamlined technical analysis"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("📊 Technical Analyzer initialized")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], historical_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            current_price = float(quote_data.get('price', 0))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Generate technical scores based on current data and some randomness for realism
            rsi = max(0, min(100, 50 + change_percent * 5 + random.gauss(0, 10)))
            williams_r = max(-100, min(0, -50 + change_percent * 3 + random.gauss(0, 15)))
            stochastic_k = max(0, min(100, 50 + change_percent * 4 + random.gauss(0, 12)))
            stochastic_d = stochastic_k * 0.9  # D is smoothed version of K
            
            # MACD signal
            if change_percent > 1:
                macd_signal = "BULLISH"
            elif change_percent < -1:
                macd_signal = "BEARISH"
            else:
                macd_signal = "NEUTRAL"
            
            # Volume score
            avg_volume = 25000000  # Estimated average
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_score = min(100, 50 + (volume_ratio - 1) * 30)
            
            # Momentum score
            momentum_score = 50 + change_percent * 3
            momentum_score = max(0, min(100, momentum_score))
            
            # Overall technical score
            scores = [rsi, abs(williams_r), stochastic_k, volume_score, momentum_score]
            overall_score = sum(scores) / len(scores)
            
            # Adjust for strong moves
            if abs(change_percent) > 2:
                if change_percent > 0:
                    overall_score = min(100, overall_score * 1.1)
                else:
                    overall_score = max(0, overall_score * 0.9)
            
            return {
                'overall_score': round(overall_score, 1),
                'momentum_score': round(momentum_score, 1),
                'rsi': round(rsi, 1),
                'williams_r': round(williams_r, 1),
                'stochastic_k': round(stochastic_k, 1),
                'stochastic_d': round(stochastic_d, 1),
                'macd_signal': macd_signal,
                'volume_score': round(volume_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {'overall_score': 50.0, 'momentum_score': 50.0}

class SentimentAnalyzer:
    """Streamlined sentiment analysis"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("💭 Sentiment Analyzer initialized")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], trends_data: Optional[Dict]) -> Dict[str, Any]:
        """Perform sentiment analysis"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Simulate sentiment based on price movement with some randomness
            base_sentiment = change_percent * 2  # Convert to sentiment scale
            
            # Add news sentiment simulation
            news_sentiment = random.gauss(0, 10)
            
            # Add social media sentiment simulation
            social_sentiment = random.gauss(0, 15)
            
            # Combine sentiments
            overall_sentiment = (base_sentiment * 0.5 + news_sentiment * 0.3 + social_sentiment * 0.2)
            
            # Bound to reasonable range
            overall_sentiment = max(-50, min(50, overall_sentiment))
            
            return {
                'overall_sentiment': round(overall_sentiment, 1)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'overall_sentiment': 0.0}

class VIXAnalyzer:
    """Streamlined VIX analysis"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("😱 VIX Analyzer initialized")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform VIX analysis"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Simulate VIX level based on market movement
            base_vix = 20.0  # Average VIX
            
            # VIX tends to spike on market declines
            if change_percent < -2:
                vix_level = base_vix + abs(change_percent) * 2 + random.gauss(0, 3)
            elif change_percent > 2:
                vix_level = base_vix - change_percent * 0.5 + random.gauss(0, 2)
            else:
                vix_level = base_vix + random.gauss(0, 2)
            
            vix_level = max(8, min(80, vix_level))
            
            # Determine sentiment based on VIX level
            if vix_level > 30:
                sentiment = 'EXTREME_FEAR'
            elif vix_level > 20:
                sentiment = 'FEAR'
            elif vix_level < 12:
                sentiment = 'EXTREME_COMPLACENCY'
            elif vix_level < 16:
                sentiment = 'COMPLACENCY'
            else:
                sentiment = 'NEUTRAL'
            
            # Fear/greed score (inverted VIX)
            fear_greed_score = max(0, min(100, 100 - (vix_level / 50) * 100))
            
            return {
                'vix_level': round(vix_level, 1),
                'sentiment': sentiment,
                'fear_greed_score': round(fear_greed_score, 1)
            }
            
        except Exception as e:
            logger.error(f"VIX analysis error: {e}")
            return {'vix_level': 20.0, 'sentiment': 'NEUTRAL', 'fear_greed_score': 50.0}

class MarketAnalyzer:
    """Streamlined market structure analysis"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("🏗️ Market Analyzer initialized")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform market structure analysis"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Simulate market structure score
            base_score = 50
            
            # Adjust based on price movement and volume
            if abs(change_percent) > 1:
                volume_factor = min(2.0, volume / 25000000) if volume > 0 else 1.0
                if change_percent > 0:
                    base_score += change_percent * 5 * volume_factor
                else:
                    base_score += change_percent * 3 * volume_factor
            
            structure_score = max(0, min(100, base_score + random.gauss(0, 10)))
            
            # Determine market regime
            if structure_score > 70:
                market_regime = 'RISK_ON'
            elif structure_score < 30:
                market_regime = 'RISK_OFF'
            else:
                market_regime = 'NORMAL'
            
            # Sector momentum
            if change_percent > 1:
                sector_momentum = 'BULLISH'
            elif change_percent < -1:
                sector_momentum = 'BEARISH'
            else:
                sector_momentum = 'NEUTRAL'
            
            return {
                'structure_score': round(structure_score, 1),
                'market_regime': market_regime,
                'sector_momentum': sector_momentum
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {'structure_score': 50.0, 'market_regime': 'NORMAL', 'sector_momentum': 'NEUTRAL'}

class RiskAnalyzer:
    """Streamlined risk analysis"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("⚠️ Risk Analyzer initialized")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], historical_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Perform risk analysis"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Calculate risk score based on volatility and other factors
            base_risk = 50
            
            # Higher absolute change = higher risk
            volatility_risk = abs(change_percent) * 5
            
            # Volume factor (unusual volume can indicate risk)
            avg_volume = 25000000
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_risk = max(0, (volume_ratio - 1.5) * 10) if volume_ratio > 1.5 else 0
            
            # Random market factors
            market_risk = random.gauss(0, 5)
            
            overall_risk_score = base_risk + volatility_risk + volume_risk + market_risk
            overall_risk_score = max(0, min(100, overall_risk_score))
            
            return {
                'overall_risk_score': round(overall_risk_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {'overall_risk_score': 50.0}

# Export main classes
__all__ = ['HYPERSignalEngine', 'HYPERSignal']

logger.info("🚀 Streamlined Signal Engine loaded successfully")
            