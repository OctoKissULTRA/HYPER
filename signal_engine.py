# signal_engine.py - HYPERtrends v4.0 Signal Engine with Alpaca Integration

import logging
import os
import requests

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1381403158907060304/4-wDpJTQ7kPHyOFCxydfGSD5E-AY9mThcBDPbpDW6a8KqFEN9A9003hpOepb7qoFaYQi"

if confidence >= 70:
    def send_discord_alert(symbol: str, signal: str, confidence: float):
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={
            "username": "HYPERtrends Bot",
            "embeds": [{
                "title": f"📡 Signal Alert: {symbol}",
                "description": f"**Signal:** `{signal}`\n**Confidence:** `{confidence:.1f}%`",
                "color": 3066993 if "BUY" in signal else 15158332,
                "footer": { "text": "HYPERtrends v4.0" }
            }]
        })
    except Exception as e:
        print(f"❌ Discord alert failed: {e}")

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import all modular components
from technical_indicators import AdvancedTechnicalAnalyzer, TechnicalAnalysis
from sentiment_analysis import AdvancedSentimentAnalyzer, SentimentAnalysis
from vix_analysis import AdvancedVIXAnalyzer, VIXAnalysis
from market_structure import AdvancedMarketStructureAnalyzer, MarketStructureAnalysis
from risk_analysis import AdvancedRiskAnalyzer, RiskAnalysis

import config

logger = logging.getLogger(__name__)

@dataclass
class HYPERSignal:
    """Enhanced HYPER trading signal - Alpaca Production Ready v4.0"""
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

    # Advanced technical indicators
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

    # Alpaca-specific features
    bid_ask_spread: float = 0.0
    market_hours: str = "UNKNOWN"
    data_quality: str = "unknown"

    # Component analysis results
    technical_analysis: Optional[TechnicalAnalysis] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    vix_analysis: Optional[VIXAnalysis] = None
    market_structure_analysis: Optional[MarketStructureAnalysis] = None
    risk_analysis: Optional[RiskAnalysis] = None

    # Supporting data
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Enhanced features for Alpaca integration
    enhanced_features: Dict[str, Any] = field(default_factory=dict)

class HYPERSignalEngine:
    """Production HYPER Signal Engine with Alpaca Integration"""

    def __init__(self):
        self.config = config
        self.signal_cache = {}
        self.cache_duration = 30  # 30 seconds
        self.generation_count = 0
        self.last_generation_time = None
        
        # Initialize all modular analyzers
        logger.info("🚀 Initializing HYPERtrends v4.0 Signal Engine with Alpaca support...")
        
        # Technical Analysis
        if config.is_feature_enabled('technical_indicators'):
            self.technical_analyzer = AdvancedTechnicalAnalyzer(config.TECHNICAL_PARAMS)
            logger.info("✅ Technical Analyzer loaded (25+ indicators)")
        else:
            self.technical_analyzer = None
            logger.info("⚠️ Technical Analyzer disabled")
        
        # Sentiment Analysis
        if config.is_feature_enabled('sentiment_analysis'):
            self.sentiment_analyzer = AdvancedSentimentAnalyzer(config.SENTIMENT_CONFIG)
            logger.info("✅ Sentiment Analyzer loaded (multi-source NLP)")
        else:
            self.sentiment_analyzer = None
            logger.info("⚠️ Sentiment Analyzer disabled")
        
        # VIX Analysis
        if config.is_feature_enabled('vix_analysis'):
            self.vix_analyzer = AdvancedVIXAnalyzer(config.VIX_CONFIG)
            logger.info("✅ VIX Analyzer loaded (fear/greed detection)")
        else:
            self.vix_analyzer = None
            logger.info("⚠️ VIX Analyzer disabled")
        
        # Market Structure Analysis
        if config.is_feature_enabled('market_structure'):
            self.market_structure_analyzer = AdvancedMarketStructureAnalyzer(config.MARKET_STRUCTURE_CONFIG)
            logger.info("✅ Market Structure Analyzer loaded")
        else:
            self.market_structure_analyzer = None
            logger.info("⚠️ Market Structure Analyzer disabled")
        
        # Risk Analysis
        if config.is_feature_enabled('risk_analysis'):
            self.risk_analyzer = AdvancedRiskAnalyzer(config.RISK_CONFIG)
            logger.info("✅ Risk Analyzer loaded")
        else:
            self.risk_analyzer = None
            logger.info("⚠️ Risk Analyzer disabled")
        
        logger.info("🌟 HYPERtrends Signal Engine initialized successfully!")

    async def generate_signal(self, symbol: str, quote_data: Dict[str, Any], 
                             trends_data: Optional[Dict] = None,
                             historical_data: Optional[List[Dict]] = None) -> HYPERSignal:
        """Generate comprehensive HYPER signal using Alpaca data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{time.time() // self.cache_duration}"
            if cache_key in self.signal_cache:
                logger.debug(f"📋 Using cached signal for {symbol}")
                return self.signal_cache[cache_key]
            
            logger.debug(f"🎯 Generating Alpaca-powered signal for {symbol}...")
            start_time = time.time()
            
            # Extract Alpaca-specific data features
            alpaca_features = self._extract_alpaca_features(quote_data)
            
            # Initialize component results
            technical_analysis = None
            sentiment_analysis = None
            vix_analysis = None
            market_structure_analysis = None
            risk_analysis = None
            
            # Run all enabled analyzers concurrently
            analysis_tasks = []
            
            if self.technical_analyzer:
                analysis_tasks.append(
                    self._run_technical_analysis(symbol, quote_data, historical_data)
                )
            
            if self.sentiment_analyzer:
                analysis_tasks.append(
                    self._run_sentiment_analysis(symbol, quote_data, trends_data)
                )
            
            if self.vix_analyzer:
                analysis_tasks.append(
                    self._run_vix_analysis(symbol, quote_data)
                )
            
            if self.market_structure_analyzer:
                analysis_tasks.append(
                    self._run_market_structure_analysis(symbol, quote_data)
                )
            
            if self.risk_analyzer:
                analysis_tasks.append(
                    self._run_risk_analysis(symbol, quote_data, historical_data)
                )
            
            # Execute all analyses concurrently
            if analysis_tasks:
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # Process results
                result_index = 0
                if self.technical_analyzer:
                    technical_analysis = results[result_index] if not isinstance(results[result_index], Exception) else None
                    result_index += 1
                
                if self.sentiment_analyzer:
                    sentiment_analysis = results[result_index] if not isinstance(results[result_index], Exception) else None
                    result_index += 1
                
                if self.vix_analyzer:
                    vix_analysis = results[result_index] if not isinstance(results[result_index], Exception) else None
                    result_index += 1
                
                if self.market_structure_analyzer:
                    market_structure_analysis = results[result_index] if not isinstance(results[result_index], Exception) else None
                    result_index += 1
                
                if self.risk_analyzer:
                    risk_analysis = results[result_index] if not isinstance(results[result_index], Exception) else None
                    result_index += 1
            
            # Extract component scores with Alpaca enhancements
            component_scores = self._extract_component_scores(
                technical_analysis, sentiment_analysis, vix_analysis, 
                market_structure_analysis, risk_analysis, alpaca_features
            )
            
            # Calculate weighted overall signal
            overall_signal = self._calculate_weighted_signal(component_scores, quote_data, alpaca_features)
            
            # Generate comprehensive insights
            reasons, warnings, recommendations = self._generate_comprehensive_insights(
                technical_analysis, sentiment_analysis, vix_analysis,
                market_structure_analysis, risk_analysis, overall_signal, alpaca_features
            )
            
            # Assess data quality with Alpaca metrics
            data_quality = self._assess_overall_data_quality(
                technical_analysis, sentiment_analysis, vix_analysis,
                market_structure_analysis, risk_analysis, alpaca_features
            )
            
            # Create enhanced signal
            signal = HYPERSignal(
                symbol=symbol,
                signal_type=overall_signal['signal_type'],
                confidence=overall_signal['confidence'],
                direction=overall_signal['direction'],
                price=float(quote_data.get('price', 0)),
                timestamp=datetime.now().isoformat(),
                
                # Component scores
                technical_score=component_scores['technical'],
                sentiment_score=component_scores['sentiment'],
                momentum_score=component_scores['momentum'],
                ml_score=component_scores['ml'],
                vix_score=component_scores['vix'],
                market_structure_score=component_scores['market_structure'],
                risk_score=component_scores['risk'],
                
                # Enhanced indicators
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
                
                # Alpaca-specific features
                bid_ask_spread=alpaca_features.get('spread_bps', 0.0),
                market_hours=alpaca_features.get('market_hours', 'UNKNOWN'),
                data_quality=data_quality,
                
                # Component analysis results
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                vix_analysis=vix_analysis,
                market_structure_analysis=market_structure_analysis,
                risk_analysis=risk_analysis,
                
                # Supporting data
                reasons=reasons,
                warnings=warnings,
                recommendations=recommendations,
                
                # Enhanced features
                enhanced_features={
                    'generation_time': time.time() - start_time,
                    'components_used': self._get_active_components(),
                    'alpaca_features': alpaca_features,
                    'data_sources': quote_data.get('enhanced_features', {}),
                    'analysis_depth': 'comprehensive',
                    'generation_count': self.generation_count,
                    'cache_key': cache_key
                }
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

    def _extract_alpaca_features(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Alpaca-specific features from quote data"""
        enhanced_features = quote_data.get('enhanced_features', {})
        
        return {
            'data_source': quote_data.get('data_source', 'unknown'),
            'data_freshness': enhanced_features.get('data_freshness', 'unknown'),
            'market_hours': enhanced_features.get('market_hours', 'UNKNOWN'),
            'spread_bps': enhanced_features.get('spread_bps', 10.0),
            'bid': quote_data.get('bid', 0.0),
            'ask': quote_data.get('ask', 0.0),
            'bid_size': quote_data.get('bid_size', 0),
            'ask_size': quote_data.get('ask_size', 0),
            'is_live_data': quote_data.get('data_source', '').startswith('alpaca'),
            'data_quality_score': self._calculate_alpaca_data_quality(quote_data)
        }

    def _calculate_alpaca_data_quality(self, quote_data: Dict[str, Any]) -> float:
        """Calculate data quality score for Alpaca data"""
        score = 0.0
        
        # Data source quality
        data_source = quote_data.get('data_source', '')
        if data_source.startswith('alpaca'):
            score += 40.0
        elif data_source == 'enhanced_simulation':
            score += 25.0
        else:
            score += 10.0
        
        # Price data completeness
        if quote_data.get('price', 0) > 0:
            score += 20.0
        
        # OHLC data availability
        if all(quote_data.get(k, 0) > 0 for k in ['open', 'high', 'low']):
            score += 15.0
        
        # Volume data
        if quote_data.get('volume', 0) > 1000:
            score += 10.0
        
        # Bid/Ask data
        if quote_data.get('bid', 0) > 0 and quote_data.get('ask', 0) > 0:
            score += 10.0
        
        # Timestamp freshness
        if quote_data.get('timestamp'):
            score += 5.0
        
        return min(100.0, score)

    async def generate_all_signals(self, data_aggregator) -> Dict[str, HYPERSignal]:
        """Generate signals for all configured tickers using data aggregator"""
        logger.info(f"🎯 Generating signals for {len(config.TICKERS)} tickers...")
        
        signals = {}
        
        # Generate signals concurrently
        signal_tasks = []
        for symbol in config.TICKERS:
            task = self._generate_single_signal_with_data(symbol, data_aggregator)
            signal_tasks.append(task)
        
        # Execute all signal generations concurrently
        signal_results = await asyncio.gather(*signal_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(signal_results):
            symbol = config.TICKERS[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Signal generation failed for {symbol}: {result}")
                signals[symbol] = self._generate_fallback_signal(symbol, {'price': 100.0})
            else:
                signals[symbol] = result
        
        logger.info(f"✅ Generated {len(signals)} signals successfully")
        return signals

    async def _generate_single_signal_with_data(self, symbol: str, data_aggregator) -> HYPERSignal:
        """Generate signal for single symbol with data aggregator"""
        try:
            # Get comprehensive data
            data = await data_aggregator.get_comprehensive_data(symbol)
            
            # Generate signal
            return await self.generate_signal(
                symbol=symbol,
                quote_data=data.get('quote', {}),
                trends_data=data.get('trends', {}),
                historical_data=data.get('historical', [])
            )
            
        except Exception as e:
            logger.error(f"❌ Signal generation with data failed for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, {'price': 100.0})

    # Individual analysis method wrappers
    async def _run_technical_analysis(self, symbol: str, quote_data: Dict[str, Any], 
                                     historical_data: Optional[List[Dict]]) -> Optional[TechnicalAnalysis]:
        """Run technical analysis"""
        try:
            return await self.technical_analyzer.analyze(symbol, quote_data, historical_data)
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return None

    async def _run_sentiment_analysis(self, symbol: str, quote_data: Dict[str, Any], 
                                     trends_data: Optional[Dict]) -> Optional[SentimentAnalysis]:
        """Run sentiment analysis"""
        try:
            return await self.sentiment_analyzer.analyze(symbol, quote_data, trends_data)
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return None

    async def _run_vix_analysis(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[VIXAnalysis]:
        """Run VIX analysis"""
        try:
            return await self.vix_analyzer.analyze(symbol, quote_data)
        except Exception as e:
            logger.error(f"VIX analysis error for {symbol}: {e}")
            return None

    async def _run_market_structure_analysis(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[MarketStructureAnalysis]:
        """Run market structure analysis"""
        try:
            return await self.market_structure_analyzer.analyze(symbol, quote_data)
        except Exception as e:
            logger.error(f"Market structure analysis error for {symbol}: {e}")
            return None

    async def _run_risk_analysis(self, symbol: str, quote_data: Dict[str, Any], 
                                historical_data: Optional[List[Dict]]) -> Optional[RiskAnalysis]:
        """Run risk analysis"""
        try:
            return await self.risk_analyzer.analyze(symbol, quote_data, historical_data)
        except Exception as e:
            logger.error(f"Risk analysis error for {symbol}: {e}")
            return None

    def _extract_component_scores(self, technical_analysis: Optional[TechnicalAnalysis],
                                 sentiment_analysis: Optional[SentimentAnalysis],
                                 vix_analysis: Optional[VIXAnalysis],
                                 market_structure_analysis: Optional[MarketStructureAnalysis],
                                 risk_analysis: Optional[RiskAnalysis],
                                 alpaca_features: Dict[str, Any]) -> Dict[str, float]:
        """Extract component scores with Alpaca feature integration"""
        scores = {
            'technical': 50.0,
            'sentiment': 50.0,
            'momentum': 50.0,
            'ml': 50.0,
            'vix': 50.0,
            'market_structure': 50.0,
            'risk': 50.0
        }
        
        # Technical scores with Alpaca data quality adjustment
        if technical_analysis:
            base_score = technical_analysis.overall_score
            
            # Adjust for Alpaca data quality
            quality_multiplier = 1.0
            if alpaca_features.get('is_live_data'):
                quality_multiplier = 1.1  # 10% boost for live data
            elif alpaca_features.get('data_quality_score', 0) > 80:
                quality_multiplier = 1.05  # 5% boost for high quality
            
            scores['technical'] = min(100.0, base_score * quality_multiplier)
            scores['momentum'] = technical_analysis.momentum_analysis.get('momentum_5d', 0) + 50
            
            # Extract specific indicators
            for signal in technical_analysis.signals:
                if signal.indicator_name == "Williams_R":
                    scores['williams_r'] = signal.value
                elif signal.indicator_name == "Stochastic":
                    scores['stochastic_k'] = signal.value
                    scores['stochastic_d'] = signal.value * 0.9
                elif signal.indicator_name == "RSI":
                    scores['rsi'] = signal.value
                elif signal.indicator_name == "MACD":
                    scores['macd_signal'] = signal.direction
            
            # Volume analysis
            volume_analysis = technical_analysis.volume_analysis
            if volume_analysis and isinstance(volume_analysis, dict):
                scores['volume_score'] = volume_analysis.get('volume_quality', 50)
        
        # Sentiment scores
        if sentiment_analysis:
            scores['sentiment'] = sentiment_analysis.overall_sentiment + 50
        
        # VIX scores
        if vix_analysis:
            scores['vix'] = vix_analysis.current_signal.fear_greed_score
            scores['vix_sentiment'] = vix_analysis.current_signal.sentiment
        
        # Market structure scores
        if market_structure_analysis:
            scores['market_structure'] = market_structure_analysis.current_signal.structure_score
            scores['market_regime'] = market_structure_analysis.current_signal.market_regime
            scores['sector_momentum'] = market_structure_analysis.current_signal.rotation_theme
        
        # Risk scores (inverted - lower risk = higher score)
        if risk_analysis:
            scores['risk'] = 100 - risk_analysis.overall_risk_score
        
        # ML score enhanced with Alpaca features
        base_ml_score = (scores['technical'] + scores['sentiment']) / 2
        
        # Boost ML confidence with high-quality Alpaca data
        if alpaca_features.get('data_quality_score', 0) > 90:
            base_ml_score *= 1.15
        elif alpaca_features.get('is_live_data'):
            base_ml_score *= 1.1
        
        scores['ml'] = min(100.0, base_ml_score)
        
        return scores

    def _calculate_weighted_signal(self, component_scores: Dict[str, float], 
                                  quote_data: Dict[str, Any],
                                  alpaca_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted signal with Alpaca data quality considerations"""
        
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
        
        # Alpaca data quality adjustment
        data_quality_score = alpaca_features.get('data_quality_score', 50)
        if data_quality_score > 80:
            confidence_boost = (data_quality_score - 80) / 20 * 5  # Up to 5% boost
            weighted_score *= (1 + confidence_boost / 100)
        
        # Market hours adjustment
        market_hours = alpaca_features.get('market_hours', 'UNKNOWN')
        if market_hours in ['PRE_MARKET', 'AFTER_HOURS']:
            weighted_score *= 0.95  # Slight reduction for extended hours
        
        # Spread quality adjustment
        spread_bps = alpaca_features.get('spread_bps', 50)
        if spread_bps < 10:  # Tight spread = higher quality
            weighted_score *= 1.02
        elif spread_bps > 50:  # Wide spread = lower quality
            weighted_score *= 0.98
        
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

    def _generate_comprehensive_insights(self, technical_analysis: Optional[TechnicalAnalysis],
                                        sentiment_analysis: Optional[SentimentAnalysis],
                                        vix_analysis: Optional[VIXAnalysis],
                                        market_structure_analysis: Optional[MarketStructureAnalysis],
                                        risk_analysis: Optional[RiskAnalysis],
                                        overall_signal: Dict[str, Any],
                                        alpaca_features: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Generate comprehensive insights with Alpaca data context"""
        reasons = []
        warnings = []
        recommendations = []
        
        # Data quality insights
        if alpaca_features.get('is_live_data'):
            reasons.append("📈 Live Alpaca data provides real-time accuracy")
        
        data_quality = alpaca_features.get('data_quality_score', 0)
        if data_quality > 90:
            reasons.append(f"🎯 Excellent data quality ({data_quality:.0f}/100)")
        elif data_quality < 60:
            warnings.append(f"⚠️ Lower data quality ({data_quality:.0f}/100)")
        
        # Market hours context
        market_hours = alpaca_features.get('market_hours', 'UNKNOWN')
        if market_hours in ['PRE_MARKET', 'AFTER_HOURS']:
            warnings.append(f"🕐 Extended hours trading ({market_hours.lower()})")
            recommendations.append("Consider tighter stops during extended hours")
        
        # Spread analysis
        spread_bps = alpaca_features.get('spread_bps', 0)
        if spread_bps > 30:
            warnings.append(f"📊 Wide bid-ask spread ({spread_bps:.1f} bps)")
            recommendations.append("Consider limit orders due to wide spread")
        elif spread_bps < 5:
            reasons.append(f"✅ Tight spread ({spread_bps:.1f} bps) - good liquidity")
        
        # Technical insights
        if technical_analysis:
            if technical_analysis.overall_score > 75:
                reasons.append(f"📊 Strong technical setup ({technical_analysis.overall_score:.0f}/100)")
            elif technical_analysis.overall_score < 35:
                warnings.append(f"📊 Weak technical setup ({technical_analysis.overall_score:.0f}/100)")
        
        # Sentiment insights
        if sentiment_analysis:
            if abs(sentiment_analysis.overall_sentiment) > 40:
                direction = "bullish" if sentiment_analysis.overall_sentiment > 0 else "bearish"
                reasons.append(f"💭 Strong {direction} sentiment")
            
            if sentiment_analysis.contrarian_signals:
                warnings.extend([f"🔄 {signal}" for signal in sentiment_analysis.contrarian_signals[:2]])
        
        # VIX insights
        if vix_analysis:
            vix_signal = vix_analysis.current_signal
            if vix_signal.contrarian_signal in ["STRONG_BUY", "STRONG_SELL"]:
                reasons.append(f"😱 VIX contrarian signal: {vix_signal.contrarian_signal}")
            
            if vix_analysis.risk_warnings:
                warnings.extend([f"⚡ {warning}" for warning in vix_analysis.risk_warnings[:2]])
        
        # Market structure insights
        if market_structure_analysis:
            structure_signal = market_structure_analysis.current_signal
            if structure_signal.structure_score > 80:
                reasons.append(f"🏗️ Strong market structure ({structure_signal.structure_score:.0f}/100)")
            elif structure_signal.structure_score < 40:
                warnings.append(f"🏗️ Weak market structure ({structure_signal.structure_score:.0f}/100)")
        
        # Risk insights
        if risk_analysis:
            if risk_analysis.risk_level == "LOW":
                reasons.append("✅ Low risk environment")
            elif risk_analysis.risk_level in ["HIGH", "EXTREME"]:
                warnings.append(f"⚠️ {risk_analysis.risk_level.lower()} risk detected")
            
            # Position sizing recommendations
            if risk_analysis.position_risk.position_size_recommendation:
                recommended_size = risk_analysis.position_risk.position_size_recommendation
                recommendations.append(f"📊 Suggested position size: {recommended_size:.1%}")
        
        # Overall signal insights
        if overall_signal['confidence'] > 85:
            reasons.append(f"🎯 High confidence signal ({overall_signal['confidence']:.0f}%)")
        elif overall_signal['confidence'] < 40:
            warnings.append(f"❓ Low confidence signal ({overall_signal['confidence']:.0f}%)")
        
        # Alpaca-specific recommendations
        if alpaca_features.get('is_live_data'):
            recommendations.append("📈 Live data enables real-time execution")
        
        if market_hours == 'REGULAR_HOURS':
            recommendations.append("✅ Optimal trading hours for execution")
        
        return reasons[:5], warnings[:3], recommendations[:4]

    def _assess_overall_data_quality(self, technical_analysis: Optional[TechnicalAnalysis],
                                    sentiment_analysis: Optional[SentimentAnalysis],
                                    vix_analysis: Optional[VIXAnalysis],
                                    market_structure_analysis: Optional[MarketStructureAnalysis],
                                    risk_analysis: Optional[RiskAnalysis],
                                    alpaca_features: Dict[str, Any]) -> str:
        """Assess overall data quality with Alpaca metrics"""
        
        quality_scores = []
        quality_weights = []
        
        # Alpaca data quality (highest weight)
        alpaca_quality = alpaca_features.get('data_quality_score', 50)
        quality_scores.append(alpaca_quality)
        quality_weights.append(0.4)
        
        # Technical data quality
        if technical_analysis:
            tech_signals = len(technical_analysis.signals)
            tech_quality = min(100, 50 + tech_signals * 3)  # More signals = better quality
            quality_scores.append(tech_quality)
            quality_weights.append(0.25)
        
        # Sentiment data quality
        if sentiment_analysis:
            sent_signals = len(sentiment_analysis.signals)
            sent_quality = min(100, 40 + sent_signals * 8)
            quality_scores.append(sent_quality)
            quality_weights.append(0.2)
        
        # VIX data quality
        if vix_analysis:
            quality_scores.append(85)  # VIX data is generally reliable
            quality_weights.append(0.1)
        
        # Market structure data quality
        if market_structure_analysis:
            quality_scores.append(75)
            quality_weights.append(0.05)
        
        if quality_scores:
            weighted_quality = sum(s * w for s, w in zip(quality_scores, quality_weights)) / sum(quality_weights)
            
            if weighted_quality >= 90:
                return "excellent"
            elif weighted_quality >= 75:
                return "good"
            elif weighted_quality >= 60:
                return "fair"
            elif weighted_quality >= 45:
                return "acceptable"
            else:
                return "poor"
        else:
            return "unknown"

    def _get_active_components(self) -> List[str]:
        """Get list of active analysis components"""
        components = []
        if self.technical_analyzer:
            components.append("technical")
        if self.sentiment_analyzer:
            components.append("sentiment")
        if self.vix_analyzer:
            components.append("vix")
        if self.market_structure_analyzer:
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
            enhanced_features={
                'fallback': True,
                'alpaca_features': self._extract_alpaca_features(quote_data)
            }
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
                "market_structure_analyzer": self.market_structure_analyzer is not None,
                "risk_analyzer": self.risk_analyzer is not None
            }
        }

    def clear_cache(self):
        """Clear signal cache"""
        self.signal_cache.clear()
        logger.info("🗑️ Signal cache cleared")

    async def warm_up_analyzers(self):
        """Warm up all analyzers with test data"""
        logger.info("🔥 Warming up analyzers...")
        
        test_quote = {
            'symbol': 'TEST',
            'price': 100.0,
            'volume': 1000000,
            'bid': 99.9,
            'ask': 100.1,
            'data_source': 'alpaca_test',
            'enhanced_features': {
                'market_hours': 'REGULAR_HOURS',
                'spread_bps': 10.0,
                'data_freshness': 'real_time'
            }
        }
        
        try:
            await self.generate_signal('TEST', test_quote)
            logger.info("✅ Analyzers warmed up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Analyzer warm-up failed: {e}")

# Export the main engine
__all__ = ['HYPERSignalEngine', 'HYPERSignal']

logger.info("🌟 HYPERtrends v4.0 Signal Engine with Alpaca integration loaded successfully!")
logger.info("🎯 Ready for production-grade signal generation with live market data")