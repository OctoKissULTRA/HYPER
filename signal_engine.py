# signal_engine.py - Modular HYPERtrends Signal Engine v4.0
import logging
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
    """Enhanced HYPER trading signal - Production Ready v4.0"""
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
    vix_sentiment: str = "NEUTRAL"
    
    # Component analysis results
    technical_analysis: Optional[TechnicalAnalysis] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    vix_analysis: Optional[VIXAnalysis] = None
    market_structure_analysis: Optional[MarketStructureAnalysis] = None
    risk_analysis: Optional[RiskAnalysis] = None
    
    # Supporting data
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_quality: str = "unknown"
    
    # Enhanced features for integration
    enhanced_features: Dict[str, Any] = field(default_factory=dict)

class HYPERSignalEngine:
    """Modular HYPER Signal Engine - Orchestrates all analysis components"""
    
    def __init__(self):
        self.config = config
        self.signal_cache = {}
        self.cache_duration = 30  # 30 seconds
        
        # Initialize all modular analyzers
        logger.info("🚀 Initializing HYPERtrends v4.0 Modular Signal Engine...")
        
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
            logger.info("✅ Market Structure Analyzer loaded (breadth + sectors)")
        else:
            self.market_structure_analyzer = None
            logger.info("⚠️ Market Structure Analyzer disabled")
        
        # Risk Analysis
        if config.is_feature_enabled('risk_analysis'):
            self.risk_analyzer = AdvancedRiskAnalyzer(config.RISK_CONFIG)
            logger.info("✅ Risk Analyzer loaded (VaR + position sizing)")
        else:
            self.risk_analyzer = None
            logger.info("⚠️ Risk Analyzer disabled")
        
        logger.info("🌟 HYPERtrends Modular Signal Engine initialized successfully!")
    
    async def generate_signal(self, symbol: str, quote_data: Dict[str, Any], 
                             trends_data: Optional[Dict] = None,
                             historical_data: Optional[List[Dict]] = None) -> HYPERSignal:
        """Generate comprehensive HYPER signal using all modular components"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{time.time() // self.cache_duration}"
            if cache_key in self.signal_cache:
                logger.debug(f"📋 Using cached signal for {symbol}")
                return self.signal_cache[cache_key]
            
            logger.debug(f"🎯 Generating modular signal for {symbol}...")
            start_time = time.time()
            
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
            
            # Extract component scores
            component_scores = self._extract_component_scores(
                technical_analysis, sentiment_analysis, vix_analysis, 
                market_structure_analysis, risk_analysis
            )
            
            # Calculate weighted overall signal
            overall_signal = self._calculate_weighted_signal(component_scores, quote_data)
            
            # Generate comprehensive reasons and warnings
            reasons, warnings = self._generate_comprehensive_insights(
                technical_analysis, sentiment_analysis, vix_analysis,
                market_structure_analysis, risk_analysis, overall_signal
            )
            
            # Assess data quality
            data_quality = self._assess_overall_data_quality(
                technical_analysis, sentiment_analysis, vix_analysis,
                market_structure_analysis, risk_analysis
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
                vix_sentiment=component_scores.get('vix_sentiment', 'NEUTRAL'),
                
                # Component analysis results
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                vix_analysis=vix_analysis,
                market_structure_analysis=market_structure_analysis,
                risk_analysis=risk_analysis,
                
                # Supporting data
                reasons=reasons,
                warnings=warnings,
                data_quality=data_quality,
                
                # Enhanced features
                enhanced_features={
                    'generation_time': time.time() - start_time,
                    'components_used': self._get_active_components(),
                    'data_sources': quote_data.get('enhanced_features', {}),
                    'analysis_depth': 'comprehensive'
                }
            )
            
            # Cache the signal
            self.signal_cache[cache_key] = signal
            
            generation_time = time.time() - start_time
            logger.debug(f"✅ Generated {signal.signal_type} signal for {symbol} "
                        f"({signal.confidence:.0f}% confidence) in {generation_time:.2f}s")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal generation error for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, quote_data)
    
    async def generate_all_signals(self) -> Dict[str, HYPERSignal]:
        """Generate signals for all configured tickers"""
        logger.info(f"🎯 Generating signals for {len(config.TICKERS)} tickers...")
        
        signals = {}
        
        # Generate signals concurrently for better performance
        signal_tasks = []
        for symbol in config.TICKERS:
            # Note: In production, you'd get actual quote_data from your data aggregator
            # For now, using placeholder data
            quote_data = {
                'symbol': symbol,
                'price': 100.0,  # Placeholder - replace with real data
                'change_percent': 0.0,  # Placeholder - replace with real data
                'volume': 25000000,  # Placeholder - replace with real data
                'timestamp': datetime.now().isoformat()
            }
            
            signal_tasks.append(self.generate_signal(symbol, quote_data))
        
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
                                 risk_analysis: Optional[RiskAnalysis]) -> Dict[str, float]:
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
        
        # Technical scores
        if technical_analysis:
            scores['technical'] = technical_analysis.overall_score
            scores['momentum'] = technical_analysis.momentum_analysis.get('momentum_5d', 0) + 50
            
            # Extract specific indicators
            for signal in technical_analysis.signals:
                if signal.indicator_name == "Williams_R":
                    scores['williams_r'] = signal.value
                elif signal.indicator_name == "Stochastic":
                    scores['stochastic_k'] = signal.value
                    scores['stochastic_d'] = signal.value * 0.9  # Approximate %D
        
        # Sentiment scores
        if sentiment_analysis:
            scores['sentiment'] = sentiment_analysis.overall_sentiment + 50  # Convert to 0-100 scale
        
        # VIX scores
        if vix_analysis:
            scores['vix'] = vix_analysis.current_signal.fear_greed_score
            scores['vix_sentiment'] = vix_analysis.current_signal.sentiment
        
        # Market structure scores
        if market_structure_analysis:
            scores['market_structure'] = market_structure_analysis.current_signal.structure_score
        
        # Risk scores (inverted - lower risk = higher score)
        if risk_analysis:
            scores['risk'] = 100 - risk_analysis.overall_risk_score
        
        # ML score (placeholder - integrate with your ML module)
        scores['ml'] = (scores['technical'] + scores['sentiment']) / 2
        
        return scores
    
    def _calculate_weighted_signal(self, component_scores: Dict[str, float], 
                                  quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted overall signal using configured weights"""
        
        # Apply signal weights from config
        weighted_score = (
            component_scores['technical'] * config.SIGNAL_WEIGHTS['technical'] +
            component_scores['sentiment'] * config.SIGNAL_WEIGHTS['sentiment'] +
            component_scores['momentum'] * config.SIGNAL_WEIGHTS['momentum'] +
            component_scores['ml'] * config.SIGNAL_WEIGHTS['ml_prediction'] +
            component_scores['market_structure'] * config.SIGNAL_WEIGHTS['market_structure'] +
            component_scores['vix'] * config.SIGNAL_WEIGHTS['vix_sentiment'] +
            component_scores['risk'] * config.SIGNAL_WEIGHTS['risk_adjusted']
        )
        
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
                                        overall_signal: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate comprehensive reasons and warnings from all components"""
        reasons = []
        warnings = []
        
        # Technical insights
        if technical_analysis:
            if technical_analysis.overall_score > 70:
                reasons.append(f"📊 Strong technical setup ({technical_analysis.overall_score:.0f}/100)")
            elif technical_analysis.overall_score < 30:
                reasons.append(f"📊 Weak technical setup ({technical_analysis.overall_score:.0f}/100)")
            
            # Add pattern insights
            if technical_analysis.pattern_analysis.get('primary_pattern'):
                pattern = technical_analysis.pattern_analysis['primary_pattern']
                if pattern.get('confidence', 0) > 0.7:
                    reasons.append(f"📈 Strong {pattern.get('pattern', 'pattern')} detected")
        
        # Sentiment insights
        if sentiment_analysis:
            if abs(sentiment_analysis.overall_sentiment) > 30:
                direction = "bullish" if sentiment_analysis.overall_sentiment > 0 else "bearish"
                reasons.append(f"💭 Strong {direction} sentiment ({sentiment_analysis.overall_sentiment:+.0f})")
            
            # Contrarian warnings
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
            
            # Regime insights
            if structure_signal.market_regime in ["RISK_ON", "RISK_OFF"]:
                reasons.append(f"📊 {structure_signal.market_regime.replace('_', '-')} environment")
        
        # Risk insights
        if risk_analysis:
            if risk_analysis.risk_level == "LOW":
                reasons.append("✅ Low risk environment")
            elif risk_analysis.risk_level in ["HIGH", "EXTREME"]:
                warnings.append(f"⚠️ {risk_analysis.risk_level.lower()} risk detected")
            
            # Position sizing insights
            if risk_analysis.position_risk.concentration_risk == "HIGH":
                warnings.append("⚖️ High concentration risk - reduce position size")
        
        # Overall signal insights
        if overall_signal['confidence'] > 80:
            reasons.append(f"🎯 High confidence signal ({overall_signal['confidence']:.0f}%)")
        elif overall_signal['confidence'] < 40:
            warnings.append(f"❓ Low confidence signal ({overall_signal['confidence']:.0f}%)")
        
        return reasons[:5], warnings[:3]  # Limit to top insights
    
    def _assess_overall_data_quality(self, technical_analysis: Optional[TechnicalAnalysis],
                                    sentiment_analysis: Optional[SentimentAnalysis],
                                    vix_analysis: Optional[VIXAnalysis],
                                    market_structure_analysis: Optional[MarketStructureAnalysis],
                                    risk_analysis: Optional[RiskAnalysis]) -> str:
        """Assess overall data quality across all components"""
        
        quality_scores = []
        quality_weights = []
        
        # Technical data quality
        if technical_analysis:
            tech_signals = len(technical_analysis.signals)
            tech_quality = "excellent" if tech_signals > 15 else "good" if tech_signals > 10 else "fair"
            quality_scores.append({"excellent": 90, "good": 75, "fair": 60}.get(tech_quality, 50))
            quality_weights.append(0.3)
        
        # Sentiment data quality
        if sentiment_analysis:
            sent_signals = len(sentiment_analysis.signals)
            sent_quality = "excellent" if sent_signals > 3 else "good" if sent_signals > 2 else "fair"
            quality_scores.append({"excellent": 85, "good": 70, "fair": 55}.get(sent_quality, 50))
            quality_weights.append(0.25)
        
        # VIX data quality
        if vix_analysis:
            quality_scores.append(80)  # VIX data is generally reliable
            quality_weights.append(0.15)
        
        # Market structure data quality
        if market_structure_analysis:
            quality_scores.append(75)  # Structure data is moderately reliable
            quality_weights.append(0.2)
        
        # Risk data quality
        if risk_analysis:
            quality_scores.append(70)  # Risk calculations are estimation-based
            quality_weights.append(0.1)
        
        if quality_scores:
            weighted_quality = sum(s * w for s, w in zip(quality_scores, quality_weights)) / sum(quality_weights)
            
            if weighted_quality >= 85:
                return "excellent"
            elif weighted_quality >= 70:
                return "good"
            elif weighted_quality >= 55:
                return "fair"
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
            data_quality="poor",
            enhanced_features={'fallback': True}
        )

# Export the main engine
__all__ = ['HYPERSignalEngine', 'HYPERSignal']

logger.info("🌟 HYPERtrends v4.0 Modular Signal Engine loaded successfully!")
logger.info("🎯 Ready for advanced multi-component signal generation")
