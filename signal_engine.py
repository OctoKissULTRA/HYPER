import logging
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HYPERSignal:
    """HYPER trading signal with full context"""
    symbol: str
    signal_type: str  # HYPER_BUY, SOFT_BUY, HOLD, SOFT_SELL, HYPER_SELL
    confidence: float  # 0-100
    direction: str    # UP, DOWN, NEUTRAL
    price: float
    timestamp: str
    
    # Signal component scores
    technical_score: float
    momentum_score: float
    trends_score: float
    volume_score: float
    ml_score: float
    
    # Supporting data
    indicators: Dict[str, Any]
    reasons: List[str]
    warnings: List[str]
    data_quality: str

class TechnicalAnalyzer:
    """Technical analysis engine"""
    
    def analyze_price_action(self, quote_data: Dict) -> Dict[str, Any]:
        """Analyze price action from quote data"""
        if not quote_data or quote_data.get('price', 0) <= 0:
            return self._empty_technical_analysis()
        
        try:
            price = quote_data['price']
            change = float(quote_data.get('change', 0))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            high = quote_data.get('high', price)
            low = quote_data.get('low', price)
            
            # Calculate technical indicators
            signals = []
            score = 50  # Start neutral
            
            # Price momentum analysis
            if change_percent > 2.0:
                signals.append("Strong upward momentum")
                score += 20
            elif change_percent > 0.5:
                signals.append("Positive momentum")
                score += 10
            elif change_percent < -2.0:
                signals.append("Strong downward momentum")
                score -= 20
            elif change_percent < -0.5:
                signals.append("Negative momentum")
                score -= 10
            
            # Volume analysis
            if volume > 10000000:  # High volume
                signals.append("High volume confirmation")
                if change_percent > 0:
                    score += 15
                else:
                    score -= 15
            elif volume < 1000000:  # Low volume
                signals.append("Low volume - weak signal")
                score -= 5
            
            # Range analysis
            if high > 0 and low > 0:
                range_percent = ((high - low) / price) * 100
                if range_percent > 3:
                    signals.append("High volatility")
                    score += 5
                elif range_percent < 1:
                    signals.append("Low volatility")
                    score -= 5
            
            # Generate RSI-like indicator from price action
            rsi = self._calculate_pseudo_rsi(change_percent)
            if rsi > 70:
                signals.append("Overbought conditions")
                score -= 10
            elif rsi < 30:
                signals.append("Oversold conditions")
                score += 10
            
            direction = 'UP' if score > 55 else 'DOWN' if score < 45 else 'NEUTRAL'
            
            return {
                'score': max(0, min(100, score)),
                'rsi': rsi,
                'volume_ratio': volume / 10000000,  # Normalize volume
                'range_percent': ((high - low) / price) * 100 if price > 0 else 0,
                'signals': signals,
                'direction': direction,
                'change_percent': change_percent,
                'volume': volume
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return self._empty_technical_analysis()
    
    def _calculate_pseudo_rsi(self, change_percent: float) -> float:
        """Calculate a pseudo-RSI from recent change"""
        # Simple RSI approximation based on recent change
        if change_percent == 0:
            return 50
        
        # Map change percent to RSI-like value
        rsi = 50 + (change_percent * 10)  # Each 1% change = 10 RSI points
        return max(0, min(100, rsi))
    
    def _empty_technical_analysis(self) -> Dict[str, Any]:
        """Return empty technical analysis"""
        return {
            'score': 50,
            'rsi': 50,
            'volume_ratio': 1,
            'range_percent': 0,
            'signals': ['No technical data available'],
            'direction': 'NEUTRAL',
            'change_percent': 0,
            'volume': 0
        }

class SentimentAnalyzer:
    """Sentiment analysis from trends data"""
    
    def analyze_trends_sentiment(self, trends_data: Dict, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from trends data"""
        if not trends_data or 'keyword_data' not in trends_data:
            return self._empty_sentiment_analysis()
        
        try:
            keyword_data = trends_data['keyword_data']
            signals = []
            score = 50  # Start neutral
            
            total_momentum = 0
            total_velocity = 0
            keyword_count = 0
            
            for keyword, data in keyword_data.items():
                momentum = data.get('momentum', 0)
                velocity = data.get('velocity', 0)
                
                total_momentum += momentum
                total_velocity += velocity
                keyword_count += 1
                
                # Generate signals based on trends
                if momentum > 50:
                    signals.append(f"High interest in {keyword}")
                    score += 10
                elif momentum < -20:
                    signals.append(f"Declining interest in {keyword}")
                    score -= 8
                
                if velocity > 20:
                    signals.append(f"Accelerating interest in {keyword}")
                    score += 5
            
            if keyword_count > 0:
                avg_momentum = total_momentum / keyword_count
                avg_velocity = total_velocity / keyword_count
                
                # Apply sentiment scoring
                if avg_momentum > 100:
                    signals.append("Extreme hype detected")
                    score -= 15  # Contrarian signal
                elif avg_momentum > 25:
                    score += 15
                elif avg_momentum < -25:
                    score -= 10
                
                direction = 'UP' if score > 55 else 'DOWN' if score < 45 else 'NEUTRAL'
            else:
                avg_momentum = 0
                avg_velocity = 0
                direction = 'NEUTRAL'
            
            return {
                'score': max(0, min(100, score)),
                'momentum': avg_momentum,
                'velocity': avg_velocity,
                'signals': signals,
                'direction': direction,
                'keywords_analyzed': keyword_count
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._empty_sentiment_analysis()
    
    def _empty_sentiment_analysis(self) -> Dict[str, Any]:
        """Return empty sentiment analysis"""
        return {
            'score': 50,
            'momentum': 0,
            'velocity': 0,
            'signals': ['No sentiment data available'],
            'direction': 'NEUTRAL',
            'keywords_analyzed': 0
        }

class RiskAnalyzer:
    """Risk and fake-out detection"""
    
    def analyze_risk(self, technical_data: Dict, sentiment_data: Dict, quote_data: Dict) -> Dict[str, Any]:
        """Analyze risk factors"""
        warnings = []
        confidence_penalty = 0
        
        try:
            # Volume analysis
            volume = quote_data.get('volume', 0) if quote_data else 0
            if volume < 1000000:
                warnings.append("Low volume - potential fake breakout")
                confidence_penalty += 0.15
            
            # Extreme price movements
            change_percent = abs(float(quote_data.get('change_percent', 0))) if quote_data else 0
            if change_percent > 5:
                warnings.append("Extreme price movement - high risk")
                confidence_penalty += 0.20
            
            # Sentiment extremes
            sentiment_momentum = abs(sentiment_data.get('momentum', 0))
            if sentiment_momentum > 200:
                warnings.append("Extreme sentiment - contrarian risk")
                confidence_penalty += 0.25
            
            # Technical divergence
            tech_direction = technical_data.get('direction', 'NEUTRAL')
            sentiment_direction = sentiment_data.get('direction', 'NEUTRAL')
            
            if (tech_direction != sentiment_direction and 
                tech_direction != 'NEUTRAL' and 
                sentiment_direction != 'NEUTRAL'):
                warnings.append("Technical vs sentiment divergence")
                confidence_penalty += 0.10
            
            return {
                'warnings': warnings,
                'confidence_penalty': confidence_penalty,
                'risk_score': min(100, confidence_penalty * 100)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {
                'warnings': ['Risk analysis failed'],
                'confidence_penalty': 0.1,
                'risk_score': 10
            }

class HYPERSignalEngine:
    """Main signal generation engine"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from data_sources import HYPERDataAggregator
        self.data_aggregator = HYPERDataAggregator()
        
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        # Signal weights
        self.weights = {
            'technical': 0.35,
            'momentum': 0.25,
            'trends': 0.15,
            'volume': 0.15,
            'ml': 0.10
        }
        
        # Confidence thresholds
        self.thresholds = {
            'HYPER_BUY': 85,
            'SOFT_BUY': 65,
            'HOLD': 35,
            'SOFT_SELL': 65,
            'HYPER_SELL': 85
        }
        
        logger.info("🧠 HYPER Signal Engine initialized")
    
    async def generate_signal(self, symbol: str) -> HYPERSignal:
        """Generate a comprehensive trading signal"""
        logger.info(f"🎯 Generating signal for {symbol}")
        
        try:
            # Get comprehensive data
            data = await self.data_aggregator.get_comprehensive_data(symbol)
            
            if not data or data.get('data_quality') == 'error':
                return self._create_error_signal(symbol, "Data retrieval failed")
            
            quote_data = data.get('quote')
            trends_data = data.get('trends')
            
            if not quote_data:
                return self._create_error_signal(symbol, "No quote data available")
            
            # Run all analyses
            technical_analysis = self.technical_analyzer.analyze_price_action(quote_data)
            sentiment_analysis = self.sentiment_analyzer.analyze_trends_sentiment(trends_data, symbol)
            risk_analysis = self.risk_analyzer.analyze_risk(technical_analysis, sentiment_analysis, quote_data)
            
            # Calculate component scores
            technical_score = technical_analysis['score']
            momentum_score = self._calculate_momentum_score(quote_data)
            trends_score = sentiment_analysis['score']
            volume_score = self._calculate_volume_score(quote_data)
            ml_score = 50 + random.uniform(-10, 10)  # Mock ML score
            
            # Calculate weighted confidence
            weighted_confidence = (
                technical_score * self.weights['technical'] +
                momentum_score * self.weights['momentum'] +
                trends_score * self.weights['trends'] +
                volume_score * self.weights['volume'] +
                ml_score * self.weights['ml']
            )
            
            # Apply risk penalty
            confidence_penalty = risk_analysis['confidence_penalty']
            final_confidence = max(0, weighted_confidence * (1 - confidence_penalty))
            
            # Determine direction and signal type
            direction = self._determine_direction(technical_analysis, sentiment_analysis, momentum_score)
            signal_type = self._classify_signal(final_confidence, direction)
            
            # Compile reasons
            reasons = []
            reasons.extend(technical_analysis.get('signals', []))
            reasons.extend(sentiment_analysis.get('signals', []))
            
            if momentum_score > 70:
                reasons.append("Strong price momentum detected")
            elif momentum_score < 30:
                reasons.append("Weak price momentum")
            
            # Create the signal
            signal = HYPERSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=round(final_confidence, 1),
                direction=direction,
                price=quote_data['price'],
                timestamp=datetime.now().isoformat(),
                technical_score=technical_score,
                momentum_score=momentum_score,
                trends_score=trends_score,
                volume_score=volume_score,
                ml_score=ml_score,
                indicators={
                    'rsi': technical_analysis.get('rsi', 50),
                    'volume_ratio': technical_analysis.get('volume_ratio', 1),
                    'change_percent': technical_analysis.get('change_percent', 0),
                    'trend_momentum': sentiment_analysis.get('momentum', 0)
                },
                reasons=reasons[:3],  # Limit to top 3 reasons
                warnings=risk_analysis['warnings'],
                data_quality=data.get('data_quality', 'unknown')
            )
            
            logger.info(f"✅ Generated {signal.signal_type} signal for {symbol} with {signal.confidence}% confidence")
            return signal
            
        except Exception as e:
            logger.error(f"💥 Signal generation error for {symbol}: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return self._create_error_signal(symbol, f"Generation error: {str(e)}")
    
    def _calculate_momentum_score(self, quote_data: Dict) -> float:
        """Calculate momentum score from price data"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            # Convert change percent to 0-100 score
            score = 50 + (change_percent * 10)  # Each 1% = 10 points
            return max(0, min(100, score))
        except:
            return 50
    
    def _calculate_volume_score(self, quote_data: Dict) -> float:
        """Calculate volume score"""
        try:
            volume = quote_data.get('volume', 0)
            if volume > 20000000:
                return 90  # Very high volume
            elif volume > 10000000:
                return 75  # High volume
            elif volume > 5000000:
                return 60  # Normal volume
            elif volume > 1000000:
                return 45  # Low volume
            else:
                return 25  # Very low volume
        except:
            return 50
    
    def _determine_direction(self, technical: Dict, sentiment: Dict, momentum: float) -> str:
        """Determine overall signal direction"""
        directions = []
        
        if technical.get('direction') == 'UP':
            directions.append('UP')
        elif technical.get('direction') == 'DOWN':
            directions.append('DOWN')
        
        if sentiment.get('direction') == 'UP':
            directions.append('UP')
        elif sentiment.get('direction') == 'DOWN':
            directions.append('DOWN')
        
        if momentum > 60:
            directions.append('UP')
        elif momentum < 40:
            directions.append('DOWN')
        
        up_count = directions.count('UP')
        down_count = directions.count('DOWN')
        
        if up_count > down_count:
            return 'UP'
        elif down_count > up_count:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _classify_signal(self, confidence: float, direction: str) -> str:
        """Classify signal based on confidence and direction"""
        if direction == 'UP':
            if confidence >= self.thresholds['HYPER_BUY']:
                return 'HYPER_BUY'
            elif confidence >= self.thresholds['SOFT_BUY']:
                return 'SOFT_BUY'
        elif direction == 'DOWN':
            if confidence >= self.thresholds['HYPER_SELL']:
                return 'HYPER_SELL'
            elif confidence >= self.thresholds['SOFT_SELL']:
                return 'SOFT_SELL'
        
        return 'HOLD'
    
    def _create_error_signal(self, symbol: str, error_reason: str) -> HYPERSignal:
        """Create an error/fallback signal"""
        return HYPERSignal(
            symbol=symbol,
            signal_type='HOLD',
            confidence=0.0,
            direction='NEUTRAL',
            price=0.0,
            timestamp=datetime.now().isoformat(),
            technical_score=50,
            momentum_score=50,
            trends_score=50,
            volume_score=50,
            ml_score=50,
            indicators={},
            reasons=[error_reason],
            warnings=['Data unavailable'],
            data_quality='error'
        )
    
    async def generate_all_signals(self) -> Dict[str, HYPERSignal]:
        """Generate signals for all tickers"""
        import config
        tickers = config.TICKERS
        logger.info(f"🎯 Generating signals for {len(tickers)} tickers: {tickers}")
        
        signals = {}
        for ticker in tickers:
            try:
                signal = await self.generate_signal(ticker)
                signals[ticker] = signal
            except Exception as e:
                logger.error(f"❌ Failed to generate signal for {ticker}: {e}")
                signals[ticker] = self._create_error_signal(ticker, f"Generation failed: {str(e)}")
        
        logger.info(f"✅ Generated {len(signals)} signals")
        return signals
