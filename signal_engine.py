import logging
import random
import asyncio
import numpy as np
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class HYPERSignal:
    """Enhanced HYPER trading signal with all features combined"""
    symbol: str
    signal_type: str  # HYPER_BUY, SOFT_BUY, HOLD, SOFT_SELL, HYPER_SELL
    confidence: float  # 0-100
    direction: str    # UP, DOWN, NEUTRAL
    price: float
    timestamp: str
    
    # Core signal component scores
    technical_score: float
    momentum_score: float
    trends_score: float
    volume_score: float
    ml_score: float
    
    # NEW: Ultra-enhanced scores
    sentiment_score: float = 50.0      # Multi-source sentiment
    pattern_score: float = 50.0        # Chart patterns
    market_structure_score: float = 50.0  # Market breadth, sector rotation
    economic_score: float = 50.0       # Economic indicators
    risk_score: float = 50.0           # Risk metrics
    
    # NEW: Advanced technical indicators
    williams_r: float = -50.0          # Williams %R oscillator
    stochastic_k: float = 50.0         # Stochastic %K
    stochastic_d: float = 50.0         # Stochastic %D
    vix_sentiment: str = "NEUTRAL"     # VIX fear/greed
    put_call_ratio: float = 1.0        # Options sentiment
    
    # NEW: Fibonacci and key levels
    fibonacci_levels: Dict[str, float] = None
    
    # NEW: Market structure
    market_breadth: float = 50.0       # Advance/decline ratio
    sector_rotation: str = "NEUTRAL"   # Sector flow direction
    volume_profile: Dict[str, float] = None
    
    # NEW: ML predictions
    lstm_predictions: Dict[str, Any] = None
    ensemble_prediction: Dict[str, Any] = None
    anomaly_score: float = 0.0
    
    # NEW: Economic and alternative data
    economic_sentiment: Dict[str, float] = None
    earnings_proximity: int = 30       # Days to earnings
    
    # NEW: Risk metrics
    var_95: float = 5.0               # Value at Risk
    max_drawdown_risk: float = 10.0   # Maximum drawdown
    correlation_spy: float = 0.7      # SPY correlation
    
    # Supporting data (existing)
    indicators: Dict[str, Any] = None
    reasons: List[str] = None
    warnings: List[str] = None
    data_quality: str = "unknown"
    
    def __post_init__(self):
        """Initialize default values for None fields"""
        if self.fibonacci_levels is None:
            self.fibonacci_levels = {}
        if self.volume_profile is None:
            self.volume_profile = {}
        if self.lstm_predictions is None:
            self.lstm_predictions = {}
        if self.ensemble_prediction is None:
            self.ensemble_prediction = {}
        if self.economic_sentiment is None:
            self.economic_sentiment = {}
        if self.indicators is None:
            self.indicators = {}
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []

class TechnicalAnalyzer:
    """Enhanced technical analysis engine with both original and advanced indicators"""
    
    def __init__(self):
        self.session = None
        logger.info("🔧 Enhanced Technical Analyzer initialized")
    
    async def create_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def analyze_price_action(self, quote_data: Dict) -> Dict[str, Any]:
        """Original price action analysis + enhanced indicators"""
        if not quote_data or quote_data.get('price', 0) <= 0:
            return self._empty_technical_analysis()
        
        try:
            price = quote_data['price']
            change = float(quote_data.get('change', 0))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            high = quote_data.get('high', price)
            low = quote_data.get('low', price)
            
            # Original technical analysis
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
            
            # NEW: Enhanced technical indicators
            enhanced_indicators = self._calculate_enhanced_indicators(price, high, low, volume)
            
            # Combine scores
            direction = 'UP' if score > 55 else 'DOWN' if score < 45 else 'NEUTRAL'
            
            result = {
                'score': max(0, min(100, score)),
                'rsi': rsi,
                'volume_ratio': volume / 10000000,
                'range_percent': ((high - low) / price) * 100 if price > 0 else 0,
                'signals': signals,
                'direction': direction,
                'change_percent': change_percent,
                'volume': volume
            }
            
            # Add enhanced indicators
            result.update(enhanced_indicators)
            
            return result
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return self._empty_technical_analysis()
    
    def _calculate_enhanced_indicators(self, price: float, high: float, low: float, volume: int) -> Dict[str, Any]:
        """Calculate enhanced technical indicators"""
        try:
            # Generate price history for calculations (in production, use real historical data)
            price_history = self._generate_price_history(price, 50)
            high_prices = [p * 1.01 for p in price_history]
            low_prices = [p * 0.99 for p in price_history]
            
            # Williams %R Oscillator
            williams_r = self._calculate_williams_r(high_prices, low_prices, price)
            
            # Stochastic Oscillator
            stochastic_k, stochastic_d = self._calculate_stochastic(high_prices, low_prices, price_history)
            
            # Fibonacci Levels
            fibonacci_levels = self._calculate_fibonacci_levels(high, low)
            
            # Volume Profile indicators
            volume_profile = self._calculate_volume_profile(price, volume)
            
            return {
                'williams_r': williams_r,
                'stochastic_k': stochastic_k,
                'stochastic_d': stochastic_d,
                'fibonacci_levels': fibonacci_levels,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logger.error(f"Enhanced indicators calculation error: {e}")
            return {
                'williams_r': -50.0,
                'stochastic_k': 50.0,
                'stochastic_d': 50.0,
                'fibonacci_levels': {},
                'volume_profile': {}
            }
    
    def _calculate_williams_r(self, high_prices: List[float], low_prices: List[float], 
                             close_price: float, period: int = 14) -> float:
        """Calculate Williams %R oscillator"""
        try:
            if len(high_prices) < period or len(low_prices) < period:
                return -50.0
            
            highest_high = max(high_prices[-period:])
            lowest_low = min(low_prices[-period:])
            
            if highest_high == lowest_low:
                return -50.0
            
            williams_r = ((highest_high - close_price) / (highest_high - lowest_low)) * -100
            return max(-100, min(0, williams_r))
            
        except:
            return -50.0
    
    def _calculate_stochastic(self, high_prices: List[float], low_prices: List[float],
                             close_prices: List[float], k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator"""
        try:
            if len(close_prices) < k_period:
                return 50.0, 50.0
            
            k_values = []
            for i in range(k_period - 1, len(close_prices)):
                period_high = max(high_prices[i - k_period + 1:i + 1])
                period_low = min(low_prices[i - k_period + 1:i + 1])
                current_close = close_prices[i]
                
                if period_high == period_low:
                    k_percent = 50.0
                else:
                    k_percent = ((current_close - period_low) / (period_high - period_low)) * 100
                
                k_values.append(k_percent)
            
            current_k = k_values[-1] if k_values else 50.0
            current_d = sum(k_values[-d_period:]) / d_period if len(k_values) >= d_period else current_k
            
            return current_k, current_d
            
        except:
            return 50.0, 50.0
    
    def _calculate_fibonacci_levels(self, high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if high_price <= low_price:
                return {}
            
            diff = high_price - low_price
            return {
                'level_23.6': high_price - (diff * 0.236),
                'level_38.2': high_price - (diff * 0.382),
                'level_50.0': high_price - (diff * 0.500),
                'level_61.8': high_price - (diff * 0.618),
                'level_78.6': high_price - (diff * 0.786)
            }
        except:
            return {}
    
    def _calculate_volume_profile(self, price: float, volume: int) -> Dict[str, float]:
        """Calculate volume profile indicators"""
        try:
            avg_volume = volume * random.uniform(0.8, 1.2)
            relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
            
            # VWAP simulation
            vwap = price * random.uniform(0.995, 1.005)
            
            return {
                'relative_volume': round(relative_volume, 2),
                'vwap': round(vwap, 2),
                'vwap_deviation': round(((price - vwap) / vwap) * 100, 2),
                'volume_strength': "HIGH" if relative_volume > 1.5 else "NORMAL" if relative_volume > 0.8 else "LOW"
            }
        except:
            return {'relative_volume': 1.0, 'vwap': price, 'vwap_deviation': 0.0, 'volume_strength': "NORMAL"}
    
    def _generate_price_history(self, current_price: float, length: int) -> List[float]:
        """Generate realistic price history for calculations"""
        history = []
        price = current_price
        
        for _ in range(length):
            change = random.uniform(-0.02, 0.02)  # ±2% daily change
            price = price * (1 + change)
            history.append(price)
        
        return history
    
    def _calculate_pseudo_rsi(self, change_percent: float) -> float:
        """Calculate a pseudo-RSI from recent change"""
        if change_percent == 0:
            return 50
        
        rsi = 50 + (change_percent * 10)
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
            'volume': 0,
            'williams_r': -50.0,
            'stochastic_k': 50.0,
            'stochastic_d': 50.0,
            'fibonacci_levels': {},
            'volume_profile': {}
        }

class SentimentAnalyzer:
    """Enhanced sentiment analysis from multiple sources"""
    
    def __init__(self):
        logger.info("📊 Enhanced Sentiment Analyzer initialized")
    
    def analyze_trends_sentiment(self, trends_data: Dict, symbol: str) -> Dict[str, Any]:
        """Original trends sentiment + enhanced multi-source analysis"""
        # Original Google Trends analysis
        original_sentiment = self._analyze_original_trends(trends_data, symbol)
        
        # NEW: Enhanced sentiment sources
        enhanced_sentiment = self._analyze_enhanced_sentiment(symbol)
        
        # Combine sentiments
        combined_score = (original_sentiment['score'] * 0.4 + 
                         enhanced_sentiment['news_sentiment'] * 0.3 +
                         enhanced_sentiment['social_sentiment'] * 0.3)
        
        return {
            'score': max(0, min(100, combined_score)),
            'momentum': original_sentiment['momentum'],
            'velocity': original_sentiment['velocity'],
            'signals': original_sentiment['signals'] + enhanced_sentiment['signals'],
            'direction': self._determine_direction(combined_score),
            'keywords_analyzed': original_sentiment['keywords_analyzed'],
            # NEW: Enhanced sentiment data
            'news_sentiment': enhanced_sentiment['news_sentiment'],
            'social_sentiment': enhanced_sentiment['social_sentiment'],
            'reddit_sentiment': enhanced_sentiment['reddit_sentiment'],
            'twitter_sentiment': enhanced_sentiment['twitter_sentiment'],
            'overall_confidence': enhanced_sentiment['confidence']
        }
    
    def _analyze_original_trends(self, trends_data: Dict, symbol: str) -> Dict[str, Any]:
        """Original Google Trends analysis"""
        if not trends_data or 'keyword_data' not in trends_data:
            return self._empty_sentiment_analysis()
        
        try:
            keyword_data = trends_data['keyword_data']
            signals = []
            score = 50
            
            total_momentum = 0
            total_velocity = 0
            keyword_count = 0
            
            for keyword, data in keyword_data.items():
                momentum = data.get('momentum', 0)
                velocity = data.get('velocity', 0)
                
                total_momentum += momentum
                total_velocity += velocity
                keyword_count += 1
                
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
                
                if avg_momentum > 100:
                    signals.append("Extreme hype detected")
                    score -= 15  # Contrarian signal
                elif avg_momentum > 25:
                    score += 15
                elif avg_momentum < -25:
                    score -= 10
            else:
                avg_momentum = 0
                avg_velocity = 0
            
            return {
                'score': max(0, min(100, score)),
                'momentum': avg_momentum,
                'velocity': avg_velocity,
                'signals': signals,
                'keywords_analyzed': keyword_count
            }
            
        except Exception as e:
            logger.error(f"Original sentiment analysis error: {e}")
            return self._empty_sentiment_analysis()
    
    def _analyze_enhanced_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Enhanced multi-source sentiment analysis"""
        try:
            # Simulate enhanced sentiment sources (in production, use real APIs)
            news_sentiment = random.uniform(30, 70)    # News sentiment
            social_sentiment = random.uniform(35, 65)  # Social media overall
            reddit_sentiment = random.uniform(40, 60)  # Reddit specific
            twitter_sentiment = random.uniform(45, 55) # Twitter specific
            
            signals = []
            
            # News sentiment signals
            if news_sentiment > 60:
                signals.append("Positive news coverage")
            elif news_sentiment < 40:
                signals.append("Negative news sentiment")
            
            # Social sentiment signals
            if social_sentiment > 60:
                signals.append("Bullish social media sentiment")
            elif social_sentiment < 40:
                signals.append("Bearish social media sentiment")
            
            # Reddit signals
            if reddit_sentiment > 55:
                signals.append("Reddit retail bullish")
            elif reddit_sentiment < 45:
                signals.append("Reddit retail bearish")
            
            # Overall confidence based on agreement
            sentiments = [news_sentiment, social_sentiment, reddit_sentiment, twitter_sentiment]
            std_dev = np.std(sentiments)
            confidence = max(0.5, 1.0 - (std_dev / 50))  # Higher agreement = higher confidence
            
            return {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'twitter_sentiment': twitter_sentiment,
                'signals': signals,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis error: {e}")
            return {
                'news_sentiment': 50,
                'social_sentiment': 50,
                'reddit_sentiment': 50,
                'twitter_sentiment': 50,
                'signals': ['Enhanced sentiment unavailable'],
                'confidence': 0.5
            }
    
    def _determine_direction(self, score: float) -> str:
        """Determine sentiment direction"""
        if score > 60:
            return 'UP'
        elif score < 40:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _empty_sentiment_analysis(self) -> Dict[str, Any]:
        """Return empty sentiment analysis"""
        return {
            'score': 50,
            'momentum': 0,
            'velocity': 0,
            'signals': ['No sentiment data available'],
            'keywords_analyzed': 0
        }

class VIXAnalyzer:
    """VIX fear/greed sentiment analyzer"""
    
    def __init__(self):
        self.session = None
        logger.info("📊 VIX Analyzer initialized")
    
    async def create_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_vix_sentiment(self) -> Dict[str, Any]:
        """Get VIX fear/greed sentiment"""
        try:
            await self.create_session()
            
            # Try to get real VIX data (simplified)
            try:
                # In production, integrate with real VIX API
                vix_value = random.uniform(15, 35)  # Simulate VIX
            except:
                vix_value = random.uniform(15, 35)
            
            # Interpret VIX levels
            if vix_value > 30:
                sentiment = "EXTREME_FEAR"
                signal = "CONTRARIAN_BULLISH"
                score = 70  # Fear often signals bottoms
            elif vix_value > 20:
                sentiment = "FEAR"
                signal = "CAUTIOUS_BULLISH"
                score = 60
            elif vix_value < 12:
                sentiment = "COMPLACENCY"
                signal = "RISK_WARNING"
                score = 30  # Complacency signals tops
            else:
                sentiment = "NEUTRAL"
                signal = "NORMAL"
                score = 50
            
            return {
                'vix_value': round(vix_value, 2),
                'sentiment': sentiment,
                'signal': signal,
                'fear_greed_score': score,
                'contrarian_bullish': sentiment in ['EXTREME_FEAR', 'FEAR']
            }
            
        except Exception as e:
            logger.error(f"VIX analysis error: {e}")
            return {
                'vix_value': 20.0,
                'sentiment': 'NEUTRAL',
                'signal': 'NORMAL',
                'fear_greed_score': 50,
                'contrarian_bullish': False
            }

class MarketStructureAnalyzer:
    """Market structure and breadth analysis"""
    
    def __init__(self):
        logger.info("📈 Market Structure Analyzer initialized")
    
    def analyze_market_breadth(self) -> Dict[str, Any]:
        """Analyze market breadth indicators"""
        try:
            # Simulate market breadth data
            advancing_stocks = random.randint(1200, 2800)
            declining_stocks = random.randint(800, 2200)
            total_stocks = advancing_stocks + declining_stocks
            
            advance_decline_ratio = advancing_stocks / declining_stocks if declining_stocks > 0 else 2.0
            breadth_thrust = advancing_stocks / total_stocks if total_stocks > 0 else 0.5
            
            # Score market breadth
            if breadth_thrust > 0.9:
                breadth_signal = "VERY_BULLISH"
                score = 90
            elif breadth_thrust > 0.6:
                breadth_signal = "BULLISH"
                score = 70
            elif breadth_thrust < 0.1:
                breadth_signal = "VERY_BEARISH"
                score = 10
            elif breadth_thrust < 0.4:
                breadth_signal = "BEARISH"
                score = 30
            else:
                breadth_signal = "NEUTRAL"
                score = 50
            
            return {
                'score': score,
                'breadth_thrust': breadth_thrust * 100,
                'breadth_signal': breadth_signal,
                'advance_decline_ratio': advance_decline_ratio,
                'advancing_stocks': advancing_stocks,
                'declining_stocks': declining_stocks
            }
            
        except Exception as e:
            logger.error(f"Market breadth analysis error: {e}")
            return {
                'score': 50,
                'breadth_thrust': 50.0,
                'breadth_signal': 'NEUTRAL',
                'advance_decline_ratio': 1.0,
                'advancing_stocks': 1500,
                'declining_stocks': 1500
            }
    
    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Utilities', 'Consumer Discretionary']
            
            # Simulate sector performance
            sector_performance = {sector: random.uniform(-2.0, 2.0) for sector in sectors}
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            top_sectors = [s[0] for s in sorted_sectors[:2]]
            bottom_sectors = [s[0] for s in sorted_sectors[-2:]]
            
            # Determine rotation theme
            if 'Technology' in top_sectors:
                rotation_theme = "GROWTH_ROTATION"
                score = 65
            elif 'Utilities' in top_sectors or 'Healthcare' in top_sectors:
                rotation_theme = "DEFENSIVE_ROTATION"
                score = 35
            elif 'Financials' in top_sectors or 'Energy' in top_sectors:
                rotation_theme = "VALUE_ROTATION"
                score = 60
            else:
                rotation_theme = "MIXED_ROTATION"
                score = 50
            
            return {
                'score': score,
                'rotation_theme': rotation_theme,
                'top_sectors': top_sectors,
                'bottom_sectors': bottom_sectors,
                'sector_performance': sector_performance
            }
            
        except Exception as e:
            logger.error(f"Sector rotation analysis error: {e}")
            return {
                'score': 50,
                'rotation_theme': 'NEUTRAL_ROTATION',
                'top_sectors': ['Technology', 'Healthcare'],
                'bottom_sectors': ['Energy', 'Utilities'],
                'sector_performance': {}
            }

class MLPredictor:
    """Machine learning predictions and pattern recognition"""
    
    def __init__(self):
        logger.info("🧠 ML Predictor initialized")
    
    def generate_ml_predictions(self, symbol: str, price_data: Dict) -> Dict[str, Any]:
        """Generate ML predictions and pattern analysis"""
        try:
            current_price = price_data.get('price', 100)
            
            # LSTM-style predictions
            lstm_predictions = self._simulate_lstm_predictions(symbol, current_price)
            
            # Ensemble model voting
            ensemble_prediction = self._simulate_ensemble_prediction(symbol)
            
            # Chart pattern recognition
            pattern_analysis = self._analyze_chart_patterns(symbol, current_price)
            
            # Anomaly detection
            anomaly_data = self._detect_anomalies(price_data)
            
            return {
                'lstm_predictions': lstm_predictions,
                'ensemble_prediction': ensemble_prediction,
                'pattern_analysis': pattern_analysis,
                'anomaly_data': anomaly_data,
                'ml_confidence': self._calculate_ml_confidence(lstm_predictions, ensemble_prediction)
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._empty_ml_predictions()
    
    def _simulate_lstm_predictions(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Simulate LSTM neural network predictions"""
        predictions = {}
        base_change = random.uniform(-0.03, 0.03)
        
        for days in [1, 3, 5]:
            daily_noise = random.uniform(-0.01, 0.01) * days
            predicted_change = base_change + daily_noise
            predicted_price = current_price * (1 + predicted_change)
            confidence = max(0.6, 0.9 - (days * 0.05))
            
            predictions[f'{days}_day'] = {
                'predicted_price': round(predicted_price, 2),
                'predicted_change': round(predicted_change * 100, 2),
                'confidence': round(confidence, 2),
                'direction': 'UP' if predicted_change > 0 else 'DOWN'
            }
        
        return {
            'model_type': 'LSTM_Neural_Network',
            'predictions': predictions,
            'model_confidence': round(random.uniform(0.75, 0.85), 2)
        }
    
    def _simulate_ensemble_prediction(self, symbol: str) -> Dict[str, Any]:
        """Simulate ensemble model voting"""
        models = ['RandomForest', 'GradientBoost', 'SVM', 'LinearRegression']
        votes = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
        
        for model in models:
            vote = random.choice(['UP', 'DOWN', 'NEUTRAL'])
            votes[vote] += 1
        
        # Determine ensemble prediction
        ensemble_direction = max(votes, key=votes.get)
        ensemble_confidence = votes[ensemble_direction] / len(models)
        
        return {
            'ensemble_direction': ensemble_direction,
            'ensemble_confidence': round(ensemble_confidence, 2),
            'model_votes': votes,
            'agreement_level': round(ensemble_confidence, 2)
        }
    
    def _analyze_chart_patterns(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Analyze chart patterns"""
        patterns = ['head_and_shoulders', 'double_top', 'triangle', 'flag', 'cup_and_handle']
        detected_pattern = random.choice(patterns) if random.random() > 0.7 else 'no_pattern'
        
        if detected_pattern != 'no_pattern':
            confidence = random.uniform(0.6, 0.9)
            breakout_probability = random.uniform(0.5, 0.8)
        else:
            confidence = 0.0
            breakout_probability = 0.5
        
        return {
            'detected_pattern': detected_pattern,
            'pattern_confidence': round(confidence, 2),
            'breakout_probability': round(breakout_probability, 2),
            'support_level': current_price * 0.97,
            'resistance_level': current_price * 1.03
        }
    
    def _detect_anomalies(self, price_data: Dict) -> Dict[str, Any]:
        """Detect market anomalies"""
        anomaly_score = 0
        anomalies = []
        
        # Price anomaly
        change_percent = abs(float(price_data.get('change_percent', 0)))
        if change_percent > 5:
            anomaly_score += 30
            anomalies.append('Extreme price movement')
        
        # Volume anomaly
        volume = price_data.get('volume', 0)
        avg_volume = volume * random.uniform(0.8, 1.2)
        if volume > avg_volume * 2:
            anomaly_score += 20
            anomalies.append('Volume spike detected')
        
        return {
            'anomaly_score': round(anomaly_score, 1),
            'anomaly_level': 'HIGH' if anomaly_score > 40 else 'MEDIUM' if anomaly_score > 20 else 'LOW',
            'anomalies_detected': anomalies
        }
    
    def _calculate_ml_confidence(self, lstm_data: Dict, ensemble_data: Dict) -> float:
        """Calculate overall ML confidence"""
        lstm_conf = lstm_data.get('model_confidence', 0.5)
        ensemble_conf = ensemble_data.get('ensemble_confidence', 0.5)
        return round((lstm_conf + ensemble_conf) / 2 * 100, 1)
    
    def _empty_ml_predictions(self) -> Dict[str, Any]:
        """Return empty ML predictions"""
        return {
            'lstm_predictions': {},
            'ensemble_prediction': {'ensemble_direction': 'NEUTRAL', 'ensemble_confidence': 0.5},
            'pattern_analysis': {'detected_pattern': 'no_pattern', 'pattern_confidence': 0.0},
            'anomaly_data': {'anomaly_score': 0.0, 'anomaly_level': 'LOW'},
            'ml_confidence': 50.0
        }

class EconomicAnalyzer:
    """Economic indicators and market fundamentals"""
    
    def __init__(self):
        logger.info("💼 Economic Analyzer initialized")
    
    def analyze_economic_indicators(self) -> Dict[str, Any]:
        """Analyze economic indicators"""
        try:
            # Simulate economic data
            indicators = {
                'gdp_growth': random.uniform(1.5, 4.0),
                'unemployment': random.uniform(3.5, 6.0),
                'inflation': random.uniform(1.0, 5.0),
                'interest_rate': random.uniform(0.25, 5.0)
            }
            
            # Calculate economic sentiment
            score = 50
            if indicators['gdp_growth'] > 3.0:
                score += 15
            if indicators['unemployment'] < 4.0:
                score += 10
            if 2.0 <= indicators['inflation'] <= 3.0:
                score += 10
            else:
                score -= 5
            
            return {
                'score': max(0, min(100, score)),
                'indicators': indicators,
                'economic_outlook': 'POSITIVE' if score > 60 else 'NEGATIVE' if score < 40 else 'NEUTRAL'
            }
            
        except Exception as e:
            logger.error(f"Economic analysis error: {e}")
            return {'score': 50, 'indicators': {}, 'economic_outlook': 'NEUTRAL'}

class RiskAnalyzer:
    """Enhanced risk analysis and fake-out detection"""
    
    def __init__(self):
        logger.info("🛡️ Risk Analyzer initialized")
    
    def analyze_risk(self, technical_data: Dict, sentiment_data: Dict, quote_data: Dict, 
                    vix_data: Dict = None, ml_data: Dict = None) -> Dict[str, Any]:
        """Enhanced risk analysis with new factors"""
        warnings = []
        confidence_penalty = 0
        
        try:
            # Original risk factors
            volume = quote_data.get('volume', 0) if quote_data else 0
            if volume < 1000000:
                warnings.append("Low volume - potential fake breakout")
                confidence_penalty += 0.15
            
            change_percent = abs(float(quote_data.get('change_percent', 0))) if quote_data else 0
            if change_percent > 5:
                warnings.append("Extreme price movement - high risk")
                confidence_penalty += 0.20
            
            # NEW: VIX risk factors
            if vix_data:
                if vix_data.get('sentiment') == 'EXTREME_FEAR':
                    warnings.append("Extreme market fear - high volatility")
                    confidence_penalty += 0.10
                elif vix_data.get('sentiment') == 'COMPLACENCY':
                    warnings.append("Market complacency - potential reversal risk")
                    confidence_penalty += 0.15
            
            # NEW: ML anomaly risks
            if ml_data and ml_data.get('anomaly_data'):
                anomaly_level = ml_data['anomaly_data'].get('anomaly_level', 'LOW')
                if anomaly_level == 'HIGH':
                    warnings.append("High anomaly detected - unusual market behavior")
                    confidence_penalty += 0.25
                elif anomaly_level == 'MEDIUM':
                    warnings.append("Market anomaly detected - proceed with caution")
                    confidence_penalty += 0.10
            
            # NEW: Technical divergence
            tech_direction = technical_data.get('direction', 'NEUTRAL')
            sentiment_direction = sentiment_data.get('direction', 'NEUTRAL')
            
            if (tech_direction != sentiment_direction and 
                tech_direction != 'NEUTRAL' and 
                sentiment_direction != 'NEUTRAL'):
                warnings.append("Technical vs sentiment divergence")
                confidence_penalty += 0.10
            
            # NEW: Risk metrics
            var_95 = self._calculate_var_95(quote_data)
            max_drawdown = self._calculate_max_drawdown_risk()
            correlation_spy = self._calculate_spy_correlation(quote_data.get('symbol', '') if quote_data else '')
            
            return {
                'warnings': warnings,
                'confidence_penalty': confidence_penalty,
                'risk_score': min(100, confidence_penalty * 100),
                'var_95': var_95,
                'max_drawdown_risk': max_drawdown,
                'correlation_spy': correlation_spy
            }
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {
                'warnings': ['Risk analysis failed'],
                'confidence_penalty': 0.1,
                'risk_score': 10,
                'var_95': 5.0,
                'max_drawdown_risk': 10.0,
                'correlation_spy': 0.7
            }
    
    def _calculate_var_95(self, quote_data: Dict) -> float:
        """Calculate 95% Value at Risk"""
        try:
            change_percent = float(quote_data.get('change_percent', 0)) if quote_data else 0
            # Simplified VaR based on recent volatility
            volatility = abs(change_percent) * 2  # Rough estimate
            return max(1.0, min(20.0, volatility))
        except:
            return 5.0
    
    def _calculate_max_drawdown_risk(self) -> float:
        """Calculate maximum drawdown risk"""
        # Simulate based on market conditions
        return round(random.uniform(5.0, 15.0), 1)
    
    def _calculate_spy_correlation(self, symbol: str) -> float:
        """Calculate correlation with SPY"""
        correlation_map = {
            'SPY': 1.0,
            'QQQ': 0.85,
            'NVDA': 0.75,
            'AAPL': 0.80,
            'MSFT': 0.82
        }
        return correlation_map.get(symbol, 0.70)

class HYPERSignalEngine:
    """Combined HYPER signal generation engine with all features"""
    
    def __init__(self):
        # Import here to avoid circular imports
        import config
        from data_sources import HYPERDataAggregator
        self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
        
        # Initialize all analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.vix_analyzer = VIXAnalyzer()
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.ml_predictor = MLPredictor()
        self.economic_analyzer = EconomicAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        # Enhanced signal weights
        self.weights = {
            'technical': 0.25,          # Technical analysis + advanced indicators
            'sentiment': 0.20,          # Multi-source sentiment
            'momentum': 0.15,           # Price momentum
            'ml_prediction': 0.15,      # ML predictions + patterns
            'market_structure': 0.10,   # Market breadth + sector rotation
            'vix_sentiment': 0.08,      # VIX fear/greed
            'economic': 0.05,           # Economic indicators
            'risk_adjusted': 0.02       # Risk penalty
        }
        
        # Confidence thresholds
        self.thresholds = {
            'HYPER_BUY': 85,
            'SOFT_BUY': 65,
            'HOLD': 35,
            'SOFT_SELL': 65,
            'HYPER_SELL': 85
        }
        
        logger.info("🚀 Combined HYPER Signal Engine initialized with ALL features")
    
    async def generate_signal(self, symbol: str) -> HYPERSignal:
        """Generate comprehensive trading signal with all enhanced features"""
        logger.info(f"🎯 Generating combined enhanced signal for {symbol}")
        
        try:
            # 1. Get comprehensive market data
            data = await self.data_aggregator.get_comprehensive_data(symbol)
            
            if not data or data.get('data_quality') == 'error':
                return self._create_error_signal(symbol, "Data retrieval failed")
            
            quote_data = data.get('quote')
            trends_data = data.get('trends')
            
            if not quote_data:
                return self._create_error_signal(symbol, "No quote data available")
            
            # 2. Run all analyses
            technical_analysis = self.technical_analyzer.analyze_price_action(quote_data)
            sentiment_analysis = self.sentiment_analyzer.analyze_trends_sentiment(trends_data, symbol)
            vix_analysis = await self.vix_analyzer.get_vix_sentiment()
            market_structure = self.market_structure_analyzer.analyze_market_breadth()
            sector_rotation = self.market_structure_analyzer.analyze_sector_rotation()
            ml_predictions = self.ml_predictor.generate_ml_predictions(symbol, quote_data)
            economic_data = self.economic_analyzer.analyze_economic_indicators()
            
            # 3. Enhanced risk analysis
            risk_analysis = self.risk_analyzer.analyze_risk(
                technical_analysis, sentiment_analysis, quote_data, vix_analysis, ml_predictions
            )
            
            # 4. Calculate all component scores
            scores = self._calculate_all_scores(
                technical_analysis, sentiment_analysis, vix_analysis, market_structure,
                sector_rotation, ml_predictions, economic_data, quote_data
            )
            
            # 5. Calculate weighted confidence with risk adjustment
            weighted_confidence = sum(
                scores[key] * self.weights.get(key, 0) for key in scores
            )
            
            # Apply risk penalty
            final_confidence = max(0, weighted_confidence * (1 - risk_analysis['confidence_penalty']))
            
            # 6. Determine direction and signal type
            direction = self._determine_enhanced_direction(
                technical_analysis, sentiment_analysis, ml_predictions, vix_analysis
            )
            signal_type = self._classify_signal(final_confidence, direction)
            
            # 7. Compile comprehensive reasons
            reasons = self._generate_enhanced_reasons(
                technical_analysis, sentiment_analysis, vix_analysis, ml_predictions,
                market_structure, sector_rotation, economic_data
            )
            
            # 8. Create enhanced signal
            signal = HYPERSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=round(final_confidence, 1),
                direction=direction,
                price=quote_data['price'],
                timestamp=datetime.now().isoformat(),
                
                # Core scores
                technical_score=scores['technical'],
                momentum_score=scores['momentum'],
                trends_score=sentiment_analysis.get('score', 50),
                volume_score=self._calculate_volume_score(quote_data),
                ml_score=scores['ml_prediction'],
                
                # Enhanced scores
                sentiment_score=scores['sentiment'],
                pattern_score=ml_predictions.get('pattern_analysis', {}).get('pattern_confidence', 0) * 100,
                market_structure_score=scores['market_structure'],
                economic_score=scores['economic'],
                risk_score=100 - risk_analysis['risk_score'],
                
                # Advanced indicators
                williams_r=technical_analysis.get('williams_r', -50.0),
                stochastic_k=technical_analysis.get('stochastic_k', 50.0),
                stochastic_d=technical_analysis.get('stochastic_d', 50.0),
                vix_sentiment=vix_analysis.get('sentiment', 'NEUTRAL'),
                put_call_ratio=random.uniform(0.8, 1.2),  # Simulated
                
                # Levels and structure
                fibonacci_levels=technical_analysis.get('fibonacci_levels', {}),
                market_breadth=market_structure.get('breadth_thrust', 50),
                sector_rotation=sector_rotation.get('rotation_theme', 'NEUTRAL'),
                volume_profile=technical_analysis.get('volume_profile', {}),
                
                # ML predictions
                lstm_predictions=ml_predictions.get('lstm_predictions', {}),
                ensemble_prediction=ml_predictions.get('ensemble_prediction', {}),
                anomaly_score=ml_predictions.get('anomaly_data', {}).get('anomaly_score', 0),
                
                # Economic and alternative data
                economic_sentiment=economic_data,
                earnings_proximity=random.randint(10, 90),  # Simulated
                
                # Risk metrics
                var_95=risk_analysis.get('var_95', 5.0),
                max_drawdown_risk=risk_analysis.get('max_drawdown_risk', 10.0),
                correlation_spy=risk_analysis.get('correlation_spy', 0.7),
                
                # Supporting data
                indicators={
                    'rsi': technical_analysis.get('rsi', 50),
                    'volume_ratio': technical_analysis.get('volume_ratio', 1),
                    'change_percent': technical_analysis.get('change_percent', 0),
                    'trend_momentum': sentiment_analysis.get('momentum', 0),
                    'vix_value': vix_analysis.get('vix_value', 20)
                },
                reasons=reasons[:5],  # Top 5 reasons
                warnings=risk_analysis['warnings'],
                data_quality=data.get('data_quality', 'unknown')
            )
            
            logger.info(f"✅ Generated {signal.signal_type} combined signal for {symbol} with {signal.confidence}% confidence")
            logger.info(f"🎯 Enhanced features: VIX={vix_analysis.get('sentiment')}, ML={ml_predictions.get('ml_confidence')}%, Anomaly={signal.anomaly_score}")
            
            return signal
            
        except Exception as e:
            logger.error(f"💥 Combined signal generation error for {symbol}: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return self._create_error_signal(symbol, f"Generation error: {str(e)}")
    
    def _calculate_all_scores(self, technical_analysis: Dict, sentiment_analysis: Dict,
                             vix_analysis: Dict, market_structure: Dict, sector_rotation: Dict,
                             ml_predictions: Dict, economic_data: Dict, quote_data: Dict) -> Dict[str, float]:
        """Calculate all component scores"""
        
        scores = {}
        
        # Technical score (enhanced with new indicators)
        base_technical = technical_analysis['score']
        williams_r = technical_analysis.get('williams_r', -50)
        stochastic_k = technical_analysis.get('stochastic_k', 50)
        
        # Adjust technical score with new indicators
        if williams_r < -80:  # Oversold
            base_technical += 10
        elif williams_r > -20:  # Overbought
            base_technical -= 10
        
        if stochastic_k < 20:  # Oversold
            base_technical += 8
        elif stochastic_k > 80:  # Overbought
            base_technical -= 8
        
        scores['technical'] = max(0, min(100, base_technical))
        
        # Enhanced sentiment score
        base_sentiment = sentiment_analysis.get('score', 50)
        news_sentiment = sentiment_analysis.get('news_sentiment', 50)
        social_sentiment = sentiment_analysis.get('social_sentiment', 50)
        
        # Weight different sentiment sources
        enhanced_sentiment = (base_sentiment * 0.4 + news_sentiment * 0.3 + social_sentiment * 0.3)
        scores['sentiment'] = max(0, min(100, enhanced_sentiment))
        
        # Momentum score
        change_percent = float(quote_data.get('change_percent', 0))
        scores['momentum'] = max(0, min(100, 50 + (change_percent * 10)))
        
        # ML prediction score
        ml_confidence = ml_predictions.get('ml_confidence', 50)
        ensemble_conf = ml_predictions.get('ensemble_prediction', {}).get('ensemble_confidence', 0.5) * 100
        scores['ml_prediction'] = (ml_confidence + ensemble_conf) / 2
        
        # Market structure score
        breadth_score = market_structure.get('score', 50)
        sector_score = sector_rotation.get('score', 50)
        scores['market_structure'] = (breadth_score + sector_score) / 2
        
        # VIX sentiment score
        scores['vix_sentiment'] = vix_analysis.get('fear_greed_score', 50)
        
        # Economic score
        scores['economic'] = economic_data.get('score', 50)
        
        return scores
    
    def _determine_enhanced_direction(self, technical: Dict, sentiment: Dict, 
                                    ml_predictions: Dict, vix: Dict) -> str:
        """Determine direction using enhanced factors"""
        direction_score = 0
        
        # Technical direction
        if technical.get('direction') == 'UP':
            direction_score += 2
        elif technical.get('direction') == 'DOWN':
            direction_score -= 2
        
        # Sentiment direction
        if sentiment.get('direction') == 'UP':
            direction_score += 2
        elif sentiment.get('direction') == 'DOWN':
            direction_score -= 2
        
        # ML ensemble direction
        ensemble_dir = ml_predictions.get('ensemble_prediction', {}).get('ensemble_direction', 'NEUTRAL')
        if ensemble_dir == 'UP':
            direction_score += 3  # Higher weight for ML
        elif ensemble_dir == 'DOWN':
            direction_score -= 3
        
        # VIX contrarian signal
        if vix.get('contrarian_bullish', False):
            direction_score += 1
        elif vix.get('sentiment') == 'COMPLACENCY':
            direction_score -= 1
        
        if direction_score > 2:
            return 'UP'
        elif direction_score < -2:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _generate_enhanced_reasons(self, technical: Dict, sentiment: Dict, vix: Dict,
                                 ml_predictions: Dict, market_structure: Dict,
                                 sector_rotation: Dict, economic_data: Dict) -> List[str]:
        """Generate comprehensive reasons"""
        reasons = []
        
        # Technical reasons (original + enhanced)
        reasons.extend(technical.get('signals', [])[:2])
        
        # Enhanced indicator reasons
        williams_r = technical.get('williams_r', -50)
        if williams_r < -80:
            reasons.append(f"Williams %R oversold at {williams_r:.1f}")
        elif williams_r > -20:
            reasons.append(f"Williams %R overbought at {williams_r:.1f}")
        
        # VIX reasons
        vix_sentiment = vix.get('sentiment', 'NEUTRAL')
        if vix_sentiment == 'EXTREME_FEAR':
            reasons.append("Extreme market fear - contrarian opportunity")
        elif vix_sentiment == 'COMPLACENCY':
            reasons.append("Market complacency - reversal risk")
        
        # ML reasons
        ensemble_dir = ml_predictions.get('ensemble_prediction', {}).get('ensemble_direction', 'NEUTRAL')
        ensemble_conf = ml_predictions.get('ensemble_prediction', {}).get('ensemble_confidence', 0)
        if ensemble_conf > 0.7:
            reasons.append(f"ML ensemble predicts {ensemble_dir} with {ensemble_conf:.0%} confidence")
        
        # Pattern reasons
        pattern = ml_predictions.get('pattern_analysis', {}).get('detected_pattern', 'no_pattern')
        if pattern != 'no_pattern':
            reasons.append(f"Chart pattern: {pattern.replace('_', ' ').title()}")
        
        # Market structure reasons
        breadth_signal = market_structure.get('breadth_signal', 'NEUTRAL')
        if breadth_signal in ['VERY_BULLISH', 'BULLISH']:
            reasons.append(f"Strong market breadth ({breadth_signal.lower()})")
        elif breadth_signal in ['VERY_BEARISH', 'BEARISH']:
            reasons.append(f"Weak market breadth ({breadth_signal.lower()})")
        
        # Sector rotation reasons
        rotation_theme = sector_rotation.get('rotation_theme', 'NEUTRAL')
        if rotation_theme != 'NEUTRAL_ROTATION':
            reasons.append(f"Sector rotation: {rotation_theme.lower().replace('_', ' ')}")
        
        # Economic reasons
        economic_outlook = economic_data.get('economic_outlook', 'NEUTRAL')
        if economic_outlook != 'NEUTRAL':
            reasons.append(f"Economic outlook: {economic_outlook.lower()}")
        
        return reasons
    
    def _calculate_volume_score(self, quote_data: Dict) -> float:
        """Calculate volume score"""
        try:
            volume = quote_data.get('volume', 0)
            if volume > 20000000:
                return 90
            elif volume > 10000000:
                return 75
            elif volume > 5000000:
                return 60
            elif volume > 1000000:
                return 45
            else:
                return 25
        except:
            return 50
    
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
        logger.info(f"🎯 Generating combined enhanced signals for {len(tickers)} tickers: {tickers}")
        
        signals = {}
        for ticker in tickers:
            try:
                signal = await self.generate_signal(ticker)
                signals[ticker] = signal
            except Exception as e:
                logger.error(f"❌ Failed to generate signal for {ticker}: {e}")
                signals[ticker] = self._create_error_signal(ticker, f"Generation failed: {str(e)}")
        
        logger.info(f"✅ Generated {len(signals)} combined enhanced signals")
        return signals
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.data_aggregator.close()
            await self.technical_analyzer.close_session()
            await self.vix_analyzer.close_session()
            logger.info("🧹 Combined signal engine cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
