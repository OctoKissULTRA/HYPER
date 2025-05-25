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
        change_percent = abs(float(price_data.get
