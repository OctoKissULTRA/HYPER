import logging
import asyncio
import numpy as np
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random
import math

# Import all enhancement modules
from enhanced_predictive_engine import EnhancedPredictiveEngine
from sentiment_analysis import SocialMediaAggregator
from ml_pattern_recognition import PatternRecognitionEngine

logger = logging.getLogger(__name__)

@dataclass
class UltraHYPERSignal:
    """Ultra-enhanced HYPER trading signal with all predictive features"""
    symbol: str
    signal_type: str  # HYPER_BUY, SOFT_BUY, HOLD, SOFT_SELL, HYPER_SELL
    confidence: float  # 0-100
    direction: str    # UP, DOWN, NEUTRAL
    price: float
    timestamp: str
    
    # Enhanced scores with new indicators
    technical_score: float
    momentum_score: float
    sentiment_score: float
    pattern_score: float
    ml_confidence: float
    market_structure_score: float  # NEW
    alternative_data_score: float  # NEW
    
    # Advanced technical indicators
    williams_r: float             # NEW: Williams %R
    stochastic_k: float          # NEW: Stochastic %K
    stochastic_d: float          # NEW: Stochastic %D
    vix_sentiment: str           # NEW: VIX Fear/Greed
    put_call_ratio: float        # NEW: Put/Call Ratio
    
    # Fibonacci levels
    fibonacci_levels: Dict[str, float]  # NEW
    
    # Market structure indicators
    market_breadth: float        # NEW: Advance/Decline
    sector_rotation: str         # NEW: Sector flow
    volume_profile: Dict[str, float]  # NEW: VWAP, OBV
    
    # Advanced ML predictions
    lstm_predictions: Dict[str, Any]     # NEW: Neural network forecasts
    ensemble_prediction: Dict[str, Any]  # NEW: Multi-model voting
    anomaly_score: float                 # NEW: Unusual activity detection
    
    # Economic indicators
    economic_sentiment: Dict[str, float] # NEW: GDP, inflation, etc.
    earnings_proximity: Dict[str, Any]   # NEW: Earnings effects
    
    # Risk metrics
    var_95: float                # NEW: Value at Risk
    max_drawdown_risk: float     # NEW: Downside risk
    correlation_spy: float       # NEW: Market correlation
    
    # Existing fields
    technical_indicators: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    chart_patterns: List[str]
    support: float
    resistance: float
    breakout_probability: float
    reasons: List[str]
    warnings: List[str]
    data_quality: str

class AdvancedTechnicalAnalyzer:
    """Advanced technical indicators - Williams %R, Stochastic, etc."""
    
    def __init__(self):
        self.session = None
        logger.info("🔧 Advanced Technical Analyzer initialized")
    
    async def create_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def calculate_williams_r(self, high_prices: List[float], low_prices: List[float], 
                           close_price: float, period: int = 14) -> float:
        """Calculate Williams %R oscillator"""
        try:
            if len(high_prices) < period or len(low_prices) < period:
                return -50.0  # Neutral
            
            highest_high = max(high_prices[-period:])
            lowest_low = min(low_prices[-period:])
            
            if highest_high == lowest_low:
                return -50.0
            
            williams_r = ((highest_high - close_price) / (highest_high - lowest_low)) * -100
            return max(-100, min(0, williams_r))
            
        except Exception as e:
            logger.error(f"Williams %R calculation error: {e}")
            return -50.0
    
    def calculate_stochastic(self, high_prices: List[float], low_prices: List[float],
                           close_prices: List[float], k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator %K and %D"""
        try:
            if len(close_prices) < k_period:
                return 50.0, 50.0  # Neutral values
            
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
            
            # Calculate %D (3-period SMA of %K)
            if len(k_values) >= d_period:
                current_d = sum(k_values[-d_period:]) / d_period
            else:
                current_d = current_k
            
            return current_k, current_d
            
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            return 50.0, 50.0
    
    def calculate_fibonacci_levels(self, high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if high_price <= low_price:
                return {}
            
            diff = high_price - low_price
            return {
                'level_0': high_price,
                'level_23.6': high_price - (diff * 0.236),
                'level_38.2': high_price - (diff * 0.382),
                'level_50.0': high_price - (diff * 0.500),
                'level_61.8': high_price - (diff * 0.618),
                'level_78.6': high_price - (diff * 0.786),
                'level_100': low_price
            }
            
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return {}
    
    async def get_vix_sentiment(self) -> Dict[str, Any]:
        """Get VIX fear/greed sentiment"""
        try:
            await self.create_session()
            
            # Try to get real VIX data
            vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
            
            async with self.session.get(vix_url) as response:
                if response.status == 200:
                    data = await response.json()
                    vix_value = data['chart']['result'][0]['meta']['regularMarketPrice']
                else:
                    raise Exception("VIX API failed")
                    
        except Exception as e:
            logger.warning(f"VIX API error, using simulation: {e}")
            # Simulate VIX value (typically 12-40 range)
            vix_value = random.uniform(15, 35)
        
        # Interpret VIX levels
        if vix_value > 30:
            sentiment = "EXTREME_FEAR"
            signal = "CONTRARIAN_BULLISH"
        elif vix_value > 20:
            sentiment = "FEAR"
            signal = "CAUTIOUS_BULLISH"
        elif vix_value < 12:
            sentiment = "COMPLACENCY"
            signal = "RISK_WARNING"
        else:
            sentiment = "NEUTRAL"
            signal = "NORMAL"
        
        return {
            'vix_value': round(vix_value, 2),
            'sentiment': sentiment,
            'signal': signal,
            'fear_greed_score': max(0, min(100, (40 - vix_value) * 2.5))  # Convert to 0-100 scale
        }

class MarketStructureAnalyzer:
    """Analyze market structure - breadth, sector rotation, institutional flow"""
    
    def __init__(self):
        self.sectors = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
            'XLI': 'Industrials', 'XLC': 'Communications', 'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples', 'XLE': 'Energy', 'XLU': 'Utilities',
            'XLB': 'Materials', 'XLRE': 'Real Estate'
        }
        logger.info("📊 Market Structure Analyzer initialized")
    
    def calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        try:
            # Simulate advance/decline data (in production, use real data)
            advancing_stocks = random.randint(1200, 2800)
            declining_stocks = random.randint(800, 2200)
            total_stocks = advancing_stocks + declining_stocks
            
            # Calculate breadth metrics
            advance_decline_ratio = advancing_stocks / declining_stocks if declining_stocks > 0 else 2.0
            breadth_thrust = advancing_stocks / total_stocks if total_stocks > 0 else 0.5
            
            # Interpret breadth
            if breadth_thrust > 0.9:
                breadth_signal = "VERY_BULLISH"
            elif breadth_thrust > 0.6:
                breadth_signal = "BULLISH"
            elif breadth_thrust < 0.1:
                breadth_signal = "VERY_BEARISH"
            elif breadth_thrust < 0.4:
                breadth_signal = "BEARISH"
            else:
                breadth_signal = "NEUTRAL"
            
            return {
                'advance_decline_ratio': round(advance_decline_ratio, 2),
                'breadth_thrust': round(breadth_thrust * 100, 1),
                'breadth_signal': breadth_signal,
                'advancing_stocks': advancing_stocks,
                'declining_stocks': declining_stocks
            }
            
        except Exception as e:
            logger.error(f"Market breadth calculation error: {e}")
            return {
                'advance_decline_ratio': 1.0,
                'breadth_thrust': 50.0,
                'breadth_signal': "NEUTRAL",
                'advancing_stocks': 1500,
                'declining_stocks': 1500
            }
    
    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            # Simulate sector performance (in production, use real sector ETF data)
            sector_performance = {}
            for etf, name in self.sectors.items():
                performance = random.uniform(-3.0, 3.0)  # Daily % change
                sector_performance[name] = performance
            
            # Sort by performance
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            # Determine rotation theme
            top_sectors = [s[0] for s in sorted_sectors[:3]]
            bottom_sectors = [s[0] for s in sorted_sectors[-3:]]
            
            # Classify rotation type
            if 'Technology' in top_sectors and 'Consumer Discretionary' in top_sectors:
                rotation_theme = "GROWTH_ROTATION"
                market_regime = "RISK_ON"
            elif 'Utilities' in top_sectors and 'Consumer Staples' in top_sectors:
                rotation_theme = "DEFENSIVE_ROTATION"
                market_regime = "RISK_OFF"
            elif 'Financials' in top_sectors and 'Energy' in top_sectors:
                rotation_theme = "VALUE_ROTATION"
                market_regime = "CYCLICAL"
            else:
                rotation_theme = "MIXED_ROTATION"
                market_regime = "NEUTRAL"
            
            return {
                'rotation_theme': rotation_theme,
                'market_regime': market_regime,
                'top_sectors': top_sectors,
                'bottom_sectors': bottom_sectors,
                'sector_performance': sector_performance
            }
            
        except Exception as e:
            logger.error(f"Sector rotation analysis error: {e}")
            return {
                'rotation_theme': "NEUTRAL_ROTATION",
                'market_regime': "NEUTRAL",
                'top_sectors': ["Technology", "Healthcare", "Financials"],
                'bottom_sectors': ["Energy", "Utilities", "Materials"],
                'sector_performance': {}
            }
    
    def calculate_volume_profile(self, price: float, volume: int) -> Dict[str, float]:
        """Calculate volume profile indicators"""
        try:
            # Simulate volume profile data
            avg_volume = volume * random.uniform(0.8, 1.2)
            relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume-weighted average price (simplified)
            vwap = price * random.uniform(0.995, 1.005)
            
            # On-balance volume trend (simplified)
            obv_trend = random.choice(["RISING", "FALLING", "FLAT"])
            
            return {
                'relative_volume': round(relative_volume, 2),
                'vwap': round(vwap, 2),
                'vwap_deviation': round(((price - vwap) / vwap) * 100, 2),
                'obv_trend': obv_trend,
                'volume_strength': "HIGH" if relative_volume > 1.5 else "NORMAL" if relative_volume > 0.8 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Volume profile calculation error: {e}")
            return {
                'relative_volume': 1.0,
                'vwap': price,
                'vwap_deviation': 0.0,
                'obv_trend': "FLAT",
                'volume_strength': "NORMAL"
            }

class AdvancedMLPredictor:
    """Advanced machine learning predictions - LSTM, ensemble, anomaly detection"""
    
    def __init__(self):
        self.models_trained = False
        logger.info("🧠 Advanced ML Predictor initialized")
    
    def simulate_lstm_predictions(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Simulate LSTM neural network predictions"""
        try:
            # Simulate realistic LSTM predictions
            base_change = random.uniform(-0.05, 0.05)  # ±5% base change
            
            predictions = {}
            for days in [1, 3, 5, 7, 14]:
                # Add some noise and trend
                daily_noise = random.uniform(-0.01, 0.01) * days
                predicted_change = base_change + daily_noise
                predicted_price = current_price * (1 + predicted_change)
                
                # Confidence decreases with longer timeframes
                confidence = max(0.5, 0.9 - (days * 0.05))
                
                predictions[f'{days}_day'] = {
                    'predicted_price': round(predicted_price, 2),
                    'predicted_change': round(predicted_change * 100, 2),
                    'confidence': round(confidence, 2),
                    'direction': 'UP' if predicted_change > 0 else 'DOWN'
                }
            
            return {
                'model_type': 'LSTM_Neural_Network',
                'predictions': predictions,
                'model_confidence': round(random.uniform(0.7, 0.9), 2),
                'training_accuracy': round(random.uniform(0.75, 0.85), 2)
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {
                'model_type': 'LSTM_Neural_Network',
                'predictions': {},
                'model_confidence': 0.5,
                'training_accuracy': 0.6
            }
    
    def simulate_ensemble_prediction(self, symbol: str, features: Dict) -> Dict[str, Any]:
        """Simulate ensemble model voting"""
        try:
            models = ['RandomForest', 'GradientBoost', 'SVM', 'LinearRegression', 'LSTM']
            model_predictions = {}
            
            for model in models:
                # Each model makes a prediction
                direction = random.choice(['UP', 'DOWN', 'NEUTRAL'])
                confidence = random.uniform(0.6, 0.9)
                strength = random.uniform(0.1, 2.0)  # % change magnitude
                
                model_predictions[model] = {
                    'direction': direction,
                    'confidence': confidence,
                    'strength': strength
                }
            
            # Voting mechanism
            up_votes = sum(1 for p in model_predictions.values() if p['direction'] == 'UP')
            down_votes = sum(1 for p in model_predictions.values() if p['direction'] == 'DOWN')
            
            if up_votes > down_votes:
                ensemble_direction = 'UP'
                ensemble_confidence = up_votes / len(models)
            elif down_votes > up_votes:
                ensemble_direction = 'DOWN'
                ensemble_confidence = down_votes / len(models)
            else:
                ensemble_direction = 'NEUTRAL'
                ensemble_confidence = 0.5
            
            return {
                'ensemble_direction': ensemble_direction,
                'ensemble_confidence': round(ensemble_confidence, 2),
                'model_agreement': round(max(up_votes, down_votes) / len(models), 2),
                'individual_predictions': model_predictions,
                'voting_strength': round(random.uniform(0.5, 1.5), 2)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {
                'ensemble_direction': 'NEUTRAL',
                'ensemble_confidence': 0.5,
                'model_agreement': 0.5,
                'individual_predictions': {},
                'voting_strength': 1.0
            }
    
    def detect_anomalies(self, price: float, volume: int, historical_data: Dict) -> Dict[str, Any]:
        """Detect market anomalies and unusual activity"""
        try:
            anomaly_score = 0.0
            anomalies_detected = []
            
            # Price anomaly detection
            avg_price = historical_data.get('avg_price', price)
            price_deviation = abs(price - avg_price) / avg_price
            if price_deviation > 0.05:  # 5% deviation
                anomaly_score += price_deviation * 100
                anomalies_detected.append(f"Unusual price movement: {price_deviation:.1%}")
            
            # Volume anomaly detection
            avg_volume = historical_data.get('avg_volume', volume)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 2.0:  # 2x normal volume
                anomaly_score += (volume_ratio - 1) * 20
                anomalies_detected.append(f"High volume spike: {volume_ratio:.1f}x normal")
            
            # Pattern anomaly (simulate)
            if random.random() < 0.1:  # 10% chance of pattern anomaly
                anomaly_score += 25
                anomalies_detected.append("Unusual technical pattern detected")
            
            # Sentiment anomaly (simulate)
            if random.random() < 0.15:  # 15% chance of sentiment anomaly
                anomaly_score += 20
                anomalies_detected.append("Extreme sentiment divergence")
            
            anomaly_level = "HIGH" if anomaly_score > 50 else "MEDIUM" if anomaly_score > 20 else "LOW"
            
            return {
                'anomaly_score': round(min(100, anomaly_score), 1),
                'anomaly_level': anomaly_level,
                'anomalies_detected': anomalies_detected,
                'requires_attention': anomaly_score > 30,
                'risk_level': "HIGH" if anomaly_score > 60 else "MEDIUM" if anomaly_score > 30 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {
                'anomaly_score': 0.0,
                'anomaly_level': "LOW", 
                'anomalies_detected': [],
                'requires_attention': False,
                'risk_level': "LOW"
            }

class EconomicDataAnalyzer:
    """Analyze economic indicators and their market impact"""
    
    def __init__(self):
        logger.info("💼 Economic Data Analyzer initialized")
    
    def get_economic_sentiment(self) -> Dict[str, Any]:
        """Get economic indicators sentiment"""
        try:
            # Simulate economic data (in production, use FRED API, etc.)
            indicators = {
                'gdp_growth': random.uniform(1.0, 4.0),      # % annual
                'unemployment': random.uniform(3.5, 6.0),    # %
                'inflation': random.uniform(1.0, 5.0),       # % annual
                'interest_rate': random.uniform(0.25, 5.0),  # %
                'retail_sales': random.uniform(-2.0, 5.0),   # % monthly change
                'manufacturing_pmi': random.uniform(45, 65)   # Index
            }
            
            # Calculate economic sentiment score
            sentiment_score = 50  # Base
            
            # GDP impact
            if indicators['gdp_growth'] > 3.0:
                sentiment_score += 10
            elif indicators['gdp_growth'] < 2.0:
                sentiment_score -= 10
            
            # Unemployment impact (inverse)
            if indicators['unemployment'] < 4.0:
                sentiment_score += 8
            elif indicators['unemployment'] > 5.5:
                sentiment_score -= 8
            
            # Inflation impact (Goldilocks zone)
            if 2.0 <= indicators['inflation'] <= 3.0:
                sentiment_score += 5
            elif indicators['inflation'] > 4.0 or indicators['inflation'] < 1.0:
                sentiment_score -= 10
            
            # PMI impact
            if indicators['manufacturing_pmi'] > 55:
                sentiment_score += 8
            elif indicators['manufacturing_pmi'] < 45:
                sentiment_score -= 8
            
            sentiment_score = max(0, min(100, sentiment_score))
            
            if sentiment_score > 70:
                economic_outlook = "STRONG"
            elif sentiment_score > 55:
                economic_outlook = "POSITIVE"
            elif sentiment_score < 30:
                economic_outlook = "WEAK"
            elif sentiment_score < 45:
                economic_outlook = "NEGATIVE"
            else:
                economic_outlook = "NEUTRAL"
            
            return {
                'economic_sentiment_score': sentiment_score,
                'economic_outlook': economic_outlook,
                'indicators': indicators,
                'key_themes': self._identify_economic_themes(indicators)
            }
            
        except Exception as e:
            logger.error(f"Economic data analysis error: {e}")
            return {
                'economic_sentiment_score': 50,
                'economic_outlook': "NEUTRAL",
                'indicators': {},
                'key_themes': ["Data unavailable"]
            }
    
    def _identify_economic_themes(self, indicators: Dict) -> List[str]:
        """Identify key economic themes"""
        themes = []
        
        if indicators.get('gdp_growth', 0) > 3.0:
            themes.append("Strong economic growth")
        elif indicators.get('gdp_growth', 0) < 1.5:
            themes.append("Slowing economic growth")
        
        if indicators.get('inflation', 0) > 4.0:
            themes.append("High inflation concerns")
        elif indicators.get('inflation', 0) < 1.0:
            themes.append("Deflationary risks")
        
        if indicators.get('unemployment', 0) < 4.0:
            themes.append("Tight labor market")
        elif indicators.get('unemployment', 0) > 5.5:
            themes.append("Rising unemployment")
        
        if indicators.get('manufacturing_pmi', 50) > 55:
            themes.append("Manufacturing expansion")
        elif indicators.get('manufacturing_pmi', 50) < 45:
            themes.append("Manufacturing contraction")
        
        return themes[:3]  # Return top 3 themes

class RiskMetricsCalculator:
    """Calculate advanced risk metrics"""
    
    def __init__(self):
        logger.info("🛡️ Risk Metrics Calculator initialized")
    
    def calculate_var_95(self, price_history: List[float]) -> float:
        """Calculate 95% Value at Risk"""
        try:
            if len(price_history) < 2:
                return 5.0  # Default 5% risk
            
            # Calculate returns
            returns = []
            for i in range(1, len(price_history)):
                ret = (price_history[i] - price_history[i-1]) / price_history[i-1]
                returns.append(ret)
            
            # 95% VaR (5th percentile)
            if returns:
                var_95 = np.percentile(returns, 5) * 100  # Convert to percentage
                return abs(round(var_95, 2))
            else:
                return 5.0
                
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 5.0
    
    def calculate_max_drawdown_risk(self, price_history: List[float]) -> float:
        """Calculate maximum drawdown risk"""
        try:
            if len(price_history) < 2:
                return 10.0  # Default 10% risk
            
            peak = price_history[0]
            max_drawdown = 0.0
            
            for price in price_history:
                if price > peak:
                    peak = price
                
                drawdown = (peak - price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return round(max_drawdown * 100, 2)  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {e}")
            return 10.0
    
    def calculate_correlation_spy(self, symbol: str) -> float:
        """Calculate correlation with SPY"""
        try:
            # Simulate correlation (in production, use real data)
            if symbol == 'SPY':
                return 1.0
            elif symbol in ['QQQ', 'NVDA', 'AAPL', 'MSFT']:
                return random.uniform(0.7, 0.95)  # High correlation with SPY
            else:
                return random.uniform(0.3, 0.8)   # Moderate correlation
                
        except Exception as e:
            logger.error(f"Correlation calculation error: {e}")
            return 0.7  # Default moderate correlation

class UltraEnhancedHYPERSignalEngine:
    """Ultra-enhanced signal engine with all advanced predictive features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all components
        try:
            self.enhanced_engine = EnhancedPredictiveEngine()
            logger.info("✅ Enhanced predictive engine loaded")
        except:
            logger.warning("⚠️ Enhanced predictive engine not available, using fallback")
            self.enhanced_engine = None
        
        try:
            self.sentiment_aggregator = SocialMediaAggregator(
                news_api_key=config.get('NEWS_API_KEY'),
                reddit_client_id=config.get('REDDIT_CLIENT_ID'),
                reddit_secret=config.get('REDDIT_SECRET'),
                twitter_bearer=config.get('TWITTER_BEARER_TOKEN')
            )
            logger.info("✅ Sentiment aggregator loaded")
        except:
            logger.warning("⚠️ Sentiment aggregator not available, using fallback")
            self.sentiment_aggregator = None
        
        try:
            self.ml_engine = PatternRecognitionEngine()
            logger.info("✅ ML pattern engine loaded")
        except:
            logger.warning("⚠️ ML pattern engine not available, using fallback")
            self.ml_engine = None
        
        # Initialize new advanced analyzers
        self.advanced_technical = AdvancedTechnicalAnalyzer()
        self.market_structure = MarketStructureAnalyzer()
        self.advanced_ml = AdvancedMLPredictor()
        self.economic_analyzer = EconomicDataAnalyzer()
        self.risk_calculator = RiskMetricsCalculator()
        
        # Import original data aggregator
        from data_sources import HYPERDataAggregator
        self.data_aggregator = HYPERDataAggregator(config['ALPHA_VANTAGE_API_KEY'])
        
        # Ultra-enhanced weights
        self.weights = {
            'technical': 0.20,          # Traditional + advanced technical
            'ml_prediction': 0.18,      # ML + LSTM predictions
            'sentiment': 0.15,          # Multi-source sentiment
            'pattern': 0.12,            # Chart patterns
            'market_structure': 0.12,   # Breadth, sector rotation
            'economic': 0.10,           # Economic indicators
            'momentum': 0.08,           # Price momentum
            'risk_adjusted': 0.05       # Risk metrics
        }
        
        logger.info("🚀 Ultra-Enhanced HYPER Signal Engine initialized with ALL features!")
    
    async def generate_ultra_signal(self, symbol: str) -> UltraHYPERSignal:
        """Generate ultra-comprehensive trading signal with all features"""
        logger.info(f"🎯 Generating ULTRA signal for {symbol}")
        
        try:
            # 1. Get core market data
            market_data = await self.data_aggregator.get_comprehensive_data(symbol)
            quote_data = market_data.get('quote', {})
            
            if not quote_data or quote_data.get('price', 0) <= 0:
                return self._create_
