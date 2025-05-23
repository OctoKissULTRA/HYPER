# ============================================
# HYPER TRADING SYSTEM - Signal Engine
# The BRAIN that generates trading signals
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import asyncio

from config import config
from data_sources import HYPERDataAggregator

logger = logging.getLogger(__name__)

# ========================================
# SIGNAL DATA STRUCTURES
# ========================================

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

# ========================================
# TECHNICAL ANALYSIS ENGINE
# ========================================

class TechnicalAnalyzer:
    """Advanced technical analysis for HYPER signals"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicators"""
        ema_fast = TechnicalAnalyzer.calculate_ema(prices, fast)
        ema_slow = TechnicalAnalyzer.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalyzer.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'width': (std * std_dev * 2) / sma * 100
        }
    
    def analyze_technicals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive technical analysis"""
        if df is None or len(df) < 30:
            return self._empty_technical_analysis()
        
        try:
            prices = df['close']
            volume = df['volume']
            
            # Calculate all indicators
            rsi = self.calculate_rsi(prices, config.TECHNICAL_PARAMS['rsi_period'])
            ema_short = self.calculate_ema(prices, config.TECHNICAL_PARAMS['ema_short'])
            ema_long = self.calculate_ema(prices, config.TECHNICAL_PARAMS['ema_long'])
            macd_data = self.calculate_macd(prices)
            bb_data = self.calculate_bollinger_bands(prices)
            
            # Volume analysis
            volume_ma = volume.rolling(window=config.TECHNICAL_PARAMS['volume_ma_period']).mean()
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # Latest values
            latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
            latest_price = prices.iloc[-1]
            latest_ema_short = ema_short.iloc[-1] if not ema_short.empty else latest_price
            latest_ema_long = ema_long.iloc[-1] if not ema_long.empty else latest_price
            latest_macd = macd_data['histogram'].iloc[-1] if not macd_data['histogram'].empty else 0
            
            # Bollinger Bands position
            bb_upper = bb_data['upper'].iloc[-1] if not bb_data['upper'].empty else latest_price
            bb_lower = bb_data['lower'].iloc[-1] if not bb_data['lower'].empty else latest_price
            bb_position = self._get_bb_position(latest_price, bb_upper, bb_lower)
            
            # Generate technical signals
            signals = []
            score = 50  # Start neutral
            
            # RSI signals
            if latest_rsi < config.TECHNICAL_PARAMS['rsi_oversold']:
                signals.append("RSI oversold - bullish")
                score += 15
            elif latest_rsi > config.TECHNICAL_PARAMS['rsi_overbought']:
                signals.append("RSI overbought - bearish")
                score -= 15
            
            # EMA crossover
            if latest_ema_short > latest_ema_long:
                signals.append("EMA bullish trend")
                score += 10
            else:
                signals.append("EMA bearish trend")
                score -= 10
            
            # MACD signals
            if latest_macd > 0:
                signals.append("MACD bullish momentum")
                score += 8
            else:
                signals.append("MACD bearish momentum")
                score -= 8
            
            # Bollinger Bands
            if bb_position == "below":
                signals.append("Price below BB lower - oversold")
                score += 12
            elif bb_position == "above":
                signals.append("Price above BB upper - overbought")
                score -= 12
            
            # Volume confirmation
            if volume_ratio > 1.5:
                signals.append("High volume confirmation")
                score += 8
            elif volume_ratio < 0.7:
                signals.append("Low volume warning")
                score -= 5
            
            return {
                'score': max(0, min(100, score)),
                'rsi': latest_rsi,
                'ema_short': latest_ema_short,
                'ema_long': latest_ema_long,
                'macd_histogram': latest_macd,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'signals': signals,
                'direction': 'UP' if score > 55 else 'DOWN' if score < 45 else 'NEUTRAL'
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return self._empty_technical_analysis()
    
    def _get_bb_position(self, price: float, upper: float, lower: float) -> str:
        """Determine Bollinger Bands position"""
        if price > upper:
            return "above"
        elif price < lower:
            return "below"
        else:
            return "middle"
    
    def _empty_technical_analysis(self) -> Dict[str, Any]:
        """Return empty technical analysis"""
        return {
            'score': 50,
            'rsi': 50,
            'ema_short': 0,
            'ema_long': 0,
            'macd_histogram': 0,
            'bb_position': 'middle',
            'volume_ratio': 1,
            'signals': [],
            'direction': 'NEUTRAL'
        }

# ========================================
# SENTIMENT ANALYZER
# ========================================

class SentimentAnalyzer:
    """Google Trends sentiment analysis"""
    
    def analyze_trends_sentiment(self, trend_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze Google Trends sentiment"""
        if not trend_data or 'trend_analysis' not in trend_data:
            return self._empty_sentiment_analysis()
        
        try:
            analysis = trend_data['trend_analysis']
            keywords = config.get_ticker_keywords(symbol)
            
            # Aggregate sentiment across keywords
            total_momentum = 0
            total_velocity = 0
            total_acceleration = 0
            keyword_count = 0
            
            signals = []
            
            for keyword in keywords:
                if keyword in analysis:
                    kw_analysis = analysis[keyword]
                    total_momentum += kw_analysis.get('momentum', 0)
                    total_velocity += kw_analysis.get('velocity', 0)
                    total_acceleration += kw_analysis.get('acceleration', 0)
                    keyword_count += 1
                    
                    # Generate keyword-specific signals
                    if kw_analysis.get('momentum', 0) > 50:
                        signals.append(f"High search interest: {keyword}")
                    elif kw_analysis.get('momentum', 0) < -20:
                        signals.append(f"Declining interest: {keyword}")
            
            if keyword_count == 0:
                return self._empty_sentiment_analysis()
            
            # Calculate averages
            avg_momentum = total_momentum / keyword_count
            avg_velocity = total_velocity / keyword_count
            avg_acceleration = total_acceleration / keyword_count
            
            # Generate sentiment score (0-100)
            score = 50  # Start neutral
            
            # Momentum contribution
            if avg_momentum > 100:
                score += 20  # Very high interest
                signals.append("Extreme search volume spike")
            elif avg_momentum > 25:
                score += 10  # Moderate interest
                signals.append("Increased search interest")
            elif avg_momentum < -25:
                score -= 10  # Declining interest
                signals.append("Decreasing search interest")
            
            # Velocity contribution
            if avg_velocity > 50:
                score += 10  # Accelerating interest
                signals.append("Accelerating search trends")
            elif avg_velocity < -50:
                score -= 10  # Decelerating interest
                signals.append("Search interest cooling")
            
            # Contrarian indicators (extreme sentiment warning)
            if avg_momentum > 300:  # Extreme spike
                score -= 15  # Potential top
                signals.append("⚠️ Extreme hype detected - contrarian signal")
            
            direction = 'UP' if score > 55 else 'DOWN' if score < 45 else 'NEUTRAL'
            
            return {
                'score': max(0, min(100, score)),
                'momentum': avg_momentum,
                'velocity': avg_velocity,
                'acceleration': avg_acceleration,
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
            'acceleration': 0,
            'signals': [],
            'direction': 'NEUTRAL',
            'keywords_analyzed': 0
        }

# ========================================
# FAKE-OUT DETECTOR
# ========================================

class FakeOutDetector:
    """Detects manipulation and fake signals"""
    
    def detect_fake_patterns(self, technical_data: Dict, sentiment_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Detect potential fake-out patterns"""
        warnings = []
        suspicion_score = 0
        confidence_penalty = 0
        
        try:
            # Check volume divergence
            volume_ratio = technical_data.get('volume_ratio', 1)
            if volume_ratio < config.FAKE_OUT_FILTERS['volume_threshold']:
                warnings.append("Low volume breakout - potential fake-out")
                suspicion_score += 25
                confidence_penalty += 0.15
            
            # Check extreme sentiment
            sentiment_momentum = sentiment_data.get('momentum', 0)
            if abs(sentiment_momentum) > 400:  # Extreme spike
                warnings.append("Extreme sentiment spike - manipulation risk")
                suspicion_score += 30
                confidence_penalty += 0.20
            
            # Check technical divergences
            rsi = technical_data.get('rsi', 50)
            bb_position = technical_data.get('bb_position', 'middle')
            
            if rsi > 85 and bb_position == 'above':
                warnings.append("Extreme overbought conditions")
                suspicion_score += 20
                confidence_penalty += 0.10
            elif rsi < 15 and bb_position == 'below':
                warnings.append("Extreme oversold conditions")
                suspicion_score += 20
                confidence_penalty += 0.10
            
            # Check sentiment vs price divergence
            tech_direction = technical_data.get('direction', 'NEUTRAL')
            sentiment_direction = sentiment_data.get('direction', 'NEUTRAL')
            
            if tech_direction != sentiment_direction and tech_direction != 'NEUTRAL' and sentiment_direction != 'NEUTRAL':
                warnings.append("Technical vs sentiment divergence")
                suspicion_score += 15
                confidence_penalty += 0.10
            
            return {
                'suspicion_score': suspicion_score,
                'confidence_penalty': confidence_penalty,
                'warnings': warnings,
                'is_suspicious': suspicion_score > 50
            }
            
        except Exception as e:
            logger.error(f"Fake-out detection error: {e}")
            return {
                'suspicion_score': 0,
                'confidence_penalty': 0,
                'warnings': [],
                'is_suspicious': False
            }

# ========================================
# MAIN SIGNAL ENGINE
# ========================================

class HYPERSignalEngine:
    """Main signal generation engine"""
    
    def __init__(self):
        self.data_aggregator = HYPERDataAggregator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fake_out_detector = FakeOutDetector()
        
    async def generate_signal(self, symbol: str) -> HYPERSignal:
        """Generate comprehensive HYPER signal"""
        try:
            # Get all data for symbol
            comprehensive_data = await self.data_aggregator.get_comprehensive_data(symbol)
            
            if not comprehensive_data or comprehensive_data.get('data_quality') == 'poor':
                return self._create_no_signal(symbol, "Insufficient data")
            
            # Extract components
            quote = comprehensive_data.get('quote')
            intraday = comprehensive_data.get('intraday')
            trends = comprehensive_data
            
            if not quote:
                return self._create_no_signal(symbol, "No quote data")
            
            # Run analysis components
            technical_analysis = self.technical_analyzer.analyze_technicals(intraday)
            sentiment_analysis = self.sentiment_analyzer.analyze_trends_sentiment(trends, symbol)
            fake_out_analysis = self.fake_out_detector.detect_fake_patterns(
                technical_analysis, sentiment_analysis, quote
            )
            
            # Calculate weighted scores
            technical_score = technical_analysis['score']
            momentum_score = self._calculate_momentum_score(quote, intraday)
            trends_score = sentiment_analysis['score']
            volume_score = self._calculate_volume_score(technical_analysis)
            ml_score = 50  # TODO: Implement ML predictions
            
            # Apply signal weights
            weighted_score = (
                technical_score * config.SIGNAL_WEIGHTS['technical_analysis'] +
                momentum_score * config.SIGNAL_WEIGHTS['alpha_vantage_momentum'] +
                trends_score * config.SIGNAL_WEIGHTS['google_trends'] +
                volume_score * config.SIGNAL_WEIGHTS['volume_analysis'] +
                ml_score * config.SIGNAL_WEIGHTS['ml_ensemble']
            )
            
            # Apply fake-out penalty
            confidence_penalty = fake_out_analysis['confidence_penalty']
            final_confidence = max(0, weighted_score * (1 - confidence_penalty))
            
            # Determine direction
            direction = self._determine_direction(technical_analysis, sentiment_analysis, momentum_score)
            
            # Classify signal
            signal_type = config.is_high_confidence_signal(final_confidence, direction)
            
            # Compile reasons
            reasons = []
            reasons.extend(technical_analysis.get('signals', []))
            reasons.extend(sentiment_analysis.get('signals', []))
            if momentum_score > 60:
                reasons.append("Strong price momentum")
            elif momentum_score < 40:
                reasons.append("Weak price momentum")
            
            # Create signal
            return HYPERSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=round(final_confidence, 1),
                direction=direction,
                price=quote['price'],
                timestamp=datetime.now().isoformat(),
                technical_score=technical_score,
                momentum_score=momentum_score,
                trends_score=trends_score,
                volume_score=volume_score,
                ml_score=ml_score,
                indicators={
                    'rsi': technical_analysis.get('rsi', 50),
                    'macd': technical_analysis.get('macd_histogram', 0),
                    'bb_position': technical_analysis.get('bb_position', 'middle'),
                    'volume_ratio': technical_analysis.get('volume_ratio', 1),
                    'trend_momentum': sentiment_analysis.get('momentum', 0)
                },
                reasons=reasons,
                warnings=fake_out_analysis['warnings'],
                data_quality=comprehensive_data.get('data_quality', 'unknown')
            )
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return self._create_no_signal(symbol, f"Error: {str(e)}")
    
    def _calculate_momentum_score(self, quote: Dict, intraday: pd.DataFrame) -> float:
        """Calculate momentum score from price action"""
        try:
            if not quote:
                return 50
            
            change_percent = quote.get('change_percent', 0)
            
            # Base score from daily change
            score = 50 + (change_percent * 5)  # Each 1% = 5 points
            
            # Intraday momentum
            if intraday is not None and len(intraday) > 5:
                recent_change = ((intraday['close'].iloc[-1] / intraday['close'].iloc[-5]) - 1) * 100
                score += recent_change * 3
            
            return max(0, min(100, score))
            
        except Exception:
            return 50
    
    def _calculate_volume_score(self, technical_analysis: Dict) -> float:
        """Calculate volume-based score"""
        volume_ratio = technical_analysis.get('volume_ratio', 1)
        
        if volume_ratio > 2:
            return 80  # High volume
        elif volume_ratio > 1.5:
            return 70
        elif volume_ratio > 1.2:
            return 60
        elif volume_ratio < 0.7:
            return 30  # Low volume
        else:
            return 50  # Normal volume
    
    def _determine_direction(self, technical: Dict, sentiment: Dict, momentum: float) -> str:
        """Determine overall signal direction"""
        directions = [
            technical.get('direction', 'NEUTRAL'),
            sentiment.get('direction', 'NEUTRAL')
        ]
        
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
    
    def _create_no_signal(self, symbol: str, reason: str) -> HYPERSignal:
        """Create a HOLD signal when no data available"""
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
            reasons=[reason],
            warnings=[],
            data_quality='poor'
        )
    
    async def generate_all_signals(self) -> Dict[str, HYPERSignal]:
        """Generate signals for all configured tickers"""
        tasks = [self.generate_signal(ticker) for ticker in config.TICKERS]
        signals = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for i, ticker in enumerate(config.TICKERS):
            if not isinstance(signals[i], Exception):
                result[ticker] = signals[i]
            else:
                logger.error(f"Error generating signal for {ticker}: {signals[i]}")
                result[ticker] = self._create_no_signal(ticker, "Generation failed")
        
        return result
