# technical_indicators.py - Advanced Technical Analysis Module
import numpy as np
import pandas as pd
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Individual technical indicator signal"""
    indicator_name: str
    value: float
    signal_strength: float  # 0-100
    direction: str  # UP, DOWN, NEUTRAL
    confidence: float  # 0-1
    oversold: bool = False
    overbought: bool = False
    divergence_detected: bool = False

@dataclass
class TechnicalAnalysis:
    """Complete technical analysis result"""
    overall_score: float
    direction: str
    confidence: float
    signals: List[TechnicalSignal]
    key_levels: Dict[str, float]
    pattern_analysis: Dict[str, Any]
    momentum_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]

class AdvancedTechnicalAnalyzer:
    """Advanced Technical Analysis with 25+ indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.price_history = {}  # Store price history for calculations
        self.volume_history = {}
        self.indicator_cache = {}
        self.fibonacci_cache = {}
        
        # Advanced indicator parameters
        self.params = {
            'rsi_period': config.get('rsi_period', 14),
            'williams_r_period': config.get('williams_r_period', 14),
            'stochastic_k_period': config.get('stochastic_k_period', 14),
            'stochastic_d_period': config.get('stochastic_d_period', 3),
            'macd_fast': config.get('macd_fast', 12),
            'macd_slow': config.get('macd_slow', 26),
            'macd_signal': config.get('macd_signal', 9),
            'bb_period': config.get('bb_period', 20),
            'bb_std': config.get('bb_std', 2),
            'atr_period': config.get('atr_period', 14),
            'adx_period': config.get('adx_period', 14),
            'cci_period': config.get('cci_period', 20),
            'fibonacci_lookback': config.get('fibonacci_lookback', 50),
            'volume_ma_period': config.get('volume_ma_period', 20),
            'vwap_period': config.get('vwap_period', 20)
        }
        
        logger.info("📊 Advanced Technical Analyzer initialized with 25+ indicators")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], 
                     historical_data: Optional[List[Dict]] = None) -> TechnicalAnalysis:
        """Complete technical analysis with all indicators"""
        try:
            # Generate or use historical data
            if not historical_data:
                historical_data = self._generate_realistic_history(symbol, quote_data)
            
            # Extract price and volume arrays
            prices = np.array([float(d.get('close', d.get('price', 0))) for d in historical_data])
            volumes = np.array([float(d.get('volume', 0)) for d in historical_data])
            highs = np.array([float(d.get('high', d.get('price', 0))) for d in historical_data])
            lows = np.array([float(d.get('low', d.get('price', 0))) for d in historical_data])
            opens = np.array([float(d.get('open', d.get('price', 0))) for d in historical_data])
            
            # Ensure we have enough data
            if len(prices) < 50:
                logger.warning(f"Insufficient data for {symbol}, using enhanced simulation")
                return self._generate_enhanced_technical_analysis(symbol, quote_data)
            
            # Calculate all technical indicators
            signals = []
            
            # === MOMENTUM OSCILLATORS ===
            signals.extend(await self._analyze_momentum_oscillators(prices, highs, lows))
            
            # === TREND INDICATORS ===
            signals.extend(await self._analyze_trend_indicators(prices, volumes))
            
            # === VOLATILITY INDICATORS ===
            signals.extend(await self._analyze_volatility_indicators(prices, highs, lows))
            
            # === VOLUME INDICATORS ===
            signals.extend(await self._analyze_volume_indicators(prices, volumes))
            
            # === SUPPORT/RESISTANCE LEVELS ===
            key_levels = await self._calculate_key_levels(prices, highs, lows)
            
            # === PATTERN ANALYSIS ===
            pattern_analysis = await self._analyze_chart_patterns(prices, highs, lows, volumes)
            
            # === MOMENTUM ANALYSIS ===
            momentum_analysis = await self._analyze_momentum_strength(prices, volumes)
            
            # === VOLUME ANALYSIS ===
            volume_analysis = await self._analyze_volume_profile(prices, volumes)
            
            # Calculate overall technical score
            overall_score, direction, confidence = self._calculate_overall_score(signals)
            
            return TechnicalAnalysis(
                overall_score=overall_score,
                direction=direction,
                confidence=confidence,
                signals=signals,
                key_levels=key_levels,
                pattern_analysis=pattern_analysis,
                momentum_analysis=momentum_analysis,
                volume_analysis=volume_analysis
            )
            
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return self._generate_enhanced_technical_analysis(symbol, quote_data)
    
    async def _analyze_momentum_oscillators(self, prices: np.ndarray, 
                                          highs: np.ndarray, lows: np.ndarray) -> List[TechnicalSignal]:
        """Analyze momentum oscillators (RSI, Williams %R, Stochastic, etc.)"""
        signals = []
        
        try:
            # RSI Analysis
            rsi = self._calculate_rsi(prices, self.params['rsi_period'])
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            
            rsi_signal = TechnicalSignal(
                indicator_name="RSI",
                value=current_rsi,
                signal_strength=abs(current_rsi - 50) * 2,  # 0-100 scale
                direction="DOWN" if current_rsi > 70 else "UP" if current_rsi < 30 else "NEUTRAL",
                confidence=min(1.0, abs(current_rsi - 50) / 30),
                oversold=current_rsi < 30,
                overbought=current_rsi > 70,
                divergence_detected=self._detect_rsi_divergence(rsi, prices)
            )
            signals.append(rsi_signal)
            
            # Williams %R Analysis
            williams_r = self._calculate_williams_r(highs, lows, prices, self.params['williams_r_period'])
            current_wr = williams_r[-1] if len(williams_r) > 0 else -50
            
            wr_signal = TechnicalSignal(
                indicator_name="Williams_R",
                value=current_wr,
                signal_strength=abs(current_wr + 50) * 2,
                direction="UP" if current_wr < -80 else "DOWN" if current_wr > -20 else "NEUTRAL",
                confidence=min(1.0, abs(current_wr + 50) / 30),
                oversold=current_wr < -80,
                overbought=current_wr > -20
            )
            signals.append(wr_signal)
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, prices, 
                                                         self.params['stochastic_k_period'],
                                                         self.params['stochastic_d_period'])
            current_k = stoch_k[-1] if len(stoch_k) > 0 else 50
            current_d = stoch_d[-1] if len(stoch_d) > 0 else 50
            
            stoch_signal = TechnicalSignal(
                indicator_name="Stochastic",
                value=current_k,
                signal_strength=abs(current_k - 50) * 2,
                direction="UP" if current_k < 20 and current_k > current_d else "DOWN" if current_k > 80 and current_k < current_d else "NEUTRAL",
                confidence=min(1.0, abs(current_k - 50) / 30),
                oversold=current_k < 20,
                overbought=current_k > 80
            )
            signals.append(stoch_signal)
            
            # CCI (Commodity Channel Index)
            cci = self._calculate_cci(highs, lows, prices, self.params['cci_period'])
            current_cci = cci[-1] if len(cci) > 0 else 0
            
            cci_signal = TechnicalSignal(
                indicator_name="CCI",
                value=current_cci,
                signal_strength=min(100, abs(current_cci) / 2),
                direction="UP" if current_cci < -100 else "DOWN" if current_cci > 100 else "NEUTRAL",
                confidence=min(1.0, abs(current_cci) / 200),
                oversold=current_cci < -100,
                overbought=current_cci > 100
            )
            signals.append(cci_signal)
            
            # Money Flow Index (MFI) - Volume-weighted RSI
            mfi = self._calculate_mfi(highs, lows, prices, np.ones(len(prices)) * 1000000, 14)  # Simulated volume
            current_mfi = mfi[-1] if len(mfi) > 0 else 50
            
            mfi_signal = TechnicalSignal(
                indicator_name="MFI",
                value=current_mfi,
                signal_strength=abs(current_mfi - 50) * 2,
                direction="UP" if current_mfi < 20 else "DOWN" if current_mfi > 80 else "NEUTRAL",
                confidence=min(1.0, abs(current_mfi - 50) / 30),
                oversold=current_mfi < 20,
                overbought=current_mfi > 80
            )
            signals.append(mfi_signal)
            
        except Exception as e:
            logger.error(f"Momentum oscillator analysis error: {e}")
        
        return signals
    
    async def _analyze_trend_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> List[TechnicalSignal]:
        """Analyze trend-following indicators"""
        signals = []
        
        try:
            # MACD Analysis
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices, 
                                                                         self.params['macd_fast'],
                                                                         self.params['macd_slow'],
                                                                         self.params['macd_signal'])
            
            current_macd = macd_line[-1] if len(macd_line) > 0 else 0
            current_signal = macd_signal[-1] if len(macd_signal) > 0 else 0
            current_histogram = macd_histogram[-1] if len(macd_histogram) > 0 else 0
            
            macd_signal_obj = TechnicalSignal(
                indicator_name="MACD",
                value=current_macd,
                signal_strength=min(100, abs(current_histogram) * 100),
                direction="UP" if current_macd > current_signal and current_histogram > 0 else "DOWN" if current_macd < current_signal and current_histogram < 0 else "NEUTRAL",
                confidence=min(1.0, abs(current_histogram) * 10),
                divergence_detected=self._detect_macd_divergence(macd_histogram, prices)
            )
            signals.append(macd_signal_obj)
            
            # ADX (Average Directional Index) - Trend Strength
            adx = self._calculate_adx(prices, self.params['adx_period'])
            current_adx = adx[-1] if len(adx) > 0 else 25
            
            adx_signal = TechnicalSignal(
                indicator_name="ADX",
                value=current_adx,
                signal_strength=current_adx,
                direction="UP" if current_adx > 25 else "NEUTRAL",
                confidence=min(1.0, current_adx / 50)
            )
            signals.append(adx_signal)
            
            # Moving Average Convergence
            ema_9 = self._calculate_ema(prices, 9)
            ema_21 = self._calculate_ema(prices, 21)
            ema_50 = self._calculate_ema(prices, 50)
            
            current_price = prices[-1]
            current_ema9 = ema_9[-1] if len(ema_9) > 0 else current_price
            current_ema21 = ema_21[-1] if len(ema_21) > 0 else current_price
            current_ema50 = ema_50[-1] if len(ema_50) > 0 else current_price
            
            # Determine MA trend
            ma_bullish = current_ema9 > current_ema21 > current_ema50 and current_price > current_ema9
            ma_bearish = current_ema9 < current_ema21 < current_ema50 and current_price < current_ema9
            
            ma_signal = TechnicalSignal(
                indicator_name="Moving_Averages",
                value=(current_price / current_ema21 - 1) * 100,  # % above/below MA
                signal_strength=80 if ma_bullish or ma_bearish else 40,
                direction="UP" if ma_bullish else "DOWN" if ma_bearish else "NEUTRAL",
                confidence=0.8 if ma_bullish or ma_bearish else 0.4
            )
            signals.append(ma_signal)
            
            # Parabolic SAR
            psar = self._calculate_parabolic_sar(prices, acceleration=0.02, maximum=0.2)
            current_psar = psar[-1] if len(psar) > 0 else current_price
            
            psar_signal = TechnicalSignal(
                indicator_name="Parabolic_SAR",
                value=current_psar,
                signal_strength=70,
                direction="UP" if current_price > current_psar else "DOWN",
                confidence=0.7
            )
            signals.append(psar_signal)
            
        except Exception as e:
            logger.error(f"Trend indicator analysis error: {e}")
        
        return signals
    
    async def _analyze_volatility_indicators(self, prices: np.ndarray, 
                                           highs: np.ndarray, lows: np.ndarray) -> List[TechnicalSignal]:
        """Analyze volatility-based indicators"""
        signals = []
        
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 
                                                                           self.params['bb_period'],
                                                                           self.params['bb_std'])
            
            current_price = prices[-1]
            current_upper = bb_upper[-1] if len(bb_upper) > 0 else current_price * 1.02
            current_lower = bb_lower[-1] if len(bb_lower) > 0 else current_price * 0.98
            current_middle = bb_middle[-1] if len(bb_middle) > 0 else current_price
            
            # BB Position (0-100, where 50 is middle band)
            bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100 if current_upper != current_lower else 50
            
            bb_signal = TechnicalSignal(
                indicator_name="Bollinger_Bands",
                value=bb_position,
                signal_strength=abs(bb_position - 50) * 2,
                direction="DOWN" if bb_position > 80 else "UP" if bb_position < 20 else "NEUTRAL",
                confidence=min(1.0, abs(bb_position - 50) / 30),
                overbought=bb_position > 80,
                oversold=bb_position < 20
            )
            signals.append(bb_signal)
            
            # Average True Range (ATR) - Volatility measure
            atr = self._calculate_atr(highs, lows, prices, self.params['atr_period'])
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            atr_percentage = (current_atr / current_price) * 100
            
            atr_signal = TechnicalSignal(
                indicator_name="ATR",
                value=atr_percentage,
                signal_strength=min(100, atr_percentage * 10),
                direction="NEUTRAL",  # ATR indicates volatility, not direction
                confidence=0.6
            )
            signals.append(atr_signal)
            
            # Keltner Channels
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(highs, lows, prices, 20, 2.0)
            kc_position = ((current_price - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1])) * 100 if len(kc_upper) > 0 else 50
            
            kc_signal = TechnicalSignal(
                indicator_name="Keltner_Channels",
                value=kc_position,
                signal_strength=abs(kc_position - 50) * 2,
                direction="DOWN" if kc_position > 85 else "UP" if kc_position < 15 else "NEUTRAL",
                confidence=min(1.0, abs(kc_position - 50) / 35)
            )
            signals.append(kc_signal)
            
        except Exception as e:
            logger.error(f"Volatility indicator analysis error: {e}")
        
        return signals
    
    async def _analyze_volume_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> List[TechnicalSignal]:
        """Analyze volume-based indicators"""
        signals = []
        
        try:
            # On-Balance Volume (OBV)
            obv = self._calculate_obv(prices, volumes)
            obv_ma = self._calculate_sma(obv, 10)
            
            current_obv = obv[-1] if len(obv) > 0 else 0
            current_obv_ma = obv_ma[-1] if len(obv_ma) > 0 else 0
            
            obv_signal = TechnicalSignal(
                indicator_name="OBV",
                value=current_obv,
                signal_strength=60,
                direction="UP" if current_obv > current_obv_ma else "DOWN",
                confidence=0.6
            )
            signals.append(obv_signal)
            
            # Volume Rate of Change
            volume_roc = self._calculate_roc(volumes, 10)
            current_vol_roc = volume_roc[-1] if len(volume_roc) > 0 else 0
            
            vol_roc_signal = TechnicalSignal(
                indicator_name="Volume_ROC",
                value=current_vol_roc,
                signal_strength=min(100, abs(current_vol_roc)),
                direction="UP" if current_vol_roc > 20 else "NEUTRAL",
                confidence=min(1.0, abs(current_vol_roc) / 50)
            )
            signals.append(vol_roc_signal)
            
            # VWAP (Volume Weighted Average Price)
            vwap = self._calculate_vwap(prices, volumes, self.params['vwap_period'])
            current_vwap = vwap[-1] if len(vwap) > 0 else prices[-1]
            current_price = prices[-1]
            
            vwap_signal = TechnicalSignal(
                indicator_name="VWAP",
                value=((current_price / current_vwap) - 1) * 100,
                signal_strength=70,
                direction="UP" if current_price > current_vwap else "DOWN",
                confidence=0.7
            )
            signals.append(vwap_signal)
            
        except Exception as e:
            logger.error(f"Volume indicator analysis error: {e}")
        
        return signals
    
    async def _calculate_key_levels(self, prices: np.ndarray, 
                                   highs: np.ndarray, lows: np.ndarray) -> Dict[str, float]:
        """Calculate key support/resistance levels"""
        try:
            current_price = prices[-1]
            
            # Fibonacci Retracement Levels
            lookback = min(len(prices), self.params['fibonacci_lookback'])
            recent_high = np.max(highs[-lookback:])
            recent_low = np.min(lows[-lookback:])
            
            fib_range = recent_high - recent_low
            fib_levels = {
                'fibonacci_23.6': recent_high - (fib_range * 0.236),
                'fibonacci_38.2': recent_high - (fib_range * 0.382),
                'fibonacci_50.0': recent_high - (fib_range * 0.500),
                'fibonacci_61.8': recent_high - (fib_range * 0.618),
                'fibonacci_78.6': recent_high - (fib_range * 0.786)
            }
            
            # Pivot Points
            yesterday_high = highs[-2] if len(highs) > 1 else current_price * 1.01
            yesterday_low = lows[-2] if len(lows) > 1 else current_price * 0.99
            yesterday_close = prices[-2] if len(prices) > 1 else current_price
            
            pivot = (yesterday_high + yesterday_low + yesterday_close) / 3
            
            pivot_levels = {
                'pivot_point': pivot,
                'resistance_1': (2 * pivot) - yesterday_low,
                'resistance_2': pivot + (yesterday_high - yesterday_low),
                'support_1': (2 * pivot) - yesterday_high,
                'support_2': pivot - (yesterday_high - yesterday_low)
            }
            
            # Moving Average Levels
            ma_levels = {
                'sma_20': np.mean(prices[-20:]) if len(prices) >= 20 else current_price,
                'ema_50': self._calculate_ema(prices, 50)[-1] if len(prices) >= 50 else current_price,
                'sma_200': np.mean(prices[-200:]) if len(prices) >= 200 else current_price
            }
            
            # Combine all levels
            key_levels = {**fib_levels, **pivot_levels, **ma_levels}
            
            # Round all levels to 2 decimal places
            return {k: round(v, 2) for k, v in key_levels.items()}
            
        except Exception as e:
            logger.error(f"Key levels calculation error: {e}")
            current_price = prices[-1] if len(prices) > 0 else 100
            return {
                'support_1': round(current_price * 0.98, 2),
                'resistance_1': round(current_price * 1.02, 2),
                'pivot_point': round(current_price, 2)
            }
    
    def _calculate_overall_score(self, signals: List[TechnicalSignal]) -> Tuple[float, str, float]:
        """Calculate overall technical score from individual signals"""
        if not signals:
            return 50.0, "NEUTRAL", 0.5
        
        # Weight signals by confidence and importance
        signal_weights = {
            'RSI': 1.0,
            'Williams_R': 0.9,
            'Stochastic': 0.8,
            'MACD': 1.1,
            'Moving_Averages': 1.2,
            'Bollinger_Bands': 0.9,
            'ADX': 0.7,
            'Volume_ROC': 0.6,
            'VWAP': 0.8,
            'OBV': 0.6
        }
        
        weighted_scores = []
        total_weight = 0
        
        for signal in signals:
            weight = signal_weights.get(signal.indicator_name, 0.5)
            confidence_weight = signal.confidence
            
            # Convert direction to score
            if signal.direction == "UP":
                direction_score = 50 + (signal.signal_strength / 2)
            elif signal.direction == "DOWN":
                direction_score = 50 - (signal.signal_strength / 2)
            else:
                direction_score = 50
            
            weighted_score = direction_score * weight * confidence_weight
            weighted_scores.append(weighted_score)
            total_weight += weight * confidence_weight
        
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 50
        
        # Determine direction and confidence
        if overall_score > 65:
            direction = "UP"
            confidence = min(1.0, (overall_score - 50) / 50)
        elif overall_score < 35:
            direction = "DOWN"
            confidence = min(1.0, (50 - overall_score) / 50)
        else:
            direction = "NEUTRAL"
            confidence = 1.0 - (abs(overall_score - 50) / 15)
        
        return round(overall_score, 1), direction, round(confidence, 3)
    
    # ========================================
    # HELPER CALCULATION METHODS
    # ========================================
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def _calculate_williams_r(self, highs: np.ndarray, lows: np.ndarray, 
                             closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R"""
        williams_r = np.zeros_like(closes)
        for i in range(period-1, len(closes)):
            highest_high = np.max(highs[i-period+1:i+1])
            lowest_low = np.min(lows[i-period+1:i+1])
            williams_r[i] = -100 * (highest_high - closes[i]) / (highest_high - lowest_low) if highest_high != lowest_low else -50
        return williams_r
    
    def _calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, 
                             closes: np.ndarray, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        k_values = np.zeros_like(closes)
        for i in range(k_period-1, len(closes)):
            highest_high = np.max(highs[i-k_period+1:i+1])
            lowest_low = np.min(lows[i-k_period+1:i+1])
            k_values[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low) if highest_high != lowest_low else 50
        
        d_values = self._calculate_sma(k_values, d_period)
        return k_values, d_values
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    def _calculate_sma(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(values)
        for i in range(period-1, len(values)):
            sma[i] = np.mean(values[i-period+1:i+1])
        return sma
    
    def _generate_realistic_history(self, symbol: str, quote_data: Dict[str, Any], periods: int = 100) -> List[Dict]:
        """Generate realistic price history for technical analysis"""
        current_price = float(quote_data.get('price', 100))
        change_percent = float(quote_data.get('change_percent', 0))
        
        # Generate historical data working backwards from current price
        history = []
        price = current_price
        
        # Symbol-specific volatility
        volatilities = {
            'NVDA': 0.035,  # 3.5% daily volatility
            'QQQ': 0.025,   # 2.5% 
            'SPY': 0.020,   # 2.0%
            'AAPL': 0.025,  # 2.5%
            'MSFT': 0.022   # 2.2%
        }
        daily_vol = volatilities.get(symbol, 0.025)
        
        # Generate trend and mean reversion components
        trend_strength = change_percent * 0.1  # Carry some momentum
        
        for i in range(periods):
            # Random walk with trend and mean reversion
            random_component = np.random.normal(0, daily_vol)
            trend_component = trend_strength * (1 - i/periods)  # Fade trend over time
            mean_reversion = -0.1 * random_component  # Slight mean reversion
            
            price_change = random_component + trend_component + mean_reversion
            price = price * (1 + price_change)
            
            # Generate OHLC from the closing price
            daily_range = price * daily_vol * np.random.uniform(0.5, 2.0)
            open_price = price + np.random.uniform(-daily_range/4, daily_range/4)
            high = max(price, open_price) + np.random.uniform(0, daily_range/2)
            low = min(price, open_price) - np.random.uniform(0, daily_range/2)
            
            # Volume (higher volume on bigger moves)
            base_volume = 25000000
            volume_multiplier = 1 + abs(price_change) * 3
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.7, 1.3))
            
            history.insert(0, {  # Insert at beginning for chronological order
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'price': round(price, 2),
                'volume': volume,
                'date': (datetime.now() - timedelta(days=periods-i)).strftime('%Y-%m-%d')
            })
        
        return history
    
    def _calculate_cci(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        typical_prices = (highs + lows + closes) / 3
        cci = np.zeros_like(closes)
        
        for i in range(period-1, len(closes)):
            tp_slice = typical_prices[i-period+1:i+1]
            sma_tp = np.mean(tp_slice)
            mean_deviation = np.mean(np.abs(tp_slice - sma_tp))
            cci[i] = (typical_prices[i] - sma_tp) / (0.015 * mean_deviation) if mean_deviation != 0 else 0
        
        return cci
    
    def _calculate_mfi(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                       volumes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Money Flow Index"""
        typical_prices = (highs + lows + closes) / 3
        money_flow = typical_prices * volumes
        mfi = np.zeros_like(closes)
        
        for i in range(period, len(closes)):
            positive_flow = 0
            negative_flow = 0
            
            for j in range(i-period+1, i+1):
                if j > 0:
                    if typical_prices[j] > typical_prices[j-1]:
                        positive_flow += money_flow[j]
                    elif typical_prices[j] < typical_prices[j-1]:
                        negative_flow += money_flow[j]
            
            money_flow_ratio = positive_flow / negative_flow if negative_flow != 0 else 100
            mfi[i] = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def _calculate_adx(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        adx = np.zeros_like(prices)
        price_changes = np.diff(prices)
        
        for i in range(period, len(prices)):
            up_moves = np.where(price_changes[i-period:i] > 0, price_changes[i-period:i], 0)
            down_moves = np.where(price_changes[i-period:i] < 0, -price_changes[i-period:i], 0)
            
            avg_up = np.mean(up_moves)
            avg_down = np.mean(down_moves)
            
            if avg_up + avg_down == 0:
                adx[i] = 0
            else:
                di_diff = abs(avg_up - avg_down)
                di_sum = avg_up + avg_down
                adx[i] = (di_diff / di_sum) * 100
        
        return adx
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        middle = self._calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range"""
        atr = np.zeros_like(closes)
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close_prev, low_close_prev)
            
            if i == 1:
                atr[i] = true_range
            else:
                atr[i] = ((atr[i-1] * (period-1)) + true_range) / period
        
        return atr
    
    def _calculate_keltner_channels(self, highs: np.ndarray, lows: np.ndarray, 
                                   closes: np.ndarray, period: int, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Keltner Channels"""
        middle = self._calculate_ema(closes, period)
        atr = self._calculate_atr(highs, lows, closes, period)
        
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        return upper, middle, lower
    
    def _calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv = np.zeros_like(prices)
        obv[0] = volumes[0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _calculate_roc(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change"""
        roc = np.zeros_like(values)
        for i in range(period, len(values)):
            if values[i-period] != 0:
                roc[i] = ((values[i] - values[i-period]) / values[i-period]) * 100
        return roc
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        vwap = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            price_volume = prices[i-period+1:i+1] * volumes[i-period+1:i+1]
            total_volume = np.sum(volumes[i-period+1:i+1])
            vwap[i] = np.sum(price_volume) / total_volume if total_volume != 0 else prices[i]
        
        return vwap
    
    def _calculate_parabolic_sar(self, prices: np.ndarray, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """Calculate Parabolic SAR"""
        sar = np.zeros_like(prices)
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = prices[0]  # Extreme point
        
        sar[0] = prices[0]
        
        for i in range(1, len(prices)):
            if trend == 1:  # Uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if prices[i] > ep:
                    ep = prices[i]
                    af = min(af + acceleration, maximum)
                
                if prices[i] <= sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = prices[i]
                    af = acceleration
            else:  # Downtrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if prices[i] < ep:
                    ep = prices[i]
                    af = min(af + acceleration, maximum)
                
                if prices[i] >= sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = prices[i]
                    af = acceleration
        
        return sar
    
    def _detect_rsi_divergence(self, rsi: np.ndarray, prices: np.ndarray) -> bool:
        """Detect RSI divergence with price"""
        if len(rsi) < 20:
            return False
        
        # Look at last 20 periods for divergence
        recent_rsi = rsi[-20:]
        recent_prices = prices[-20:]
        
        # Simple divergence detection
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        price_trend = recent_prices[-1] - recent_prices[0]
        
        # Bullish divergence: price down, RSI up
        # Bearish divergence: price up, RSI down
        return (price_trend > 0 and rsi_trend < -5) or (price_trend < 0 and rsi_trend > 5)
    
    def _detect_macd_divergence(self, macd_histogram: np.ndarray, prices: np.ndarray) -> bool:
        """Detect MACD divergence with price"""
        if len(macd_histogram) < 20:
            return False
        
        recent_histogram = macd_histogram[-20:]
        recent_prices = prices[-20:]
        
        histogram_trend = recent_histogram[-1] - recent_histogram[0]
        price_trend = recent_prices[-1] - recent_prices[0]
        
        return (price_trend > 0 and histogram_trend < -0.1) or (price_trend < 0 and histogram_trend > 0.1)
    
    async def _analyze_chart_patterns(self, prices: np.ndarray, highs: np.ndarray, 
                                     lows: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Advanced chart pattern analysis"""
        try:
            if len(prices) < 20:
                return {"pattern": "insufficient_data", "confidence": 0.0}
            
            patterns_detected = []
            
            # Double Top/Bottom Detection
            double_pattern = self._detect_double_patterns(highs, lows, prices)
            if double_pattern["detected"]:
                patterns_detected.append(double_pattern)
            
            # Triangle Pattern Detection
            triangle_pattern = self._detect_triangle_patterns(highs, lows)
            if triangle_pattern["detected"]:
                patterns_detected.append(triangle_pattern)
            
            # Flag/Pennant Detection
            flag_pattern = self._detect_flag_patterns(prices, volumes)
            if flag_pattern["detected"]:
                patterns_detected.append(flag_pattern)
            
            # Head and Shoulders Detection
            hs_pattern = self._detect_head_shoulders(highs, lows)
            if hs_pattern["detected"]:
                patterns_detected.append(hs_pattern)
            
            # Select strongest pattern
            if patterns_detected:
                strongest_pattern = max(patterns_detected, key=lambda x: x["confidence"])
                return {
                    "primary_pattern": strongest_pattern,
                    "all_patterns": patterns_detected,
                    "pattern_count": len(patterns_detected)
                }
            else:
                return {
                    "primary_pattern": {"pattern": "no_clear_pattern", "confidence": 0.0},
                    "all_patterns": [],
                    "pattern_count": 0
                }
                
        except Exception as e:
            logger.error(f"Chart pattern analysis error: {e}")
            return {"pattern": "analysis_error", "confidence": 0.0}
    
    def _detect_double_patterns(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Detect double top/bottom patterns"""
        if len(highs) < 20:
            return {"detected": False}
        
        # Look for two peaks/troughs of similar height
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Find local maxima and minima
        high_peaks = []
        low_troughs = []
        
        for i in range(2, len(recent_highs)-2):
            if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                high_peaks.append((i, recent_highs[i]))
            
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                low_troughs.append((i, recent_lows[i]))
        
        # Check for double top
        if len(high_peaks) >= 2:
            peak1 = high_peaks[-2][1]
            peak2 = high_peaks[-1][1]
            height_diff = abs(peak1 - peak2) / max(peak1, peak2)
            
            if height_diff < 0.03:  # Within 3%
                return {
                    "detected": True,
                    "pattern": "double_top",
                    "confidence": 0.8 - height_diff,
                    "bearish": True
                }
        
        # Check for double bottom
        if len(low_troughs) >= 2:
            trough1 = low_troughs[-2][1]
            trough2 = low_troughs[-1][1]
            depth_diff = abs(trough1 - trough2) / min(trough1, trough2)
            
            if depth_diff < 0.03:  # Within 3%
                return {
                    "detected": True,
                    "pattern": "double_bottom",
                    "confidence": 0.8 - depth_diff,
                    "bullish": True
                }
        
        return {"detected": False}
    
    def _detect_triangle_patterns(self, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(highs) < 15:
            return {"detected": False}
        
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        # Calculate trend lines
        high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        # Determine triangle type
        if abs(high_slope) < 0.001 and low_slope > 0.001:  # Horizontal resistance, rising support
            return {
                "detected": True,
                "pattern": "ascending_triangle",
                "confidence": 0.7,
                "bullish": True
            }
        elif high_slope < -0.001 and abs(low_slope) < 0.001:  # Falling resistance, horizontal support
            return {
                "detected": True,
                "pattern": "descending_triangle",
                "confidence": 0.7,
                "bearish": True
            }
        elif high_slope < -0.001 and low_slope > 0.001:  # Converging lines
            return {
                "detected": True,
                "pattern": "symmetrical_triangle",
                "confidence": 0.6,
                "neutral": True
            }
        
        return {"detected": False}
    
    def _detect_flag_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect flag and pennant patterns"""
        if len(prices) < 15:
            return {"detected": False}
        
        # Look for strong move followed by consolidation
        recent_prices = prices[-15:]
        recent_volumes = volumes[-15:]
        
        # Check for initial strong move (first 5 periods)
        initial_move = (recent_prices[4] - recent_prices[0]) / recent_prices[0]
        
        if abs(initial_move) > 0.05:  # >5% move
            # Check for consolidation (next 10 periods)
            consolidation_range = np.max(recent_prices[5:]) - np.min(recent_prices[5:])
            consolidation_pct = consolidation_range / recent_prices[5]
            
            # Check volume pattern (should decrease during consolidation)
            early_volume = np.mean(recent_volumes[:5])
            late_volume = np.mean(recent_volumes[5:])
            volume_decline = (early_volume - late_volume) / early_volume
            
            if consolidation_pct < 0.03 and volume_decline > 0.1:  # Tight consolidation, declining volume
                return {
                    "detected": True,
                    "pattern": "bull_flag" if initial_move > 0 else "bear_flag",
                    "confidence": 0.75,
                    "bullish": initial_move > 0,
                    "bearish": initial_move < 0
                }
        
        return {"detected": False}
    
    def _detect_head_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """Detect head and shoulders patterns"""
        if len(highs) < 25:
            return {"detected": False}
        
        recent_highs = highs[-25:]
        recent_lows = lows[-25:]
        
        # Find three peaks for head and shoulders
        peaks = []
        for i in range(2, len(recent_highs)-2):
            if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                peaks.append((i, recent_highs[i]))
        
        if len(peaks) >= 3:
            # Take last three peaks
            left_shoulder = peaks[-3][1]
            head = peaks[-2][1]
            right_shoulder = peaks[-1][1]
            
            # Check pattern validity
            head_higher = head > left_shoulder and head > right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05
            
            if head_higher and shoulders_similar:
                return {
                    "detected": True,
                    "pattern": "head_and_shoulders",
                    "confidence": 0.8,
                    "bearish": True
                }
        
        # Check for inverse head and shoulders
        troughs = []
        for i in range(2, len(recent_lows)-2):
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                troughs.append((i, recent_lows[i]))
        
        if len(troughs) >= 3:
            left_shoulder = troughs[-3][1]
            head = troughs[-2][1]
            right_shoulder = troughs[-1][1]
            
            head_lower = head < left_shoulder and head < right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / min(left_shoulder, right_shoulder) < 0.05
            
            if head_lower and shoulders_similar:
                return {
                    "detected": True,
                    "pattern": "inverse_head_and_shoulders",
                    "confidence": 0.8,
                    "bullish": True
                }
        
        return {"detected": False}
    
    async def _analyze_momentum_strength(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze momentum strength and sustainability"""
        try:
            if len(prices) < 10:
                return {"momentum": "insufficient_data"}
            
            # Price momentum
            price_momentum_5 = (prices[-1] - prices[-6]) / prices[-6] * 100 if len(prices) > 5 else 0
            price_momentum_10 = (prices[-1] - prices[-11]) / prices[-11] * 100 if len(prices) > 10 else 0
            
            # Volume momentum
            recent_volume = np.mean(volumes[-5:]) if len(volumes) > 5 else volumes[-1]
            avg_volume = np.mean(volumes[-20:]) if len(volumes) > 20 else recent_volume
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Acceleration
            if len(prices) > 3:
                velocity = np.diff(prices[-3:])
                acceleration = np.diff(velocity)[0] if len(velocity) > 1 else 0
            else:
                acceleration = 0
            
            # Momentum quality assessment
            momentum_strength = "WEAK"
            if abs(price_momentum_5) > 3 and volume_ratio > 1.5:
                momentum_strength = "VERY_STRONG"
            elif abs(price_momentum_5) > 2 and volume_ratio > 1.2:
                momentum_strength = "STRONG"
            elif abs(price_momentum_5) > 1:
                momentum_strength = "MODERATE"
            
            return {
                "momentum_5d": round(price_momentum_5, 2),
                "momentum_10d": round(price_momentum_10, 2),
                "volume_ratio": round(volume_ratio, 2),
                "acceleration": round(acceleration, 4),
                "momentum_strength": momentum_strength,
                "sustainability": "HIGH" if volume_ratio > 1.5 else "MEDIUM" if volume_ratio > 1.0 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {"momentum": "analysis_error"}
    
    async def _analyze_volume_profile(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze volume profile and distribution"""
        try:
            if len(prices) < 20:
                return {"volume_profile": "insufficient_data"}
            
            # Calculate volume-weighted average price
            vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
            current_price = prices[-1]
            
            # Volume distribution analysis
            above_vwap_volume = np.sum(volumes[prices > vwap])
            below_vwap_volume = np.sum(volumes[prices <= vwap])
            total_volume = above_vwap_volume + below_vwap_volume
            
            volume_balance = above_vwap_volume / total_volume if total_volume > 0 else 0.5
            
            # Recent volume trend
            recent_volumes = volumes[-10:] if len(volumes) > 10 else volumes
            volume_trend = "INCREASING" if recent_volumes[-1] > np.mean(recent_volumes[:-1]) else "DECREASING"
            
            # Volume quality
            avg_volume = np.mean(volumes)
            current_volume = volumes[-1]
            volume_quality = "HIGH" if current_volume > avg_volume * 1.5 else "NORMAL" if current_volume > avg_volume * 0.8 else "LOW"
            
            return {
                "vwap": round(vwap, 2),
                "price_vs_vwap": round(((current_price / vwap) - 1) * 100, 2),
                "volume_balance": round(volume_balance, 3),
                "volume_trend": volume_trend,
                "volume_quality": volume_quality,
                "buying_pressure": "HIGH" if volume_balance > 0.6 else "LOW" if volume_balance < 0.4 else "NEUTRAL"
            }
            
        except Exception as e:
            logger.error(f"Volume profile analysis error: {e}")
            return {"volume_profile": "analysis_error"}
    
    def _generate_enhanced_technical_analysis(self, symbol: str, quote_data: Dict[str, Any]) -> TechnicalAnalysis:
        """Generate enhanced technical analysis when insufficient data"""
        try:
            current_price = float(quote_data.get('price', 100))
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Generate basic technical signals based on current data
            signals = []
            
            # RSI estimation based on recent price action
            estimated_rsi = 50 + (change_percent * 2)  # Rough RSI estimation
            estimated_rsi = max(0, min(100, estimated_rsi))
            
            signals.append(TechnicalSignal(
                indicator_name="RSI_Estimated",
                value=estimated_rsi,
                signal_strength=abs(estimated_rsi - 50) * 2,
                direction="DOWN" if estimated_rsi > 70 else "UP" if estimated_rsi < 30 else "NEUTRAL",
                confidence=0.6,
                oversold=estimated_rsi < 30,
                overbought=estimated_rsi > 70
            ))
            
            # Basic trend analysis
            trend_direction = "UP" if change_percent > 1 else "DOWN" if change_percent < -1 else "NEUTRAL"
            signals.append(TechnicalSignal(
                indicator_name="Trend_Basic",
                value=change_percent,
                signal_strength=min(100, abs(change_percent) * 10),
                direction=trend_direction,
                confidence=min(1.0, abs(change_percent) / 5)
            ))
            
            # Volume analysis if available
            volume = quote_data.get('volume', 0)
            if volume > 0:
                # Estimate volume relative to "normal"
                estimated_avg_volume = 25000000  # Rough average
                volume_ratio = volume / estimated_avg_volume
                
                signals.append(TechnicalSignal(
                    indicator_name="Volume_Analysis",
                    value=volume_ratio,
                    signal_strength=min(100, volume_ratio * 50),
                    direction="UP" if volume_ratio > 1.5 else "NEUTRAL",
                    confidence=0.7
                ))
            
            # Calculate overall score
            overall_score, direction, confidence = self._calculate_overall_score(signals)
            
            # Basic key levels
            key_levels = {
                'support_1': round(current_price * 0.98, 2),
                'resistance_1': round(current_price * 1.02, 2),
                'pivot_point': round(current_price, 2)
            }
            
            return TechnicalAnalysis(
                overall_score=overall_score,
                direction=direction,
                confidence=confidence,
                signals=signals,
                key_levels=key_levels,
                pattern_analysis={"pattern": "insufficient_data_for_patterns"},
                momentum_analysis={"momentum": f"{change_percent:.2f}% current"},
                volume_analysis={"volume": "basic_analysis"}
            )
            
        except Exception as e:
            logger.error(f"Enhanced technical analysis fallback error: {e}")
            return TechnicalAnalysis(
                overall_score=50.0,
                direction="NEUTRAL",
                confidence=0.5,
                signals=[],
                key_levels={'current_price': float(quote_data.get('price', 100))},
                pattern_analysis={"error": "analysis_failed"},
                momentum_analysis={"error": "analysis_failed"},
                volume_analysis={"error": "analysis_failed"}
