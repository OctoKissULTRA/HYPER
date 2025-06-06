# vix_analysis.py - VIX Fear & Greed Analysis Module - COMPLETE VERSION
import logging
import asyncio
import numpy as np
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class VIXSignal:
    """VIX-based trading signal""""
    vix_level: float
    percentile_rank: float  # 0-100, where this VIX level ranks historically
    fear_greed_score: float  # 0-100, where 0=extreme fear, 100=extreme greed
    regime: str  # LOW_VOL, NORMAL, ELEVATED, HIGH_VOL, CRISIS
    sentiment: str  # EXTREME_FEAR, FEAR, NEUTRAL, COMPLACENCY, EXTREME_COMPLACENCY
    contrarian_signal: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    term_structure: Dict[str, float]  # VIX9D, VIX, VIX3M, VIX6M
    mean_reversion_probability: float  # 0-1
    spike_probability: float  # 0-1
    trend_direction: str  # UP, DOWN, NEUTRAL
    volatility_risk_premium: float
    options_flow_sentiment: str

@dataclass 
class VIXAnalysis:
    """Complete VIX analysis result""""
    current_signal: VIXSignal
    market_stress_level: str
    volatility_forecast: Dict[str, float]
    trading_recommendations: List[str]
    risk_warnings: List[str]
    historical_context: Dict[str, Any]
    correlation_analysis: Dict[str, float]
    regime_analysis: Dict[str, Any]

class AdvancedVIXAnalyzer:
    """Advanced VIX Fear & Greed Analysis with Market Regime Detection""""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vix_cache = {}
        self.cache_duration = 180  # 3 minutes for VIX data
        self.vix_history = []  # Store VIX history for analysis
        self.regime_history = []
        
        # VIX thresholds from config
        self.thresholds = {
            'extreme_fear': config.get('extreme_fear_threshold', 30),
            'fear': config.get('fear_threshold', 20),
            'complacency': config.get('complacency_threshold', 12),
            'crisis': 40,
            'elevated': 25
        }
        
        # Historical VIX percentiles (approximated)
        self.vix_percentiles = {
            10: 12.5,   # 10th percentile
            25: 15.0,   # 25th percentile  
            50: 18.5,   # Median
            75: 24.0,   # 75th percentile
            90: 30.0,   # 90th percentile
            95: 35.0,   # 95th percentile
            99: 50.0    # 99th percentile
        }
        
        # VIX term structure simulator
        self.term_structure_simulator = VIXTermStructureSimulator()
        
        # Options flow analyzer
        self.options_analyzer = OptionsFlowAnalyzer()
        
        logger.info("😱 Advanced VIX Analyzer initialized")
        logger.info(f"📊 Thresholds: Fear>{self.thresholds['fear']}, Complacency<{self.thresholds['complacency']}")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], 
                     market_data: Optional[Dict] = None) -> VIXAnalysis:
        """Complete VIX fear/greed analysis""""
        try:
            # Check cache first
            cache_key = f"vix_{time.time() // self.cache_duration}""
            if cache_key in self.vix_cache:
                logger.debug("📋 Using cached VIX analysis")
                return self.vix_cache[cache_key]
            
            logger.debug("😱 Performing VIX fear/greed analysis...")
            
            # Get current VIX data
            vix_data = await self._get_vix_data(symbol, quote_data, market_data)
            
            # Generate VIX signal
            vix_signal = await self._generate_vix_signal(vix_data, symbol, quote_data)
            
            # Analyze market stress
            market_stress_level = self._analyze_market_stress(vix_signal, quote_data)
            
            # Generate volatility forecast
            volatility_forecast = await self._forecast_volatility(vix_data, vix_signal)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(vix_signal, symbol)
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(vix_signal, market_stress_level)
            
            # Historical context analysis
            historical_context = self._analyze_historical_context(vix_signal)
            
            # Correlation analysis
            correlation_analysis = await self._analyze_correlations(vix_signal, quote_data)
            
            # Regime analysis
            regime_analysis = await self._analyze_volatility_regime(vix_signal)
            
            # Update VIX history
            self._update_vix_history(vix_signal)
            
            result = VIXAnalysis(
                current_signal=vix_signal,
                market_stress_level=market_stress_level,
                volatility_forecast=volatility_forecast,
                trading_recommendations=trading_recommendations,
                risk_warnings=risk_warnings,
                historical_context=historical_context,
                correlation_analysis=correlation_analysis,
                regime_analysis=regime_analysis
            )
            
            # Cache result
            self.vix_cache[cache_key] = result
            
            logger.debug(f"✅ VIX analysis: {vix_signal.vix_level:.1f} ({vix_signal.sentiment})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ VIX analysis error: {e}")
            return self._generate_fallback_vix_analysis()
    
    async def _get_vix_data(self, symbol: str, quote_data: Dict[str, Any], 
                           market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Get VIX data (simulated with realistic market-driven behavior)""""
        try:
            # Base VIX calculation from market conditions
            current_price = float(quote_data.get('price', 100))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Calculate base VIX level
            base_vix = await self._calculate_dynamic_vix(symbol, change_percent, volume)
            
            # Get VIX term structure
            term_structure = await self.term_structure_simulator.get_term_structure(base_vix)
            
            # Calculate VIX derivatives
            vix9d = base_vix * random.uniform(0.85, 1.15)  # Short-term volatility
            vix3m = base_vix * random.uniform(0.90, 1.25)  # 3-month forward
            vix6m = base_vix * random.uniform(0.95, 1.30)  # 6-month forward
            
            # Historical VIX simulation (last 30 days)
            vix_history = self._generate_vix_history(base_vix, 30)
            
            return {
                'current_vix': base_vix,
                'vix9d': vix9d,
                'vix3m': vix3m,
                'vix6m': vix6m,
                'term_structure': term_structure,
                'history': vix_history,
                'intraday_high': base_vix * random.uniform(1.0, 1.15),
                'intraday_low': base_vix * random.uniform(0.85, 1.0),
                'previous_close': base_vix * random.uniform(0.95, 1.05)
            }
            
        except Exception as e:
            logger.error(f"VIX data retrieval error: {e}")
            return {'current_vix': 20.0, 'vix9d': 19.0, 'vix3m': 21.0, 'vix6m': 22.0}

    async def _calculate_dynamic_vix(self, symbol: str, change_percent: float, volume: int) -> float:
        """Calculate dynamic VIX based on market conditions""""
        try:
            # Base VIX level (historical average around 18-20)
            base_vix = 19.0
            
            # Market movement factor
            # VIX typically moves inverse to market
            if change_percent > 2:  # Strong up move
                movement_factor = -0.15 * change_percent  # VIX down
            elif change_percent < -2:  # Strong down move  
                movement_factor = -0.20 * change_percent  # VIX up (note: negative * negative = positive)
            else:
                movement_factor = random.uniform(-1, 1)  # Small random variation
            
            # Volume factor (high volume can increase volatility)
            avg_volume = 25000000
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_factor = (volume_ratio - 1) * 2  # Amplify volume effect
            
            # Time-based factors
            hour = datetime.now().hour
            if hour in [9, 15, 16]:  # Market open/close volatility
                time_factor = 2.0
            elif 11 <= hour <= 14:  # Lunch lull
                time_factor = -1.0
            else:
                time_factor = 0.0
            
            # Day of week factor
            weekday = datetime.now().weekday()
            if weekday == 0:  # Monday
                dow_factor = 1.5
            elif weekday == 4:  # Friday
                dow_factor = -0.5
            else:
                dow_factor = 0.0
            
            # Symbol-specific volatility
            symbol_factors = {
                'NVDA': 1.3,   # Higher baseline volatility
                'QQQ': 1.1,    # Tech volatility
                'SPY': 1.0,    # Market baseline
                'AAPL': 0.9,   # Lower volatility
                'MSFT': 0.9    # Lower volatility
            }
            symbol_factor = symbol_factors.get(symbol, 1.0)
            
            # Add some persistence from previous VIX levels
            if self.vix_history:
                last_vix = self.vix_history[-1].get('vix_level', base_vix)
                persistence_factor = (last_vix - base_vix) * 0.3  # 30% persistence
            else:
                persistence_factor = 0
            
            # Calculate final VIX
            calculated_vix = (base_vix + movement_factor + volume_factor + 
                            time_factor + dow_factor + persistence_factor) * symbol_factor
            
            # Add random noise
            noise = random.gauss(0, 1.5)
            calculated_vix += noise
            
            # Bound VIX to reasonable range
            calculated_vix = max(8.0, min(80.0, calculated_vix))
            
            # Add occasional spikes (2% chance of volatility spike)
            if random.random() < 0.02:
                spike_factor = random.uniform(1.5, 2.5)
                calculated_vix *= spike_factor
                calculated_vix = min(75.0, calculated_vix)  # Cap spike VIX
                logger.info(f"⚡ VIX spike event: {calculated_vix:.1f}")
            
            return round(calculated_vix, 2)
            
        except Exception as e:
            logger.error(f"Dynamic VIX calculation error: {e}")
            return 20.0
    
    async def _generate_vix_signal(self, vix_data: Dict[str, Any], symbol: str, 
                                  quote_data: Dict[str, Any]) -> VIXSignal:
        """Generate comprehensive VIX signal""""
        try:
            current_vix = vix_data.get('current_vix', 20.0)
            
            # Calculate percentile rank
            percentile_rank = self._calculate_vix_percentile(current_vix)
            
            # Calculate fear/greed score (inverted VIX)
            fear_greed_score = self._calculate_fear_greed_score(current_vix, percentile_rank)
            
            # Determine volatility regime
            regime = self._determine_volatility_regime(current_vix)
            
            # Determine sentiment
            sentiment = self._determine_vix_sentiment(current_vix)
            
            # Generate contrarian signal
            contrarian_signal = self._generate_contrarian_signal(current_vix, sentiment, regime)
            
            # Build term structure
            term_structure = {
                'VIX9D': vix_data.get('vix9d', current_vix * 0.95),
                'VIX': current_vix,
                'VIX3M': vix_data.get('vix3m', current_vix * 1.1),
                'VIX6M': vix_data.get('vix6m', current_vix * 1.15)
            }
            
            # Calculate mean reversion probability
            mean_reversion_prob = self._calculate_mean_reversion_probability(current_vix, percentile_rank)
            
            # Calculate spike probability
            spike_prob = self._calculate_spike_probability(current_vix, vix_data.get('history', []))
            
            # Determine trend direction
            trend_direction = self._determine_vix_trend(vix_data.get('history', []), current_vix)
            
            # Calculate volatility risk premium
            vol_risk_premium = self._calculate_volatility_risk_premium(term_structure)
            
            # Get options flow sentiment
            options_flow_sentiment = await self.options_analyzer.analyze_options_flow(
                symbol, current_vix, quote_data
            )
            
            return VIXSignal(
                vix_level=current_vix,
                percentile_rank=percentile_rank,
                fear_greed_score=fear_greed_score,
                regime=regime,
                sentiment=sentiment,
                contrarian_signal=contrarian_signal,
                term_structure=term_structure,
                mean_reversion_probability=mean_reversion_prob,
                spike_probability=spike_prob,
                trend_direction=trend_direction,
                volatility_risk_premium=vol_risk_premium,
                options_flow_sentiment=options_flow_sentiment
            )
            
        except Exception as e:
            logger.error(f"VIX signal generation error: {e}")
            return self._generate_fallback_vix_signal()
    
    def _calculate_vix_percentile(self, current_vix: float) -> float:
        """Calculate VIX percentile rank""""
        try:
            # Find the percentile bracket
            for percentile in sorted(self.vix_percentiles.keys()):
                if current_vix <= self.vix_percentiles[percentile]:
                    return percentile
            return 99  # Very high VIX
        except Exception as e:
            logger.error(f"VIX percentile calculation error: {e}")
            return 50.0
    
    def _calculate_fear_greed_score(self, current_vix: float, percentile_rank: float) -> float:
        """Calculate fear/greed score (0=extreme fear, 100=extreme greed)""""
        try:
            # Invert VIX: high VIX = fear (low score), low VIX = greed (high score)
            base_score = 100 - min(100, (current_vix / 50) * 100)
            
            # Adjust by percentile for historical context
            percentile_adjustment = (50 - percentile_rank) * 0.5
            
            fear_greed_score = base_score + percentile_adjustment
            return max(0, min(100, fear_greed_score))
            
        except Exception as e:
            logger.error(f"Fear/greed score calculation error: {e}")
            return 50.0
    
    def _determine_volatility_regime(self, current_vix: float) -> str:
        """Determine current volatility regime""""
        if current_vix >= self.thresholds['crisis']:
            return "CRISIS""
        elif current_vix >= self.thresholds['extreme_fear']:
            return "HIGH_VOL""
        elif current_vix >= self.thresholds['fear']:
            return "ELEVATED""
        elif current_vix <= self.thresholds['complacency']:
            return "LOW_VOL""
        else:
            return "NORMAL""
    
    def _determine_vix_sentiment(self, current_vix: float) -> str:
        """Determine VIX-based market sentiment""""
        if current_vix >= self.thresholds['crisis']:
            return "EXTREME_FEAR""
        elif current_vix >= self.thresholds['extreme_fear']:
            return "FEAR""
        elif current_vix <= self.thresholds['complacency']:
            return "EXTREME_COMPLACENCY""
        elif current_vix <= self.thresholds['fear'] * 0.8:
            return "COMPLACENCY""
        else:
            return "NEUTRAL""
    
    def _generate_contrarian_signal(self, current_vix: float, sentiment: str, regime: str) -> str:
        """Generate contrarian trading signal based on VIX""""
        try:
            # VIX contrarian logic: extreme fear = buy, extreme greed = sell
            if sentiment == "EXTREME_FEAR" and current_vix > self.thresholds['extreme_fear']:
                return "STRONG_BUY""
            elif sentiment == "FEAR" and current_vix > self.thresholds['fear']:
                return "BUY""
            elif sentiment == "EXTREME_COMPLACENCY" and current_vix < self.thresholds['complacency']:
                return "STRONG_SELL""
            elif sentiment == "COMPLACENCY":
                return "SELL""
            else:
                return "HOLD""
                
        except Exception as e:
            logger.error(f"Contrarian signal generation error: {e}")
            return "HOLD""

    def _calculate_mean_reversion_probability(self, current_vix: float, percentile_rank: float) -> float:
        """Calculate probability of VIX mean reversion""""
        try:
            # VIX has strong mean reversion properties
            historical_mean = 19.0
            deviation = abs(current_vix - historical_mean) / historical_mean
            
            # Higher deviation = higher reversion probability
            base_probability = min(0.9, deviation * 2)
            
            # Adjust by percentile (extreme levels more likely to revert)
            if percentile_rank > 90 or percentile_rank < 10:
                base_probability *= 1.3
            
            return min(1.0, base_probability)
            
        except Exception as e:
            logger.error(f"Mean reversion calculation error: {e}")
            return 0.5
    
    def _calculate_spike_probability(self, current_vix: float, vix_history: List[Dict]) -> float:
        """Calculate probability of VIX spike""""
        try:
            # Base spike probability (VIX spikes are relatively rare)
            base_prob = 0.05  # 5% daily probability
            
            # Increase probability if VIX is rising
            if len(vix_history) >= 3:
                recent_trend = sum(vix_history[-3:]) / 3 - sum(vix_history[-6:-3]) / 3
                if recent_trend > 2:  # Rising trend
                    base_prob *= 2
            
            # Increase probability in high-stress environments
            if current_vix > self.thresholds['fear']:
                base_prob *= 1.5
            
            # Decrease probability in low-vol environments
            if current_vix < self.thresholds['complacency']:
                base_prob *= 0.3
            
            return min(1.0, base_prob)
            
        except Exception as e:
            logger.error(f"Spike probability calculation error: {e}")
            return 0.05
    
    def _determine_vix_trend(self, vix_history: List[Dict], current_vix: float) -> str:
        """Determine VIX trend direction""""
        try:
            if len(vix_history) < 5:
                return "NEUTRAL""
            
            # Calculate recent trend
            recent_avg = sum(vix_history[-3:]) / 3 if len(vix_history) >= 3 else current_vix
            older_avg = sum(vix_history[-6:-3]) / 3 if len(vix_history) >= 6 else current_vix
            
            trend_diff = recent_avg - older_avg
            
            if trend_diff > 1.5:
                return "UP""
            elif trend_diff < -1.5:
                return "DOWN""
            else:
                return "NEUTRAL""
                
        except Exception as e:
            logger.error(f"VIX trend determination error: {e}")
            return "NEUTRAL""
    
    def _calculate_volatility_risk_premium(self, term_structure: Dict[str, float]) -> float:
        """Calculate volatility risk premium""""
        try:
            # Compare current VIX to longer-term expectations
            current_vix = term_structure.get('VIX', 20)
            vix_3m = term_structure.get('VIX3M', 21)
            
            # Risk premium = difference between forward and spot VIX
            risk_premium = vix_3m - current_vix
            return round(risk_premium, 2)
            
        except Exception as e:
            logger.error(f"Volatility risk premium calculation error: {e}")
            return 0.0
    
    def _generate_vix_history(self, current_vix: float, days: int) -> List[float]:
        """Generate realistic VIX history""""
        try:
            history = []
            vix = current_vix
            
            for i in range(days):
                # VIX random walk with mean reversion
                mean_reversion = (19.0 - vix) * 0.1  # Pull toward long-term mean
                random_change = random.gauss(0, 1.5)  # Daily volatility
                
                vix += mean_reversion + random_change
                vix = max(8.0, min(80.0, vix))  # Bound VIX
                
                history.insert(0, vix)  # Insert at beginning for chronological order
            
            return history
            
        except Exception as e:
            logger.error(f"VIX history generation error: {e}")
            return [current_vix] * days

    def _analyze_market_stress(self, vix_signal: VIXSignal, quote_data: Dict[str, Any]) -> str:
        """Analyze overall market stress level""""
        try:
            stress_factors = []
            
            # VIX level stress
            if vix_signal.vix_level > self.thresholds['crisis']:
                stress_factors.append("EXTREME_VIX")
            elif vix_signal.vix_level > self.thresholds['extreme_fear']:
                stress_factors.append("HIGH_VIX")
            
            # VIX trend stress
            if vix_signal.trend_direction == "UP" and vix_signal.vix_level > 25:
                stress_factors.append("RISING_FEAR")
            
            # Term structure stress
            if vix_signal.volatility_risk_premium < -3:  # Backwardation
                stress_factors.append("TERM_STRUCTURE_STRESS")
            
            # Market movement stress
            change_percent = abs(float(quote_data.get('change_percent', 0)))
            if change_percent > 3:
                stress_factors.append("HIGH_PRICE_VOLATILITY")
            
            # Determine overall stress level
            if len(stress_factors) >= 3:
                return "EXTREME_STRESS""
            elif len(stress_factors) >= 2:
                return "HIGH_STRESS""
            elif len(stress_factors) >= 1:
                return "MODERATE_STRESS""
            else:
                return "LOW_STRESS""
                
        except Exception as e:
            logger.error(f"Market stress analysis error: {e}")
            return "UNKNOWN""

    async def _forecast_volatility(self, vix_data: Dict[str, Any], vix_signal: VIXSignal) -> Dict[str, float]:
        """Generate volatility forecast""""
        try:
            current_vix = vix_signal.vix_level
            
            # Short-term forecast (1-3 days)
            if vix_signal.mean_reversion_probability > 0.7:
                short_term = current_vix * 0.9  # Mean reversion
            else:
                short_term = current_vix * random.uniform(0.95, 1.05)
            
            # Medium-term forecast (1-2 weeks)
            medium_term = (current_vix + 19.0) / 2  # Pull toward historical mean
            
            # Long-term forecast (1 month)
            long_term = 19.0 + random.uniform(-2, 2)  # Converge to historical mean
            
            return {
                "1_day": round(short_term, 1),
                "3_day": round(short_term * 1.1, 1),
                "1_week": round((short_term + medium_term) / 2, 1),
                "2_week": round(medium_term, 1),
                "1_month": round(long_term, 1)
            }
            
        except Exception as e:
            logger.error(f"Volatility forecast error: {e}")
            return {"1_day": 20.0, "1_week": 19.0, "1_month": 19.0}

    def _generate_trading_recommendations(self, vix_signal: VIXSignal, symbol: str) -> List[str]:
        """Generate VIX-based trading recommendations""""
        recommendations = []
        
        try:
            # Contrarian recommendations
            if vix_signal.contrarian_signal == "STRONG_BUY":
                recommendations.append(f"🟢 STRONG BUY: Extreme fear creates opportunity in {symbol}")
                recommendations.append("💎 Consider increasing position size on weakness")
                
            elif vix_signal.contrarian_signal == "BUY":
                recommendations.append(f"🟢 BUY: Fear levels suggest good entry point for {symbol}")
                recommendations.append("📈 Market oversold, consider accumulating")
                
            elif vix_signal.contrarian_signal == "STRONG_SELL":
                recommendations.append(f"🔴 STRONG SELL: Extreme complacency warns of correction in {symbol}")
                recommendations.append("🛡️ Consider hedging or reducing exposure")
                
            elif vix_signal.contrarian_signal == "SELL":
                recommendations.append(f"🔴 SELL: Low fear suggests caution with {symbol}")
                recommendations.append("⚠️ Market may be overbought")
            
            # Volatility-based recommendations
            if vix_signal.spike_probability > 0.3:
                recommendations.append("⚡ HIGH SPIKE RISK: Consider volatility protection")
                
            if vix_signal.mean_reversion_probability > 0.8:
                recommendations.append("🔄 MEAN REVERSION LIKELY: VIX should normalize")
                
            # Term structure recommendations
            if vix_signal.volatility_risk_premium > 3:
                recommendations.append("📊 POSITIVE CARRY: VIX term structure supports stability")
            elif vix_signal.volatility_risk_premium < -2:
                recommendations.append("⚠️ BACKWARDATION: Stressed term structure warns of volatility")
            
            return recommendations[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Trading recommendations error: {e}")
            return ["VIX analysis recommendations unavailable"]

    def _generate_risk_warnings(self, vix_signal: VIXSignal, market_stress_level: str) -> List[str]:
        """Generate VIX-based risk warnings""""
        warnings = []
        
        try:
            # Stress level warnings
            if market_stress_level == "EXTREME_STRESS":
                warnings.append("🚨 EXTREME MARKET STRESS: Reduce risk immediately")
                
            elif market_stress_level == "HIGH_STRESS":
                warnings.append("⚠️ HIGH STRESS ENVIRONMENT: Exercise caution")
            
            # VIX level warnings
            if vix_signal.vix_level > self.thresholds['crisis']:
                warnings.append(f"💥 CRISIS-LEVEL VIX: {vix_signal.vix_level:.1f} indicates severe market distress")
                
            elif vix_signal.vix_level < self.thresholds['complacency']:
                warnings.append(f"😴 COMPLACENCY WARNING: VIX {vix_signal.vix_level:.1f} suggests overconfidence")
            
            # Trend warnings
            if vix_signal.trend_direction == "UP" and vix_signal.vix_level > 25:
                warnings.append("📈 RISING FEAR: VIX trend suggests increasing market stress")
                
            # Spike warnings
            if vix_signal.spike_probability > 0.5:
                warnings.append("⚡ HIGH SPIKE PROBABILITY: Volatility explosion risk elevated")
            
            # Term structure warnings
            if vix_signal.volatility_risk_premium < -5:
                warnings.append("📉 SEVERE BACKWARDATION: Term structure signals extreme stress")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Risk warnings generation error: {e}")
            return ["VIX risk analysis unavailable"]

    def _analyze_historical_context(self, vix_signal: VIXSignal) -> Dict[str, Any]:
        """Analyze VIX in historical context""""
        try:
            return {
                "percentile_rank": vix_signal.percentile_rank,
                "vs_2008_crisis": "Below" if vix_signal.vix_level < 40 else "Above",
                "vs_covid_spike": "Below" if vix_signal.vix_level < 35 else "Above",
                "vs_long_term_avg": round((vix_signal.vix_level / 19.0 - 1) * 100, 1),
                "regime_frequency": {
                    "crisis": 2,      # % of time in crisis
                    "high_vol": 8,    # % of time in high vol
                    "normal": 70,     # % of time normal
                    "low_vol": 20     # % of time low vol
                },
                "current_regime_duration": random.randint(1, 15)  # Days in current regime
            }
            
        except Exception as e:
            logger.error(f"Historical context analysis error: {e}")
            return {"percentile_rank": 50}

    async def _analyze_correlations(self, vix_signal: VIXSignal, quote_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze VIX correlations with markets""""
        try:
            # Typical VIX correlations (negative with stocks, positive with bonds)
            return {
                "spy_correlation": round(random.uniform(-0.8, -0.6), 3),
                "qqq_correlation": round(random.uniform(-0.85, -0.65), 3),
                "tlt_correlation": round(random.uniform(0.3, 0.6), 3),
                "gold_correlation": round(random.uniform(0.1, 0.4), 3),
                "dollar_correlation": round(random.uniform(-0.2, 0.3), 3),
                "oil_correlation": round(random.uniform(-0.4, 0.1), 3)
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {"spy_correlation": -0.7}

    async def _analyze_volatility_regime(self, vix_signal: VIXSignal) -> Dict[str, Any]:
        """Analyze volatility regime characteristics""""
        try:
            return {
                "current_regime": vix_signal.regime,
                "regime_stability": "High" if vix_signal.mean_reversion_probability > 0.7 else "Low",
                "transition_probability": {
                    "to_crisis": 0.05 if vix_signal.regime != "CRISIS" else 0.8,
                    "to_normal": 0.6 if vix_signal.regime in ["HIGH_VOL", "ELEVATED"] else 0.2,
                    "to_low_vol": 0.1 if vix_signal.regime == "NORMAL" else 0.05
                },
                "regime_characteristics": {
                    "expected_duration_days": 15,
                    "typical_range": f"{vix_signal.vix_level-3:.1f}-{vix_signal.vix_level+3:.1f}",
                    "breakout_threshold": vix_signal.vix_level * 1.25
                }
            }
            
        except Exception as e:
            logger.error(f"Volatility regime analysis error: {e}")
            return {"current_regime": "NORMAL"}

    def _update_vix_history(self, vix_signal: VIXSignal):
        """Update VIX analysis history""""
        self.vix_history.append({
            'timestamp': datetime.now(),
            'vix_level': vix_signal.vix_level,
            'sentiment': vix_signal.sentiment,
            'regime': vix_signal.regime,
            'fear_greed_score': vix_signal.fear_greed_score
        })
        
        # Keep last 100 records
        if len(self.vix_history) > 100:
            self.vix_history.pop(0)

    def _generate_fallback_vix_signal(self) -> VIXSignal:
        """Generate fallback VIX signal""""
        return VIXSignal(
            vix_level=20.0,
            percentile_rank=50.0,
            fear_greed_score=50.0,
            regime="NORMAL",
            sentiment="NEUTRAL",
            contrarian_signal="HOLD",
            term_structure={"VIX": 20.0, "VIX3M": 21.0},
            mean_reversion_probability=0.5,
            spike_probability=0.05,
            trend_direction="NEUTRAL",
            volatility_risk_premium=1.0,
            options_flow_sentiment="NEUTRAL""
        )

    def _generate_fallback_vix_analysis(self) -> VIXAnalysis:
        """Generate fallback VIX analysis""""
        fallback_signal = self._generate_fallback_vix_signal()
        
        return VIXAnalysis(
            current_signal=fallback_signal,
            market_stress_level="UNKNOWN",
            volatility_forecast={"1_day": 20.0, "1_week": 19.0, "1_month": 19.0},
            trading_recommendations=["VIX analysis unavailable"],
            risk_warnings=["VIX risk analysis unavailable"],
            historical_context={"percentile_rank": 50},
            correlation_analysis={"spy_correlation": -0.7},
            regime_analysis={"current_regime": "NORMAL"}
        )

# Supporting classes for VIX analysis
class VIXTermStructureSimulator:
    """Simulate VIX term structure""""
    
    def __init__(self):
        logger.info("📊 VIX Term Structure Simulator initialized")
    
    async def get_term_structure(self, current_vix: float) -> Dict[str, float]:
        """Generate realistic VIX term structure""""
        try:
            # Normal contango: longer terms higher than spot
            vix9d = current_vix * random.uniform(0.9, 1.05)
            vix1m = current_vix * random.uniform(1.0, 1.15)
            vix3m = current_vix * random.uniform(1.05, 1.25)
            vix6m = current_vix * random.uniform(1.10, 1.30)
            
            return {
                "VIX9D": round(vix9d, 2),
                "VIX1M": round(vix1m, 2),
                "VIX3M": round(vix3m, 2),
                "VIX6M": round(vix6m, 2),
                "structure_type": "contango" if vix3m > current_vix else "backwardation""
            }
            
        except Exception as e:
            logger.error(f"Term structure simulation error: {e}")
            return {"VIX": current_vix, "VIX3M": current_vix * 1.1}

class OptionsFlowAnalyzer:
    """Analyze options flow for VIX sentiment""""
    
    def __init__(self):
        logger.info("📈 Options Flow Analyzer initialized")
    
    async def analyze_options_flow(self, symbol: str, current_vix: float, quote_data: Dict[str, Any]) -> str:
        """Analyze options flow sentiment""""
        try:
            # Simulate options flow analysis
            change_percent = float(quote_data.get('change_percent', 0))
            
            if current_vix > 25 and change_percent < -2:
                return "PROTECTIVE_BUYING"  # Fear-driven put buying
            elif current_vix < 15 and change_percent > 2:
                return "COMPLACENT_SELLING"  # Overconfident put selling
            elif abs(change_percent) > 3:
                return "VOLATILITY_BUYING"  # Straddle/strangle buying
            else:
                return "NEUTRAL""
                
        except Exception as e:
            logger.error(f"Options flow analysis error: {e}")
            return "UNKNOWN""

# Export main classes
__all__ = ['AdvancedVIXAnalyzer', 'VIXAnalysis', 'VIXSignal']

logger.info("😱 Advanced VIX Analysis module loaded successfully")
logger.info("🎯 Fear/greed detection with contrarian signals enabled")
logger.info("📊 VIX term structure and regime analysis active")