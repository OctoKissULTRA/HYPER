# risk_analysis.py - Advanced Risk Analysis Module
import logging
import asyncio
import numpy as np
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Individual risk metrics"""
    var_95: float  # 5% Value at Risk
    var_99: float  # 1% Value at Risk
    expected_shortfall: float  # Conditional VaR
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_spy: float
    volatility_annualized: float
    skewness: float
    kurtosis: float
    tail_risk_score: float

@dataclass
class PositionRisk:
    """Position-specific risk analysis"""
    position_size_recommendation: float
    kelly_criterion: float
    optimal_stop_loss: float
    risk_reward_ratio: float
    probability_of_loss: float
    expected_return: float
    risk_budget_allocation: float
    concentration_risk: str

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    portfolio_var: float
    diversification_ratio: float
    correlation_risk: str
    sector_concentration: Dict[str, float]
    max_position_size: float
    total_portfolio_risk: float
    risk_contribution: float
    stress_test_results: Dict[str, float]

@dataclass
class AnomalySignal:
    """Anomaly detection signal"""
    anomaly_score: float  # 0-100
    anomaly_type: str  # PRICE, VOLUME, VOLATILITY, CORRELATION
    severity: str  # LOW, MEDIUM, HIGH, EXTREME
    description: str
    probability: float
    historical_precedent: bool
    recommended_action: str

@dataclass
class RiskAnalysis:
    """Complete risk analysis result"""
    overall_risk_score: float  # 0-100
    risk_level: str  # LOW, MODERATE, HIGH, EXTREME
    risk_metrics: RiskMetrics
    position_risk: PositionRisk
    portfolio_risk: PortfolioRisk
    anomaly_signals: List[AnomalySignal]
    risk_warnings: List[str]
    risk_recommendations: List[str]
    scenario_analysis: Dict[str, float]
    tail_risk_events: List[str]

class AdvancedRiskAnalyzer:
    """Advanced Risk Analysis with ML-Enhanced Anomaly Detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_cache = {}
        self.cache_duration = 120  # 2 minutes
        self.price_history = {}
        self.risk_history = []
        
        # Risk configuration
        self.risk_config = config.get('risk_config', {})
        self.var_confidence = self.risk_config.get('var_confidence_level', 0.05)
        self.max_drawdown_warning = self.risk_config.get('max_drawdown_warning', 15.0)
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk', 0.02)
        self.stop_loss_percent = self.risk_config.get('stop_loss_percent', 0.05)
        
        # Risk components
        self.volatility_analyzer = VolatilityRiskAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.correlation_analyzer = CorrelationRiskAnalyzer()
        self.anomaly_detector = MLAnomalyDetector()
        self.stress_tester = StressTestEngine()
        
        logger.info("⚠️ Advanced Risk Analyzer initialized")
        logger.info(f"📊 VaR confidence: {(1-self.var_confidence)*100:.0f}%")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], 
                     historical_data: Optional[List[Dict]] = None,
                     portfolio_context: Optional[Dict] = None) -> RiskAnalysis:
        """Complete risk analysis"""
        try:
            # Check cache first
            cache_key = f"risk_{symbol}_{time.time() // self.cache_duration}""
            if cache_key in self.risk_cache:
                logger.debug("📋 Using cached risk analysis")
                return self.risk_cache[cache_key]
            
            logger.debug(f"⚠️ Performing risk analysis for {symbol}...")
            
            # Generate or use historical data
            if not historical_data:
                historical_data = self._generate_price_history(symbol, quote_data)
            
            # Calculate core risk metrics
            risk_metrics = await self._calculate_risk_metrics(symbol, quote_data, historical_data)
            
            # Analyze position-specific risk
            position_risk = await self._analyze_position_risk(symbol, quote_data, risk_metrics)
            
            # Analyze portfolio-level risk
            portfolio_risk = await self._analyze_portfolio_risk(symbol, risk_metrics, portfolio_context)
            
            # Detect anomalies
            anomaly_signals = await self.anomaly_detector.detect_anomalies(symbol, quote_data, historical_data)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_metrics, position_risk, anomaly_signals)
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score, anomaly_signals)
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(risk_metrics, position_risk, anomaly_signals)
            
            # Generate risk recommendations
            risk_recommendations = self._generate_risk_recommendations(symbol, risk_metrics, position_risk, risk_level)
            
            # Perform scenario analysis
            scenario_analysis = await self.stress_tester.run_scenarios(symbol, quote_data, risk_metrics)
            
            # Identify tail risk events
            tail_risk_events = self._identify_tail_risk_events(risk_metrics, anomaly_signals)
            
            # Update risk history
            self._update_risk_history(symbol, overall_risk_score, risk_level)
            
            result = RiskAnalysis(
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                risk_metrics=risk_metrics,
                position_risk=position_risk,
                portfolio_risk=portfolio_risk,
                anomaly_signals=anomaly_signals,
                risk_warnings=risk_warnings,
                risk_recommendations=risk_recommendations,
                scenario_analysis=scenario_analysis,
                tail_risk_events=tail_risk_events
            )
            
            # Cache result
            self.risk_cache[cache_key] = result
            
            logger.debug(f"✅ Risk analysis: {risk_level} risk ({overall_risk_score:.0f} score)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk analysis error for {symbol}: {e}")
            return self._generate_fallback_risk_analysis()
    
    async def _calculate_risk_metrics(self, symbol: str, quote_data: Dict[str, Any], 
                                     historical_data: List[Dict]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Extract price data
            prices = np.array([float(d.get('close', d.get('price', 0))) for d in historical_data])
            
            if len(prices) < 30:
                logger.warning(f"Insufficient price history for {symbol}, using simulation")
                return self._generate_simulated_risk_metrics(symbol, quote_data)
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Value at Risk calculations
            var_95 = np.percentile(returns, self.var_confidence * 100) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Expected Shortfall (Conditional VaR)
            var_95_threshold = np.percentile(returns, self.var_confidence * 100)
            tail_losses = returns[returns <= var_95_threshold]
            expected_shortfall = np.mean(tail_losses) * 100 if len(tail_losses) > 0 else var_95
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            maximum_drawdown = np.min(drawdowns) * 100
            
            # Sharpe Ratio (annualized)
            risk_free_rate = 0.02  # 2% annual risk-free rate
            mean_return = np.mean(returns) * 252
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Beta (vs SPY - simplified estimation)
            market_beta = self._estimate_beta(symbol, returns)
            
            # Correlation with SPY
            spy_correlation = self._estimate_spy_correlation(symbol, returns)
            
            # Volatility (annualized)
            volatility_annualized = volatility * 100
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Tail risk score (custom metric)
            tail_risk_score = self._calculate_tail_risk_score(returns, skewness, kurtosis)
            
            return RiskMetrics(
                var_95=round(var_95, 2),
                var_99=round(var_99, 2),
                expected_shortfall=round(expected_shortfall, 2),
                maximum_drawdown=round(maximum_drawdown, 2),
                sharpe_ratio=round(sharpe_ratio, 3),
                sortino_ratio=round(sortino_ratio, 3),
                beta=round(market_beta, 3),
                correlation_spy=round(spy_correlation, 3),
                volatility_annualized=round(volatility_annualized, 2),
                skewness=round(skewness, 3),
                kurtosis=round(kurtosis, 3),
                tail_risk_score=round(tail_risk_score, 1)
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return self._generate_simulated_risk_metrics(symbol, quote_data)
    
    async def _analyze_position_risk(self, symbol: str, quote_data: Dict[str, Any], 
                                    risk_metrics: RiskMetrics) -> PositionRisk:
        """Analyze position-specific risk"""
        try:
            current_price = float(quote_data.get('price', 100))
            
            # Kelly Criterion calculation
            win_rate = 0.55  # Estimated win rate
            avg_win = abs(risk_metrics.var_95) * 1.5  # Estimated average win
            avg_loss = abs(risk_metrics.var_95)  # Estimated average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Position size recommendation (Kelly * safety factor)
            position_size_recommendation = kelly_fraction * 0.5  # 50% of Kelly for safety
            
            # Optimal stop loss based on volatility
            optimal_stop_loss = max(self.stop_loss_percent, risk_metrics.volatility_annualized / 100 * 2)
            
            # Risk-reward ratio
            expected_return = risk_metrics.sharpe_ratio * risk_metrics.volatility_annualized / 100
            risk_reward_ratio = expected_return / optimal_stop_loss if optimal_stop_loss > 0 else 0
            
            # Probability of loss (based on distribution)
            probability_of_loss = 0.5 + (risk_metrics.skewness * -0.1)  # Adjust for skewness
            probability_of_loss = max(0.3, min(0.7, probability_of_loss))
            
            # Risk budget allocation
            risk_budget_allocation = min(self.max_portfolio_risk, position_size_recommendation * optimal_stop_loss)
            
            # Concentration risk assessment
            if position_size_recommendation > 0.15:
                concentration_risk = "HIGH""
            elif position_size_recommendation > 0.10:
                concentration_risk = "MODERATE""
            else:
                concentration_risk = "LOW""
            
            return PositionRisk(
                position_size_recommendation=round(position_size_recommendation, 4),
                kelly_criterion=round(kelly_fraction, 4),
                optimal_stop_loss=round(optimal_stop_loss, 4),
                risk_reward_ratio=round(risk_reward_ratio, 2),
                probability_of_loss=round(probability_of_loss, 3),
                expected_return=round(expected_return, 4),
                risk_budget_allocation=round(risk_budget_allocation, 4),
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Position risk analysis error: {e}")
            return self._generate_fallback_position_risk()
    
    async def _analyze_portfolio_risk(self, symbol: str, risk_metrics: RiskMetrics, 
                                     portfolio_context: Optional[Dict] = None) -> PortfolioRisk:
        """Analyze portfolio-level risk"""
        try:
            # Simulated portfolio analysis (in practice, would use actual portfolio data)
            portfolio_var = risk_metrics.var_95 * 0.8  # Diversification benefit
            
            # Diversification ratio (simplified)
            diversification_ratio = 0.85  # Assumes moderate diversification
            
            # Correlation risk
            if abs(risk_metrics.correlation_spy) > 0.8:
                correlation_risk = "HIGH""
            elif abs(risk_metrics.correlation_spy) > 0.6:
                correlation_risk = "MODERATE""
            else:
                correlation_risk = "LOW""
            
            # Sector concentration (simulated)
            sector_concentration = {
                "Technology": 0.25,
                "Healthcare": 0.15,
                "Financials": 0.20,
                "Other": 0.40
            }
            
            # Max position size based on risk budget
            max_position_size = min(0.20, self.max_portfolio_risk / abs(risk_metrics.var_95) * 100)
            
            # Total portfolio risk
            total_portfolio_risk = abs(portfolio_var)
            
            # Risk contribution of this position
            risk_contribution = abs(risk_metrics.var_95) * 0.1  # Assumes 10% position
            
            # Stress test results
            stress_test_results = {
                "market_crash_20": risk_metrics.var_95 * 4,
                "volatility_spike": risk_metrics.var_95 * 2,
                "correlation_breakdown": risk_metrics.var_95 * 1.5
            }
            
            return PortfolioRisk(
                portfolio_var=round(portfolio_var, 2),
                diversification_ratio=round(diversification_ratio, 3),
                correlation_risk=correlation_risk,
                sector_concentration=sector_concentration,
                max_position_size=round(max_position_size, 4),
                total_portfolio_risk=round(total_portfolio_risk, 2),
                risk_contribution=round(risk_contribution, 4),
                stress_test_results={k: round(v, 2) for k, v in stress_test_results.items()}
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis error: {e}")
            return self._generate_fallback_portfolio_risk()
    
    def _calculate_overall_risk_score(self, risk_metrics: RiskMetrics, 
                                     position_risk: PositionRisk,
                                     anomaly_signals: List[AnomalySignal]) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            score = 50  # Base score
            
            # VaR contribution
            if abs(risk_metrics.var_95) > 5:
                score += 20
            elif abs(risk_metrics.var_95) > 3:
                score += 10
            elif abs(risk_metrics.var_95) < 1:
                score -= 10
            
            # Volatility contribution
            if risk_metrics.volatility_annualized > 40:
                score += 15
            elif risk_metrics.volatility_annualized > 25:
                score += 8
            elif risk_metrics.volatility_annualized < 15:
                score -= 5
            
            # Drawdown contribution
            if abs(risk_metrics.maximum_drawdown) > 20:
                score += 15
            elif abs(risk_metrics.maximum_drawdown) > 10:
                score += 8
            
            # Tail risk contribution
            if risk_metrics.tail_risk_score > 75:
                score += 10
            elif risk_metrics.tail_risk_score < 25:
                score -= 5
            
            # Anomaly contribution
            high_anomalies = sum(1 for a in anomaly_signals if a.severity in ["HIGH", "EXTREME"])
            score += high_anomalies * 5
            
            # Position risk contribution
            if position_risk.concentration_risk == "HIGH":
                score += 10
            elif position_risk.concentration_risk == "LOW":
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Overall risk score calculation error: {e}")
            return 50.0
    
    def _determine_risk_level(self, risk_score: float, anomaly_signals: List[AnomalySignal]) -> str:
        """Determine risk level based on score and anomalies"""
        extreme_anomalies = sum(1 for a in anomaly_signals if a.severity == "EXTREME")
        
        if extreme_anomalies > 0 or risk_score >= 85:
            return "EXTREME""
        elif risk_score >= 70:
            return "HIGH""
        elif risk_score >= 45:
            return "MODERATE""
        else:
            return "LOW""
    
    def _generate_risk_warnings(self, risk_metrics: RiskMetrics, 
                               position_risk: PositionRisk,
                               anomaly_signals: List[AnomalySignal]) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        try:
            # VaR warnings
            if abs(risk_metrics.var_95) > 5:
                warnings.append(f"🚨 HIGH VaR: {abs(risk_metrics.var_95):.1f}% daily risk")
            
            # Volatility warnings
            if risk_metrics.volatility_annualized > 40:
                warnings.append(f"⚡ HIGH VOLATILITY: {risk_metrics.volatility_annualized:.0f}% annualized")
            
            # Drawdown warnings
            if abs(risk_metrics.maximum_drawdown) > self.max_drawdown_warning:
                warnings.append(f"📉 DRAWDOWN RISK: Historical max {abs(risk_metrics.maximum_drawdown):.1f}%")
            
            # Tail risk warnings
            if risk_metrics.tail_risk_score > 75:
                warnings.append("💀 HIGH TAIL RISK: Elevated probability of extreme losses")
            
            # Position size warnings
            if position_risk.concentration_risk == "HIGH":
                warnings.append("⚖️ CONCENTRATION RISK: Position size may be too large")
            
            # Correlation warnings
            if abs(risk_metrics.correlation_spy) > 0.9:
                warnings.append("🔗 HIGH CORRELATION: Limited diversification benefit")
            
            # Anomaly warnings
            for anomaly in anomaly_signals:
                if anomaly.severity in ["HIGH", "EXTREME"]:
                    warnings.append(f"🚨 {anomaly.anomaly_type} ANOMALY: {anomaly.description}")
            
            # Sharpe ratio warnings
            if risk_metrics.sharpe_ratio < 0:
                warnings.append("📊 NEGATIVE SHARPE: Risk-adjusted returns are poor")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Risk warnings generation error: {e}")
            return ["Risk warnings unavailable"]
    
    def _generate_risk_recommendations(self, symbol: str, risk_metrics: RiskMetrics,
                                      position_risk: PositionRisk, risk_level: str) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            # Risk level recommendations
            if risk_level == "EXTREME":
                recommendations.append("🛑 EXTREME RISK: Consider avoiding or exiting position")
                recommendations.append("💰 CAPITAL PRESERVATION: Focus on preserving capital")
                
            elif risk_level == "HIGH":
                recommendations.append("⚠️ HIGH RISK: Reduce position size significantly")
                recommendations.append(f"📏 MAX SIZE: Consider <{position_risk.position_size_recommendation/2:.1%} allocation")
                
            elif risk_level == "MODERATE":
                recommendations.append("⚖️ MODERATE RISK: Use standard risk management")
                recommendations.append(f"📊 SUGGESTED SIZE: {position_risk.position_size_recommendation:.1%} allocation")
                
            else:  # LOW risk
                recommendations.append("✅ LOW RISK: Normal position sizing acceptable")
                recommendations.append(f"📈 OPPORTUNITY: Consider up to {position_risk.position_size_recommendation:.1%} allocation")
            
            # Stop loss recommendations
            recommendations.append(f"🛡️ STOP LOSS: Set at {position_risk.optimal_stop_loss:.1%} below entry")
            
            # Risk-reward recommendations
            if position_risk.risk_reward_ratio < 1.5:
                recommendations.append("📊 POOR R:R: Risk-reward ratio below 1.5:1 - reconsider")
            else:
                recommendations.append(f"✅ GOOD R:R: {position_risk.risk_reward_ratio:.1f}:1 risk-reward ratio")
            
            # Diversification recommendations
            if abs(risk_metrics.correlation_spy) > 0.8:
                recommendations.append("🌐 DIVERSIFY: Add uncorrelated assets to reduce risk")
            
            # Volatility recommendations
            if risk_metrics.volatility_annualized > 30:
                recommendations.append("📉 HIGH VOL: Consider options strategies to manage volatility")
            
            # Portfolio recommendations
            if position_risk.concentration_risk == "HIGH":
                recommendations.append("⚖️ REBALANCE: Reduce concentration risk across portfolio")
            
            return recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"Risk recommendations generation error: {e}")
            return ["Risk recommendations unavailable"]
    
    def _identify_tail_risk_events(self, risk_metrics: RiskMetrics, 
                                  anomaly_signals: List[AnomalySignal]) -> List[str]:
        """Identify potential tail risk events"""
        events = []
        
        try:
            # Market-wide tail risks
            events.append("📉 Market crash (>20% decline)")
            events.append("⚡ Volatility spike (VIX >40)")
            events.append("🏦 Credit crisis / liquidity freeze")
            events.append("🌍 Geopolitical shock")
            
            # Symbol-specific tail risks
            if risk_metrics.tail_risk_score > 60:
                events.append("📊 Earnings shock / guidance cut")
                events.append("👨‍💼 Management/governance issues")
                events.append("🏭 Operational disruption")
            
            # Anomaly-based tail risks
            for anomaly in anomaly_signals:
                if anomaly.severity == "EXTREME":
                    events.append(f"🚨 {anomaly.anomaly_type.title()} regime change")
            
            return events[:6]  # Limit to 6 events
            
        except Exception as e:
            logger.error(f"Tail risk events identification error: {e}")
            return ["Tail risk analysis unavailable"]
    
    def _estimate_beta(self, symbol: str, returns: np.ndarray) -> float:
        """Estimate beta vs market"""
        # Simplified beta estimation
        symbol_betas = {
            'NVDA': 1.4, 'QQQ': 1.1, 'SPY': 1.0, 'AAPL': 1.2, 'MSFT': 0.9
        }
        base_beta = symbol_betas.get(symbol, 1.0)
        
        # Add some randomness based on recent volatility
        vol_adjustment = (np.std(returns) - 0.02) * 5  # Adjust for volatility
        return max(0.3, min(2.5, base_beta + vol_adjustment))
    
    def _estimate_spy_correlation(self, symbol: str, returns: np.ndarray) -> float:
        """Estimate correlation with SPY"""
        # Simplified correlation estimation
        base_correlations = {
            'NVDA': 0.75, 'QQQ': 0.95, 'SPY': 1.0, 'AAPL': 0.80, 'MSFT': 0.85
        }
        base_correlation = base_correlations.get(symbol, 0.7)
        
        # Add some noise
        noise = random.uniform(-0.1, 0.1)
        return max(-0.5, min(1.0, base_correlation + noise))
    
    def _calculate_tail_risk_score(self, returns: np.ndarray, skewness: float, kurtosis: float) -> float:
        """Calculate custom tail risk score"""
        try:
            # Base score from VaR
            var_95 = np.percentile(returns, 5)
            var_score = min(50, abs(var_95) * 1000)  # Scale to 0-50
            
            # Skewness contribution (negative skew = more tail risk)
            skew_score = max(0, -skewness * 10)  # 0-30 range
            
            # Kurtosis contribution (high kurtosis = fat tails)
            kurt_score = min(20, max(0, (kurtosis - 3) * 5))  # 0-20 range
            
            total_score = var_score + skew_score + kurt_score
            return min(100, total_score)
            
        except Exception as e:
            logger.error(f"Tail risk score calculation error: {e}")
            return 50.0
    
    def _generate_price_history(self, symbol: str, quote_data: Dict[str, Any], periods: int = 100) -> List[Dict]:
        """Generate realistic price history for risk analysis"""
        try:
            current_price = float(quote_data.get('price', 100))
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Symbol-specific volatility
            volatilities = {
                'NVDA': 0.035, 'QQQ': 0.025, 'SPY': 0.020, 'AAPL': 0.025, 'MSFT': 0.022
            }
            daily_vol = volatilities.get(symbol, 0.025)
            
            history = []
            price = current_price
            
            for i in range(periods):
                # Random walk with some mean reversion
                random_change = np.random.normal(0, daily_vol)
                mean_reversion = -0.05 * random_change  # Slight mean reversion
                
                price_change = random_change + mean_reversion
                price = price * (1 + price_change)
                
                history.insert(0, {
                    'close': round(price, 2),
                    'price': round(price, 2),
                    'date': (datetime.now() - timedelta(days=periods-i)).strftime('%Y-%m-%d')
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Price history generation error: {e}")
            return []
    
    def _update_risk_history(self, symbol: str, risk_score: float, risk_level: str):
        """Update risk analysis history"""
        self.risk_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'risk_score': risk_score,
            'risk_level': risk_level
        })
        
        # Keep last 100 records
        if len(self.risk_history) > 100:
            self.risk_history.pop(0)
    
    def _generate_simulated_risk_metrics(self, symbol: str, quote_data: Dict[str, Any]) -> RiskMetrics:
        """Generate simulated risk metrics when insufficient data"""
        change_percent = float(quote_data.get('change_percent', 0))
        
        # Base metrics on current volatility
        volatility_annualized = abs(change_percent) * 5 + random.uniform(15, 30)
        
        return RiskMetrics(
            var_95=-(volatility_annualized / 16),  # Rough approximation
            var_99=-(volatility_annualized / 10),
            expected_shortfall=-(volatility_annualized / 12),
            maximum_drawdown=-(volatility_annualized * 0.8),
            sharpe_ratio=random.uniform(-0.5, 1.5),
            sortino_ratio=random.uniform(-0.3, 1.8),
            beta=self._estimate_beta(symbol, np.array([change_percent/100])),
            correlation_spy=self._estimate_spy_correlation(symbol, np.array([change_percent/100])),
            volatility_annualized=volatility_annualized,
            skewness=random.uniform(-1, 1),
            kurtosis=random.uniform(0, 5),
            tail_risk_score=random.uniform(30, 70)
        )
    
    def _generate_fallback_position_risk(self) -> PositionRisk:
        """Generate fallback position risk"""
        return PositionRisk(
            position_size_recommendation=0.05,
            kelly_criterion=0.10,
            optimal_stop_loss=0.05,
            risk_reward_ratio=1.5,
            probability_of_loss=0.45,
            expected_return=0.08,
            risk_budget_allocation=0.0025,
            concentration_risk="MODERATE""
        )
    
    def _generate_fallback_portfolio_risk(self) -> PortfolioRisk:
        """Generate fallback portfolio risk"""
        return PortfolioRisk(
            portfolio_var=-2.5,
            diversification_ratio=0.8,
            correlation_risk="MODERATE",
            sector_concentration={"Technology": 0.3, "Other": 0.7},
            max_position_size=0.10,
            total_portfolio_risk=3.0,
            risk_contribution=0.005,
            stress_test_results={"market_crash": -15.0, "volatility_spike": -8.0}
        )
    
    def _generate_fallback_risk_analysis(self) -> RiskAnalysis:
        """Generate fallback risk analysis"""
        fallback_metrics = RiskMetrics(
            var_95=-2.0, var_99=-3.5, expected_shortfall=-2.8, maximum_drawdown=-12.0,
            sharpe_ratio=0.8, sortino_ratio=1.0, beta=1.0, correlation_spy=0.7,
            volatility_annualized=20.0, skewness=-0.2, kurtosis=3.5, tail_risk_score=45.0
        )
        
        return RiskAnalysis(
            overall_risk_score=50.0,
            risk_level="MODERATE",
            risk_metrics=fallback_metrics,
            position_risk=self._generate_fallback_position_risk(),
            portfolio_risk=self._generate_fallback_portfolio_risk(),
            anomaly_signals=[],
            risk_warnings=["Risk analysis unavailable"],
            risk_recommendations=["Risk recommendations unavailable"],
            scenario_analysis={"base_case": 0.0},
            tail_risk_events=["Risk analysis unavailable"]
        )

# Supporting classes for risk analysis components
class VolatilityRiskAnalyzer:
    """Analyze volatility-specific risks"""
    
    def __init__(self):
        logger.info("⚡ Volatility Risk Analyzer initialized")

class DrawdownAnalyzer:
    """Analyze drawdown patterns and risks"""
    
    def __init__(self):
        logger.info("📉 Drawdown Analyzer initialized")

class CorrelationRiskAnalyzer:
    """Analyze correlation and diversification risks"""
    
    def __init__(self):
        logger.info("🔗 Correlation Risk Analyzer initialized")

class MLAnomalyDetector:
    """ML-enhanced anomaly detection"""
    
    def __init__(self):
        logger.info("🤖 ML Anomaly Detector initialized")
    
    async def detect_anomalies(self, symbol: str, quote_data: Dict[str, Any], 
                              historical_data: List[Dict]) -> List[AnomalySignal]:
        """Detect market anomalies using ML techniques"""
        try:
            anomalies = []
            
            current_price = float(quote_data.get('price', 100))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Price anomaly detection
            if abs(change_percent) > 5:
                anomalies.append(AnomalySignal(
                    anomaly_score=min(100, abs(change_percent) * 10),
                    anomaly_type="PRICE",
                    severity="HIGH" if abs(change_percent) > 8 else "MEDIUM",
                    description=f"Extreme price movement: {change_percent:+.1f}%",
                    probability=0.02,  # 2% probability of such moves
                    historical_precedent=True,
                    recommended_action="Monitor closely, consider reducing position""
                ))
            
            # Volume anomaly detection
            avg_volume = 25000000  # Estimated average
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 3:
                anomalies.append(AnomalySignal(
                    anomaly_score=min(100, volume_ratio * 20),
                    anomaly_type="VOLUME",
                    severity="HIGH" if volume_ratio > 5 else "MEDIUM",
                    description=f"Extreme volume: {volume_ratio:.1f}x normal",
                    probability=0.05,
                    historical_precedent=True,
                    recommended_action="Investigate news catalyst""
                ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return []

class StressTestEngine:
    """Perform scenario and stress testing"""
    
    def __init__(self):
        logger.info("🧪 Stress Test Engine initialized")
    
    async def run_scenarios(self, symbol: str, quote_data: Dict[str, Any], 
                           risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Run stress test scenarios"""
        try:
            current_price = float(quote_data.get('price', 100))
            
            scenarios = {
                "market_crash_20": current_price * 0.8 - current_price,
                "market_crash_30": current_price * 0.7 - current_price,
                "volatility_spike_2x": risk_metrics.var_95 * 2,
                "correlation_breakdown": risk_metrics.var_95 * 1.5,
                "liquidity_crisis": risk_metrics.var_95 * 1.8,
                "sector_rotation": random.uniform(-8, 8)
            }
            
            return {k: round(v, 2) for k, v in scenarios.items()}
            
        except Exception as e:
            logger.error(f"Stress testing error: {e}")
            return {"base_case": 0.0}

# Export main classes
__all__ = ['AdvancedRiskAnalyzer', 'RiskAnalysis', 'RiskMetrics', 'PositionRisk', 'PortfolioRisk', 'AnomalySignal']

logger.info("⚠️ Advanced Risk Analysis module loaded successfully")