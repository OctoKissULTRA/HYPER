# market_structure.py - Market Structure Analysis Module
import logging
import asyncio
import numpy as np
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class SectorData:
    """Individual sector performance data""""
    sector: str
    performance_1d: float
    performance_5d: float
    performance_1m: float
    relative_strength: float  # vs market
    momentum_score: float
    rotation_signal: str  # INFLOW, OUTFLOW, NEUTRAL
    leadership_rank: int  # 1-11 ranking
    volume_ratio: float
    institutional_activity: str

@dataclass
class BreadthSignal:
    """Market breadth signal""""
    advancing_stocks: int
    declining_stocks: int
    unchanged_stocks: int
    advance_decline_ratio: float
    advance_decline_line: float
    breadth_thrust: bool
    breadth_divergence: bool
    market_participation: str  # BROAD, NARROW, MIXED
    strength_score: float  # 0-100

@dataclass
class MarketStructureSignal:
    """Complete market structure signal""""
    breadth_signal: BreadthSignal
    sector_rotation: Dict[str, SectorData]
    leadership_analysis: Dict[str, Any]
    institutional_flow: Dict[str, Any]
    market_regime: str  # RISK_ON, RISK_OFF, TRANSITION, UNCERTAINTY
    rotation_theme: str  # GROWTH_TO_VALUE, TECH_TO_CYCLICAL, etc.
    structure_score: float  # 0-100
    momentum_quality: str  # STRONG, MODERATE, WEAK, DETERIORATING
    risk_assessment: Dict[str, Any]

@dataclass
class MarketStructureAnalysis:
    """Complete market structure analysis""""
    current_signal: MarketStructureSignal
    market_health: str  # HEALTHY, CAUTION, UNHEALTHY, DETERIORATING
    participation_quality: str
    rotation_opportunities: List[str]
    risk_warnings: List[str]
    strategic_recommendations: List[str]
    correlation_breakdown: Dict[str, float]
    structural_trends: Dict[str, Any]

class AdvancedMarketStructureAnalyzer:
    """Advanced Market Structure and Sector Rotation Analysis""""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.structure_cache = {}
        self.cache_duration = 180  # 3 minutes
        self.sector_history = {}
        self.breadth_history = []
        
        # Market structure thresholds
        self.thresholds = config.get('market_structure_config', {})
        self.breadth_very_bullish = self.thresholds.get('breadth_very_bullish', 0.9)
        self.breadth_bullish = self.thresholds.get('breadth_bullish', 0.6)
        self.breadth_bearish = self.thresholds.get('breadth_bearish', 0.4)
        self.breadth_very_bearish = self.thresholds.get('breadth_very_bearish', 0.1)
        
        # Sector definitions
        self.sectors = [
            'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples',
            'Energy', 'Utilities', 'Real Estate', 'Materials'"
        ]
        
        # Component analyzers
        self.breadth_analyzer = BreadthAnalyzer()
        self.sector_analyzer = SectorRotationAnalyzer(self.sectors)
        self.institutional_analyzer = InstitutionalFlowAnalyzer()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        
        logger.info("🏗️ Advanced Market Structure Analyzer initialized")
        logger.info(f"📊 Tracking {len(self.sectors)} sectors with microstructure analysis")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], 
                     market_data: Optional[Dict] = None) -> MarketStructureAnalysis:
        """Complete market structure analysis""""
        try:
            # Check cache first
            cache_key = f"structure_{time.time() // self.cache_duration}""
            if cache_key in self.structure_cache:
                logger.debug("📋 Using cached market structure analysis")
                return self.structure_cache[cache_key]
            
            logger.debug("🏗️ Performing comprehensive market structure analysis...")
            
            # Analyze market breadth
            breadth_signal = await self.breadth_analyzer.analyze_breadth(symbol, quote_data)
            
            # Analyze sector rotation
            sector_rotation = await self.sector_analyzer.analyze_sector_rotation(symbol, quote_data)
            
            # Analyze leadership
            leadership_analysis = await self._analyze_market_leadership(sector_rotation, breadth_signal)
            
            # Analyze institutional flow
            institutional_flow = await self.institutional_analyzer.analyze_flow(symbol, quote_data, sector_rotation)
            
            # Determine market regime
            market_regime = self._determine_market_regime(breadth_signal, sector_rotation, institutional_flow)
            
            # Identify rotation theme
            rotation_theme = self._identify_rotation_theme(sector_rotation)
            
            # Calculate structure score
            structure_score = self._calculate_structure_score(breadth_signal, sector_rotation, institutional_flow)
            
            # Assess momentum quality
            momentum_quality = self._assess_momentum_quality(breadth_signal, sector_rotation)
            
            # Risk assessment
            risk_assessment = self._perform_risk_assessment(breadth_signal, sector_rotation, market_regime)
            
            # Create market structure signal
            market_signal = MarketStructureSignal(
                breadth_signal=breadth_signal,
                sector_rotation=sector_rotation,
                leadership_analysis=leadership_analysis,
                institutional_flow=institutional_flow,
                market_regime=market_regime,
                rotation_theme=rotation_theme,
                structure_score=structure_score,
                momentum_quality=momentum_quality,
                risk_assessment=risk_assessment
            )
            
            # Analyze overall market health
            market_health = self._analyze_market_health(market_signal)
            
            # Assess participation quality
            participation_quality = self._assess_participation_quality(breadth_signal, sector_rotation)
            
            # Generate rotation opportunities
            rotation_opportunities = self._identify_rotation_opportunities(sector_rotation, rotation_theme)
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(market_signal, market_health)
            
            # Generate strategic recommendations
            strategic_recommendations = self._generate_strategic_recommendations(market_signal, symbol)
            
            # Analyze correlation breakdown
            correlation_breakdown = self._analyze_correlation_breakdown(sector_rotation)
            
            # Identify structural trends
            structural_trends = self._identify_structural_trends(market_signal)
            
            # Add microstructure analysis
            microstructure_data = await self.microstructure_analyzer.analyze_microstructure(symbol, quote_data)
            structural_trends['microstructure'] = microstructure_data
            
            # Update history
            self._update_structure_history(market_signal)
            
            result = MarketStructureAnalysis(
                current_signal=market_signal,
                market_health=market_health,
                participation_quality=participation_quality,
                rotation_opportunities=rotation_opportunities,
                risk_warnings=risk_warnings,
                strategic_recommendations=strategic_recommendations,
                correlation_breakdown=correlation_breakdown,
                structural_trends=structural_trends
            )
            
            # Cache result
            self.structure_cache[cache_key] = result
            
            logger.debug(f"✅ Market structure: {market_regime} regime, {structure_score:.0f} score")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Market structure analysis error: {e}")
            return self._generate_fallback_analysis()
    
    async def _analyze_market_leadership(self, sector_rotation: Dict[str, SectorData], 
                                        breadth_signal: BreadthSignal) -> Dict[str, Any]:
        """Analyze market leadership characteristics""""
        try:
            # Find leading sectors
            leading_sectors = []
            lagging_sectors = []
            
            for sector_name, sector_data in sector_rotation.items():
                if sector_data.leadership_rank <= 3:
                    leading_sectors.append({
                        'sector': sector_name,
                        'rank': sector_data.leadership_rank,
                        'relative_strength': sector_data.relative_strength,
                        'momentum': sector_data.momentum_score
                    })
                elif sector_data.leadership_rank >= 9:
                    lagging_sectors.append({
                        'sector': sector_name,
                        'rank': sector_data.leadership_rank,
                        'relative_strength': sector_data.relative_strength,
                        'momentum': sector_data.momentum_score
                    })
            
            # Analyze leadership quality
            leadership_concentrated = len(leading_sectors) <= 2
            leadership_rotation = any(s['momentum'] < 0 for s in leading_sectors[:2]) if leading_sectors else False
            
            # Determine leadership theme
            if leading_sectors:
                top_sector = leading_sectors[0]['sector']
                if top_sector in ['Technology', 'Communication Services', 'Consumer Discretionary']:
                    leadership_theme = "GROWTH_LEADERSHIP""
                elif top_sector in ['Financials', 'Energy', 'Materials', 'Industrials']:
                    leadership_theme = "VALUE_LEADERSHIP""
                elif top_sector in ['Utilities', 'Consumer Staples', 'Real Estate']:
                    leadership_theme = "DEFENSIVE_LEADERSHIP""
                else:
                    leadership_theme = "MIXED_LEADERSHIP""
            else:
                leadership_theme = "NO_CLEAR_LEADERSHIP""
            
            return {
                'leading_sectors': leading_sectors,
                'lagging_sectors': lagging_sectors,
                'leadership_theme': leadership_theme,
                'leadership_concentrated': leadership_concentrated,
                'leadership_rotation': leadership_rotation,
                'leadership_quality': "STRONG" if len(leading_sectors) >= 2 and not leadership_rotation else "WEAK""
            }
            
        except Exception as e:
            logger.error(f"Leadership analysis error: {e}")
            return {'leadership_theme': 'UNKNOWN', 'leadership_quality': 'UNKNOWN'}
    
    def _determine_market_regime(self, breadth_signal: BreadthSignal, 
                                sector_rotation: Dict[str, SectorData],
                                institutional_flow: Dict[str, Any]) -> str:
        """Determine current market regime""""
        try:
            regime_scores = {
                'RISK_ON': 0,
                'RISK_OFF': 0,
                'TRANSITION': 0,
                'UNCERTAINTY': 0
            }
            
            # Breadth contribution
            if breadth_signal.advance_decline_ratio > 1.5:
                regime_scores['RISK_ON'] += 3
            elif breadth_signal.advance_decline_ratio < 0.7:
                regime_scores['RISK_OFF'] += 3
            else:
                regime_scores['TRANSITION'] += 2
            
            # Sector rotation contribution
            growth_sectors = ['Technology', 'Consumer Discretionary', 'Communication Services']
            defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            cyclical_sectors = ['Financials', 'Energy', 'Materials', 'Industrials']
            
            growth_performance = np.mean([sector_rotation[s].relative_strength for s in growth_sectors if s in sector_rotation])
            defensive_performance = np.mean([sector_rotation[s].relative_strength for s in defensive_sectors if s in sector_rotation])
            cyclical_performance = np.mean([sector_rotation[s].relative_strength for s in cyclical_sectors if s in sector_rotation])
            
            if growth_performance > 0 and cyclical_performance > 0:
                regime_scores['RISK_ON'] += 2
            elif defensive_performance > growth_performance:
                regime_scores['RISK_OFF'] += 2
            else:
                regime_scores['TRANSITION'] += 1
            
            # Institutional flow contribution
            inst_sentiment = institutional_flow.get('sentiment', 'NEUTRAL')
            if inst_sentiment == 'BULLISH':
                regime_scores['RISK_ON'] += 1
            elif inst_sentiment == 'BEARISH':
                regime_scores['RISK_OFF'] += 1
            else:
                regime_scores['UNCERTAINTY'] += 1
            
            # Return regime with highest score
            return max(regime_scores, key=regime_scores.get)
            
        except Exception as e:
            logger.error(f"Market regime determination error: {e}")
            return "UNCERTAINTY""
    
    def _identify_rotation_theme(self, sector_rotation: Dict[str, SectorData]) -> str:
        """Identify current sector rotation theme""""
        try:
            # Sort sectors by performance
            sorted_sectors = sorted(sector_rotation.items(), 
                                  key=lambda x: x[1].relative_strength, reverse=True)
            
            top_3 = [s[0] for s in sorted_sectors[:3]]
            bottom_3 = [s[0] for s in sorted_sectors[-3:]]
            
            # Define sector groups
            growth_sectors = ['Technology', 'Consumer Discretionary', 'Communication Services']
            value_sectors = ['Financials', 'Energy', 'Materials']
            defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            cyclical_sectors = ['Industrials', 'Materials', 'Energy']
            
            # Analyze rotation patterns
            growth_in_top = len([s for s in top_3 if s in growth_sectors])
            value_in_top = len([s for s in top_3 if s in value_sectors])
            defensive_in_top = len([s for s in top_3 if s in defensive_sectors])
            cyclical_in_top = len([s for s in top_3 if s in cyclical_sectors])
            
            # Determine rotation theme
            if growth_in_top >= 2:
                return "GROWTH_MOMENTUM""
            elif value_in_top >= 2:
                return "VALUE_ROTATION""
            elif defensive_in_top >= 2:
                return "DEFENSIVE_ROTATION""
            elif cyclical_in_top >= 2:
                return "CYCLICAL_ROTATION""
            elif 'Technology' in top_3 and 'Financials' in bottom_3:
                return "TECH_TO_VALUE""
            elif 'Financials' in top_3 and 'Technology' in bottom_3:
                return "VALUE_TO_TECH""
            else:
                return "MIXED_ROTATION""
                
        except Exception as e:
            logger.error(f"Rotation theme identification error: {e}")
            return "UNCLEAR_ROTATION""
    
    def _calculate_structure_score(self, breadth_signal: BreadthSignal,
                                  sector_rotation: Dict[str, SectorData],
                                  institutional_flow: Dict[str, Any]) -> float:
        """Calculate overall market structure health score""""
        try:
            score = 0
            
            # Breadth component (40%)
            if breadth_signal.advance_decline_ratio > 1.5:
                score += 40
            elif breadth_signal.advance_decline_ratio > 1.0:
                score += 25
            elif breadth_signal.advance_decline_ratio > 0.7:
                score += 15
            else:
                score += 5
            
            # Sector participation component (30%)
            positive_sectors = sum(1 for s in sector_rotation.values() if s.relative_strength > 0)
            sector_participation = positive_sectors / len(sector_rotation) if sector_rotation else 0
            
            if sector_participation > 0.7:
                score += 30
            elif sector_participation > 0.5:
                score += 20
            elif sector_participation > 0.3:
                score += 10
            else:
                score += 0
            
            # Leadership quality component (20%)
            strong_momentum_sectors = sum(1 for s in sector_rotation.values() if s.momentum_score > 70)
            if strong_momentum_sectors >= 3:
                score += 20
            elif strong_momentum_sectors >= 2:
                score += 15
            elif strong_momentum_sectors >= 1:
                score += 10
            else:
                score += 0
            
            # Institutional flow component (10%)
            inst_sentiment = institutional_flow.get('sentiment', 'NEUTRAL')
            if inst_sentiment == 'BULLISH':
                score += 10
            elif inst_sentiment == 'NEUTRAL':
                score += 5
            else:
                score += 0
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Structure score calculation error: {e}")
            return 50.0
    
    def _assess_momentum_quality(self, breadth_signal: BreadthSignal,
                                sector_rotation: Dict[str, SectorData]) -> str:
        """Assess quality of market momentum""""
        try:
            quality_factors = []
            
            # Breadth quality
            if breadth_signal.market_participation == "BROAD":
                quality_factors.append("BROAD_PARTICIPATION")
            elif breadth_signal.market_participation == "NARROW":
                quality_factors.append("NARROW_PARTICIPATION")
            
            # Divergence check
            if breadth_signal.breadth_divergence:
                quality_factors.append("BREADTH_DIVERGENCE")
            
            # Sector momentum distribution
            high_momentum_sectors = sum(1 for s in sector_rotation.values() if s.momentum_score > 70)
            low_momentum_sectors = sum(1 for s in sector_rotation.values() if s.momentum_score < 30)
            
            if high_momentum_sectors >= 4:
                quality_factors.append("STRONG_SECTOR_MOMENTUM")
            elif low_momentum_sectors >= 4:
                quality_factors.append("WEAK_SECTOR_MOMENTUM")
            
            # Institutional participation
            inst_vol_ratio = sum(s.volume_ratio for s in sector_rotation.values()) / len(sector_rotation)
            if inst_vol_ratio > 1.3:
                quality_factors.append("HIGH_INSTITUTIONAL_ACTIVITY")
            
            # Assess overall quality
            strong_factors = ["BROAD_PARTICIPATION", "STRONG_SECTOR_MOMENTUM", "HIGH_INSTITUTIONAL_ACTIVITY"]
            weak_factors = ["NARROW_PARTICIPATION", "BREADTH_DIVERGENCE", "WEAK_SECTOR_MOMENTUM"]
            
            strong_count = sum(1 for f in strong_factors if f in quality_factors)
            weak_count = sum(1 for f in weak_factors if f in quality_factors)
            
            if strong_count >= 2 and weak_count == 0:
                return "STRONG""
            elif strong_count >= 1 and weak_count <= 1:
                return "MODERATE""
            elif weak_count >= 2:
                return "DETERIORATING""
            else:
                return "WEAK""
                
        except Exception as e:
            logger.error(f"Momentum quality assessment error: {e}")
            return "UNKNOWN""
    
    def _perform_risk_assessment(self, breadth_signal: BreadthSignal,
                                sector_rotation: Dict[str, SectorData],
                                market_regime: str) -> Dict[str, Any]:
        """Perform market structure risk assessment""""
        try:
            risk_factors = []
            risk_level = "LOW""
            
            # Breadth risks
            if breadth_signal.breadth_divergence:
                risk_factors.append("Market breadth diverging from price")
                risk_level = "MEDIUM""
            
            if breadth_signal.market_participation == "NARROW":
                risk_factors.append("Narrow market participation")
                risk_level = "MEDIUM""
            
            # Sector concentration risks
            top_sector_performance = max(s.relative_strength for s in sector_rotation.values())
            if top_sector_performance > 15:  # >15% outperformance
                risk_factors.append("Excessive sector concentration")
                risk_level = "MEDIUM""
            
            # Rotation risks
            defensive_outperformance = any(s.relative_strength > 5 for s in sector_rotation.values() 
                                         if s.sector in ['Utilities', 'Consumer Staples']):
            if defensive_outperformance and market_regime != "RISK_OFF":
                risk_factors.append("Defensive sectors outperforming in risk-on environment")
                risk_level = "MEDIUM""
            
            # Momentum deterioration
            negative_momentum_sectors = sum(1 for s in sector_rotation.values() if s.momentum_score < 40)
            if negative_momentum_sectors >= len(sector_rotation) * 0.6:  # >60% weak momentum
                risk_factors.append("Broad momentum deterioration")
                risk_level = "HIGH""
            
            # Set risk level based on factors
            if len(risk_factors) >= 3:
                risk_level = "HIGH""
            elif len(risk_factors) >= 2:
                risk_level = "MEDIUM""
            elif len(risk_factors) >= 1:
                risk_level = "LOW""
            else:
                risk_level = "MINIMAL""
            
            return {
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'factor_count': len(risk_factors),
                'concentration_risk': top_sector_performance > 10,
                'breadth_risk': breadth_signal.breadth_divergence,
                'momentum_risk': negative_momentum_sectors >= 4
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {'risk_level': 'UNKNOWN', 'risk_factors': []}
    
    def _analyze_market_health(self, market_signal: MarketStructureSignal) -> str:
        """Analyze overall market health""""
        try:
            health_score = market_signal.structure_score
            
            # Adjust for specific factors
            if market_signal.breadth_signal.breadth_divergence:
                health_score -= 15
            
            if market_signal.momentum_quality == "DETERIORATING":
                health_score -= 20
            elif market_signal.momentum_quality == "STRONG":
                health_score += 10
            
            if market_signal.market_regime == "RISK_OFF":
                health_score -= 10
            elif market_signal.market_regime == "RISK_ON":
                health_score += 5
            
            # Determine health level
            if health_score >= 80:
                return "HEALTHY""
            elif health_score >= 60:
                return "CAUTION""
            elif health_score >= 40:
                return "UNHEALTHY""
            else:
                return "DETERIORATING""
                
        except Exception as e:
            logger.error(f"Market health analysis error: {e}")
            return "UNKNOWN""
    
    def _assess_participation_quality(self, breadth_signal: BreadthSignal,
                                     sector_rotation: Dict[str, SectorData]) -> str:
        """Assess quality of market participation""""
        try:
            # Count positive sectors
            positive_sectors = sum(1 for s in sector_rotation.values() if s.relative_strength > 0)
            participation_rate = positive_sectors / len(sector_rotation) if sector_rotation else 0
            
            # Assess breadth
            ad_ratio = breadth_signal.advance_decline_ratio
            
            if participation_rate > 0.7 and ad_ratio > 1.3:
                return "EXCELLENT_BROAD_PARTICIPATION""
            elif participation_rate > 0.6 and ad_ratio > 1.1:
                return "GOOD_PARTICIPATION""
            elif participation_rate > 0.4 and ad_ratio > 0.9:
                return "MODERATE_PARTICIPATION""
            elif participation_rate < 0.3 or ad_ratio < 0.7:
                return "POOR_NARROW_PARTICIPATION""
            else:
                return "MIXED_PARTICIPATION""
                
        except Exception as e:
            logger.error(f"Participation quality assessment error: {e}")
            return "UNKNOWN""
    
    def _identify_rotation_opportunities(self, sector_rotation: Dict[str, SectorData],
                                        rotation_theme: str) -> List[str]:
        """Identify sector rotation opportunities""""
        opportunities = []
        
        try:
            # Sort sectors by relative strength
            sorted_sectors = sorted(sector_rotation.items(), 
                                  key=lambda x: x[1].relative_strength, reverse=True)
            
            # Identify emerging opportunities
            for sector_name, sector_data in sorted_sectors:
                if sector_data.rotation_signal == "INFLOW" and sector_data.momentum_score > 60:
                    opportunities.append(f"🟢 {sector_name}: Strong inflows with good momentum")
                elif sector_data.leadership_rank <= 3 and sector_data.relative_strength > 3:
                    opportunities.append(f"📈 {sector_name}: Top 3 leadership with strong relative performance")
            
            # Theme-based opportunities
            if rotation_theme == "VALUE_ROTATION":
                value_sectors = ['Financials', 'Energy', 'Materials']
                for sector in value_sectors:
                    if sector in sector_rotation and sector_rotation[sector].momentum_score > 50:
                        opportunities.append(f"💰 Value rotation play: {sector} showing momentum")
            
            elif rotation_theme == "DEFENSIVE_ROTATION":
                defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
                for sector in defensive_sectors:
                    if sector in sector_rotation and sector_rotation[sector].relative_strength > 0:
                        opportunities.append(f"🛡️ Defensive play: {sector} outperforming")
            
            # Contrarian opportunities
            underperforming_sectors = [s for s, d in sorted_sectors[-3:] 
                                     if d.momentum_score < 30 and d.relative_strength < -5]:
            for sector_name, _ in underperforming_sectors:
                opportunities.append(f"🔄 Contrarian opportunity: {sector_name} oversold")
            
            return opportunities[:6]  # Limit to top 6 opportunities
            
        except Exception as e:
            logger.error(f"Rotation opportunities identification error: {e}")
            return ["Analysis unavailable"]
    
    def _generate_risk_warnings(self, market_signal: MarketStructureSignal,
                               market_health: str) -> List[str]:
        """Generate market structure risk warnings""""
        warnings = []
        
        try:
            # Health-based warnings
            if market_health == "DETERIORATING":
                warnings.append("🚨 MARKET STRUCTURE DETERIORATING: Multiple negative factors present")
            elif market_health == "UNHEALTHY":
                warnings.append("⚠️ UNHEALTHY MARKET STRUCTURE: Exercise caution")
            
            # Breadth warnings
            if market_signal.breadth_signal.breadth_divergence:
                warnings.append("📉 BREADTH DIVERGENCE: Market breadth not confirming price action")
            
            if market_signal.breadth_signal.market_participation == "NARROW":
                warnings.append("🎯 NARROW PARTICIPATION: Few stocks driving market gains")
            
            # Momentum warnings
            if market_signal.momentum_quality == "DETERIORATING":
                warnings.append("📉 MOMENTUM DETERIORATING: Sector momentum weakening broadly")
            
            # Regime warnings
            if market_signal.market_regime == "RISK_OFF":
                warnings.append("🛡️ RISK-OFF ENVIRONMENT: Defensive positioning recommended")
            elif market_signal.market_regime == "UNCERTAINTY":
                warnings.append("❓ UNCERTAIN REGIME: Mixed signals across market structure")
            
            # Sector concentration warnings
            risk_assessment = market_signal.risk_assessment
            if risk_assessment.get('concentration_risk', False):
                warnings.append("⚖️ CONCENTRATION RISK: Excessive reliance on few sectors")
            
            # Structural warnings
            if market_signal.structure_score < 40:
                warnings.append("🏗️ WEAK STRUCTURE: Poor underlying market foundation")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Risk warnings generation error: {e}")
            return ["Risk analysis unavailable"]
    
    def _generate_strategic_recommendations(self, market_signal: MarketStructureSignal,
                                           symbol: str) -> List[str]:
        """Generate strategic recommendations based on market structure""""
        recommendations = []
        
        try:
            # Regime-based recommendations
            if market_signal.market_regime == "RISK_ON":
                recommendations.append("🟢 RISK-ON REGIME: Consider growth and cyclical exposure")
                recommendations.append(f"📈 {symbol}: Favorable environment for risk assets")
                
            elif market_signal.market_regime == "RISK_OFF":
                recommendations.append("🛡️ RISK-OFF REGIME: Focus on defensive positioning")
                recommendations.append(f"⚠️ {symbol}: Consider reducing position size or hedging")
                
            elif market_signal.market_regime == "TRANSITION":
                recommendations.append("🔄 TRANSITION REGIME: Balanced approach recommended")
                recommendations.append(f"⚖️ {symbol}: Monitor for directional clarity")
            
            # Rotation-based recommendations
            rotation_theme = market_signal.rotation_theme
            if rotation_theme == "GROWTH_MOMENTUM":
                recommendations.append("🚀 GROWTH MOMENTUM: Technology and growth sectors favored")
            elif rotation_theme == "VALUE_ROTATION":
                recommendations.append("💰 VALUE ROTATION: Financials, energy, materials in favor")
            elif rotation_theme == "DEFENSIVE_ROTATION":
                recommendations.append("🛡️ DEFENSIVE ROTATION: Utilities, staples, healthcare preferred")
            
            # Structure-based recommendations
            if market_signal.structure_score >= 80:
                recommendations.append("💪 STRONG STRUCTURE: Increase risk tolerance and position sizing")
            elif market_signal.structure_score <= 40:
                recommendations.append("⚠️ WEAK STRUCTURE: Reduce risk and focus on capital preservation")
            
            # Momentum-based recommendations
            if market_signal.momentum_quality == "STRONG":
                recommendations.append("⚡ STRONG MOMENTUM: Trend-following strategies favored")
            elif market_signal.momentum_quality == "DETERIORATING":
                recommendations.append("📉 DETERIORATING MOMENTUM: Consider contrarian positioning")
            
            # Breadth-based recommendations
            if market_signal.breadth_signal.market_participation == "BROAD":
                recommendations.append("🌊 BROAD PARTICIPATION: Sustainable move, maintain exposure")
            elif market_signal.breadth_signal.market_participation == "NARROW":
                recommendations.append("🎯 NARROW PARTICIPATION: Be selective, focus on leaders")
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            logger.error(f"Strategic recommendations generation error: {e}")
            return ["Strategic analysis unavailable"]
    
    def _analyze_correlation_breakdown(self, sector_rotation: Dict[str, SectorData]) -> Dict[str, float]:
        """Analyze sector correlation patterns""""
        try:
            # Calculate dispersion of sector performance
            performances = [s.relative_strength for s in sector_rotation.values()]
            performance_std = np.std(performances) if performances else 0
            
            # High dispersion = lower correlation (more differentiation)
            # Low dispersion = higher correlation (moving together)
            estimated_correlation = max(0.3, min(0.9, 0.8 - (performance_std / 10)))
            
            correlations = {
                'average_sector_correlation': round(estimated_correlation, 3),
                'performance_dispersion': round(performance_std, 2),
                'correlation_regime': "HIGH" if estimated_correlation > 0.7 else "MEDIUM" if estimated_correlation > 0.5 else "LOW",
                'tech_vs_financials': round(random.uniform(-0.2, 0.4), 3),
                'growth_vs_value': round(random.uniform(-0.3, 0.3), 3),
                'cyclical_vs_defensive': round(random.uniform(-0.4, 0.2), 3)
            }
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation breakdown analysis error: {e}")
            return {'average_sector_correlation': 0.6}
    
    def _identify_structural_trends(self, market_signal: MarketStructureSignal) -> Dict[str, Any]:
        """Identify longer-term structural trends""""
        try:
            trends = {}
            
            # Rotation durability
            if market_signal.rotation_theme in ["GROWTH_MOMENTUM", "VALUE_ROTATION"]:
                trends['rotation_durability'] = "STRONG" if market_signal.structure_score > 70 else "WEAK""
            else:
                trends['rotation_durability'] = "UNCERTAIN""
            
            # Leadership sustainability
            leadership_quality = market_signal.leadership_analysis.get('leadership_quality', 'UNKNOWN')
            trends['leadership_sustainability'] = leadership_quality
            
            # Regime stability
            if market_signal.structure_score > 75:
                trends['regime_stability'] = "STABLE""
            elif market_signal.structure_score < 45:
                trends['regime_stability'] = "UNSTABLE""
            else:
                trends['regime_stability'] = "TRANSITIONAL""
            
            # Institutional engagement
            inst_sentiment = market_signal.institutional_flow.get('sentiment', 'NEUTRAL')
            trends['institutional_engagement'] = "HIGH" if inst_sentiment == "BULLISH" else "LOW" if inst_sentiment == "BEARISH" else "MODERATE""
            
            return trends
            
        except Exception as e:
            logger.error(f"Structural trends identification error: {e}")
            return {'trends': 'analysis_unavailable'}
    
    def _update_structure_history(self, market_signal: MarketStructureSignal):
        """Update market structure analysis history""""
        history_entry = {
            'timestamp': datetime.now(),
            'regime': market_signal.market_regime,
            'structure_score': market_signal.structure_score,
            'breadth_ratio': market_signal.breadth_signal.advance_decline_ratio,
            'rotation_theme': market_signal.rotation_theme
        }
        
        self.breadth_history.append(history_entry)
        
        # Keep last 100 records
        if len(self.breadth_history) > 100:
            self.breadth_history.pop(0)
    
    def _generate_fallback_analysis(self) -> MarketStructureAnalysis:
        """Generate fallback analysis when main analysis fails""""
        fallback_breadth = BreadthSignal(
            advancing_stocks=1500,
            declining_stocks=1500,
            unchanged_stocks=0,
            advance_decline_ratio=1.0,
            advance_decline_line=0.0,
            breadth_thrust=False,
            breadth_divergence=False,
            market_participation="MIXED",
            strength_score=50.0
        )
        
        fallback_sectors = {}
        for sector in self.sectors:
            fallback_sectors[sector] = SectorData(
                sector=sector,
                performance_1d=0.0,
                performance_5d=0.0,
                performance_1m=0.0,
                relative_strength=0.0,
                momentum_score=50.0,
                rotation_signal="NEUTRAL",
                leadership_rank=6,
                volume_ratio=1.0,
                institutional_activity="NEUTRAL""
            )
        
        fallback_signal = MarketStructureSignal(
            breadth_signal=fallback_breadth,
            sector_rotation=fallback_sectors,
            leadership_analysis={'leadership_theme': 'UNKNOWN'},
            institutional_flow={'sentiment': 'NEUTRAL'},
            market_regime="UNCERTAINTY",
            rotation_theme="MIXED_ROTATION",
            structure_score=50.0,
            momentum_quality="UNKNOWN",
            risk_assessment={'risk_level': 'UNKNOWN', 'risk_factors': []}
        )
        
        return MarketStructureAnalysis(
            current_signal=fallback_signal,
            market_health="UNKNOWN",
            participation_quality="UNKNOWN",
            rotation_opportunities=["Analysis unavailable"],
            risk_warnings=["Analysis unavailable"],
            strategic_recommendations=["Analysis unavailable"],
            correlation_breakdown={'average_sector_correlation': 0.6},
            structural_trends={'trends': 'unavailable'}
        )

class BreadthAnalyzer:
    """Analyze market breadth indicators""""
    
    def __init__(self):
        self.breadth_cache = {}
        logger.info("📊 Breadth Analyzer initialized")
    
    async def analyze_breadth(self, symbol: str, quote_data: Dict[str, Any]) -> BreadthSignal:
        """Analyze market breadth indicators""""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Simulate market breadth based on individual stock performance
            if change_percent > 2:  # Strong up day
                advancing_ratio = random.uniform(0.7, 0.9)
            elif change_percent > 0:  # Modest up day
                advancing_ratio = random.uniform(0.55, 0.75)
            elif change_percent > -2:  # Modest down day
                advancing_ratio = random.uniform(0.35, 0.55)
            else:  # Strong down day
                advancing_ratio = random.uniform(0.1, 0.4)
            
            # Total stocks (simplified)
            total_stocks = 3000
            advancing_stocks = int(total_stocks * advancing_ratio)
            declining_stocks = total_stocks - advancing_stocks - random.randint(0, 100)
            unchanged_stocks = total_stocks - advancing_stocks - declining_stocks
            
            # Calculate A/D ratio
            ad_ratio = advancing_stocks / declining_stocks if declining_stocks > 0 else 10.0
            
            # Simulate A/D line (cumulative)
            ad_line_change = advancing_stocks - declining_stocks
            ad_line = ad_line_change  # Simplified, normally cumulative
            
            # Breadth thrust (90% up in 10 days)
            breadth_thrust = advancing_ratio > 0.9
            
            # Breadth divergence detection
            breadth_divergence = self._detect_breadth_divergence(change_percent, advancing_ratio)
            
            # Market participation assessment
            if advancing_ratio > 0.7:
                participation = "BROAD""
            elif advancing_ratio < 0.4:
                participation = "NARROW""
            else:
                participation = "MIXED""
            
            # Strength score
            strength_score = advancing_ratio * 100
            
            return BreadthSignal(
                advancing_stocks=advancing_stocks,
                declining_stocks=declining_stocks,
                unchanged_stocks=unchanged_stocks,
                advance_decline_ratio=round(ad_ratio, 3),
                advance_decline_line=ad_line,
                breadth_thrust=breadth_thrust,
                breadth_divergence=breadth_divergence,
                market_participation=participation,
                strength_score=round(strength_score, 1)
            )
            
        except Exception as e:
            logger.error(f"Breadth analysis error: {e}")
            return self._generate_fallback_breadth()
    
    def _detect_breadth_divergence(self, price_change: float, breadth_ratio: float) -> bool:
        """Detect breadth divergence from price action""""
        # Positive price with poor breadth = bearish divergence
        # Negative price with good breadth = bullish divergence
        
        if price_change > 1 and breadth_ratio < 0.6:  # Up move, poor breadth
            return True
        elif price_change < -1 and breadth_ratio > 0.5:  # Down move, decent breadth
            return True
        else:
            return False
    
    def _generate_fallback_breadth(self) -> BreadthSignal:
        """Generate fallback breadth signal""""
        return BreadthSignal(
            advancing_stocks=1500,
            declining_stocks=1500,
            unchanged_stocks=0,
            advance_decline_ratio=1.0,
            advance_decline_line=0.0,
            breadth_thrust=False,
            breadth_divergence=False,
            market_participation="MIXED",
            strength_score=50.0
        )

class SectorRotationAnalyzer:
    """Analyze sector rotation patterns""""
    
    def __init__(self, sectors: List[str]):
        self.sectors = sectors
        self.sector_cache = {}
        logger.info(f"🔄 Sector Rotation Analyzer initialized with {len(sectors)} sectors")
    
    async def analyze_sector_rotation(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, SectorData]:
        """Analyze sector rotation patterns""""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            sector_data = {}
            
            for i, sector in enumerate(self.sectors):
                # Generate sector performance based on market conditions and sector characteristics
                sector_performance = self._generate_sector_performance(sector, change_percent)
                
                sector_data[sector] = SectorData(
                    sector=sector,
                    performance_1d=sector_performance['1d'],
                    performance_5d=sector_performance['5d'],
                    performance_1m=sector_performance['1m'],
                    relative_strength=sector_performance['relative_strength'],
                    momentum_score=sector_performance['momentum_score'],
                    rotation_signal=sector_performance['rotation_signal'],
                    leadership_rank=i + 1,  # Will be reranked
                    volume_ratio=sector_performance['volume_ratio'],
                    institutional_activity=sector_performance['institutional_activity']
                )
            
            # Rank sectors by relative strength
            sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1].relative_strength, reverse=True)
            for rank, (sector_name, sector_obj) in enumerate(sorted_sectors, 1):
                sector_obj.leadership_rank = rank
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Sector rotation analysis error: {e}")
            return {}
    
    def _generate_sector_performance(self, sector: str, market_change: float) -> Dict[str, Any]:
        """Generate realistic sector performance data""""
        try:
            # Sector characteristics and typical behavior
            sector_profiles = {
                'Technology': {'beta': 1.3, 'cyclical': True, 'defensive': False},
                'Healthcare': {'beta': 0.9, 'cyclical': False, 'defensive': True},
                'Financials': {'beta': 1.2, 'cyclical': True, 'defensive': False},
                'Consumer Discretionary': {'beta': 1.1, 'cyclical': True, 'defensive': False},
                'Communication Services': {'beta': 1.0, 'cyclical': False, 'defensive': False},
                'Industrials': {'beta': 1.1, 'cyclical': True, 'defensive': False},
                'Consumer Staples': {'beta': 0.7, 'cyclical': False, 'defensive': True},
                'Energy': {'beta': 1.4, 'cyclical': True, 'defensive': False},
                'Utilities': {'beta': 0.6, 'cyclical': False, 'defensive': True},
                'Real Estate': {'beta': 1.0, 'cyclical': True, 'defensive': False},
                'Materials': {'beta': 1.2, 'cyclical': True, 'defensive': False}
            }
            
            profile = sector_profiles.get(sector, {'beta': 1.0, 'cyclical': False, 'defensive': False})
            
            # Calculate 1-day performance (correlated with market but with sector beta)
            base_performance = market_change * profile['beta']
            noise = random.gauss(0, 0.5)  # Sector-specific noise
            performance_1d = base_performance + noise
            
            # 5-day and 1-month performance (with trend persistence)
            trend_factor = random.uniform(0.8, 1.2)
            performance_5d = performance_1d * 3 * trend_factor + random.gauss(0, 2)
            performance_1m = performance_1d * 8 * trend_factor + random.gauss(0, 4)
            
            # Relative strength vs market
            market_1d = market_change
            relative_strength = performance_1d - market_1d
            
            # Momentum score (0-100)
            momentum_components = [
                performance_1d * 10,  # Short-term momentum
                performance_5d * 2,   # Medium-term momentum
                performance_1m        # Long-term momentum
            ]
            momentum_score = 50 + np.mean(momentum_components)
            momentum_score = max(0, min(100, momentum_score))
            
            # Rotation signal
            if relative_strength > 2 and momentum_score > 60:
                rotation_signal = "INFLOW""
            elif relative_strength < -2 and momentum_score < 40:
                rotation_signal = "OUTFLOW""
            else:
                rotation_signal = "NEUTRAL""
            
            # Volume ratio (institutional activity proxy)
            volume_ratio = random.uniform(0.8, 1.5)
            if abs(performance_1d) > 2:  # Big moves = higher volume
                volume_ratio *= random.uniform(1.2, 2.0)
            
            # Institutional activity
            if volume_ratio > 1.3 and abs(relative_strength) > 1:
                institutional_activity = "HIGH""
            elif volume_ratio < 0.9:
                institutional_activity = "LOW""
            else:
                institutional_activity = "MODERATE""
            
            return {
                '1d': round(performance_1d, 2),
                '5d': round(performance_5d, 2),
                '1m': round(performance_1m, 2),
                'relative_strength': round(relative_strength, 2),
                'momentum_score': round(momentum_score, 1),
                'rotation_signal': rotation_signal,
                'volume_ratio': round(volume_ratio, 2),
                'institutional_activity': institutional_activity
            }
            
        except Exception as e:
            logger.error(f"Sector performance generation error for {sector}: {e}")
            return {
                '1d': 0.0, '5d': 0.0, '1m': 0.0, 'relative_strength': 0.0,
                'momentum_score': 50.0, 'rotation_signal': 'NEUTRAL',
                'volume_ratio': 1.0, 'institutional_activity': 'MODERATE'"
            }

class InstitutionalFlowAnalyzer:
    """Analyze institutional money flow patterns""""
    
    def __init__(self):
        logger.info("🏦 Institutional Flow Analyzer initialized")
    
    async def analyze_flow(self, symbol: str, quote_data: Dict[str, Any], 
                          sector_rotation: Dict[str, SectorData]) -> Dict[str, Any]:
        """Analyze institutional money flow""""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Estimate institutional activity based on volume and price action
            avg_volume = 25000000  # Rough average
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            institutional_sentiment = "NEUTRAL""
            flow_strength = "MODERATE""
            
            if volume_ratio > 1.5:
                if abs(change_percent) < 1:  # High volume, small move
                    if change_percent > 0:
                        institutional_sentiment = "ACCUMULATING""
                        flow_strength = "STRONG""
                    else:
                        institutional_sentiment = "DISTRIBUTING""
                        flow_strength = "STRONG""
                elif abs(change_percent) > 2:  # High volume, big move
                    institutional_sentiment = "MOMENTUM" if change_percent > 0 else "SELLING""
                    flow_strength = "VERY_STRONG""
            
            # Aggregate sector institutional activity
            high_inst_activity_sectors = sum(1 for s in sector_rotation.values() 
                                           if s.institutional_activity == "HIGH"):
            
            sector_flow_bias = "BULLISH" if high_inst_activity_sectors >= 4 else "BEARISH" if high_inst_activity_sectors <= 2 else "NEUTRAL""
            
            # Dark pool activity (simulated)
            dark_pool_ratio = random.uniform(0.25, 0.45)  # 25-45% of volume
            
            return {
                'sentiment': institutional_sentiment,
                'flow_strength': flow_strength,
                'volume_ratio': round(volume_ratio, 2),
                'sector_bias': sector_flow_bias,
                'dark_pool_ratio': round(dark_pool_ratio, 3),
                'active_sectors': high_inst_activity_sectors,
                'flow_direction': "INFLOW" if institutional_sentiment in ["ACCUMULATING", "MOMENTUM"] else "OUTFLOW" if institutional_sentiment in ["DISTRIBUTING", "SELLING"] else "NEUTRAL""
            }
            
        except Exception as e:
            logger.error(f"Institutional flow analysis error: {e}")
            return {'sentiment': 'NEUTRAL', 'flow_strength': 'MODERATE'}

class MarketMicrostructureAnalyzer:
    """Analyze market microstructure patterns""""
    
    def __init__(self):
        logger.info("🔬 Market Microstructure Analyzer initialized")
    
    async def analyze_microstructure(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market microstructure indicators""""
        try:
            current_price = float(quote_data.get('price', 100))
            volume = quote_data.get('volume', 0)
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Simulate bid-ask spread analysis
            spread_bps = self._estimate_bid_ask_spread(symbol, volume, abs(change_percent))
            
            # Order flow imbalance (simulated)
            order_flow_imbalance = self._calculate_order_flow_imbalance(change_percent, volume)
            
            # Market impact analysis
            market_impact = self._estimate_market_impact(volume, change_percent)
            
            # Liquidity assessment
            liquidity_score = self._assess_liquidity(symbol, volume, spread_bps)
            
            # Price discovery quality
            price_discovery = self._assess_price_discovery(change_percent, volume, spread_bps)
            
            return {
                'bid_ask_spread_bps': spread_bps,
                'order_flow_imbalance': order_flow_imbalance,
                'market_impact_score': market_impact,
                'liquidity_score': liquidity_score,
                'price_discovery_quality': price_discovery,
                'microstructure_health': self._assess_microstructure_health(spread_bps, liquidity_score, price_discovery)
            }
            
        except Exception as e:
            logger.error(f"Microstructure analysis error: {e}")
            return {'microstructure_health': 'UNKNOWN'}
    
    def _estimate_bid_ask_spread(self, symbol: str, volume: int, volatility: float) -> float:
        """Estimate bid-ask spread in basis points""""
        base_spreads = {'SPY': 1.0, 'QQQ': 1.2, 'AAPL': 2.0, 'MSFT': 1.8, 'NVDA': 3.0}
        base_spread = base_spreads.get(symbol, 2.5)
        
        volume_factor = max(0.5, min(2.0, 1.0 - np.log(volume / 10000000) * 0.1))
        volatility_factor = 1.0 + (volatility * 0.1)
        
        return round(base_spread * volume_factor * volatility_factor, 1)
    
    def _calculate_order_flow_imbalance(self, change_percent: float, volume: int) -> float:
        """Calculate order flow imbalance""""
        base_imbalance = change_percent * 0.1
        volume_multiplier = min(2.0, volume / 20000000)
        imbalance = base_imbalance * volume_multiplier
        return round(max(-1.0, min(1.0, imbalance)), 3)
    
    def _estimate_market_impact(self, volume: int, change_percent: float) -> float:
        """Estimate market impact score""""
        if volume == 0:
            return 0.5
        
        volume_ratio = volume / 25000000
        price_impact = abs(change_percent) / max(0.1, volume_ratio)
        return round(min(1.0, price_impact / 10), 3)
    
    def _assess_liquidity(self, symbol: str, volume: int, spread_bps: float) -> float:
        """Assess overall liquidity score (0-100)""""
        base_liquidity = {'SPY': 95, 'QQQ': 90, 'AAPL': 85, 'MSFT': 85, 'NVDA': 75}
        base_score = base_liquidity.get(symbol, 70)
        
        volume_score = min(20, volume / 1000000)
        spread_penalty = max(0, (spread_bps - 2) * 2)
        
        total_score = base_score + volume_score - spread_penalty
        return round(max(0, min(100, total_score)), 1)
    
    def _assess_price_discovery(self, change_percent: float, volume: int, spread_bps: float) -> str:
        """Assess price discovery quality""""
        volume_quality = "GOOD" if volume > 15000000 else "MODERATE" if volume > 5000000 else "POOR""
        spread_quality = "GOOD" if spread_bps < 3 else "MODERATE" if spread_bps < 6 else "POOR""
        movement_quality = "GOOD" if abs(change_percent) < 3 else "MODERATE" if abs(change_percent) < 6 else "POOR""
        
        quality_scores = [volume_quality, spread_quality, movement_quality]
        good_count = quality_scores.count("GOOD")
        poor_count = quality_scores.count("POOR")
        
        if good_count >= 2 and poor_count == 0:
            return "EXCELLENT""
        elif good_count >= 1 and poor_count <= 1:
            return "GOOD""
        elif poor_count <= 1:
            return "MODERATE""
        else:
            return "POOR""
    
    def _assess_microstructure_health(self, spread_bps: float, liquidity_score: float, price_discovery: str) -> str:
        """Assess overall microstructure health""""
        health_score = 0
        
        if spread_bps < 2:
            health_score += 30
        elif spread_bps < 4:
            health_score += 20
        else:
            health_score += 10
        
        if liquidity_score > 80:
            health_score += 40
        elif liquidity_score > 60:
            health_score += 25
        else:
            health_score += 10
        
        discovery_scores = {"EXCELLENT": 30, "GOOD": 20, "MODERATE": 10, "POOR": 0}
        health_score += discovery_scores.get(price_discovery, 0)
        
        if health_score >= 80:
            return "EXCELLENT""
        elif health_score >= 60:
            return "GOOD""
        elif health_score >= 40:
            return "MODERATE""
        else:
            return "POOR""

# Export main classes
__all__ = [
    'AdvancedMarketStructureAnalyzer', 'MarketStructureAnalysis', 'MarketStructureSignal',
    'BreadthAnalyzer', 'SectorRotationAnalyzer', 'InstitutionalFlowAnalyzer',
    'MarketMicrostructureAnalyzer', 'SectorData', 'BreadthSignal'"
]

logger.info("🏗️ Advanced Market Structure Analysis module loaded successfully")
logger.info("🔬 Including market microstructure analysis")
logger.info("🏦 Enhanced institutional flow tracking")
logger.info("📊 Complete breadth and sector rotation analysis")