# ===== main_py_chunk1.py =====
# main.py - HYPERtrends v4.0 Modular Application - COMPLETE VERSION
import os
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import configuration and modular components
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine, HYPERSignal  # Updated modular engine
from model_testing import ModelTester, TestingAPI
from ml_learning import integrate_ml_learning, MLEnhancedSignalEngine, LearningAPI

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG.get("level", "INFO")),
    format=config.LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="🌟 HYPERtrends v4.0 - Modular Edition",
    description="AI-powered trading signals with advanced modular architecture",
    version="4.0.0-MODULAR"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving index.html and assets)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ========================================
# GLOBAL STATE - Enhanced for Modular Architecture
# ========================================
class ModularHYPERState:
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None  # Now modular!
        self.ml_enhanced_engine = None
        self.model_tester = None
        self.learning_api = None
        self.testing_api = None
        
        self.current_signals = {}
        self.enhanced_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.startup_time = datetime.now()
        
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "successful_cycles": 0,
            "errors_encountered": 0,
            "average_confidence": 0.0,
            "last_error": None,
            "data_source_status": config.get_data_source_status(),
            "robinhood_available": config.has_robinhood_credentials(),
            "modular_components": [],
            "component_performance": {}
        }
    
    async def initialize(self):
        logger.info("🚀 Initializing HYPERtrends v4.0 Modular System...")
        
        try:
            # Initialize data aggregator
            self.data_aggregator = HYPERDataAggregator()
            if hasattr(self.data_aggregator, 'initialize'):
                await self.data_aggregator.initialize()
            
            # Initialize modular signal engine
            self.signal_engine = HYPERSignalEngine()
            
            # Track active components
            if hasattr(self.signal_engine, '_get_active_components'):
                self.stats["modular_components"] = self.signal_engine._get_active_components()
            
            # Initialize testing framework
            try:
                self.model_tester = ModelTester(self.signal_engine)
                self.testing_api = TestingAPI(self.model_tester)
                logger.info("✅ Testing framework initialized")
            except Exception as e:
                logger.warning(f"⚠️ Testing framework failed: {e}")
            
            # Initialize ML learning
            try:
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, self.model_tester
                )
                logger.info("✅ ML learning integration successful")
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed: {e}")
            
            # Validate configuration
            config.validate_config()
            
            # Log initialization status
            component_count = len(self.stats.get("modular_components", []))
            logger.info(f"✅ Modular system initialized with {component_count} active components")
            
            if config.has_robinhood_credentials():
                logger.info("✅ Robinhood credentials available - real data mode")
            else:
                logger.info("ℹ️ Using dynamic simulation mode")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Modular initialization failed: {e}")
            return False

hyper_state = ModularHYPERState()

# ===== main_py_chunk2.py =====
# ========================================
# WEBSOCKET MANAGER - Enhanced for Modular Data
# ========================================
class ModularConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        hyper_state.stats["clients_connected"] = len(self.active_connections)
        logger.info(f"Client connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.stats["clients_connected"] = len(self.active_connections)
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        message_json = json.dumps(message, default=str)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def serialize_modular_signals(self, signals):
        """Enhanced serialization for modular signals"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal_data in signals.items():
            try:
                if hasattr(signal_data, '__dict__'):
                    signal = signal_data
                    
                    # Basic signal data
                    signal_dict = {
                        "symbol": signal.symbol,
                        "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                        "confidence": float(getattr(signal, 'confidence', 0)),
                        "direction": getattr(signal, 'direction', 'NEUTRAL'),
                        "price": float(getattr(signal, 'price', 0)),
                        "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                        
                        # Component scores
                        "technical_score": float(getattr(signal, 'technical_score', 50)),
                        "sentiment_score": float(getattr(signal, 'sentiment_score', 50)),
                        "momentum_score": float(getattr(signal, 'momentum_score', 50)),
                        "ml_score": float(getattr(signal, 'ml_score', 50)),
                        "vix_score": float(getattr(signal, 'vix_score', 50)),
                        "market_structure_score": float(getattr(signal, 'market_structure_score', 50)),
                        "risk_score": float(getattr(signal, 'risk_score', 50)),
                        
                        # Enhanced indicators
                        "williams_r": float(getattr(signal, 'williams_r', -50)),
                        "stochastic_k": float(getattr(signal, 'stochastic_k', 50)),
                        "stochastic_d": float(getattr(signal, 'stochastic_d', 50)),
                        "vix_sentiment": getattr(signal, 'vix_sentiment', 'NEUTRAL'),
                        
                        # Supporting data
                        "data_quality": getattr(signal, 'data_quality', 'unknown'),
                        "reasons": getattr(signal, 'reasons', [])[:3],  # Top 3 reasons
                        "warnings": getattr(signal, 'warnings', [])[:2],  # Top 2 warnings
                        
                        # Modular insights
                        "enhanced_features": getattr(signal, 'enhanced_features', {}),
                        "components_used": getattr(signal, 'enhanced_features', {}).get('components_used', [])
                    }
                    
                    # Add component analysis summaries (without full objects to reduce payload)
                    if hasattr(signal, 'technical_analysis') and signal.technical_analysis:
                        signal_dict["technical_summary"] = {
                            "overall_score": signal.technical_analysis.overall_score,
                            "direction": signal.technical_analysis.direction,
                            "key_levels": signal.technical_analysis.key_levels
                        }
                    
                    if hasattr(signal, 'sentiment_analysis') and signal.sentiment_analysis:
                        signal_dict["sentiment_summary"] = {
                            "overall_sentiment": signal.sentiment_analysis.overall_sentiment,
                            "retail_sentiment": signal.sentiment_analysis.retail_sentiment,
                            "fear_greed_indicator": signal.sentiment_analysis.fear_greed_indicator
                        }
                    
                    if hasattr(signal, 'vix_analysis') and signal.vix_analysis:
                        signal_dict["vix_summary"] = {
                            "vix_level": signal.vix_analysis.current_signal.vix_level,
                            "sentiment": signal.vix_analysis.current_signal.sentiment,
                            "contrarian_signal": signal.vix_analysis.current_signal.contrarian_signal
                        }
                    
                    if hasattr(signal, 'market_structure_analysis') and signal.market_structure_analysis:
                        signal_dict["market_structure_summary"] = {
                            "market_regime": signal.market_structure_analysis.current_signal.market_regime,
                            "structure_score": signal.market_structure_analysis.current_signal.structure_score,
                            "rotation_theme": signal.market_structure_analysis.current_signal.rotation_theme
                        }
                    
                    if hasattr(signal, 'risk_analysis') and signal.risk_analysis:
                        signal_dict["risk_summary"] = {
                            "risk_level": signal.risk_analysis.risk_level,
                            "overall_risk_score": signal.risk_analysis.overall_risk_score,
                            "position_size_recommendation": signal.risk_analysis.position_risk.position_size_recommendation
                        }
                    
                    serialized[symbol] = signal_dict
                else:
                    serialized[symbol] = signal_data
            except Exception as e:
                logger.error(f"Error serializing modular signal for {symbol}: {e}")
                serialized[symbol] = {"symbol": symbol, "error": str(e)}
        
        return serialized

manager = ModularConnectionManager()

# ===== main_py_chunk3.py =====
# ========================================
# ENHANCED SIGNAL GENERATION LOOP
# ========================================
async def modular_signal_generation_loop():
    logger.info("🚀 Starting modular signal generation loop...")
    
    while hyper_state.is_running:
        try:
            start_time = time.time()
            
            # Get comprehensive data for all symbols
            all_signals = {}
            component_performance = {}
            
            for symbol in config.TICKERS:
                try:
                    # Get comprehensive data
                    symbol_data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
                    
                    # Generate modular signal
                    signal_start = time.time()
                    signal = await hyper_state.signal_engine.generate_signal(
                        symbol, 
                        symbol_data.get('quote', {}),
                        symbol_data.get('trends', {}),
                        symbol_data.get('historical_data', [])
                    )
                    signal_time = time.time() - signal_start
                    
                    all_signals[symbol] = signal
                    component_performance[symbol] = {
                        'generation_time': signal_time,
                        'components_used': signal.enhanced_features.get('components_used', []),
                        'data_quality': signal.data_quality
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Signal generation failed for {symbol}: {e}")
                    # Generate fallback signal
                    all_signals[symbol] = hyper_state.signal_engine._generate_fallback_signal(
                        symbol, {'price': 100.0}
                    )
            
            # Enhanced signals (ML if available)
            enhanced_signals = {}
            if hyper_state.ml_enhanced_engine:
                try:
                    for symbol, base_signal in all_signals.items():
                        enhanced_signal = await hyper_state.ml_enhanced_engine.enhanced_signal_generation(symbol)
                        enhanced_signals[symbol] = enhanced_signal
                except Exception as e:
                    logger.error(f"ML enhancement failed: {e}")
            
            # Update state
            hyper_state.current_signals = all_signals
            hyper_state.enhanced_signals = enhanced_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(all_signals)
            hyper_state.stats["successful_cycles"] += 1
            hyper_state.stats["component_performance"] = component_performance
            
            # Calculate average confidence
            confidences = [s.confidence for s in all_signals.values() if hasattr(s, 'confidence')]
            hyper_state.stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0
            
            generation_time = time.time() - start_time
            
            # Create enhanced signal summary
            signal_summary = []
            for symbol, signal in all_signals.items():
                signal_type = getattr(signal, 'signal_type', 'HOLD')
                confidence = getattr(signal, 'confidence', 0)
                components = len(getattr(signal, 'enhanced_features', {}).get('components_used', []))
                signal_summary.append(f"{symbol}:{signal_type}({confidence:.0f}%,{components}c)")
            
            logger.info(f"📊 Generated modular signals: {', '.join(signal_summary)} ({generation_time:.2f}s)")
            
            # Broadcast enhanced update
            await manager.broadcast({
                "type": "modular_signal_update",
                "signals": manager.serialize_modular_signals(all_signals),
                "enhanced_signals": manager.serialize_modular_signals(enhanced_signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(),
                "generation_time": generation_time,
                "modular_info": {
                    "active_components": hyper_state.stats.get("modular_components", []),
                    "component_performance": component_performance,
                    "data_source_status": config.get_data_source_status(),
                    "version": "4.0.0-MODULAR"
                }
            })
            
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            hyper_state.stats["errors_encountered"] += 1
            hyper_state.stats["last_error"] = str(e)
            logger.error(f"💥 Modular signal generation error: {e}")
            await asyncio.sleep(30)

# ========================================
# ENHANCED API ROUTES
# ========================================

@app.get("/health/modular")
async def get_modular_health():
    """Enhanced health check for modular system"""
    symbol = "QQQ"
    logger.info("🔍 /health/modular check triggered")
    try:
        data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
        
        # Test modular signal generation
        signal = await hyper_state.signal_engine.generate_signal(
            symbol, data.get("quote", {}), data.get("trends", {})
        )
        
        return {
            "status": "online",
            "version": "4.0.0-MODULAR",
            "symbol": symbol,
            "modular_components": hyper_state.stats.get("modular_components", []),
            "component_count": len(hyper_state.stats.get("modular_components", [])),
            "signal_generated": {
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "components_used": signal.enhanced_features.get('components_used', []),
                "data_quality": signal.data_quality
            },
            "data_source": {
                "status": config.get_data_source_status(),
                "robinhood_available": config.has_robinhood_credentials(),
                "source": data.get("quote", {}).get("data_source", "unknown")
            },
            "performance": {
                "generation_time": signal.enhanced_features.get('generation_time', 0),
                "cache_enabled": True,
                "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None
            }
        }
    except Exception as e:
        logger.error(f"❌ Modular health check failed: {e}")
        return {
            "status": "error",
            "version": "4.0.0-MODULAR",
            "error": str(e)
        }

# ===== main_py_chunk4.py =====
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HYPERtrends v4.0 dashboard"""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
    
    # Enhanced fallback HTML for v4.0
    fallback_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends v4.0 - Modular Edition</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #000428, #004e92); color: #00ffff; text-align: center; padding: 50px; margin: 0; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ font-size: 3.5em; margin-bottom: 10px; text-shadow: 0 0 20px #00ffff; }}
            h2 {{ font-size: 1.8em; margin-bottom: 30px; opacity: 0.9; }}
            .status {{ background: rgba(0, 255, 255, 0.1); padding: 25px; border-radius: 15px; margin: 25px 0; border: 2px solid #00ffff; box-shadow: 0 0 30px rgba(0, 255, 255, 0.3); }}
            .components {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .component {{ background: rgba(0, 100, 255, 0.2); padding: 15px; border-radius: 10px; border: 1px solid #0066ff; }}
            .version {{ font-size: 0.9em; opacity: 0.7; margin-top: 20px; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }}
            .pulse {{ animation: pulse 2s infinite; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="pulse">🌟 HYPERtrends</h1>
            <h2>v4.0 Modular Edition</h2>
            <div class="status">
                <h3>🚀 Modular Architecture Active</h3>
                <p>Advanced AI-powered trading signals with modular components</p>
                <p><strong>Data Source:</strong> Robinhood + Dynamic Simulation</p>
                <p><strong>Robinhood:</strong> {'✅ Available' if config.has_robinhood_credentials() else '❌ Not configured'}</p>
                <p><strong>Status:</strong> <span id="status" class="pulse">Initializing...</span></p>
                
                <div class="components">
                    <div class="component">📊 Technical Analysis<br><small>25+ Indicators</small></div>
                    <div class="component">💭 Sentiment Analysis<br><small>Multi-source NLP</small></div>
                    <div class="component">😱 VIX Fear/Greed<br><small>Contrarian Signals</small></div>
                    <div class="component">🏗️ Market Structure<br><small>Breadth + Sectors</small></div>
                    <div class="component">⚠️ Risk Analysis<br><small>VaR + Position Sizing</small></div>
                </div>
                
                <div class="version">
                    <p>Dashboard loading... Please ensure index.html is deployed correctly.</p>
                    <p>API Health: <a href="/health/modular" style="color: #00ffff;">/health/modular</a></p>
                    <p>API Docs: <a href="/docs" style="color: #00ffff;">/docs</a></p>
                </div>
            </div>
        </div>
        <script>
            fetch('/health/modular').then(r => r.json()).then(data => {{
                document.getElementById('status').textContent = data.status + ' (' + data.component_count + ' components)';
                document.getElementById('status').className = data.status === 'online' ? 'pulse' : '';
            }}).catch(e => {{
                document.getElementById('status').textContent = 'Error';
                document.getElementById('status').className = '';
            }});
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=fallback_html)

@app.get("/health")
async def health_check():
    """Standard health check with modular info"""
    uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
    
    return {
        "status": "healthy",
        "version": "4.0.0-MODULAR",
        "environment": config.ENVIRONMENT,
        "demo_mode": config.DEMO_MODE,
        "modular_architecture": {
            "enabled": True,
            "active_components": hyper_state.stats.get("modular_components", []),
            "component_count": len(hyper_state.stats.get("modular_components", [])),
            "total_possible_components": 5
        },
        "data_source": {
            "primary": "robinhood",
            "fallback": "dynamic_simulation",
            "robinhood_credentials": config.has_robinhood_credentials(),
            "status": config.get_data_source_status()
        },
        "is_running": hyper_state.is_running,
        "uptime_seconds": uptime,
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "tickers": config.TICKERS,
        "connected_clients": len(manager.active_connections)
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current modular signals"""
    return {
        "signals": manager.serialize_modular_signals(hyper_state.current_signals),
        "enhanced_signals": manager.serialize_modular_signals(hyper_state.enhanced_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "modular_info": {
            "version": "4.0.0-MODULAR",
            "active_components": hyper_state.stats.get("modular_components", []),
            "component_performance": hyper_state.stats.get("component_performance", {}),
            "data_source_status": config.get_data_source_status()
        }
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get detailed modular signal for specific symbol"""
    symbol = symbol.upper()
    if symbol not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not tracked")
    
    if symbol in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol]
        serialized = manager.serialize_modular_signals({symbol: signal})
        
        # Add detailed component analysis if available
        detailed_signal = serialized[symbol].copy()
        
        if hasattr(signal, 'technical_analysis') and signal.technical_analysis:
            detailed_signal["detailed_technical"] = {
                "signals": [{"name": s.indicator_name, "value": s.value, "direction": s.direction} 
                           for s in signal.technical_analysis.signals[:10]],
                "key_levels": signal.technical_analysis.key_levels,
                "pattern_analysis": signal.technical_analysis.pattern_analysis
            }
        
        if hasattr(signal, 'risk_analysis') and signal.risk_analysis:
            detailed_signal["detailed_risk"] = {
                "position_sizing": {
                    "recommended_size": signal.risk_analysis.position_risk.position_size_recommendation,
                    "kelly_criterion": signal.risk_analysis.position_risk.kelly_criterion,
                    "optimal_stop_loss": signal.risk_analysis.position_risk.optimal_stop_loss
                },
                "risk_warnings": signal.risk_analysis.risk_warnings[:3]
            }
        
        return detailed_signal
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.get("/api/components/status")
async def get_component_status():
    """Get status of all modular components"""
    if not hyper_state.signal_engine:
        raise HTTPException(status_code=503, detail="Signal engine not initialized")
    
    component_status = {}
    
    # Check each component
    if hasattr(hyper_state.signal_engine, 'technical_analyzer') and hyper_state.signal_engine.technical_analyzer:
        component_status["technical"] = {"enabled": True, "status": "active", "indicators": "25+"}
    else:
        component_status["technical"] = {"enabled": False, "status": "disabled"}
    
    if hasattr(hyper_state.signal_engine, 'sentiment_analyzer') and hyper_state.signal_engine.sentiment_analyzer:
        component_status["sentiment"] = {"enabled": True, "status": "active", "sources": "multi-source"}
    else:
        component_status["sentiment"] = {"enabled": False, "status": "disabled"}
    
    if hasattr(hyper_state.signal_engine, 'vix_analyzer') and hyper_state.signal_engine.vix_analyzer:
        component_status["vix"] = {"enabled": True, "status": "active", "features": "fear/greed + contrarian"}
    else:
        component_status["vix"] = {"enabled": False, "status": "disabled"}
    
    if hasattr(hyper_state.signal_engine, 'market_structure_analyzer') and hyper_state.signal_engine.market_structure_analyzer:
        component_status["market_structure"] = {"enabled": True, "status": "active", "features": "breadth + sectors"}
    else:
        component_status["market_structure"] = {"enabled": False, "status": "disabled"}
    
    if hasattr(hyper_state.signal_engine, 'risk_analyzer') and hyper_state.signal_engine.risk_analyzer:
        component_status["risk"] = {"enabled": True, "status": "active", "features": "VaR + position sizing"}
    else:
        component_status["risk"] = {"enabled": False, "status": "disabled"}
    
    return {
        "components": component_status,
        "total_enabled": sum(1 for c in component_status.values() if c.get("enabled", False)),
        "performance": hyper_state.stats.get("component_performance", {}),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None
    }

@app.post("/api/start")
async def start_system():
    """Start the modular system"""
    if hyper_state.is_running:
        return {"status": "

# ===== main_py_final.py =====
@app.post("/api/start")
async def start_system():
    """Start the modular system"""
    if hyper_state.is_running:
        return {"status": "already_running", "version": "4.0.0-MODULAR"}
    
    if not hyper_state.signal_engine:
        success = await hyper_state.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize modular system")
    
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    asyncio.create_task(modular_signal_generation_loop())
    
    return {
        "status": "started", 
        "version": "4.0.0-MODULAR",
        "message": "Modular signal generation started",
        "active_components": hyper_state.stats.get("modular_components", []),
        "data_source": config.get_data_source_status()
    }

@app.post("/api/stop")
async def stop_system():
    """Stop the modular system"""
    hyper_state.is_running = False
    return {"status": "stopped", "version": "4.0.0-MODULAR"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket with modular data"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                    "version": "4.0.0-MODULAR",
                    "active_components": hyper_state.stats.get("modular_components", []),
                    "data_source_status": config.get_data_source_status()
                }))
            elif message.get("type") == "get_component_status":
                component_status = await get_component_status()
                await websocket.send_text(json.dumps({
                    "type": "component_status",
                    "data": component_status,
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Model testing endpoints (if available)
@app.get("/api/testing/status")
async def get_testing_status():
    if hyper_state.testing_api:
        return await hyper_state.testing_api.get_test_status()
    return {"status": "unavailable", "message": "Testing framework not initialized"}

@app.post("/api/testing/backtest")
async def run_backtest(days: int = 7):
    if hyper_state.testing_api:
        return await hyper_state.testing_api.run_quick_backtest(days)
    raise HTTPException(status_code=503, detail="Testing framework not available")

# ML learning endpoints (if available)  
@app.get("/api/ml/status")
async def get_ml_status():
    if hyper_state.learning_api:
        return await hyper_state.learning_api.get_ml_status()
    return {"status": "unavailable", "message": "ML learning not initialized"}

@app.get("/api/ml/performance")
async def get_ml_performance():
    if hyper_state.learning_api:
        return await hyper_state.learning_api.get_model_performance()
    raise HTTPException(status_code=503, detail="ML learning not available")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting HYPERtrends v4.0 Modular System...")
    
    success = await hyper_state.initialize()
    if success:
        try:
            # Generate initial signals
            initial_signals = await hyper_state.signal_engine.generate_all_signals()
            hyper_state.current_signals = initial_signals
            hyper_state.last_update = datetime.now()
            
            active_components = len(hyper_state.stats.get("modular_components", []))
            logger.info(f"✅ Initial signals generated with {active_components} components")
        except Exception as e:
            logger.error(f"Failed to generate initial signals: {e}")
        
        hyper_state.is_running = True
        hyper_state.stats["uptime_start"] = datetime.now()
        asyncio.create_task(modular_signal_generation_loop())
        
        logger.info("🔥 HYPERtrends v4.0 Modular System auto-started!")
        logger.info(f"🎯 Active components: {', '.join(hyper_state.stats.get('modular_components', []))}")
        
        if config.has_robinhood_credentials():
            logger.info("📱 Robinhood integration: ACTIVE")
        else:
            logger.info("🔄 Dynamic simulation: ACTIVE")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("⏸️ Shutting down HYPERtrends v4.0 Modular System...")
    hyper_state.is_running = False
    
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, 'close'):
        await hyper_state.data_aggregator.close()
    
    logger.info("👋 HYPERtrends v4.0 Modular System shutdown complete")

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    logger.info("🌟 Starting HYPERtrends v4.0 - Modular Edition")
    logger.info(f"🔧 Environment: {config.ENVIRONMENT}")
    logger.info(f"📱 Robinhood: {'✅ Available' if config.has_robinhood_credentials() else '❌ Not configured'}")
    logger.info(f"🔄 Demo Mode: {config.DEMO_MODE}")
    
    uvicorn.run(
        app,
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG["reload"],
        access_log=config.SERVER_CONFIG["access_log"]
    )

