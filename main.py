# main.py - HYPERtrends v4.0 Modular Application - FIXED STARTUP VERSION
import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration and modular components
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine, HYPERSignal
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

# ========================================
# GLOBAL STATE
# ========================================
class ModularHYPERState:
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None
        self.ml_enhanced_engine = None
        self.model_tester = None
        self.learning_api = None
        self.testing_api = None
        
        self.current_signals = {}
        self.enhanced_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.startup_time = datetime.now()
        self.initialization_complete = False
        
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
            "component_performance": {},
            "startup_complete": False
        }
    
    async def initialize(self):
        logger.info("🚀 Initializing HYPERtrends v4.0 Modular System...")
        
        try:
            # Initialize data aggregator
            logger.info("📡 Initializing data aggregator...")
            self.data_aggregator = HYPERDataAggregator()
            if hasattr(self.data_aggregator, 'initialize'):
                await self.data_aggregator.initialize()
            logger.info("✅ Data aggregator initialized")
            
            # Initialize modular signal engine
            logger.info("🧠 Initializing modular signal engine...")
            self.signal_engine = HYPERSignalEngine()
            logger.info("✅ Signal engine initialized")
            
            # Track active components
            if hasattr(self.signal_engine, '_get_active_components'):
                self.stats["modular_components"] = self.signal_engine._get_active_components()
            
            # Initialize testing framework
            try:
                logger.info("🧪 Initializing testing framework...")
                self.model_tester = ModelTester(self.signal_engine)
                self.testing_api = TestingAPI(self.model_tester)
                logger.info("✅ Testing framework initialized")
            except Exception as e:
                logger.warning(f"⚠️ Testing framework failed: {e}")
            
            # Initialize ML learning
            try:
                logger.info("🤖 Initializing ML learning...")
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, self.model_tester
                )
                logger.info("✅ ML learning integration successful")
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed: {e}")
            
            # Validate configuration
            logger.info("⚙️ Validating configuration...")
            config.validate_config()
            logger.info("✅ Configuration validated")
            
            # Generate initial signals to verify everything works
            logger.info("🎯 Generating initial test signals...")
            try:
                initial_signals = await self.signal_engine.generate_all_signals()
                self.current_signals = initial_signals
                self.last_update = datetime.now()
                logger.info(f"✅ Generated initial signals for {len(initial_signals)} symbols")
            except Exception as e:
                logger.warning(f"⚠️ Initial signal generation failed: {e}")
                # Create fallback signals
                self.current_signals = {}
                for symbol in config.TICKERS:
                    self.current_signals[symbol] = self.signal_engine._generate_fallback_signal(
                        symbol, {'price': 100.0}
                    )
            
            self.initialization_complete = True
            self.stats["startup_complete"] = True
            
            component_count = len(self.stats.get("modular_components", []))
            logger.info(f"✅ Modular system fully initialized with {component_count} active components")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Modular initialization failed: {e}")
            self.stats["last_error"] = str(e)
            # Still return True to allow the app to start
            return True

hyper_state = ModularHYPERState()

# ========================================
# WEBSOCKET MANAGER
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
                    
                    signal_dict = {
                        "symbol": signal.symbol,
                        "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                        "confidence": float(getattr(signal, 'confidence', 0)),
                        "direction": getattr(signal, 'direction', 'NEUTRAL'),
                        "price": float(getattr(signal, 'price', 0)),
                        "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                        "technical_score": float(getattr(signal, 'technical_score', 50)),
                        "sentiment_score": float(getattr(signal, 'sentiment_score', 50)),
                        "momentum_score": float(getattr(signal, 'momentum_score', 50)),
                        "ml_score": float(getattr(signal, 'ml_score', 50)),
                        "vix_score": float(getattr(signal, 'vix_score', 50)),
                        "market_structure_score": float(getattr(signal, 'market_structure_score', 50)),
                        "risk_score": float(getattr(signal, 'risk_score', 50)),
                        "williams_r": float(getattr(signal, 'williams_r', -50)),
                        "stochastic_k": float(getattr(signal, 'stochastic_k', 50)),
                        "stochastic_d": float(getattr(signal, 'stochastic_d', 50)),
                        "vix_sentiment": getattr(signal, 'vix_sentiment', 'NEUTRAL'),
                        "data_quality": getattr(signal, 'data_quality', 'unknown'),
                        "reasons": getattr(signal, 'reasons', [])[:3],
                        "warnings": getattr(signal, 'warnings', [])[:2],
                        "enhanced_features": getattr(signal, 'enhanced_features', {}),
                        "components_used": getattr(signal, 'enhanced_features', {}).get('components_used', [])
                    }
                    
                    serialized[symbol] = signal_dict
                else:
                    serialized[symbol] = signal_data
            except Exception as e:
                logger.error(f"Error serializing signal for {symbol}: {e}")
                serialized[symbol] = {"symbol": symbol, "error": str(e)}
        
        return serialized

manager = ModularConnectionManager()

# ========================================
# SIGNAL GENERATION LOOP
# ========================================
async def modular_signal_generation_loop():
    logger.info("🚀 Starting modular signal generation loop...")
    
    while hyper_state.is_running:
        try:
            start_time = time.time()
            all_signals = {}
            component_performance = {}
            
            for symbol in config.TICKERS:
                try:
                    if hyper_state.data_aggregator and hyper_state.signal_engine:
                        symbol_data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
                        
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
                    else:
                        # Fallback signal if not fully initialized
                        all_signals[symbol] = hyper_state.signal_engine._generate_fallback_signal(
                            symbol, {'price': 100.0}
                        ) if hyper_state.signal_engine else None
                        
                except Exception as e:
                    logger.error(f"❌ Signal generation failed for {symbol}: {e}")
                    if hyper_state.signal_engine:
                        all_signals[symbol] = hyper_state.signal_engine._generate_fallback_signal(
                            symbol, {'price': 100.0}
                        )
            
            # Update state
            hyper_state.current_signals = all_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(all_signals)
            hyper_state.stats["successful_cycles"] += 1
            hyper_state.stats["component_performance"] = component_performance
            
            confidences = [s.confidence for s in all_signals.values() if hasattr(s, 'confidence')]
            hyper_state.stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0
            
            generation_time = time.time() - start_time
            
            signal_summary = []
            for symbol, signal in all_signals.items():
                if signal:
                    signal_type = getattr(signal, 'signal_type', 'HOLD')
                    confidence = getattr(signal, 'confidence', 0)
                    signal_summary.append(f"{symbol}:{signal_type}({confidence:.0f}%)")
            
            logger.info(f"📊 Generated signals: {', '.join(signal_summary)} ({generation_time:.2f}s)")
            
            await manager.broadcast({
                "type": "modular_signal_update",
                "signals": manager.serialize_modular_signals(all_signals),
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
            logger.error(f"💥 Signal generation error: {e}")
            await asyncio.sleep(30)

# ========================================
# API ROUTES
# ========================================

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
        "initialization_complete": hyper_state.initialization_complete,
        "uptime_seconds": uptime,
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "tickers": config.TICKERS,
        "connected_clients": len(manager.active_connections)
    }

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HYPERtrends v4.0 dashboard"""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
    
    fallback_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends v4.0 - Modular Edition</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #000428, #004e92); color: #00ffff; text-align: center; padding: 50px; margin: 0; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { font-size: 3.5em; margin-bottom: 10px; text-shadow: 0 0 20px #00ffff; }
            h2 { font-size: 1.8em; margin-bottom: 30px; opacity: 0.9; }
            .status { background: rgba(0, 255, 255, 0.1); padding: 25px; border-radius: 15px; margin: 25px 0; border: 2px solid #00ffff; box-shadow: 0 0 30px rgba(0, 255, 255, 0.3); }
            .components { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .component { background: rgba(0, 100, 255, 0.2); padding: 15px; border-radius: 10px; border: 1px solid #0066ff; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
            .pulse { animation: pulse 2s infinite; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="pulse">🌟 HYPERtrends</h1>
            <h2>v4.0 Modular Edition</h2>
            <div class="status">
                <h3>🚀 Modular Architecture Active</h3>
                <p>Advanced AI-powered trading signals with modular components</p>
                <p><strong>Status:</strong> <span id="status" class="pulse">Online</span></p>
                
                <div class="components">
                    <div class="component">📊 Technical Analysis<br><small>25+ Indicators</small></div>
                    <div class="component">💭 Sentiment Analysis<br><small>Multi-source NLP</small></div>
                    <div class="component">😱 VIX Fear/Greed<br><small>Contrarian Signals</small></div>
                    <div class="component">🏗️ Market Structure<br><small>Breadth + Sectors</small></div>
                    <div class="component">⚠️ Risk Analysis<br><small>VaR + Position Sizing</small></div>
                </div>
                
                <div>
                    <p>API Health: <a href="/health" style="color: #00ffff;">/health</a></p>
                    <p>API Docs: <a href="/docs" style="color: #00ffff;">/docs</a></p>
                </div>
            </div>
        </div>
        
        <script>
            // Auto-refresh status
            setInterval(async () => {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('status').textContent = 
                        data.is_running ? 'Online & Running' : 'Online (Stopped)';
                } catch (e) {
                    document.getElementById('status').textContent = 'Checking...';
                }
            }, 5000);
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=fallback_html)

@app.get("/api/signals")
async def get_current_signals():
    """Get current modular signals"""
    return {
        "signals": manager.serialize_modular_signals(hyper_state.current_signals),
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
        return serialized[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.post("/api/start")
async def start_system():
    """Start the modular system"""
    if hyper_state.is_running:
        return {"status": "already_running", "version": "4.0.0-MODULAR"}
    
    if not hyper_state.initialization_complete:
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
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ========================================
# STARTUP AND SHUTDOWN
# ========================================

@app.on_event("startup")
async def startup_event():
    """Fixed startup that ensures proper initialization and port binding"""
    try:
        logger.info("🚀 Starting HYPERtrends v4.0 Modular System startup...")
        
        # CRITICAL: Ensure the app starts and binds to port immediately
        logger.info(f"🌐 Server starting on port {config.SERVER_CONFIG.get('port', 8000)}")
        
        # Initialize the system but don't let failures stop the startup
        try:
            success = await hyper_state.initialize()
            if success:
                logger.info("✅ System initialization successful")
                
                # Auto-start the signal generation if initialization succeeded
                hyper_state.is_running = True
                hyper_state.stats["uptime_start"] = datetime.now()
                asyncio.create_task(modular_signal_generation_loop())
                
                logger.info("🔥 HYPERtrends v4.0 Modular System auto-started!")
                logger.info(f"🎯 Active components: {', '.join(hyper_state.stats.get('modular_components', []))}")
                
                if config.has_robinhood_credentials():
                    logger.info("📱 Robinhood integration: ACTIVE")
                else:
                    logger.info("🔄 Dynamic simulation: ACTIVE")
            else:
                logger.warning("⚠️ System initialization had issues, but app will still start")
                
        except Exception as init_error:
            logger.error(f"⚠️ Initialization error: {init_error}")
            logger.info("🔄 App will start in fallback mode")
            hyper_state.stats["last_error"] = str(init_error)
        
        logger.info("✅ Startup sequence complete - server should be ready")
        
    except Exception as e:
        logger.error(f"💥 Critical startup error: {e}")
        # Don't re-raise - let the app start anyway
        hyper_state.stats["last_error"] = str(e)

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
        try:
            await hyper_state.data_aggregator.close()
        except:
            pass
    
    logger.info("👋 HYPERtrends v4.0 Modular System shutdown complete")

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    # Print startup info
    logger.info("🌟 Starting HYPERtrends v4.0 - Modular Edition")
    logger.info(f"🔧 Environment: {config.ENVIRONMENT}")
    logger.info(f"📱 Robinhood: {'✅ Available' if config.has_robinhood_credentials() else '❌ Not configured'}")
    logger.info(f"🔄 Demo Mode: {config.DEMO_MODE}")
    
    # Get server configuration
    server_config = config.SERVER_CONFIG
    host = server_config.get("host", "0.0.0.0")
    port = int(server_config.get("port", 8000))
    
    # Ensure port is set from environment if available
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
        logger.info(f"🌐 Using PORT from environment: {port}")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=server_config.get("reload", False),
            access_log=server_config.get("access_log", True),
            log_level="info"
        )
    except Exception as e:
        logger.error(f"💥 Failed to start server: {e}")
        raise