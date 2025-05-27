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

# Import configuration and components
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
    title="🌟 HYPERtrends - Production",
    description="Production-grade AI-powered trading signals",
    version="3.0.0-PRODUCTION"
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
# GLOBAL STATE
# ========================================
class ProductionHYPERState:
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
        
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "successful_cycles": 0,
            "errors_encountered": 0,
            "average_confidence": 0.0,
            "last_error": None
        }
    
    async def initialize(self):
        logger.info("🚀 Initializing HYPER Trading System...")
        
        try:
            self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
            if hasattr(self.data_aggregator, 'initialize'):
                await self.data_aggregator.initialize()
            
            self.signal_engine = HYPERSignalEngine()
            
            try:
                self.model_tester = ModelTester(self.signal_engine)
                self.testing_api = TestingAPI(self.model_tester)
            except Exception as e:
                logger.warning(f"⚠️ Testing framework failed: {e}")
            
            try:
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, self.model_tester
                )
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed: {e}")
            
            config.validate_config()
            logger.info("✅ System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

hyper_state = ProductionHYPERState()

# ========================================
# WEBSOCKET MANAGER
# ========================================
class ConnectionManager:
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
    
    def serialize_signals(self, signals):
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal_data in signals.items():
            try:
                if hasattr(signal_data, '__dict__'):
                    signal = signal_data
                    serialized[symbol] = {
                        "symbol": signal.symbol,
                        "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                        "confidence": float(getattr(signal, 'confidence', 0)),
                        "direction": getattr(signal, 'direction', 'NEUTRAL'),
                        "price": float(getattr(signal, 'price', 0)),
                        "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                        "technical_score": float(getattr(signal, 'technical_score', 50)),
                        "sentiment_score": float(getattr(signal, 'sentiment_score', 50)),
                        "ml_score": float(getattr(signal, 'ml_score', 50)),
                        "williams_r": float(getattr(signal, 'williams_r', -50)),
                        "stochastic_k": float(getattr(signal, 'stochastic_k', 50)),
                        "stochastic_d": float(getattr(signal, 'stochastic_d', 50)),
                        "vix_sentiment": getattr(signal, 'vix_sentiment', 'NEUTRAL'),
                        "risk_score": float(getattr(signal, 'risk_score', 50)),
                        "data_quality": getattr(signal, 'data_quality', 'unknown'),
                        "reasons": getattr(signal, 'reasons', []),
                        "warnings": getattr(signal, 'warnings', [])
                    }
                else:
                    serialized[symbol] = signal_data
            except Exception as e:
                logger.error(f"Error serializing {symbol}: {e}")
                serialized[symbol] = {"symbol": symbol, "error": str(e)}
        
        return serialized

manager = ConnectionManager()

# ========================================
# SIGNAL GENERATION LOOP
# ========================================
async def signal_generation_loop():
    logger.info("🚀 Starting signal generation loop...")
    
    while hyper_state.is_running:
        try:
            start_time = time.time()
            
            base_signals = await hyper_state.signal_engine.generate_all_signals()
            
            enhanced_signals = {}
            if hyper_state.ml_enhanced_engine:
                try:
                    for symbol in config.TICKERS:
                        if symbol in base_signals:
                            enhanced_signal = await hyper_state.ml_enhanced_engine.enhanced_signal_generation(symbol)
                            enhanced_signals[symbol] = enhanced_signal
                except Exception as e:
                    logger.error(f"ML enhancement failed: {e}")
            
            hyper_state.current_signals = base_signals
            hyper_state.enhanced_signals = enhanced_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(base_signals)
            hyper_state.stats["successful_cycles"] += 1
            
            generation_time = time.time() - start_time
            
            signal_summary = []
            for symbol, signal in base_signals.items():
                signal_type = getattr(signal, 'signal_type', 'HOLD')
                confidence = getattr(signal, 'confidence', 0)
                signal_summary.append(f"{symbol}:{signal_type}({confidence:.0f}%)")
            
            logger.info(f"📊 Generated signals: {', '.join(signal_summary)} ({generation_time:.2f}s)")
            
            await manager.broadcast({
                "type": "signal_update",
                "signals": manager.serialize_signals(base_signals),
                "enhanced_signals": manager.serialize_signals(enhanced_signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(),
                "generation_time": generation_time
            })
            
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            hyper_state.stats["errors_encountered"] += 1
            logger.error(f"💥 Signal generation error: {e}")
            await asyncio.sleep(30)

# ========================================
# API ROUTES
# ========================================
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HYPERtrends dashboard"""
    try:
        # Try to serve from index.html file
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
    
    # Fallback minimal HTML if file not found
    fallback_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends</title>
        <style>
            body { font-family: Arial, sans-serif; background: #000; color: #00ffff; text-align: center; padding: 50px; }
            h1 { font-size: 3em; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>🌟 HYPERtrends</h1>
        <p>Dashboard loading... Please ensure index.html is deployed correctly.</p>
        <p>API Status: <span id="status">Checking...</span></p>
        <script>
            fetch('/health').then(r => r.json()).then(data => {
                document.getElementById('status').textContent = data.status;
            }).catch(e => {
                document.getElementById('status').textContent = 'Error';
            });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=fallback_html)

@app.get("/health")
async def health_check():
    uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
    
    return {
        "status": "healthy",
        "version": "3.0.0-PRODUCTION",
        "environment": config.ENVIRONMENT,
        "demo_mode": config.DEMO_MODE,
        "is_running": hyper_state.is_running,
        "uptime_seconds": uptime,
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "tickers": config.TICKERS,
        "connected_clients": len(manager.active_connections)
    }

@app.get("/api/signals")
async def get_current_signals():
    return {
        "signals": manager.serialize_signals(hyper_state.current_signals),
        "enhanced_signals": manager.serialize_signals(hyper_state.enhanced_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    symbol = symbol.upper()
    if symbol not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not tracked")
    
    if symbol in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol]
        return manager.serialize_signals({symbol: signal})[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.post("/api/start")
async def start_system():
    if hyper_state.is_running:
        return {"status": "already_running"}
    
    if not hyper_state.signal_engine:
        success = await hyper_state.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize system")
    
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    asyncio.create_task(signal_generation_loop())
    
    return {"status": "started", "message": "Signal generation started"}

@app.post("/api/stop")
async def stop_system():
    hyper_state.is_running = False
    return {"status": "stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Optional: Model testing endpoints (if available)
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

# Optional: ML learning endpoints (if available)  
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
    logger.info("🚀 Starting HYPERtrends system...")
    
    success = await hyper_state.initialize()
    if success:
        try:
            initial_signals = await hyper_state.signal_engine.generate_all_signals()
            hyper_state.current_signals = initial_signals
            hyper_state.last_update = datetime.now()
            logger.info("✅ Initial signals generated")
        except Exception as e:
            logger.error(f"Failed to generate initial signals: {e}")
        
        hyper_state.is_running = True
        hyper_state.stats["uptime_start"] = datetime.now()
        asyncio.create_task(signal_generation_loop())
        logger.info("🔥 HYPERtrends system auto-started!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("⏸️ Shutting down HYPERtrends system...")
    hyper_state.is_running = False
    
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, 'close'):
        await hyper_state.data_aggregator.close()
    
    logger.info("👋 HYPERtrends shutdown complete")

if __name__ == "__main__":
    logger.info("🚀 Starting HYPERtrends server...")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG.get("reload", False),
        log_level="info"
    )
