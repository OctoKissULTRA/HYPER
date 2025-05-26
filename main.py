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
    title="⚡ HYPER Trading System - Production",
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
                        "data_quality": getattr(signal, 'data_quality', 'unknown')
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
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HYPER Trading System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #fff; 
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { 
            font-size: 2.5em; 
            background: linear-gradient(45deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .status { 
            display: inline-block; 
            padding: 8px 16px; 
            margin: 5px; 
            border-radius: 20px; 
            font-weight: bold; 
        }
        .connected { background: #4CAF50; }
        .disconnected { background: #f44336; }
        .demo { background: #ff9800; }
        
        .signals-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .signal-card { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px; 
            padding: 20px; 
            backdrop-filter: blur(10px);
            border-left: 5px solid #ccc;
        }
        .signal-card.hyper_buy, .signal-card.soft_buy { border-left-color: #4CAF50; }
        .signal-card.hyper_sell, .signal-card.soft_sell { border-left-color: #f44336; }
        .signal-card.hold { border-left-color: #ff9800; }
        
        .symbol { font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }
        .confidence { font-size: 1.2em; margin: 10px 0; }
        .price { font-size: 1.3em; color: #00d4ff; margin: 10px 0; }
        .timestamp { color: #888; font-size: 0.8em; margin-top: 15px; }
        
        .warning { 
            background: #ff5722; 
            padding: 15px; 
            margin: 20px 0; 
            border-radius: 5px; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 HYPER Trading System</h1>
            <div id="status" class="status disconnected">Connecting...</div>
        </div>
        
        <div class="warning">
            <strong>⚠️ EDUCATIONAL PURPOSES ONLY</strong><br>
            This system provides signals for educational purposes. Not financial advice.
        </div>
        
        <div id="signals" class="signals-grid"></div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws`);
        
        ws.onopen = function() {
            document.getElementById('status').innerHTML = '✅ Connected';
            document.getElementById('status').className = 'status connected';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'signal_update') {
                displaySignals(data.signals);
            }
        };
        
        ws.onclose = function() {
            document.getElementById('status').innerHTML = '❌ Disconnected';
            document.getElementById('status').className = 'status disconnected';
        };
        
        function displaySignals(signals) {
            const container = document.getElementById('signals');
            container.innerHTML = '';
            
            for (const [symbol, signal] of Object.entries(signals)) {
                const card = document.createElement('div');
                card.className = `signal-card ${signal.signal_type.toLowerCase().replace('_', '')}`;
                
                card.innerHTML = `
                    <div class="symbol">${symbol}</div>
                    <div class="confidence">Confidence: ${signal.confidence.toFixed(1)}%</div>
                    <div class="price">$${signal.price.toFixed(2)}</div>
                    <div>Signal: ${signal.signal_type} ${signal.direction}</div>
                    <div class="timestamp">${new Date(signal.timestamp).toLocaleString()}</div>
                `;
                
                container.appendChild(card);
            }
        }
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

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

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting HYPER Trading System...")
    
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
        logger.info("🔥 HYPER system auto-started!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("⏸️ Shutting down HYPER Trading System...")
    hyper_state.is_running = False
    
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, 'close'):
        await hyper_state.data_aggregator.close()
    
    logger.info("👋 HYPER shutdown complete")

if __name__ == "__main__":
    logger.info("🚀 Starting HYPER Trading System server...")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG.get("reload", False),
        log_level="info"
    )
