# main.py - HYPERtrends v4.0 - Production Alpaca Edition

import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import configuration and components

import config

# Import core modules

from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine
from ml_learning import integrate_ml_learning
from model_testing import ModelTester, TestingAPI

# ========================================

# LOGGING SETUP

# ========================================

logging.basicConfig(
level=getattr(logging, config.LOGGING_CONFIG.get(“level”, “INFO”)),
format=config.LOGGING_CONFIG.get(“format”, “%(asctime)s - %(name)s - %(levelname)s - %(message)s”)
)
logger = logging.getLogger(**name**)

# ========================================

# FASTAPI APPLICATION

# ========================================

app = FastAPI(
title=“🚀 HYPERtrends v4.0 - Alpaca Edition”,
description=“AI-powered trading signals with Alpaca Markets integration”,
version=“4.0.0-ALPACA”,
docs_url=”/docs” if config.is_development() else None,
redoc_url=”/redoc” if config.is_development() else None
)

# CORS Configuration

app.add_middleware(
CORSMiddleware,
allow_origins=config.SECURITY_CONFIG[“cors_origins”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# ========================================

# GLOBAL STATE

# ========================================

class HYPERState:
def **init**(self):
self.is_running = False
self.initialization_complete = False
self.startup_time = datetime.now()
self.connected_clients = []
self.current_signals = {}
self.last_update = None

```
    # Components (lazy loaded)
    self.data_aggregator = None
    self.signal_engine = None
    self.ml_engine = None
    self.model_tester = None
    self.testing_api = None
    
    self.stats = {
        "status": "starting",
        "uptime_start": datetime.now(),
        "alpaca_available": config.has_alpaca_credentials(),
        "data_source_status": config.get_data_source_status(),
        "initialization_complete": False,
        "signals_generated": 0,
        "ml_predictions": 0,
        "accuracy_rate": 0.0
    }
    
    logger.info("⚡ HYPER state initialized")
```

hyper_state = HYPERState()

# ========================================

# WEBSOCKET MANAGER

# ========================================

class ConnectionManager:
def **init**(self):
self.active_connections: List[WebSocket] = []

```
async def connect(self, websocket: WebSocket):
    await websocket.accept()
    self.active_connections.append(websocket)
    hyper_state.connected_clients = self.active_connections
    logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

def disconnect(self, websocket: WebSocket):
    if websocket in self.active_connections:
        self.active_connections.remove(websocket)
        hyper_state.connected_clients = self.active_connections

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
```

manager = ConnectionManager()

# ========================================

# STARTUP & SHUTDOWN

# ========================================

@app.on_event(“startup”)
async def startup_event():
“”“Fast startup with background initialization”””
try:
logger.info(“🚀 Starting HYPERtrends v4.0 - Alpaca Edition”)

```
    # Immediate port binding
    hyper_state.stats["status"] = "online"
    
    # Schedule background initialization
    asyncio.create_task(background_initialization())
    
    logger.info("✅ Server started - Background initialization in progress")
    
except Exception as e:
    logger.error(f"❌ Startup error: {e}")
    hyper_state.stats["status"] = "error"
```

async def background_initialization():
“”“Initialize components in background”””
try:
logger.info(“🔧 Starting background initialization…”)

```
    # Initialize data aggregator
    logger.info("📡 Initializing Alpaca data aggregator...")
    hyper_state.data_aggregator = HYPERDataAggregator()
    await hyper_state.data_aggregator.initialize()
    
    # Initialize signal engine
    logger.info("🧠 Initializing signal engine...")
    hyper_state.signal_engine = HYPERSignalEngine()
    
    # Initialize ML components
    logger.info("🤖 Initializing ML components...")
    hyper_state.ml_engine, _ = integrate_ml_learning(hyper_state.signal_engine)
    
    # Initialize testing framework
    if hyper_state.signal_engine:
        hyper_state.model_tester = ModelTester(hyper_state.signal_engine)
        hyper_state.testing_api = TestingAPI(hyper_state.model_tester)
    
    # Generate initial signals
    if hyper_state.signal_engine and hyper_state.data_aggregator:
        logger.info("🎯 Generating initial signals...")
        initial_signals = await generate_all_signals()
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
    
    # Mark as fully operational
    hyper_state.is_running = True
    hyper_state.initialization_complete = True
    hyper_state.stats["initialization_complete"] = True
    hyper_state.stats["status"] = "fully_operational"
    
    logger.info("✅ Background initialization complete - System fully operational!")
    
    # Start signal generation loop
    asyncio.create_task(signal_generation_loop())
    
except Exception as e:
    logger.error(f"❌ Background initialization failed: {e}")
    hyper_state.stats["status"] = "initialization_error"
    
    # Create fallback signals
    await create_fallback_signals()
```

async def generate_all_signals() -> Dict[str, Any]:
“”“Generate signals for all tickers”””
signals = {}

```
try:
    if not hyper_state.signal_engine or not hyper_state.data_aggregator:
        return await create_fallback_signals()
    
    # Generate signals for each ticker
    for symbol in config.TICKERS:
        try:
            # Get comprehensive data
            data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
            
            # Generate signal
            signal = await hyper_state.signal_engine.generate_signal(
                symbol=symbol,
                quote_data=data.get("quote", {}),
                trends_data=data.get("trends", {}),
                historical_data=data.get("historical", [])
            )
            
            # Convert signal to serializable format
            signals[symbol] = serialize_signal(signal)
            
            # Track predictions if model tester available
            if hyper_state.model_tester:
                hyper_state.model_tester.tracker.record_prediction(signal)
            
        except Exception as e:
            logger.error(f"❌ Signal generation error for {symbol}: {e}")
            signals[symbol] = create_fallback_signal(symbol)
    
    # Update stats
    hyper_state.stats["signals_generated"] += len(signals)
    
    logger.info(f"✅ Generated {len(signals)} signals")
    return signals
    
except Exception as e:
    logger.error(f"❌ Generate all signals error: {e}")
    return await create_fallback_signals()
```

async def create_fallback_signals() -> Dict[str, Any]:
“”“Create basic fallback signals”””
signals = {}

```
for symbol in config.TICKERS:
    signals[symbol] = create_fallback_signal(symbol)

hyper_state.current_signals = signals
hyper_state.last_update = datetime.now()
hyper_state.is_running = True

logger.info("✅ Fallback signals created")
return signals
```

def create_fallback_signal(symbol: str) -> Dict[str, Any]:
“”“Create a single fallback signal”””
base_prices = {
“QQQ”: 450.25, “SPY”: 535.80, “NVDA”: 875.90,
“AAPL”: 185.45, “MSFT”: 428.75
}

```
price = base_prices.get(symbol, 100.0)

return {
    "symbol": symbol,
    "signal_type": "HOLD",
    "confidence": 50.0,
    "direction": "NEUTRAL",
    "price": price,
    "timestamp": datetime.now().isoformat(),
    "technical_score": 50.0,
    "sentiment_score": 50.0,
    "ml_score": 50.0,
    "vix_score": 50.0,
    "data_source": "fallback"
}
```

def serialize_signal(signal) -> Dict[str, Any]:
“”“Convert signal object to serializable dict”””
try:
if hasattr(signal, “**dict**”):
result = {}
for key, value in signal.**dict**.items():
if not key.startswith(”*”) and not callable(value):
if hasattr(value, “**dict**”):  # Nested objects
result[key] = {k: v for k, v in value.**dict**.items()
if not k.startswith(”*”) and not callable(v)}
else:
result[key] = value
return result
elif isinstance(signal, dict):
return signal
else:
return {“error”: “unable_to_serialize”, “type”: str(type(signal))}
except Exception as e:
logger.error(f”Signal serialization error: {e}”)
return {“error”: str(e)}

async def signal_generation_loop():
“”“Background signal generation loop”””
while hyper_state.is_running:
try:
await asyncio.sleep(config.UPDATE_INTERVALS[“signal_generation”])

```
        if hyper_state.signal_engine and hyper_state.data_aggregator:
            new_signals = await generate_all_signals()
            hyper_state.current_signals = new_signals
            hyper_state.last_update = datetime.now()
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "signal_update",
                "signals": new_signals,
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"❌ Signal generation loop error: {e}")
        await asyncio.sleep(30)  # Wait before retrying
```

# ========================================

# API ENDPOINTS

# ========================================

@app.get(”/”, response_class=HTMLResponse)
async def dashboard():
“”“Main dashboard”””
try:
with open(“index.html”, “r”) as f:
return HTMLResponse(f.read())
except FileNotFoundError:
return HTMLResponse(”<h1>HYPERtrends v4.0</h1><p>Dashboard loading…</p>”)

@app.get(”/health”)
async def health_check():
“”“Health check endpoint”””
return {
“status”: “healthy”,
“timestamp”: datetime.now().isoformat(),
“system_status”: hyper_state.stats[“status”],
“initialization_complete”: hyper_state.initialization_complete,
“uptime_seconds”: (datetime.now() - hyper_state.startup_time).total_seconds(),
“connected_clients”: len(hyper_state.connected_clients),
“signals_available”: len(hyper_state.current_signals)
}

@app.get(”/api/signals”)
async def get_signals():
“”“Get current trading signals”””
return {
“status”: “success”,
“signals”: hyper_state.current_signals,
“last_update”: hyper_state.last_update.isoformat() if hyper_state.last_update else None,
“timestamp”: datetime.now().isoformat(),
“system_status”: hyper_state.stats[“status”]
}

@app.get(”/api/signals/{symbol}”)
async def get_signal(symbol: str):
“”“Get signal for specific symbol”””
symbol = symbol.upper()

```
if symbol in hyper_state.current_signals:
    return {
        "status": "success",
        "symbol": symbol,
        "signal": hyper_state.current_signals[symbol],
        "timestamp": datetime.now().isoformat()
    }
else:
    raise HTTPException(status_code=404, detail=f"Signal for {symbol} not found")
```

@app.post(”/api/signals/refresh”)
async def refresh_signals():
“”“Manual signal refresh”””
try:
if hyper_state.signal_engine and hyper_state.data_aggregator:
new_signals = await generate_all_signals()
hyper_state.current_signals = new_signals
hyper_state.last_update = datetime.now()

```
        return {
            "status": "success",
            "message": "Signals refreshed",
            "signals": new_signals,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=503, detail="Signal engine not available")
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/api/system/status”)
async def system_status():
“”“System status information”””
return {
“status”: hyper_state.stats[“status”],
“initialization_complete”: hyper_state.initialization_complete,
“uptime”: (datetime.now() - hyper_state.startup_time).total_seconds(),
“data_source”: config.get_data_source_status(),
“alpaca_available”: config.has_alpaca_credentials(),
“connected_clients”: len(hyper_state.connected_clients),
“signals_generated”: hyper_state.stats[“signals_generated”],
“last_update”: hyper_state.last_update.isoformat() if hyper_state.last_update else None,
“components”: {
“data_aggregator”: hyper_state.data_aggregator is not None,
“signal_engine”: hyper_state.signal_engine is not None,
“ml_engine”: hyper_state.ml_engine is not None,
“model_tester”: hyper_state.model_tester is not None
}
}

# ML and Testing endpoints

@app.get(”/api/ml/status”)
async def ml_status():
“”“ML system status”””
if hyper_state.ml_engine:
# Return ML status from the learning API
return {“status”: “active”, “message”: “ML system operational”}
else:
return {“status”: “inactive”, “message”: “ML system not available”}

@app.get(”/api/testing/status”)
async def testing_status():
“”“Model testing status”””
if hyper_state.testing_api:
return await hyper_state.testing_api.get_test_status()
else:
return {“status”: “inactive”, “message”: “Testing framework not available”}

@app.get(”/api/testing/backtest”)
async def run_backtest(days: int = 7):
“”“Run backtest”””
if hyper_state.testing_api:
return await hyper_state.testing_api.run_quick_backtest(days)
else:
raise HTTPException(status_code=503, detail=“Testing framework not available”)

# ========================================

# WEBSOCKET ENDPOINT

# ========================================

@app.websocket(”/ws”)
async def websocket_endpoint(websocket: WebSocket):
“”“WebSocket endpoint for real-time updates”””
await manager.connect(websocket)
try:
# Send initial data
await websocket.send_text(json.dumps({
“type”: “initial_data”,
“signals”: hyper_state.current_signals,
“timestamp”: datetime.now().isoformat(),
“status”: hyper_state.stats[“status”]
}, default=str))

```
    # Keep connection alive
    while True:
        try:
            # Wait for ping from client or send periodic updates
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "connected_clients": len(manager.active_connections)
            }, default=str))
        except:
            break
            
except WebSocketDisconnect:
    manager.disconnect(websocket)
except Exception as e:
    logger.error(f"WebSocket error: {e}")
    manager.disconnect(websocket)
```

# ========================================

# SHUTDOWN

# ========================================

@app.on_event(“shutdown”)
async def shutdown_event():
“”“Cleanup on shutdown”””
logger.info(“🛑 Shutting down HYPERtrends…”)
hyper_state.is_running = False

```
# Cleanup components
if hyper_state.data_aggregator:
    await hyper_state.data_aggregator.cleanup()

logger.info("✅ Shutdown complete")
```

# ========================================

# MAIN

# ========================================

if **name** == “**main**”:
uvicorn.run(
app,
host=config.SERVER_CONFIG[“host”],
port=config.SERVER_CONFIG[“port”],
reload=config.SERVER_CONFIG[“reload”],
log_level=“info”
)
