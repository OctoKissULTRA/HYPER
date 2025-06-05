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
                quote_data=data.get('quote', {}),
                trends_data=data.get('trends', {}),
                historical_data=data.get('historical', [])
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
if hasattr(signal, ‘**dict**’):
result = {}
for key, value in signal.**dict**.items():
if not key.startswith(’*’) and not callable(value):
if hasattr(value, ‘**dict**’):  # Nested objects
result[key] = {k: v for k, v in value.**dict**.items()
if not k.startswith(’*’) and not callable(v)}
else:
result[key] = value
return result
elif isinstance(signal, dict):
return signal
else:
return {“error”: “unable_to_serialize”, “type”: str(type(signal))}
except Exception as e:
logger.error(f”Signal serialization error: {e}”)
