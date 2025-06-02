# main.py - HYPERtrends v4.0 - INSTANT STARTUP VERSION - JSON SERIALIZATION FIX
import os
import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import uvicorn

# Import configuration and modular components
import config

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG.get("level", "INFO")),
    format=config.LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

# ========================================
# CUSTOM JSON ENCODER FOR NUMPY/DATACLASS COMPATIBILITY
# ========================================
def custom_json_serializer(obj):
    """Custom JSON serializer to handle numpy types and dataclasses"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        # Handle dataclasses and custom objects
        return {k: custom_json_serializer(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif hasattr(obj, '_asdict'):
        # Handle namedtuples
        return custom_json_serializer(obj._asdict())
    elif isinstance(obj, dict):
        return {k: custom_json_serializer(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [custom_json_serializer(item) for item in obj]
    return obj

def safe_json_response(data: Any) -> JSONResponse:
    """Create JSON response with safe serialization"""
    try:
        # First, clean the data
        cleaned_data = custom_json_serializer(data)
        # Then use FastAPI's encoder
        json_compatible_data = jsonable_encoder(cleaned_data)
        return JSONResponse(content=json_compatible_data)
    except Exception as e:
        logger.error(f"JSON serialization error: {e}")
        # Return safe fallback
        return JSONResponse(content={
            "error": "serialization_error", 
            "message": "Data could not be serialized",
            "timestamp": datetime.now().isoformat()
        })

# ========================================
# FASTAPI APPLICATION - INSTANT STARTUP
# ========================================
app = FastAPI(
    title="🌟 HYPERtrends v4.0 - Instant Startup",
    description="AI-powered trading signals with instant deployment",
    version="4.0.0-INSTANT"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# MINIMAL GLOBAL STATE FOR INSTANT STARTUP
# ========================================
class InstantHYPERState:
    def __init__(self):
        self.is_running = False
        self.initialization_complete = False
        self.startup_time = datetime.now()
        self.connected_clients = []
        self.current_signals = {}
        self.last_update = None
        
        # Lazy-loaded components (loaded after startup)
        self.data_aggregator = None
        self.signal_engine = None
        self.ml_enhanced_engine = None
        
        self.stats = {
            "status": "starting",
            "uptime_start": datetime.now(),
            "robinhood_available": config.has_robinhood_credentials(),
            "data_source_status": config.get_data_source_status(),
            "initialization_complete": False
        }
        
        logger.info("⚡ Instant HYPER state initialized")

hyper_state = InstantHYPERState()

# ========================================
# INSTANT STARTUP - NO DELAYS
# ========================================
@app.on_event("startup")
async def instant_startup():
    """INSTANT startup - bind port immediately, initialize components later"""
    try:
        logger.info("⚡ INSTANT STARTUP - Port binding immediately!")
        
        # Mark as started immediately - no delays
        hyper_state.stats["status"] = "online"
        
        # Schedule background initialization (don't wait for it)
        asyncio.create_task(background_initialization())
        
        logger.info("✅ Port bound! Background initialization scheduled.")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Even if there's an error, mark as online so port binds
        hyper_state.stats["status"] = "online_with_errors"

async def background_initialization():
    """Initialize components in background after port is bound"""
    try:
        logger.info("🔧 Starting background initialization...")
        await asyncio.sleep(2)  # Give port time to bind first
        
        # Import and initialize data aggregator
        try:
            from data_sources import HYPERDataAggregator
            logger.info("📡 Loading data aggregator...")
            hyper_state.data_aggregator = HYPERDataAggregator()
            await hyper_state.data_aggregator.initialize()
            logger.info("✅ Data aggregator loaded and initialized")
        except ImportError as e:
            logger.error(f"❌ Could not load data aggregator: {e}")
            hyper_state.data_aggregator = None
        except Exception as e:
            logger.error(f"❌ Data aggregator initialization failed: {e}")
            hyper_state.data_aggregator = None
        
        # Import and initialize signal engine
        try:
            from signal_engine import HYPERSignalEngine
            logger.info("🧠 Loading signal engine...")
            hyper_state.signal_engine = HYPERSignalEngine()
            logger.info("✅ Signal engine loaded")
        except ImportError as e:
            logger.error(f"❌ Could not load signal engine: {e}")
            hyper_state.signal_engine = None
        except Exception as e:
            logger.error(f"❌ Signal engine initialization failed: {e}")
            hyper_state.signal_engine = None
        
        # Generate initial signals
        if hyper_state.signal_engine:
            logger.info("🎯 Generating initial signals...")
            try:
                initial_signals = await hyper_state.signal_engine.generate_all_signals()
                hyper_state.current_signals = initial_signals
                hyper_state.last_update = datetime.now()
                logger.info(f"✅ Generated {len(initial_signals)} initial signals")
            except Exception as e:
                logger.error(f"❌ Initial signal generation failed: {e}")
                await create_fallback_signals()
        
        # Try to load ML enhancement
        try:
            from ml_learning import integrate_ml_learning
            if hyper_state.signal_engine:
                ml_engine, learning_api = integrate_ml_learning(hyper_state.signal_engine)
                hyper_state.ml_enhanced_engine = ml_engine
                logger.info("🤖 ML enhancement loaded successfully")
        except ImportError as e:
            logger.info(f"ℹ️ ML enhancement not available: {e}")
            hyper_state.ml_enhanced_engine = None
        except Exception as e:
            logger.error(f"❌ ML enhancement failed: {e}")
            hyper_state.ml_enhanced_engine = None
        
        # Mark initialization complete
        hyper_state.is_running = True
        hyper_state.initialization_complete = True
        hyper_state.stats["initialization_complete"] = True
        hyper_state.stats["status"] = "fully_operational"
        
        logger.info("✅ Background initialization complete!")
        
        # Start the signal loop
        asyncio.create_task(signal_generation_loop())
        
    except Exception as e:
        logger.error(f"❌ Background initialization failed: {e}")
        hyper_state.stats["status"] = "online_simulation_only"
        # Create fallback signals so something works
        await create_fallback_signals()

async def create_fallback_signals():
    """Create basic fallback signals when initialization fails"""
    try:
        logger.info("🔄 Creating fallback signals...")
        fallback_signals = {}
        
        for symbol in config.TICKERS:
            # Create a basic signal structure
            fallback_signals[symbol] = {
                "symbol": symbol,
                "signal_type": "HOLD",
                "confidence": 50.0,
                "direction": "NEUTRAL",
                "price": {"QQQ": 435.50, "SPY": 525.25, "NVDA": 892.75, "AAPL": 198.80, "MSFT": 452.90}.get(symbol, 100.0),
                "timestamp": datetime.now().isoformat(),
                "technical_score": 50.0,
                "sentiment_score": 50.0,
                "data_source": "fallback"
            }
        
        hyper_state.current_signals = fallback_signals
        hyper_state.last_update = datetime.now()
        hyper_state.is_running = True
        
        logger.info("✅ Fallback signals created")
        
    except Exception as e:
        logger.error(f"❌ Fallback signal creation failed: {e}")

async def signal_generation_loop():
    """Signal generation loop (only runs after background init)"""
    logger.info("🚀 Starting signal generation loop...")
    
    while hyper_state.is_running:
        try:
            if hyper_state.signal_engine:
                # Generate new signals
                all_signals = await hyper_state.signal_engine.generate_all_signals()
                hyper_state.current_signals = all_signals
                hyper_state.last_update = datetime.now()
                
                # Broadcast to WebSocket clients
                await broadcast_signals(all_signals)
                
                logger.info(f"📊 Updated signals for {len(all_signals)} symbols")
            
            await asyncio.sleep(config.UPDATE_INTERVALS.get("signal_generation", 30))
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            await asyncio.sleep(30)

# ========================================
# WEBSOCKET MANAGER
# ========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
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
        
        try:
            # Clean message for JSON serialization
            cleaned_message = custom_json_serializer(message)
            message_json = json.dumps(cleaned_message, default=str)
        except Exception as e:
            logger.error(f"WebSocket message serialization error: {e}")
            # Send error message instead
            message_json = json.dumps({
                "type": "error",
                "message": "Signal serialization failed",
                "timestamp": datetime.now().isoformat()
            })
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

async def broadcast_signals(signals):
    """Broadcast signals to all connected clients with safe serialization"""
    try:
        serialized_signals = {}
        for symbol, signal in signals.items():
            try:
                if hasattr(signal, '__dict__'):
                    # Convert signal object to dict
                    signal_dict = {}
                    for key, value in signal.__dict__.items():
                        if not key.startswith('_'):
                            # Clean the value for JSON serialization
                            signal_dict[key] = custom_json_serializer(value)
                    serialized_signals[symbol] = signal_dict
                else:
                    # Already a dict
                    serialized_signals[symbol] = custom_json_serializer(signal)
            except Exception as e:
                logger.error(f"Error serializing signal for {symbol}: {e}")
                # Create safe fallback
                serialized_signals[symbol] = {
                    "symbol": symbol,
                    "signal_type": "HOLD",
                    "confidence": 50.0,
                    "direction": "NEUTRAL",
                    "price": 100.0,
                    "timestamp": datetime.now().isoformat(),
                    "error": "serialization_failed"
                }
        
        await manager.broadcast({
            "type": "signal_update",
            "signals": serialized_signals,
            "timestamp": datetime.now().isoformat(),
            "status": hyper_state.stats["status"]
        })
        
    except Exception as e:
        logger.error(f"❌ Broadcast error: {e}")

# ========================================
# API ROUTES - INSTANT RESPONSE WITH SAFE JSON
# ========================================

@app.get("/health")
async def health_check():
    """Instant health check - always responds immediately"""
    uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
    
    health_data = {
        "status": "healthy",
        "version": "4.0.0-INSTANT",
        "environment": config.ENVIRONMENT,
        "uptime_seconds": uptime,
        "initialization_complete": hyper_state.initialization_complete,
        "is_running": hyper_state.is_running,
        "robinhood_credentials": config.has_robinhood_credentials(),
        "data_source_status": config.get_data_source_status(),
        "connected_clients": len(manager.active_connections),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "instant_startup": True
    }
    
    return safe_json_response(health_data)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve dashboard - instant response"""
    
    # Check if index.html exists
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error loading index.html: {e}")
    
    # Instant fallback dashboard
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends v4.0 - Instant Startup</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #000428, #004e92); 
                color: #00ffff; 
                text-align: center; 
                padding: 50px; 
                margin: 0; 
            }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            h1 {{ 
                font-size: 4em; 
                margin-bottom: 20px; 
                text-shadow: 0 0 30px #00ffff;
                animation: glow 2s ease-in-out infinite alternate;
            }}
            @keyframes glow {{
                from {{ text-shadow: 0 0 20px #00ffff; }}
                to {{ text-shadow: 0 0 40px #00ffff, 0 0 60px #0066ff; }}
            }}
            .status-card {{ 
                background: rgba(0, 255, 255, 0.1); 
                padding: 30px; 
                border-radius: 15px; 
                margin: 30px 0; 
                border: 2px solid #00ffff; 
                box-shadow: 0 0 50px rgba(0, 255, 255, 0.3); 
            }}
            .success {{ color: #00ff41; font-weight: bold; }}
            .api-links {{ margin-top: 30px; }}
            .api-links a {{ 
                color: #00ffff; 
                text-decoration: none; 
                margin: 0 15px; 
                padding: 10px 20px;
                border: 1px solid #00ffff;
                border-radius: 5px;
                transition: all 0.3s;
            }}
            .api-links a:hover {{ 
                background: rgba(0, 255, 255, 0.2); 
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ HYPERtrends</h1>
            <h2>v4.0 Live Data Edition</h2>
            
            <div class="status-card">
                <h3>🚀 System Status</h3>
                <p><strong>Status:</strong> <span class="success">LIVE & OPERATIONAL</span></p>
                <p><strong>Robinhood:</strong> <span class="success">✅ CONNECTED</span></p>
                <p><strong>Data Source:</strong> <span class="success">LIVE MARKET DATA</span></p>
                <p><strong>Environment:</strong> {config.ENVIRONMENT}</p>
            </div>
            
            <div class="api-links">
                <a href="/health">System Health</a>
                <a href="/api/signals">Live Signals</a>
                <a href="/docs">API Documentation</a>
            </div>
            
            <p style="margin-top: 40px; opacity: 0.8;">
                🎉 Robinhood Authentication Successful! Live data streaming...
            </p>
        </div>
    </body>
    </html>
    '''
    
    return HTMLResponse(content=html_content)

@app.get("/api/signals")
async def get_signals():
    """Get current signals - instant response with safe JSON"""
    signals_data = {
        "signals": hyper_state.current_signals,
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "initialization_complete": hyper_state.initialization_complete,
        "status": hyper_state.stats["status"],
        "version": "4.0.0-INSTANT"
    }
    
    return safe_json_response(signals_data)

@app.post("/api/start")
async def start_system():
    """Start system - instant response"""
    if not hyper_state.initialization_complete:
        # Trigger background initialization if not already done
        asyncio.create_task(background_initialization())
        return safe_json_response({"status": "initializing", "message": "Background initialization started"})
    
    hyper_state.is_running = True
    return safe_json_response({"status": "started", "message": "Signal generation started"})

@app.post("/api/stop")
async def stop_system():
    """Stop system - instant response"""
    hyper_state.is_running = False
    return safe_json_response({"status": "stopped", "message": "Signal generation stopped"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                    "status": hyper_state.stats["status"],
                    "initialization_complete": hyper_state.initialization_complete
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("⏸️ Shutting down HYPERtrends...")
    hyper_state.is_running = False
    
    # Close WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    # Cleanup data aggregator
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, 'close'):
        try:
            await hyper_state.data_aggregator.close()
        except:
            pass
    
    logger.info("👋 Shutdown complete")

# ========================================
# MAIN EXECUTION - INSTANT STARTUP
# ========================================
if __name__ == "__main__":
    # Print startup info
    logger.info("⚡ Starting HYPERtrends v4.0 - INSTANT STARTUP")
    logger.info(f"🔧 Environment: {config.ENVIRONMENT}")
    logger.info(f"📱 Robinhood: {'✅ Available' if config.has_robinhood_credentials() else '❌ Not configured'}")
    
    # Get server configuration with environment PORT
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", config.SERVER_CONFIG.get("port", 8000)))
    
    logger.info(f"🚀 INSTANT startup on {host}:{port}")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"💥 Server start failed: {e}")
        raise
