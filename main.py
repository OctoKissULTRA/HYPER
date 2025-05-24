import os
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing modules with enhanced debugging
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine

# ========================================
# ENHANCED LOGGING SETUP
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# FRONTEND CONFIGURATION
# ========================================
current_dir = Path(__file__).parent
index_file = current_dir / "index.html"

logger.info(f"🔧 Current directory: {current_dir}")
logger.info(f"🔧 Index file: {index_file}")
logger.info(f"🔧 Index file exists: {index_file.exists()}")

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="HYPER Trading System",
    description="Advanced AI-powered trading signals with real-time intelligence",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# GLOBAL STATE WITH ENHANCED TRACKING
# ========================================
class HYPERState:
    """Global application state with enhanced tracking"""
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None
        self.current_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.update_task = None
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "api_calls_made": 0,
            "signals_with_data": 0,
            "fallback_signals": 0
        }
    
    async def initialize(self):
        """Initialize the HYPER system with enhanced debugging"""
        logger.info("🚀 Starting HYPER Trading System...")
        logger.info(f"Tracking tickers: {', '.join(config.TICKERS)}")
        logger.info(f"Alpha Vantage API configured: {'✅' if config.ALPHA_VANTAGE_API_KEY else '❌'}")
        
        try:
            # Initialize data aggregator
            self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
            logger.info("✅ Data aggregator initialized")
            
            # Initialize signal engine  
            self.signal_engine = HYPERSignalEngine()
            logger.info("✅ Signal engine initialized")
            
            # Validate configuration
            logger.info(f"🔧 Running from: {os.getcwd()}")
            logger.info(f"🔧 Index file exists: {index_file.exists()}")
            
            config.validate_config()
            logger.info("✅ Configuration validated successfully")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize HYPER system: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False

hyper_state = HYPERState()

# ========================================
# ENHANCED WEBSOCKET CONNECTION MANAGER
# ========================================
class ConnectionManager:
    """Enhanced WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        hyper_state.stats["clients_connected"] = len(self.active_connections)
        
        logger.info(f"New client connected. Total: {len(self.active_connections)}")
        
        # Send current data to new client
        await self.send_personal_message(websocket, {
            "type": "signal_update",
            "signals": self._serialize_signals(hyper_state.current_signals),
            "stats": hyper_state.stats.copy(),
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        message_json = json.dumps(message, default=str)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
        
        if self.active_connections:
            logger.info(f"Broadcasted signals to {len(self.active_connections)} clients")
    
    def _serialize_signals(self, signals):
        """Convert signals to JSON-serializable format"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal in signals.items():
            if hasattr(signal, '__dict__'):
                # It's a HYPERSignal object
                serialized[symbol] = {
                    "symbol": signal.symbol,
                    "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                    "confidence": float(getattr(signal, 'confidence', 0.0)),
                    "price": float(getattr(signal, 'price', 0.0)),
                    "change_percent": float(getattr(signal, 'indicators', {}).get('change_percent', 0.0)),
                    "volume": int(getattr(signal, 'indicators', {}).get('volume', 0)),
                    "trend_score": float(getattr(signal, 'trends_score', 0.0)),
                    "technical_score": float(getattr(signal, 'technical_score', 0.0)),
                    "momentum_score": float(getattr(signal, 'momentum_score', 0.0)),
                    "reasons": getattr(signal, 'reasons', []),
                    "warnings": getattr(signal, 'warnings', []),
                    "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat())
                }
            else:
                # It's already a dict
                serialized[symbol] = signal
        
        logger.info(f"📡 Serialized {len(serialized)} signals for frontend")
        return serialized

manager = ConnectionManager()

# ========================================
# ENHANCED SIGNAL GENERATION LOOP
# ========================================
async def signal_generation_loop():
    """Enhanced background signal generation with debugging"""
    logger.info("🚀 Starting HYPER signal generation loop...")
    
    while hyper_state.is_running:
        try:
            logger.info("Generating signals for all tickers...")
            
            if not hyper_state.signal_engine:
                logger.error("❌ Signal engine not initialized")
                await asyncio.sleep(30)
                continue
            
            # Generate signals with timing
            start_time = datetime.now()
            signals = await hyper_state.signal_engine.generate_all_signals()
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Update state and stats
            hyper_state.current_signals = signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(signals)
            
            # Analyze signal quality
            signals_with_data = 0
            fallback_signals = 0
            
            for symbol, signal in signals.items():
                if hasattr(signal, 'price') and signal.price > 0:
                    signals_with_data += 1
                else:
                    fallback_signals += 1
            
            hyper_state.stats["signals_with_data"] = signals_with_data
            hyper_state.stats["fallback_signals"] = fallback_signals
            
            # Log detailed signal information
            signal_details = []
            for symbol, signal in signals.items():
                if hasattr(signal, '__dict__'):
                    signal_type = getattr(signal, 'signal_type', 'HOLD')
                    confidence = getattr(signal, 'confidence', 0.0)
                    price = getattr(signal, 'price', 0.0)
                    signal_details.append(f"{symbol}: {signal_type} ({confidence:.1f}%) ${price:.2f}")
                else:
                    signal_details.append(f"{symbol}: {signal}")
            
            logger.info(f"Generated signals: {', '.join(signal_details)}")
            logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
            logger.info(f"📊 Data quality: {signals_with_data}/{len(signals)} with real data")
            
            # Broadcast to clients
            await manager.broadcast({
                "type": "signal_update",
                "signals": manager._serialize_signals(signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy()
            })
            
            # Wait for next update
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            logger.error(f"💥 Error in signal generation loop: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            await asyncio.sleep(30)  # Wait longer on error

# ========================================
# API ROUTES
# ========================================
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main trading interface"""
    try:
        if index_file.exists():
            logger.info(f"📁 Serving index.html from: {index_file}")
            with open(index_file, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"✅ Successfully loaded index.html ({len(content)} characters)")
                return HTMLResponse(content=content)
        else:
            logger.error(f"❌ index.html not found at: {index_file}")
            return HTMLResponse(
                content="<h1>HYPER Trading System</h1><p>Frontend not found</p>",
                status_code=404
            )
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Enhanced system health check"""
    return {
        "status": "healthy",
        "is_running": hyper_state.is_running,
        "connected_clients": len(manager.active_connections),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "total_signals": hyper_state.stats["total_signals_generated"],
        "signals_with_data": hyper_state.stats.get("signals_with_data", 0),
        "fallback_signals": hyper_state.stats.get("fallback_signals", 0),
        "tickers": config.TICKERS,
        "system_initialized": hyper_state.signal_engine is not None,
        "index_file_exists": index_file.exists(),
        "api_key_configured": bool(config.ALPHA_VANTAGE_API_KEY),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current signals for all tickers"""
    return {
        "signals": manager._serialize_signals(hyper_state.current_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats.copy()
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get signal for specific symbol"""
    symbol = symbol.upper()
    if symbol not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not tracked")
    
    if symbol in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol]
        return manager._serialize_signals({symbol: signal})[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.post("/api/start")
async def start_system():
    """Start the signal generation system"""
    if hyper_state.is_running:
        return {"status": "already_running", "message": "HYPER is already running"}
    
    if not hyper_state.signal_engine:
        success = await hyper_state.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize HYPER system")
    
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    
    # Start background task
    asyncio.create_task(signal_generation_loop())
    
    logger.info("🚀 HYPER signal generation started")
    return {"status": "started", "message": "HYPER signal generation started"}

@app.post("/api/stop")
async def stop_system():
    """Stop the signal generation system"""
    if not hyper_state.is_running:
        return {"status": "not_running", "message": "HYPER is not running"}
    
    hyper_state.is_running = False
    logger.info("⏸️ HYPER signal generation stopped")
    return {"status": "stopped", "message": "HYPER signal generation stopped"}

@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop all operations"""
    hyper_state.is_running = False
    logger.warning("🚨 EMERGENCY STOP ACTIVATED")
    return {"status": "emergency_stopped", "message": "Emergency stop activated"}

# ========================================
# WEBSOCKET ENDPOINT
# ========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await manager.send_personal_message(websocket, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message.get("type") == "request_signals":
                await manager.send_personal_message(websocket, {
                    "type": "signal_update",
                    "signals": manager._serialize_signals(hyper_state.current_signals),
                    "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
                    "stats": hyper_state.stats.copy()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

# ========================================
# STARTUP/SHUTDOWN EVENTS
# ========================================
@app.on_event("startup")
async def startup_event():
    """Application startup with market status detection"""
    logger.info("🚀 Starting HYPER Trading System...")
    
    # Market status check (simple version)
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 16:  # Market hours (rough)
        market_status = "OPEN"
    else:
        market_status = "CLOSED"
    
    logger.info(f"📈 Market status: {market_status}")
    
    # Initialize the system
    success = await hyper_state.initialize()
    if not success:
        logger.error("❌ Failed to initialize HYPER system")
        return
    
    # Generate initial signals
    logger.info("Generating initial signals...")
    try:
        initial_signals = await hyper_state.signal_engine.generate_all_signals()
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
        logger.info("✅ Initial signals generated successfully")
    except Exception as e:
        logger.error(f"❌ Failed to generate initial signals: {e}")
    
    # Auto-start the system
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    asyncio.create_task(signal_generation_loop())
    
    status_msg = f"Markets {market_status} - using {'live' if market_status == 'OPEN' else 'cached/demo'} data"
    logger.info(f"🔥 HYPER signal generation auto-started! ({status_msg})")
    logger.info("🎯 HYPER Trading System ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("⏸️ Shutting down HYPER Trading System...")
    hyper_state.is_running = False
    
    # Close WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    logger.info("👋 HYPER shutdown complete")

# ========================================
# MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    logger.info("🚀 Starting HYPER Trading System server...")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG.get("reload", False),
        log_level="info"
    )
