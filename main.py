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

# Import our fixed modules
from fixed_data_sources import HYPERDataAggregator
from fixed_signal_engine import HYPERSignalEngine, HYPERSignal

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================
class Config:
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")
    
    # Target tickers
    TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]
    
    # Server configuration
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Update intervals
    SIGNAL_UPDATE_INTERVAL = 30  # seconds
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY not configured")
        logger.info(f"✅ Configuration validated")
        logger.info(f"🔑 API Key: {cls.ALPHA_VANTAGE_API_KEY[:10]}...")
        logger.info(f"📊 Tickers: {', '.join(cls.TICKERS)}")

config = Config()

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
    version="2.1.0"
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
class HYPERState:
    """Global application state"""
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None
        self.current_signals: Dict[str, HYPERSignal] = {}
        self.connected_clients: List[WebSocket] = []
        self.last_update = None
        self.update_task = None
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "api_calls_made": 0
        }
    
    async def initialize(self):
        """Initialize the HYPER system"""
        logger.info("🚀 Initializing HYPER system...")
        
        try:
            self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
            self.signal_engine = HYPERSignalEngine(self.data_aggregator)
            
            logger.info("✅ HYPER system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize HYPER system: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.data_aggregator:
            await self.data_aggregator.close()
        logger.info("🔒 HYPER system cleaned up")

hyper_state = HYPERState()

# ========================================
# WEBSOCKET CONNECTION MANAGER
# ========================================
class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        hyper_state.stats["clients_connected"] = len(self.active_connections)
        
        logger.info(f"🔌 New client connected. Total: {len(self.active_connections)}")
        
        # Send current data to new client
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "Connected to HYPER Trading System",
            "signals": self._serialize_signals(hyper_state.current_signals),
            "stats": self._serialize_stats(),
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            logger.info(f"🔌 Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message, default=self._serialize_datetime))
        except Exception as e:
            logger.error(f"❌ Error sending personal message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        message_json = json.dumps(message, default=self._serialize_datetime)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"❌ Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def _serialize_signals(self, signals: Dict[str, HYPERSignal]) -> Dict[str, Dict]:
        """Convert HYPERSignal objects to JSON-serializable dict"""
        serialized = {}
        for symbol, signal in signals.items():
            serialized[symbol] = {
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "direction": signal.direction,
                "price": signal.price,
                "timestamp": signal.timestamp,
                "technical_score": signal.technical_score,
                "momentum_score": signal.momentum_score,
                "trends_score": signal.trends_score,
                "volume_score": signal.volume_score,
                "ml_score": signal.ml_score,
                "indicators": signal.indicators,
                "reasons": signal.reasons,
                "warnings": signal.warnings,
                "data_quality": signal.data_quality
            }
        return serialized
    
    def _serialize_stats(self) -> Dict:
        """Serialize stats with datetime handling"""
        return {
            "total_signals_generated": hyper_state.stats["total_signals_generated"],
            "clients_connected": hyper_state.stats["clients_connected"],
            "uptime_start": self._serialize_datetime(hyper_state.stats["uptime_start"]),
            "api_calls_made": hyper_state.stats.get("api_calls_made", 0)
        }
    
    def _serialize_datetime(self, obj):
        """Convert datetime objects to ISO format strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

manager = ConnectionManager()

# ========================================
# BACKGROUND SIGNAL GENERATION
# ========================================
async def signal_generation_loop():
    """Background task that generates signals continuously"""
    logger.info("🔄 Starting signal generation loop...")
    
    while hyper_state.is_running:
        try:
            logger.info("🎯 Generating signals for all tickers...")
            
            if not hyper_state.signal_engine:
                logger.error("❌ Signal engine not initialized")
                await asyncio.sleep(30)
                continue
            
            # Generate signals
            new_signals = await hyper_state.signal_engine.generate_all_signals(config.TICKERS)
            
            # Update state
            hyper_state.current_signals = new_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(new_signals)
            
            # Log signal summary
            signal_summary = []
            for symbol, signal in new_signals.items():
                signal_summary.append(f"{symbol}: {signal.signal_type} ({signal.confidence}%)")
            
            logger.info(f"📊 Generated signals: {', '.join(signal_summary)}")
            
            # Broadcast to WebSocket clients
            if manager.active_connections:
                await manager.broadcast({
                    "type": "signal_update",
                    "signals": manager._serialize_signals(new_signals),
                    "timestamp": hyper_state.last_update.isoformat(),
                    "stats": manager._serialize_stats()
                })
                logger.info(f"📡 Broadcasted to {len(manager.active_connections)} clients")
            
            # Wait before next update
            await asyncio.sleep(config.SIGNAL_UPDATE_INTERVAL)
            
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
                logger.info(f"✅ Loaded index.html ({len(content)} characters)")
                return HTMLResponse(content=content)
        else:
            logger.error(f"❌ index.html not found at: {index_file}")
            return HTMLResponse(
                content=f"""
                <h1>🚀 HYPER Trading System</h1>
                <p>❌ Frontend not found at: {index_file}</p>
                <p>Files in directory: {list(current_dir.iterdir())}</p>
                """,
                status_code=404
            )
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(
            content=f"<h1>Error: {str(e)}</h1>",
            status_code=500
        )

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "is_running": hyper_state.is_running,
        "connected_clients": len(manager.active_connections),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "total_signals": hyper_state.stats["total_signals_generated"],
        "tickers": config.TICKERS,
        "system_initialized": hyper_state.signal_engine is not None,
        "api_key_configured": bool(config.ALPHA_VANTAGE_API_KEY),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current signals for all tickers"""
    return {
        "signals": manager._serialize_signals(hyper_state.current_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": manager._serialize_stats()
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
    
    # Broadcast status change
    await manager.broadcast({
        "type": "system_status",
        "status": "started",
        "message": "HYPER signal generation started",
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info("🚀 HYPER signal generation started")
    return {"status": "started", "message": "HYPER signal generation started"}

@app.post("/api/stop")
async def stop_system():
    """Stop the signal generation system"""
    if not hyper_state.is_running:
        return {"status": "not_running", "message": "HYPER is not running"}
    
    hyper_state.is_running = False
    
    await manager.broadcast({
        "type": "system_status",
        "status": "stopped",
        "message": "HYPER signal generation stopped",
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info("⏸️ HYPER signal generation stopped")
    return {"status": "stopped", "message": "HYPER signal generation stopped"}

@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop all operations"""
    hyper_state.is_running = False
    
    await manager.broadcast({
        "type": "emergency_stop",
        "message": "🚨 EMERGENCY STOP ACTIVATED",
        "timestamp": datetime.now().isoformat()
    })
    
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
                    "stats": manager._serialize_stats()
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
    """Application startup"""
    logger.info("🚀 Starting HYPER Trading System...")
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return
    
    # Initialize the system
    success = await hyper_state.initialize()
    if not success:
        logger.error("❌ Failed to initialize HYPER system")
        return
    
    # Generate initial signals
    logger.info("🎯 Generating initial signals...")
    try:
        initial_signals = await hyper_state.signal_engine.generate_all_signals(config.TICKERS)
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
        logger.info("✅ Initial signals generated")
        
        # Log signal summary
        signal_summary = []
        for symbol, signal in initial_signals.items():
            signal_summary.append(f"{symbol}: {signal.signal_type} ({signal.confidence}%)")
        logger.info(f"📊 Initial signals: {', '.join(signal_summary)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate initial signals: {e}")
    
    # Auto-start the system
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    asyncio.create_task(signal_generation_loop())
    
    logger.info("🎯 HYPER Trading System ready and running!")

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
    
    # Cleanup resources
    await hyper_state.cleanup()
    
    logger.info("👋 HYPER shutdown complete")

# ========================================
# MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    logger.info("🚀 Starting HYPER Trading System server...")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
