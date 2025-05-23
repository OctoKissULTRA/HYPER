# ============================================
# HYPER TRADING SYSTEM - FastAPI Server
# Real-time signal delivery via WebSocket
# ============================================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
import os
from pathlib import Path

from config import config
from signal_engine import HYPERSignalEngine, HYPERSignal
from data_sources import HYPERDataAggregator

# ========================================
# LOGGING SETUP
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# FRONTEND PATH CONFIGURATION
# ========================================

# Get the correct paths for organized structure
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
frontend_dir = project_root / "frontend"

logger.info(f"🔧 Backend directory: {backend_dir}")
logger.info(f"🔧 Project root: {project_root}")
logger.info(f"🔧 Frontend directory: {frontend_dir}")
logger.info(f"🔧 Frontend exists: {frontend_dir.exists()}")

if frontend_dir.exists():
    index_file = frontend_dir / "index.html"
    logger.info(f"🔧 Index file exists: {index_file.exists()}")
else:
    logger.warning("⚠️ Frontend directory not found!")

# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(
    title="HYPER Trading System",
    description="Advanced AI-powered trading signals with real-time intelligence",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# SERVE FRONTEND FILES
# ========================================

# Mount static files if frontend directory exists
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend")
    logger.info("✅ Frontend files mounted at /static")
else:
    logger.warning("⚠️ Frontend directory not found - static files not mounted")

# ========================================
# GLOBAL STATE
# ========================================

class HYPERState:
    """Global application state"""
    def __init__(self):
        self.is_running = False
        self.signal_engine = HYPERSignalEngine()
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
        hyper_state.connected_clients = self.active_connections
        hyper_state.stats["clients_connected"] = len(self.active_connections)
        
        logger.info(f"New client connected. Total: {len(self.active_connections)}")
        
        # Send initial data to new client
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "Connected to HYPER Trading System",
            "current_signals": self._serialize_signals(hyper_state.current_signals),
            "stats": hyper_state.stats,
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.connected_clients = self.active_connections
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        message_json = json.dumps(message)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def _serialize_signals(self, signals: Dict[str, HYPERSignal]) -> Dict[str, Dict]:
        """Convert HYPERSignal objects to dict for JSON"""
        return {
            symbol: {
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
            for symbol, signal in signals.items()
        }

manager = ConnectionManager()

# ========================================
# BACKGROUND SIGNAL GENERATION
# ========================================

async def signal_generation_loop():
    """Background task that generates signals continuously"""
    logger.info("🚀 Starting HYPER signal generation loop...")
    
    while hyper_state.is_running:
        try:
            # Generate signals for all tickers
            logger.info("Generating signals for all tickers...")
            new_signals = await hyper_state.signal_engine.generate_all_signals()
            
            # Update state
            hyper_state.current_signals = new_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(new_signals)
            
            # Broadcast to all connected clients
            if manager.active_connections:
                await manager.broadcast({
                    "type": "signal_update",
                    "signals": manager._serialize_signals(new_signals),
                    "timestamp": hyper_state.last_update.isoformat(),
                    "stats": hyper_state.stats
                })
                
                logger.info(f"Broadcasted signals to {len(manager.active_connections)} clients")
            
            # Log signal summary
            signal_summary = []
            for symbol, signal in new_signals.items():
                signal_summary.append(f"{symbol}: {signal.signal_type} ({signal.confidence}%)")
            
            logger.info(f"Generated signals: {', '.join(signal_summary)}")
            
            # Wait before next update
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            logger.error(f"Error in signal generation loop: {e}")
            await asyncio.sleep(30)  # Wait 30 seconds on error

# ========================================
# API ROUTES
# ========================================

@app.get("/", response_class=HTMLResponse)
async def get_trading_interface():
    """Serve the main trading interface"""
    try:
        index_path = frontend_dir / "index.html"
        
        if index_path.exists():
            logger.info(f"📁 Serving index.html from: {index_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"✅ Successfully loaded index.html ({len(content)} characters)")
                return HTMLResponse(content=content)
        else:
            logger.error(f"❌ index.html not found at: {index_path}")
            logger.info(f"📁 Files in frontend dir: {list(frontend_dir.iterdir()) if frontend_dir.exists() else 'Directory does not exist'}")
            
            return HTMLResponse(
                content="""
                <h1>🚀 HYPER Trading System</h1>
                <p>❌ Frontend index.html not found</p>
                <p>Expected location: """ + str(index_path) + """</p>
                <p>Directory exists: """ + str(frontend_dir.exists()) + """</p>
                """,
                status_code=404
            )
            
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(
            content=f"""
            <h1>🚀 HYPER Trading System</h1>
            <p>❌ Error loading frontend: {str(e)}</p>
            <p>Backend directory: {backend_dir}</p>
            <p>Frontend directory: {frontend_dir}</p>
            """,
            status_code=500
        )

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "is_running": hyper_state.is_running,
        "uptime": str(datetime.now() - hyper_state.stats["uptime_start"]),
        "connected_clients": len(manager.active_connections),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "total_signals": hyper_state.stats["total_signals_generated"],
        "tickers": config.TICKERS,
        "backend_dir": str(backend_dir),
        "frontend_dir": str(frontend_dir),
        "frontend_exists": frontend_dir.exists()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current signals for all tickers"""
    return {
        "signals": manager._serialize_signals(hyper_state.current_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get signal for specific symbol"""
    if symbol.upper() not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not tracked")
    
    symbol = symbol.upper()
    if symbol in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol]
        return manager._serialize_signals({symbol: signal})[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.post("/api/start")
async def start_signal_generation(background_tasks: BackgroundTasks):
    """Start the signal generation system"""
    if hyper_state.is_running:
        return {"status": "already_running", "message": "HYPER is already running"}
    
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    
    # Start background task
    background_tasks.add_task(signal_generation_loop)
    
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
async def stop_signal_generation():
    """Stop the signal generation system"""
    if not hyper_state.is_running:
        return {"status": "not_running", "message": "HYPER is not running"}
    
    hyper_state.is_running = False
    
    # Broadcast status change
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
    
    # Broadcast emergency stop
    await manager.broadcast({
        "type": "emergency_stop",
        "message": "🚨 EMERGENCY STOP ACTIVATED",
        "timestamp": datetime.now().isoformat()
    })
    
    logger.warning("🚨 EMERGENCY STOP ACTIVATED")
    return {"status": "emergency_stopped", "message": "Emergency stop activated"}

@app.get("/api/config")
async def get_configuration():
    """Get system configuration"""
    return {
        "tickers": config.TICKERS,
        "confidence_thresholds": config.CONFIDENCE_THRESHOLDS,
        "signal_weights": config.SIGNAL_WEIGHTS,
        "update_intervals": config.UPDATE_INTERVALS,
        "technical_params": config.TECHNICAL_PARAMS
    }

@app.get("/api/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    uptime = datetime.now() - hyper_state.stats["uptime_start"]
    
    return {
        **hyper_state.stats,
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime),
        "signals_per_minute": hyper_state.stats["total_signals_generated"] / max(1, uptime.total_seconds() / 60),
        "current_signal_count": len(hyper_state.current_signals),
        "last_update_ago": (datetime.now() - hyper_state.last_update).total_seconds() if hyper_state.last_update else None
    }

# ========================================
# WEBSOCKET ENDPOINT
# ========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message(websocket, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message.get("type") == "request_signals":
                await manager.send_personal_message(websocket, {
                    "type": "signal_update",
                    "signals": manager._serialize_signals(hyper_state.current_signals),
                    "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None
                })
            
            elif message.get("type") == "request_stats":
                await manager.send_personal_message(websocket, {
                    "type": "stats_update",
                    "stats": hyper_state.stats,
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ========================================
# STARTUP/SHUTDOWN EVENTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Starting HYPER Trading System...")
    logger.info(f"Tracking tickers: {', '.join(config.TICKERS)}")
    logger.info(f"Alpha Vantage API configured: {'✅' if config.ALPHA_VANTAGE_API_KEY else '❌'}")
    
    # Log path information
    logger.info(f"🔧 Backend running from: {backend_dir}")
    logger.info(f"🔧 Frontend directory: {frontend_dir}")
    logger.info(f"🔧 Frontend exists: {frontend_dir.exists()}")
    
    # Validate configuration
    try:
        config.validate_config()
        logger.info("✅ Configuration validated successfully")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
    
    # Pre-generate initial signals
    logger.info("Generating initial signals...")
    try:
        initial_signals = await hyper_state.signal_engine.generate_all_signals()
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
        logger.info("✅ Initial signals generated successfully")
    except Exception as e:
        logger.error(f"❌ Failed to generate initial signals: {e}")
    
    logger.info("🎯 HYPER Trading System ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("⏸️ Shutting down HYPER Trading System...")
    hyper_state.is_running = False
    
    # Close all WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    logger.info("👋 HYPER Trading System shutdown complete")

# ========================================
# MAIN APPLICATION ENTRY POINT
# ========================================

if __name__ == "__main__":
    logger.info("🚀 Starting HYPER Trading System server...")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG["reload"],
        log_level="info"
    )
