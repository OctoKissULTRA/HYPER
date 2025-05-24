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

# Simple path for flat structure
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# UTILITY FUNCTIONS FOR JSON SERIALIZATION
# ========================================

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_stats(stats: Dict) -> Dict:
    """Serialize stats dictionary, converting datetime objects to strings"""
    return {
        "total_signals_generated": stats.get("total_signals_generated", 0),
        "clients_connected": stats.get("clients_connected", 0),
        "uptime_start": serialize_datetime(stats.get("uptime_start")),
        "api_calls_made": stats.get("api_calls_made", 0)
    }

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
            "stats": serialize_stats(hyper_state.stats),
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
            await websocket.send_text(json.dumps(message, default=serialize_datetime))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        
        try:
            # Use json.dumps with datetime serialization
            message_json = json.dumps(message, default=serialize_datetime)
        except Exception as e:
            logger.error(f"Error serializing message: {e}")
            return
        
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
                "timestamp": serialize_datetime(signal.timestamp) if hasattr(signal, 'timestamp') else datetime.now().isoformat(),
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
            
            # Broadcast to all connected clients with proper serialization
            if manager.active_connections:
                broadcast_message = {
                    "type": "signal_update",
                    "signals": manager._serialize_signals(new_signals),
                    "timestamp": hyper_state.last_update.isoformat(),
                    "stats": serialize_stats(hyper_state.stats)
                }
                
                await manager.broadcast(broadcast_message)
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await asyncio.sleep(30)  # Wait 30 seconds on error

# ========================================
# MARKET HOURS CHECK
# ========================================

def is_market_hours():
    """Check if markets are currently open (US Eastern Time)"""
    try:
        import pytz
        from datetime import datetime, time
        
        # US Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now_eastern = datetime.now(eastern)
        
        # Market hours: Monday-Friday, 9:30 AM - 4:00 PM ET
        weekday = now_eastern.weekday()  # 0=Monday, 6=Sunday
        current_time = now_eastern.time()
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        # Market hours check
        market_open = time(9, 30)   # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM
        
        return market_open <= current_time <= market_close
    except ImportError:
        # If pytz not available, assume markets are open for demo
        logger.warning("pytz not available, assuming markets open for demo")
        return True
    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return True  # Default to open for demo

# ========================================
# API ROUTES
# ========================================

@app.get("/", response_class=HTMLResponse)
async def get_trading_interface():
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
                content=f"""
                <h1>🚀 HYPER Trading System</h1>
                <p>❌ Frontend index.html not found</p>
                <p>Expected location: {index_file}</p>
                <p>Current directory: {current_dir}</p>
                <p>Files in directory: {list(current_dir.iterdir())}</p>
                """,
                status_code=404
            )
            
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(
            content=f"""
            <h1>🚀 HYPER Trading System</h1>
            <p>❌ Error loading frontend: {str(e)}</p>
            <p>Current directory: {current_dir}</p>
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
        "current_dir": str(current_dir),
        "index_file_exists": index_file.exists(),
        "market_open": is_market_hours()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current signals for all tickers"""
    return {
        "signals": manager._serialize_signals(hyper_state.current_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": serialize_stats(hyper_state.stats)
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
        **serialize_stats(hyper_state.stats),
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
                    "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
                    "stats": serialize_stats(hyper_state.stats)
                })
            
            elif message.get("type") == "request_stats":
                await manager.send_personal_message(websocket, {
                    "type": "stats_update",
                    "stats": serialize_stats(hyper_state.stats),
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
    
    # Check market status
    market_open = is_market_hours()
    logger.info(f"📈 Market status: {'OPEN' if market_open else 'CLOSED'}")
    
    # Log path information
    logger.info(f"🔧 Running from: {current_dir}")
    logger.info(f"🔧 Index file exists: {index_file.exists()}")
    
    # Validate configuration
    try:
        config.validate_config()
        logger.info("✅ Configuration validated successfully")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
    
    # Pre-generate initial signals (works even when markets closed)
    logger.info("Generating initial signals...")
    try:
        initial_signals = await hyper_state.signal_engine.generate_all_signals()
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
        logger.info("✅ Initial signals generated successfully")
    except Exception as e:
        logger.error(f"❌ Failed to generate initial signals: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # AUTO-START THE SYSTEM (works 24/7 for demo purposes)
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    
    # Start background signal generation
    asyncio.create_task(signal_generation_loop())
    
    if market_open:
        logger.info("🔥 HYPER signal generation auto-started! (Markets OPEN)")
    else:
        logger.info("🔥 HYPER signal generation auto-started! (Markets CLOSED - using cached/demo data)")
    
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
