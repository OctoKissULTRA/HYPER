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

# Import configuration and combined signal engine
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine, HYPERSignal

# ========================================
# ENHANCED LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG.get("level", "INFO")),
    format=config.LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    title="⚡ HYPER Trading System - Combined Enhanced",
    description="Advanced AI-powered trading signals with combined enhanced predictive capabilities",
    version="2.5.0-COMBINED"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# GLOBAL STATE WITH COMBINED ENHANCEMENTS
# ========================================
class CombinedHYPERState:
    """Combined enhanced global application state"""
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None
        self.current_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.update_task = None
        
        # Enhanced statistics
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "api_calls_made": 0,
            "signals_with_data": 0,
            "fallback_signals": 0,
            "average_confidence": 0.0,
            "high_confidence_signals": 0,
            "ml_predictions_made": 0,
            "anomalies_detected": 0,
            "vix_sentiment_updates": 0,
            "williams_r_signals": 0,
            "fibonacci_levels_calculated": 0,
            "pattern_recognitions": 0
        }
    
    async def initialize(self):
        """Initialize the Combined Enhanced HYPER system"""
        logger.info("🚀 Starting Combined Enhanced HYPER Trading System...")
        logger.info(f"Tracking tickers: {', '.join(config.TICKERS)}")
        logger.info(f"Alpha Vantage API configured: {'✅' if config.ALPHA_VANTAGE_API_KEY else '❌'}")
        
        try:
            # Initialize data aggregator
            self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
            logger.info("✅ Data aggregator initialized")
            
            # Initialize combined enhanced signal engine
            self.signal_engine = HYPERSignalEngine()
            logger.info("✅ Combined enhanced signal engine initialized")
            
            # Validate configuration
            logger.info(f"🔧 Running from: {os.getcwd()}")
            logger.info(f"🔧 Index file exists: {index_file.exists()}")
            
            config.validate_config()
            logger.info("✅ Configuration validated successfully")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Combined HYPER system: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False

hyper_state = CombinedHYPERState()

# ========================================
# ENHANCED WEBSOCKET CONNECTION MANAGER
# ========================================
class CombinedConnectionManager:
    """Combined enhanced WebSocket connection manager"""
    
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
            "signals": self._serialize_combined_signals(hyper_state.current_signals),
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
    
    def _serialize_combined_signals(self, signals):
        """Convert combined enhanced signals to JSON-serializable format"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal in signals.items():
            if hasattr(signal, '__dict__'):
                # Extract combined enhanced signal data
                serialized[symbol] = {
                    "symbol": signal.symbol,
                    "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                    "confidence": float(getattr(signal, 'confidence', 0.0)),
                    "direction": getattr(signal, 'direction', 'NEUTRAL'),
                    "price": float(getattr(signal, 'price', 0.0)),
                    "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                    
                    # Core scores
                    "technical_score": float(getattr(signal, 'technical_score', 50.0)),
                    "momentum_score": float(getattr(signal, 'momentum_score', 50.0)),
                    "sentiment_score": float(getattr(signal, 'sentiment_score', 50.0)),
                    "ml_score": float(getattr(signal, 'ml_score', 50.0)),
                    
                    # Enhanced indicators
                    "williams_r": float(getattr(signal, 'williams_r', -50.0)),
                    "stochastic_k": float(getattr(signal, 'stochastic_k', 50.0)),
                    "stochastic_d": float(getattr(signal, 'stochastic_d', 50.0)),
                    "vix_sentiment": getattr(signal, 'vix_sentiment', 'NEUTRAL'),
                    "market_breadth": float(getattr(signal, 'market_breadth', 50.0)),
                    "sector_rotation": getattr(signal, 'sector_rotation', 'NEUTRAL'),
                    "anomaly_score": float(getattr(signal, 'anomaly_score', 0.0)),
                    "pattern_score": float(getattr(signal, 'pattern_score', 50.0)),
                    "economic_score": float(getattr(signal, 'economic_score', 50.0)),
                    "var_95": float(getattr(signal, 'var_95', 5.0)),
                    "correlation_spy": float(getattr(signal, 'correlation_spy', 0.7)),
                    
                    # Supporting data
                    "fibonacci_levels": getattr(signal, 'fibonacci_levels', {}),
                    "lstm_predictions": getattr(signal, 'lstm_predictions', {}),
                    "ensemble_prediction": getattr(signal, 'ensemble_prediction', {}),
                    "volume_profile": getattr(signal, 'volume_profile', {}),
                    "economic_sentiment": getattr(signal, 'economic_sentiment', {}),
                    "reasons": getattr(signal, 'reasons', []),
                    "warnings": getattr(signal, 'warnings', []),
                    "data_quality": getattr(signal, 'data_quality', 'unknown')
                }
            else:
                # Already a dict
                serialized[symbol] = signal
        
        logger.info(f"📡 Serialized {len(serialized)} combined enhanced signals for frontend")
        return serialized

manager = CombinedConnectionManager()

# ========================================
# COMBINED ENHANCED SIGNAL GENERATION LOOP
# ========================================
async def combined_signal_generation_loop():
    """Combined enhanced background signal generation with all features"""
    logger.info("🚀 Starting Combined Enhanced HYPER signal generation loop...")
    
    while hyper_state.is_running:
        try:
            loop_start_time = datetime.now()
            logger.info("🧠 Generating combined enhanced signals for all tickers...")
            
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
            
            # Analyze signal quality and calculate enhanced statistics
            signals_with_data = 0
            fallback_signals = 0
            high_confidence_count = 0
            confidence_sum = 0
            ml_predictions = 0
            anomalies = 0
            vix_updates = 0
            williams_signals = 0
            fibonacci_calculations = 0
            pattern_recognitions = 0
            
            for symbol, signal in signals.items():
                if hasattr(signal, 'price') and signal.price > 0:
                    signals_with_data += 1
                else:
                    fallback_signals += 1
                
                confidence = getattr(signal, 'confidence', 0)
                confidence_sum += confidence
                
                if confidence >= 80:
                    high_confidence_count += 1
                
                # Count enhanced features
                if hasattr(signal, 'lstm_predictions') and signal.lstm_predictions:
                    ml_predictions += 1
                if hasattr(signal, 'anomaly_score') and signal.anomaly_score > 20:
                    anomalies += 1
                if hasattr(signal, 'vix_sentiment') and signal.vix_sentiment != 'NEUTRAL':
                    vix_updates += 1
                if hasattr(signal, 'williams_r') and signal.williams_r != -50:
                    williams_signals += 1
                if hasattr(signal, 'fibonacci_levels') and signal.fibonacci_levels:
                    fibonacci_calculations += 1
                if hasattr(signal, 'pattern_score') and signal.pattern_score > 0:
                    pattern_recognitions += 1
            
            # Update enhanced statistics
            hyper_state.stats.update({
                "signals_with_data": signals_with_data,
                "fallback_signals": fallback_signals,
                "average_confidence": confidence_sum / len(signals) if signals else 0,
                "high_confidence_signals": high_confidence_count,
                "ml_predictions_made": ml_predictions,
                "anomalies_detected": anomalies,
                "vix_sentiment_updates": vix_updates,
                "williams_r_signals": williams_signals,
                "fibonacci_levels_calculated": fibonacci_calculations,
                "pattern_recognitions": pattern_recognitions
            })
            
            # Log detailed signal information
            signal_details = []
            for symbol, signal in signals.items():
                if hasattr(signal, '__dict__'):
                    signal_type = getattr(signal, 'signal_type', 'HOLD')
                    confidence = getattr(signal, 'confidence', 0.0)
                    price = getattr(signal, 'price', 0.0)
                    williams = getattr(signal, 'williams_r', -50)
                    vix_sentiment = getattr(signal, 'vix_sentiment', 'N/A')
                    ml_conf = getattr(signal, 'ml_score', 50)
                    
                    detail = f"{symbol}: {signal_type} ({confidence:.1f}%) ${price:.2f} [W%R:{williams:.0f}, VIX:{vix_sentiment}, ML:{ml_conf:.0f}]"
                    signal_details.append(detail)
                else:
                    signal_details.append(f"{symbol}: {signal}")
            
            logger.info(f"Generated signals: {', '.join(signal_details)}")
            logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
            logger.info(f"📊 Data quality: {signals_with_data}/{len(signals)} with real data")
            logger.info(f"🎯 High confidence: {high_confidence_count}/{len(signals)} signals")
            logger.info(f"🧠 Enhanced features: ML:{ml_predictions}, VIX:{vix_updates}, Williams:{williams_signals}, Patterns:{pattern_recognitions}")
            
            # Check performance thresholds
            total_loop_time = (datetime.now() - loop_start_time).total_seconds()
            max_time = config.PERFORMANCE_THRESHOLDS.get("total_update_cycle_max_time", 30.0) if hasattr(config, 'PERFORMANCE_THRESHOLDS') else 30.0
            if total_loop_time > max_time:
                logger.warning(f"⚠️ Slow update cycle: {total_loop_time:.2f}s (threshold: {max_time}s)")
            
            # Broadcast to clients
            await manager.broadcast({
                "type": "signal_update",
                "signals": manager._serialize_combined_signals(signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(),
                "generation_time": generation_time,
                "performance": {
                    "signals_per_second": len(signals) / generation_time if generation_time > 0 else 0,
                    "total_loop_time": total_loop_time,
                    "enhanced_features_active": williams_signals + fibonacci_calculations + pattern_recognitions
                }
            })
            
            # Wait for next update
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            logger.error(f"💥 Error in combined signal generation loop: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            await asyncio.sleep(30)  # Wait longer on error

# ========================================
# API ROUTES (Enhanced)
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
                content="<h1>HYPER Trading System - Combined Enhanced</h1><p>Frontend not found</p>",
                status_code=404
            )
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Combined enhanced system health check"""
    uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
    
    return {
        "status": "healthy",
        "version": "2.5.0-COMBINED",
        "is_running": hyper_state.is_running,
        "connected_clients": len(manager.active_connections),
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "uptime_seconds": uptime,
        "statistics": hyper_state.stats,
        "performance": {
            "signals_per_minute": hyper_state.stats["total_signals_generated"] / (uptime / 60) if uptime > 0 else 0,
            "average_confidence": hyper_state.stats["average_confidence"],
            "success_rate": (hyper_state.stats["signals_with_data"] / max(1, hyper_state.stats["total_signals_generated"])) * 100,
            "enhanced_features_usage": {
                "ml_predictions": hyper_state.stats["ml_predictions_made"],
                "vix_updates": hyper_state.stats["vix_sentiment_updates"],
                "williams_signals": hyper_state.stats["williams_r_signals"],
                "fibonacci_calculations": hyper_state.stats["fibonacci_levels_calculated"],
                "pattern_recognitions": hyper_state.stats["pattern_recognitions"]
            }
        },
        "configuration": {
            "tickers": config.TICKERS,
            "signal_weights": getattr(config, 'SIGNAL_WEIGHTS', {}),
            "system_initialized": hyper_state.signal_engine is not None,
            "index_file_exists": index_file.exists(),
            "api_key_configured": bool(config.ALPHA_VANTAGE_API_KEY)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current combined enhanced signals for all tickers"""
    return {
        "signals": manager._serialize_combined_signals(hyper_state.current_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats.copy(),
        "enhanced_features": {
            "williams_r": True,
            "vix_sentiment": True,
            "fibonacci_levels": True,
            "ml_predictions": True,
            "pattern_recognition": True,
            "market_structure": True
        }
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get combined enhanced signal for specific symbol"""
    symbol = symbol.upper()
    if symbol not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not tracked")
    
    if symbol in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol]
        return manager._serialize_combined_signals({symbol: signal})[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"No signal available for {symbol}")

@app.post("/api/start")
async def start_system():
    """Start the combined enhanced signal generation system"""
    if hyper_state.is_running:
        return {"status": "already_running", "message": "Combined Enhanced HYPER is already running"}
    
    if not hyper_state.signal_engine:
        success = await hyper_state.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize Combined Enhanced HYPER system")
    
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    
    # Start background task
    asyncio.create_task(combined_signal_generation_loop())
    
    logger.info("🚀 Combined Enhanced HYPER signal generation started")
    return {"status": "started", "message": "Combined Enhanced HYPER signal generation started"}

@app.post("/api/stop")
async def stop_system():
    """Stop the signal generation system"""
    if not hyper_state.is_running:
        return {"status": "not_running", "message": "HYPER is not running"}
    
    hyper_state.is_running = False
    logger.info("⏸️ Combined Enhanced HYPER signal generation stopped")
    return {"status": "stopped", "message": "Combined Enhanced HYPER signal generation stopped"}

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
    """Combined enhanced WebSocket endpoint for real-time communication"""
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
                    "signals": manager._serialize_combined_signals(hyper_state.current_signals),
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
    """Application startup with combined enhanced features"""
    logger.info("🚀 Starting Combined Enhanced HYPER Trading System...")
    
    # Market status check
    current_hour = datetime.now().hour
    current_weekday = datetime.now().weekday()
    
    if current_weekday < 5:  # Monday-Friday
        if 9 <= current_hour <= 16:
            market_status = "OPEN"
        elif 4 <= current_hour < 9:
            market_status = "PRE_MARKET"
        elif 16 < current_hour <= 20:
            market_status = "AFTER_HOURS"
        else:
            market_status = "CLOSED"
    else:
        market_status = "WEEKEND"
    
    logger.info(f"📈 Market status: {market_status}")
    
    # Initialize the combined system
    success = await hyper_state.initialize()
    if not success:
        logger.error("❌ Failed to initialize Combined Enhanced HYPER system")
        return
    
    # Generate initial signals
    logger.info("🧠 Generating initial combined enhanced signals...")
    try:
        initial_signals = await hyper_state.signal_engine.generate_all_signals()
        hyper_state.current_signals = initial_signals
        hyper_state.last_update = datetime.now()
        
        # Log initial signal summary
        signal_types = {}
        enhanced_features = 0
        for signal in initial_signals.values():
            signal_type = getattr(signal, 'signal_type', 'UNKNOWN')
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
            # Count enhanced features
            if hasattr(signal, 'williams_r') and signal.williams_r != -50:
                enhanced_features += 1
        
        logger.info(f"📊 Initial signals: {dict(signal_types)}")
        logger.info(f"⚡ Enhanced features active: {enhanced_features}/{len(initial_signals)}")
        logger.info("✅ Initial combined enhanced signals generated successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate initial signals: {e}")
    
    # Auto-start the system
    hyper_state.is_running = True
    hyper_state.stats["uptime_start"] = datetime.now()
    asyncio.create_task(combined_signal_generation_loop())
    
    status_msg = f"Markets {market_status} - using {'live' if market_status == 'OPEN' else 'cached/demo'} data"
    logger.info(f"🔥 Combined Enhanced HYPER signal generation auto-started! ({status_msg})")
    logger.info("🎯 Combined Enhanced HYPER Trading System ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown with cleanup"""
    logger.info("⏸️ Shutting down Combined Enhanced HYPER Trading System...")
    hyper_state.is_running = False
    
    # Close WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    
    # Cleanup signal engine resources
    if hyper_state.signal_engine and hasattr(hyper_state.signal_engine, 'cleanup'):
        try:
            await hyper_state.signal_engine.cleanup()
            logger.info("🧹 Signal engine cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    # Log final statistics
    if hyper_state.stats["total_signals_generated"] > 0:
        uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
        signals_per_minute = hyper_state.stats["total_signals_generated"] / (uptime / 60) if uptime > 0 else 0
        
        logger.info(f"📊 Final statistics:")
        logger.info(f"   • Total signals generated: {hyper_state.stats['total_signals_generated']}")
        logger.info(f"   • Average confidence: {hyper_state.stats['average_confidence']:.1f}%")
        logger.info(f"   • High confidence signals: {hyper_state.stats['high_confidence_signals']}")
        logger.info(f"   • Enhanced features used: ML:{hyper_state.stats['ml_predictions_made']}, VIX:{hyper_state.stats['vix_sentiment_updates']}")
        logger.info(f"   • Signals per minute: {signals_per_minute:.1f}")
        logger.info(f"   • Uptime: {uptime:.0f} seconds")
    
    logger.info("👋 Combined Enhanced HYPER shutdown complete")

# ========================================
# MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    logger.info("🚀 Starting Combined Enhanced HYPER Trading System server...")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG.get("reload", False),
        log_level="info"
    )
