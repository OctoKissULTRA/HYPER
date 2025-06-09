# main.py - HYPERtrends v4.0 - Optimized Production Ready

import os
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import core modules
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="HYPERtrends v4.0 - Alpaca Edition",
    description="AI-powered trading signals with Alpaca Markets integration",
    version="4.0.0-ALPACA",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
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
    def __init__(self):
        self.is_running = False
        self.initialization_complete = False
        self.startup_time = datetime.now()
        self.connected_clients = []
        self.current_signals = {}
        self.last_update = None

        # Components (lazy loaded)
        self.data_aggregator = None
        self.signal_engine = None
        
        self.stats = {
            "status": "starting",
            "uptime_start": datetime.now(),
            "alpaca_available": config.has_alpaca_credentials(),
            "data_source_status": config.get_data_source_status(),
            "initialization_complete": False,
            "signals_generated": 0,
            "accuracy_rate": 0.0
        }
        
        logger.info("🚀 HYPER state initialized")

hyper_state = HYPERState()

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
        logger.info(f"📡 WebSocket connected. Total: {len(self.active_connections)}")

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

manager = ConnectionManager()

# ========================================
# STARTUP & SHUTDOWN
# ========================================
@app.on_event("startup")
async def startup_event():
    """Fast startup with background initialization"""
    try:
        logger.info("🚀 Starting HYPERtrends v4.0 - Alpaca Edition")
        
        # Print startup banner
        print_startup_info()
        
        hyper_state.stats["status"] = "online"
        asyncio.create_task(background_initialization())
        logger.info("✅ Server started - Background initialization in progress")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        hyper_state.stats["status"] = "error"

async def background_initialization():
    """Initialize components in background"""
    try:
        logger.info("🔧 Starting background initialization...")

        # Initialize data aggregator
        logger.info("📊 Initializing Alpaca data aggregator...")
        hyper_state.data_aggregator = HYPERDataAggregator()
        await hyper_state.data_aggregator.initialize()
        
        # Initialize signal engine
        logger.info("🧠 Initializing signal engine...")
        hyper_state.signal_engine = HYPERSignalEngine()
        
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
        await create_fallback_signals()

async def generate_all_signals() -> Dict[str, Any]:
    """Generate signals for all tickers"""
    signals = {}

    try:
        if not hyper_state.signal_engine or not hyper_state.data_aggregator:
            return await create_fallback_signals()
        
        # Generate signals for each ticker
        for symbol in config.TICKERS:
            try:
                # Get comprehensive data (async)
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

async def create_fallback_signals() -> Dict[str, Any]:
    """Create basic fallback signals"""
    signals = {}

    for symbol in config.TICKERS:
        signals[symbol] = create_fallback_signal(symbol)

    hyper_state.current_signals = signals
    hyper_state.last_update = datetime.now()
    hyper_state.is_running = True

    logger.info("✅ Fallback signals created")
    return signals

def create_fallback_signal(symbol: str) -> Dict[str, Any]:
    """Create a single fallback signal"""
    base_prices = {
        "QQQ": 450.25, "SPY": 535.80, "NVDA": 875.90,
        "AAPL": 185.45, "MSFT": 428.75
    }
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
        "data_source": "fallback",
        "reasons": ["System initializing"],
        "warnings": [],
        "recommendations": ["Wait for full system initialization"]
    }

def serialize_signal(signal) -> Dict[str, Any]:
    """Convert signal object to serializable dict"""
    try:
        if hasattr(signal, "__dict__"):
            result = {}
            for key, value in signal.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    if hasattr(value, "__dict__"):
                        result[key] = {k: v for k, v in value.__dict__.items() 
                                     if not k.startswith("_") and not callable(v)}
                    else:
                        result[key] = value
            return result
        elif isinstance(signal, dict):
            return signal
        else:
            return {"error": "unable_to_serialize", "type": str(type(signal))}
    except Exception as e:
        logger.error(f"❌ Signal serialization error: {e}")
        return {"error": str(e)}

async def signal_generation_loop():
    """Background signal generation loop"""
    while hyper_state.is_running:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            if hyper_state.signal_engine and hyper_state.data_aggregator:
                new_signals = await generate_all_signals()
                hyper_state.current_signals = new_signals
                hyper_state.last_update = datetime.now()
                
                # Broadcast to connected clients
                await manager.broadcast({
                    "type": "signal_update",
                    "signals": new_signals,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"❌ Signal generation loop error: {e}")
            await asyncio.sleep(30)

# ========================================
# API ENDPOINTS
# ========================================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends v4.0 - Alpaca Edition</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0a0e1a; color: #ffffff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .title { color: #00d4ff; font-size: 2.5em; margin: 0; }
            .subtitle { color: #888; margin: 10px 0; }
            .status { display: flex; justify-content: space-around; margin: 30px 0; }
            .status-card { background: #1a1f2e; padding: 20px; border-radius: 10px; min-width: 200px; }
            .signals { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .signal-card { background: #1a1f2e; padding: 20px; border-radius: 10px; border-left: 4px solid #00d4ff; }
            .signal-type { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
            .confidence { font-size: 1.5em; margin: 10px 0; }
            .buy { color: #00ff88; }
            .sell { color: #ff4444; }
            .hold { color: #ffaa00; }
            .loading { text-align: center; padding: 50px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">⚡ HYPERtrends v4.0</h1>
                <p class="subtitle">AI-Powered Trading Signals with Alpaca Markets Integration</p>
            </div>
            
            <div class="status" id="status">
                <div class="status-card">
                    <h3>System Status</h3>
                    <div id="system-status">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Data Source</h3>
                    <div id="data-source">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Last Update</h3>
                    <div id="last-update">Loading...</div>
                </div>
            </div>
            
            <div id="signals" class="signals">
                <div class="loading">🔄 Loading signals...</div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'signal_update') {
                        updateSignals(data.signals);
                    }
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected, reconnecting...');
                    setTimeout(connectWebSocket, 5000);
                };
            }
            
            function updateSignals(signals) {
                const container = document.getElementById('signals');
                container.innerHTML = '';
                
                for (const [symbol, signal] of Object.entries(signals)) {
                    const card = document.createElement('div');
                    card.className = 'signal-card';
                    
                    const typeClass = signal.signal_type.includes('BUY') ? 'buy' : 
                                     signal.signal_type.includes('SELL') ? 'sell' : 'hold';
                    
                    card.innerHTML = `
                        <div class="signal-type ${typeClass}">${symbol} - ${signal.signal_type}</div>
                        <div class="confidence ${typeClass}">${signal.confidence.toFixed(1)}% Confidence</div>
                        <div>Direction: ${signal.direction}</div>
                        <div>Price: $${signal.price}</div>
                        <div>Technical: ${signal.technical_score.toFixed(1)}</div>
                        <div>Sentiment: ${signal.sentiment_score.toFixed(1)}</div>
                        <div>Updated: ${new Date(signal.timestamp).toLocaleTimeString()}</div>
                    `;
                    
                    container.appendChild(card);
                }
            }
            
            async function loadInitialData() {
                try {
                    const response = await fetch('/api/signals');
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        updateSignals(data.signals);
                        document.getElementById('last-update').textContent = 
                            new Date(data.timestamp).toLocaleTimeString();
                    }
                    
                    const healthResponse = await fetch('/health');
                    const healthData = await healthResponse.json();
                    
                    document.getElementById('system-status').textContent = healthData.system_status;
                    document.getElementById('data-source').textContent = 'Alpaca Markets';
                    
                } catch (error) {
                    console.error('Failed to load initial data:', error);
                }
            }
            
            // Initialize
            loadInitialData();
            connectWebSocket();
            
            // Auto-refresh every 30 seconds
            setInterval(loadInitialData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_status": hyper_state.stats["status"],
        "initialization_complete": hyper_state.initialization_complete,
        "uptime_seconds": (datetime.now() - hyper_state.startup_time).total_seconds(),
        "connected_clients": len(hyper_state.connected_clients),
        "signals_available": len(hyper_state.current_signals),
        "data_source": config.get_data_source_status(),
        "version": "4.0.0-ALPACA"
    }

@app.get("/api/signals")
async def get_signals():
    """Get current trading signals"""
    return {
        "status": "success",
        "signals": hyper_state.current_signals,
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "timestamp": datetime.now().isoformat(),
        "system_status": hyper_state.stats["status"]
    }

@app.get("/api/signals/{symbol}")
async def get_signal(symbol: str):
    """Get signal for specific symbol"""
    symbol = symbol.upper()
    if symbol in hyper_state.current_signals:
        return {
            "status": "success",
            "symbol": symbol,
            "signal": hyper_state.current_signals[symbol],
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Signal for {symbol} not found")

@app.post("/api/signals/refresh")
async def refresh_signals():
    """Manually refresh all signals"""
    try:
        if hyper_state.signal_engine and hyper_state.data_aggregator:
            new_signals = await generate_all_signals()
            hyper_state.current_signals = new_signals
            hyper_state.last_update = datetime.now()
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

@app.get("/api/system/status")
async def system_status():
    """Get detailed system status"""
    return {
        "status": hyper_state.stats["status"],
        "initialization_complete": hyper_state.initialization_complete,
        "uptime": (datetime.now() - hyper_state.startup_time).total_seconds(),
        "data_source": config.get_data_source_status(),
        "alpaca_available": config.has_alpaca_credentials(),
        "connected_clients": len(hyper_state.connected_clients),
        "signals_generated": hyper_state.stats["signals_generated"],
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "components": {
            "data_aggregator": hyper_state.data_aggregator is not None,
            "signal_engine": hyper_state.signal_engine is not None,
        },
        "tickers": config.TICKERS,
        "version": "4.0.0-ALPACA"
    }

# ========================================
# WEBSOCKET ENDPOINT
# ========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            "type": "initial_data",
            "signals": hyper_state.current_signals,
            "timestamp": datetime.now().isoformat(),
            "status": hyper_state.stats["status"]
        }, default=str))
        
        # Keep connection alive
        while True:
            try:
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
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

# ========================================
# SHUTDOWN
# ========================================
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down HYPERtrends...")
    hyper_state.is_running = False
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, "cleanup"):
        await hyper_state.data_aggregator.cleanup()
    logger.info("✅ Shutdown complete")

# ========================================
# UTILITY FUNCTIONS
# ========================================
def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("⚡ HYPERtrends v4.0 - ALPACA EDITION ⚡")
    print("="*60)
    print(f"🕐 Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"🔧 Python: {os.sys.version.split()[0]}")
    print(f"📍 Port: {os.getenv('PORT', '8000')}")
    print(f"📊 Data Source: {config.get_data_source_status()}")
    print(f"📈 Tracking: {', '.join(config.TICKERS)}")
    print("="*60 + "\n")

# ========================================
# MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )