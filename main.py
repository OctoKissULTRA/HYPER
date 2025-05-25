# main.py

import os
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"], "INFO"),
    format=config.LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# ========================================
# FRONTEND CONFIGURATION
# ========================================
current_dir = Path(__file__).parent
index_file = current_dir / "index.html"

# ========================================
# FASTAPI APP
# ========================================
app = FastAPI(
    title="⚡ HYPERtrends",
    description="AI-powered combined enhanced trading signals",
    version="2.5.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ========================================
# GLOBAL STATE
# ========================================
class CombinedHYPERState:
    def __init__(self):
        self.is_running = False
        self.data_aggregator: HYPERDataAggregator = None
        self.signal_engine: HYPERSignalEngine  = None
        self.current_signals: dict           = {}
        self.last_update: str                = None
        self.stats: dict = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now().isoformat()
        }

    async def initialize(self):
        logger.info("🚀 Initializing Combined HYPERtrends…")
        config.validate_config()
        self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
        self.signal_engine   = HYPERSignalEngine()
        return True

hyper_state = CombinedHYPERState()

# ========================================
# WEBSOCKET MANAGER
# ========================================
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        hyper_state.stats["clients_connected"] = len(self.active)
        await self.send_personal(ws, {
            "type":"signal_update",
            "signals": {s.symbol:s for s in hyper_state.current_signals.values()},
            "stats": hyper_state.stats,
            "timestamp": hyper_state.last_update
        })

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
            hyper_state.stats["clients_connected"] = len(self.active)

    async def send_personal(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_text(json.dumps(msg, default=str))
        except:
            self.disconnect(ws)

    async def broadcast(self, msg: dict):
        data = json.dumps(msg, default=str)
        for c in list(self.active):
            try:
                await c.send_text(data)
            except:
                self.disconnect(c)

manager = ConnectionManager()

# ========================================
# SIGNAL GENERATION LOOP
# ========================================
async def signal_loop():
    while hyper_state.is_running:
        try:
            signals = await hyper_state.signal_engine.generate_all_signals()
            hyper_state.current_signals = signals
            hyper_state.last_update = datetime.now().isoformat()
            hyper_state.stats["total_signals_generated"] += len(signals)

            await manager.broadcast({
                "type": "signal_update",
                "signals": {s.symbol: s for s in signals.values()},
                "stats": hyper_state.stats,
                "timestamp": hyper_state.last_update
            })

            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
        except Exception as e:
            logger.error(f"Signal loop error: {e}")
            await asyncio.sleep(5)

# ========================================
# STARTUP / SHUTDOWN
# ========================================
@app.on_event("startup")
async def on_startup():
    success = await hyper_state.initialize()
    if not success:
        raise RuntimeError("Initialization failed")
    hyper_state.is_running = True
    asyncio.create_task(signal_loop())
    logger.info("🎯 Signal loop started")

@app.on_event("shutdown")
async def on_shutdown():
    hyper_state.is_running = False
    if hyper_state.data_aggregator:
        await hyper_state.data_aggregator.close()
    if hyper_state.signal_engine:
        await hyper_state.signal_engine.cleanup()
    logger.info("🛑 Shutdown complete")

# ========================================
# ENDPOINTS
# ========================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if index_file.exists():
        return HTMLResponse(index_file.read_text("utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "last_update": hyper_state.last_update,
        "stats": hyper_state.stats
    }

@app.get("/api/signals")
async def get_signals():
    return {
        "signals": {s.symbol: s for s in hyper_state.current_signals.values()},
        "timestamp": hyper_state.last_update,
        "stats": hyper_state.stats
    }

@app.get("/api/signals/{symbol}")
async def get_signal(symbol: str):
    sym = symbol.upper()
    sig = hyper_state.current_signals.get(sym)
    if not sig:
        raise HTTPException(404, f"No signal for {sym}")
    return sig

@app.post("/api/start")
async def api_start():
    if hyper_state.is_running:
        return {"status":"already_running"}
    await hyper_state.initialize()
    hyper_state.is_running = True
    asyncio.create_task(signal_loop())
    return {"status":"started"}

@app.post("/api/stop")
async def api_stop():
    if not hyper_state.is_running:
        return {"status":"not_running"}
    hyper_state.is_running = False
    return {"status":"stopped"}

@app.post("/api/emergency-stop")
async def api_emergency():
    hyper_state.is_running = False
    return {"status":"emergency_stopped"}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    uvicorn.run("main:app",
                host=config.SERVER_CONFIG["host"],
                port=config.SERVER_CONFIG["port"],
                log_level="info")
