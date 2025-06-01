# FINAL PATCHED main.py

import os
import json
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from data_sources import get_data_source
from signal_engine import HYPERSignalEngine

# State and logger setup
app = FastAPI(title="HYPERtrends v4.0")
logger = logging.getLogger("hypertrends")
logging.basicConfig(level=logging.INFO)

hyper_state = {
    "data_aggregator": None,
    "signal_engine": None,
    "current_signals": {},
    "last_update": None,
    "is_running": False,
    "initialization_complete": False,
    "connected_clients": [],
    "stats": {
        "status": "initializing",
        "initialization_complete": False,
        "last_signal_count": 0
    }
}

# Serve static HTML
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse("Frontend unavailable", status_code=500)

@app.get("/health")
async def health_check():
    return {
        "status": hyper_state["stats"]["status"],
        "initialization_complete": hyper_state["initialization_complete"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_signals():
    return {
        "signals": hyper_state["current_signals"],
        "timestamp": hyper_state["last_update"].isoformat() if hyper_state["last_update"] else None,
        "status": hyper_state["stats"]["status"],
        "version": "4.0.0"
    }

@app.post("/api/start")
async def start_system():
    if not hyper_state["initialization_complete"]:
        asyncio.create_task(background_initialization())
        return {"status": "initializing"}
    hyper_state["is_running"] = True
    return {"status": "started"}

@app.post("/api/stop")
async def stop_system():
    hyper_state["is_running"] = False
    return {"status": "stopped"}

# === WebSocket Management ===

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(json.dumps(message, default=str))
            except:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(15)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "status": hyper_state["stats"]["status"]
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# === Background Systems ===

async def background_initialization():
    logger.info("⚙️ Background system initialization started...")
    try:
        aggregator = get_data_source()
        await aggregator.initialize()

        signal_engine = HYPERSignalEngine()
        hyper_state["data_aggregator"] = aggregator
        hyper_state["signal_engine"] = signal_engine

        signals = await signal_engine.generate_all_signals()
        hyper_state["current_signals"] = signals
        hyper_state["last_update"] = datetime.now()

        hyper_state["is_running"] = True
        hyper_state["initialization_complete"] = True
        hyper_state["stats"]["initialization_complete"] = True
        hyper_state["stats"]["status"] = "fully_operational"
        logger.info("✅ HYPERtrends system initialized successfully")

        asyncio.create_task(signal_loop())

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        await create_fallback_signals()
        hyper_state["stats"]["status"] = "simulation_only"

async def signal_loop():
    logger.info("🚀 Starting signal generation loop")
    while hyper_state["is_running"]:
        try:
            engine = hyper_state["signal_engine"]
            if not engine:
                raise RuntimeError("Signal engine missing")

            new_signals = await engine.generate_all_signals()
            hyper_state["current_signals"] = new_signals
            hyper_state["last_update"] = datetime.now()
            hyper_state["stats"]["last_signal_count"] = len(new_signals)

            await manager.broadcast({
                "type": "signal_update",
                "signals": new_signals,
                "timestamp": hyper_state["last_update"].isoformat()
            })

            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])

        except Exception as e:
            logger.error(f"❌ Signal generation loop error: {e}")
            await asyncio.sleep(30)

async def create_fallback_signals():
    logger.info("🔄 Creating fallback signals...")
    now = datetime.now().isoformat()
    fallback_signals = {
        symbol: {
            "symbol": symbol,
            "signal_type": "HOLD",
            "confidence": 50.0,
            "direction": "NEUTRAL",
            "price": 100.0,
            "timestamp": now,
            "technical_score": 50.0,
            "sentiment_score": 50.0,
            "data_source": "fallback"
        }
        for symbol in config.TICKERS
    }
    hyper_state["current_signals"] = fallback_signals
    hyper_state["last_update"] = datetime.now()