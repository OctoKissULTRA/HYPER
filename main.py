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

# Logging
logging.basicConfig(level=getattr(logging, config.LOGGING_CONFIG["level"], "INFO"),
                    format=config.LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# Frontend
current_dir = Path(__file__).parent
index_file = current_dir / "index.html"

# App
app = FastAPI(title="HYPERtrends", version="2.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class CombinedHYPERState:
    def __init__(self):
        self.is_running = False
        self.data_aggregator: HYPERDataAggregator = None
        self.signal_engine: HYPERSignalEngine = None
        self.current_signals = {}
        self.last_update = None
        self.stats = {"total":0, "clients":0, "start":datetime.now().isoformat()}

    async def initialize(self):
        logger.info("Initializing system…")
        self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
        self.signal_engine = HYPERSignalEngine()
        config.validate_config()
        return True

hyper_state = CombinedHYPERState()

# WebSocket manager (simplified)
class ConnectionManager:
    def __init__(self):
        self.connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        self.stats_update()

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)
        self.stats_update()

    async def broadcast(self, msg: dict):
        data = json.dumps(msg, default=str)
        for c in self.connections:
            await c.send_text(data)

    def stats_update(self):
        hyper_state.stats["clients"] = len(self.connections)

manager = ConnectionManager()

# Signal loop
async def signal_loop():
    while hyper_state.is_running:
        signals = await hyper_state.signal_engine.generate_all_signals()
        hyper_state.current_signals = signals
        hyper_state.last_update = datetime.now().isoformat()
        hyper_state.stats["total"] += len(signals)
        await manager.broadcast({"type":"signal_update",
                                  "signals": {s.symbol:s for s in signals.values()},
                                  "stats": hyper_state.stats,
                                  "time":hyper_state.last_update})
        await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])

@app.on_event("startup")
async def on_startup():
    await hyper_state.initialize()
    hyper_state.is_running = True
    asyncio.create_task(signal_loop())

@app.on_event("shutdown")
async def on_shutdown():
    hyper_state.is_running = False
    if hyper_state.data_aggregator:
        await hyper_state.data_aggregator.close()
    if hyper_state.signal_engine:
        await hyper_state.signal_engine.cleanup()

@app.get("/", response_class=HTMLResponse)
async def root():
    if index_file.exists():
        return HTMLResponse(index_file.read_text("utf-8"))
    raise HTTPException(404, "Frontend not found")

@app.get("/health")
async def health():
    return {"status":"ok","last_update":hyper_state.last_update,"stats":hyper_state.stats}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.SERVER_CONFIG["host"], port=config.SERVER_CONFIG["port"])
