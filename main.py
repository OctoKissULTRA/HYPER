# main.py - HYPERtrends v4.1 API Backend (Alpaca + yfinance + robust status)
import os
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from data_sources import HYPERDataAggregator, SUPPORTED_SYMBOLS, get_market_status

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, getattr(config, "LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="HYPERtrends v4.1 - Alpaca Edition",
    description="AI-powered trading signals and analytics platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
def root():
    return HTMLResponse("<h1>HYPERtrends v4.1 API is running.</h1>")

@app.get("/api/ping")
def ping():
    return {"status": "ok", "ts": str(datetime.utcnow()), "data_status": get_market_status()}

@app.get("/api/status")
def status():
    # Returns status of all supported tickers
    status = HYPERDataAggregator.get_all_status()
    return {"status": status, "ts": str(datetime.utcnow()), "data_status": get_market_status()}

@app.get("/api/signals")
def get_signals():
    # This is a minimal placeholder; expand as needed for your ML engine!
    signals = {}
    for symbol in SUPPORTED_SYMBOLS:
        price_data = HYPERDataAggregator.get_current_price(symbol)
        signals[symbol] = {
            "symbol": symbol,
            "price": price_data["price"],
            "confidence": 70,  # placeholder; replace with ML signal
            "signal_type": "HOLD",  # placeholder; replace with real logic
            "timestamp": price_data["timestamp"],
        }
    return {
        "signals": signals,
        "data_status": get_market_status(),
        "ts": str(datetime.utcnow())
    }

@app.get("/api/history")
def get_history(
    symbol: str = Query(...),
    interval: str = Query("1d"),
    limit: int = Query(30)
):
    bars = HYPERDataAggregator.get_historical_bars(symbol, interval, limit)
    return {
        "symbol": symbol,
        "interval": interval,
        "data": bars,
        "data_status": get_market_status(),
        "ts": str(datetime.utcnow())
    }

# =======================
# WebSocket (minimal demo)
# =======================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Periodically push signals to the frontend
            signals = {}
            for symbol in SUPPORTED_SYMBOLS:
                price_data = HYPERDataAggregator.get_current_price(symbol)
                signals[symbol] = {
                    "symbol": symbol,
                    "price": price_data["price"],
                    "confidence": 70,  # placeholder
                    "signal_type": "HOLD",
                    "timestamp": price_data["timestamp"],
                }
            data = {
                "type": "signal_update",
                "signals": signals,
                "data_status": get_market_status(),
                "ts": str(datetime.utcnow())
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ===========================
# Error Handlers, etc
# ===========================
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return HTMLResponse(content=f"Internal error: {exc}", status_code=500)
