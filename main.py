from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import yfinance as yf
import os

app = FastAPI(title="HYPERtrends API", version="2.0")

# Global state
class TradingState:
    def __init__(self):
        self.is_trading = False
        self.portfolio_value = 100000.0
        self.positions = {}
        self.signals = []
        self.connected_clients = []

trading_state = TradingState()

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Serve the HTML interface
@app.get("/")
async def get_interface():
    return FileResponse("index.html")

# API endpoints
@app.get("/api/portfolio")
async def get_portfolio():
    return {
        "value": trading_state.portfolio_value,
        "positions": trading_state.positions,
        "is_trading": trading_state.is_trading,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    if not trading_state.is_trading:
        trading_state.is_trading = True
        background_tasks.add_task(trading_loop)
        await manager.broadcast({
            "type": "trading_status",
            "status": "started",
            "message": "Live trading started!"
        })
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/api/trading/stop")
async def stop_trading():
    trading_state.is_trading = False
    await manager.broadcast({
        "type": "trading_status", 
        "status": "stopped",
        "message": "Trading stopped"
    })
    return {"status": "stopped"}

@app.post("/api/emergency-stop")
async def emergency_stop():
    trading_state.is_trading = False
    await manager.broadcast({
        "type": "emergency_stop",
        "message": "EMERGENCY STOP ACTIVATED"
    })
    return {"status": "emergency_stopped"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message["type"] == "request_update":
                await send_full_update(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Trading logic
async def trading_loop():
    while trading_state.is_trading:
        try:
            await process_signals()
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Trading loop error: {e}")
            await asyncio.sleep(10)

async def process_signals():
    tickers = ['QQQ', 'SPY', 'NVDA', 'AAPL', 'MSFT']
    new_signals = []
    
    for ticker in tickers:
        try:
            # Simulate your model predictions
            confidence = np.random.uniform(0.5, 0.95)
            price = np.random.uniform(100, 500)
            
            signal = {
                "ticker": ticker,
                "action": "BUY" if confidence > 0.65 else "HOLD",
                "confidence": round(confidence * 100, 1),
                "price": round(price, 2),
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "rf": round(np.random.uniform(50, 95), 1),
                    "xgb": round(np.random.uniform(50, 95), 1),
                    "lstm": round(np.random.uniform(50, 95), 1)
                }
            }
            new_signals.append(signal)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    trading_state.signals = new_signals
    
    await manager.broadcast({
        "type": "signals_update",
        "signals": new_signals,
        "timestamp": datetime.now().isoformat()
    })

async def send_full_update(websocket: WebSocket):
    update = {
        "type": "full_update",
        "portfolio": {
            "value": trading_state.portfolio_value,
            "positions": trading_state.positions
        },
        "signals": trading_state.signals,
        "is_trading": trading_state.is_trading,
        "timestamp": datetime.now().isoformat()
    }
    await websocket.send_text(json.dumps(update))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
