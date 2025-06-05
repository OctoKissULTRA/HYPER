import os
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
import uvicorn

import config
from signal_engine import HYPERSignalEngine

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG if config.VERBOSE else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

signal_engine = HYPERSignalEngine()

@app.get("/")
async def get_ui():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": config.ENVIRONMENT}

@app.get("/signal")
async def get_signal(symbol: str = "QQQ"):
    result = signal_engine.generate_signal(symbol)
    return {"symbol": symbol, "signal": result}

@app.websocket("/ws/signal")
async def websocket_signal(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = signal_engine.generate_signal(data)
            await websocket.send_json({"symbol": data, "signal": result})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()