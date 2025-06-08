import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tenacity import retry, wait_exponential, stop_after_attempt
import json
from asyncio import Lock

from config import ALPACA_CONFIG, SECURITY_CONFIG, UPDATE_INTERVALS, TICKERS
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine
from ml_learning import integrate_ml_learning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HYPERtrends v4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=SECURITY_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()

class HYPERState:
    def __init__(self):
        self.is_running = False
        self.initialization_complete = False
        self.startup_time = datetime.now()
        self.connected_clients = []
        self.current_signals = {}
        self.last_update = None
        self._lock = Lock()
        self.data_aggregator = None
        self.signal_engine = None
        self.ml_engine = None
        self.stats = {
            "status": "starting",
            "uptime_start": datetime.now(),
            "alpaca_available": ALPACA_CONFIG.get("api_key") and ALPACA_CONFIG.get("secret_key"),
            "data_source_status": "initializing",
            "initialization_complete": False,
            "signals_generated": 0,
            "ml_predictions": 0,
            "accuracy_rate": 0.0
        }
        logger.info("HYPER state initialized")

hyper_state = HYPERState()

async def background_initialization():
    logger.info("Starting background initialization...")
    try:
        hyper_state.data_aggregator = HYPERDataAggregator(ALPACA_CONFIG)
        await hyper_state.data_aggregator.initialize()
        hyper_state.signal_engine = HYPERSignalEngine()
        await hyper_state.signal_engine.warm_up_analyzers()
        hyper_state.ml_engine, _ = integrate_ml_learning(hyper_state.signal_engine)
        hyper_state.stats["data_source_status"] = "Alpaca Markets (Paper Mode)"
        hyper_state.initialization_complete = True
        hyper_state.is_running = True
        hyper_state.stats["status"] = "running"
        hyper_state.stats["initialization_complete"] = True
        logger.info("Background initialization completed")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        hyper_state.stats["status"] = "error"

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_initialization())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down HYPERtrends...")
    hyper_state.is_running = False
    async with hyper_state._lock:
        if hyper_state.data_aggregator:
            await hyper_state.data_aggregator.cleanup()
        if hyper_state.ml_engine and hasattr(hyper_state.ml_engine, "cleanup"):
            await hyper_state.ml_engine.cleanup()
    logger.info("Shutdown complete")

@app.get("/health")
async def health_check():
    return {
        "status": hyper_state.stats["status"],
        "uptime": (datetime.now() - hyper_state.startup_time).total_seconds(),
        "initialization_complete": hyper_state.initialization_complete
    }

@app.get("/api/system/status")
async def system_status():
    return hyper_state.stats

@app.get("/api/signals")
async def get_signals():
    return hyper_state.current_signals

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def signal_generation_loop():
    while hyper_state.is_running:
        try:
            async with hyper_state._lock:
                await asyncio.sleep(UPDATE_INTERVALS["signal_generation"])
                if hyper_state.signal_engine and hyper_state.data_aggregator:
                    new_signals = await hyper_state.signal_engine.generate_all_signals(hyper_state.data_aggregator)
                    hyper_state.current_signals = {k: v.__dict__ for k, v in new_signals.items()}
                    hyper_state.last_update = datetime.now()
                    hyper_state.stats["signals_generated"] += len(new_signals)
                    await manager.broadcast({
                        "type": "signal_update",
                        "signals": hyper_state.current_signals,
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"Signal generation loop error: {e}")
            await asyncio.sleep(UPDATE_INTERVALS["signal_generation"])

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            await websocket.send_json({"status": "acknowledged", "message": data})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

@app.on_event("startup")
async def start_signal_loop():
    asyncio.create_task(signal_generation_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
