import os
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration and components
import config
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine, HYPERSignal
from model_testing import ModelTester, TestingAPI
from ml_learning import integrate_ml_learning, MLEnhancedSignalEngine, LearningAPI

# ========================================
# PRODUCTION LOGGING SETUP
# ========================================
def setup_logging():
    """Setup production-grade logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG.get("level", "INFO")),
        format=config.LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(config.LOGGING_CONFIG.get("file", "logs/hyper.log")),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# ========================================
# FRONTEND CONFIGURATION
# ========================================
current_dir = Path(__file__).parent
index_file = current_dir / "index.html"

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="⚡ HYPER Trading System - Production",
    description="Production-grade AI-powered trading signals",
    version="3.0.0-PRODUCTION",
    docs_url="/docs" if not config.is_production() else None,
    redoc_url="/redoc" if not config.is_production() else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.SECURITY_CONFIG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ========================================
# PRODUCTION STATE MANAGEMENT
# ========================================
class ProductionHYPERState:
    """Production-grade application state with full error handling"""
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None
        self.ml_enhanced_engine = None
        self.model_tester = None
        self.learning_api = None
        self.testing_api = None
        
        self.current_signals = {}
        self.enhanced_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.update_task = None
        self.startup_time = datetime.now()
        
        # Enhanced statistics
        self.stats = {
            "total_signals_generated": 0,
            "ml_enhanced_signals": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "api_calls_made": 0,
            "signals_with_data": 0,
            "fallback_signals": 0,
            "average_confidence": 0.0,
            "ml_average_confidence": 0.0,
            "high_confidence_signals": 0,
            "ml_predictions_made": 0,
            "ml_agreements": 0,
            "ml_disagreements": 0,
            "errors_encountered": 0,
            "successful_cycles": 0,
            "system_restarts": 0
        }
        
        # Performance tracking
        self.performance_metrics = {
            "avg_generation_time": 0.0,
            "avg_ml_enhancement_time": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "total_api_calls": 0,
            "error_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the production system"""
        logger.info("🚀 Initializing Production HYPER Trading System...")
        logger.info(f"Environment: {config.ENVIRONMENT}")
        logger.info(f"Demo Mode: {config.DEMO_MODE}")
        logger.info(f"Tracking tickers: {', '.join(config.TICKERS)}")
        
        try:
            # Initialize data aggregator
            self.data_aggregator = HYPERDataAggregator(config.ALPHA_VANTAGE_API_KEY)
            if hasattr(self.data_aggregator, 'initialize'):
                await self.data_aggregator.initialize()
            logger.info("✅ Data aggregator initialized")
            
            # Initialize signal engine
            self.signal_engine = HYPERSignalEngine()
            logger.info("✅ Signal engine initialized")
            
            # Initialize testing framework
            try:
                self.model_tester = ModelTester(self.signal_engine)
                self.testing_api = TestingAPI(self.model_tester)
                logger.info("✅ Testing framework initialized")
            except Exception as e:
                logger.warning(f"⚠️ Testing framework failed to initialize: {e}")
                self.model_tester = None
                self.testing_api = None
            
            # Initialize ML learning
            try:
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, 
                    self.model_tester
                )
                logger.info("✅ ML learning system initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed to initialize: {e}")
                self.ml_enhanced_engine = None
                self.learning_api = None
            
            # Validate configuration
            config.validate_config()
            logger.info("✅ Configuration validated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up system resources...")
        
        try:
            self.is_running = False
            
            if self.learning_api and hasattr(self.learning_api, 'cleanup'):
                await self.learning_api.cleanup()
            
            if self.data_aggregator and hasattr(self.data_aggregator, 'close'):
                await self.data_aggregator.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

hyper_state = ProductionHYPERState()

# ========================================
# WEBSOCKET CONNECTION MANAGER
# ========================================
class ProductionConnectionManager:
    """Production WebSocket manager with enhanced error handling"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.connection_count += 1
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            
            logger.info(f"Client connected (#{self.connection_count}). Active: {len(self.active_connections)}")
            
            # Send welcome message
            await self.send_personal_message(websocket, {
                "type": "connection_established",
                "message": "Connected to HYPER Trading System",
                "version": "3.0.0-PRODUCTION",
                "demo_mode": config.DEMO_MODE,
                "capabilities": {
                    "ml_learning": hyper_state.ml_enhanced_engine is not None,
                    "model_testing": hyper_state.model_tester is not None,
                    "enhanced_signals": True
                }
            })
            
        except Exception as e:
            logger.error(f"Error connecting client: {e}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            logger.info(f"Client disconnected. Active: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        message_json = json.dumps(message, default=str)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def serialize_signals(self, signals):
        """Convert signals to JSON-serializable format"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal_data in signals.items():
            try:
                if isinstance(signal_data, dict) and 'base_signal' in signal_data:
                    # Enhanced signal format
                    base = signal_data['base_signal']
                    ml_pred = signal_data.get('ml_predictions', {})
                    
                    serialized[symbol] = {
                        "symbol": base.get('symbol', symbol),
                        "signal_type": base.get('signal_type', 'HOLD'),
                        "confidence": float(signal_data.get('final_confidence', base.get('confidence', 0.0))),
                        "direction": base.get('direction', 'NEUTRAL'),
                        "price": float(base.get('price', 0.0)),
                        "timestamp": base.get('timestamp', datetime.now().isoformat()),
                        "ml_agreement": signal_data.get('ml_agreement', 'UNKNOWN'),
                        "ml_confidence": ml_pred.get('confidence', {}).get('predicted_accuracy', 0) * 100,
                        "enhanced_reasoning": signal_data.get('enhanced_reasoning', []),
                        "technical_score": float(base.get('technical_score', 50.0)),
                        "sentiment_score": float(base.get('sentiment_score', 50.0)),
                        "data_quality": base.get('data_quality', 'unknown')
                    }
                else:
                    # Regular signal format
                    if hasattr(signal_data, '__dict__'):
                        signal = signal_data
                        serialized[symbol] = {
                            "symbol": signal.symbol,
                            "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                            "confidence": float(getattr(signal, 'confidence', 0.0)),
                            "direction": getattr(signal, 'direction', 'NEUTRAL'),
                            "price": float(getattr(signal, 'price', 0.0)),
                            "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                            "technical_score": float(getattr(signal, 'technical_score', 50.0)),
                            "sentiment_score": float(getattr(signal, 'sentiment_score', 50.0)),
                            "data_quality": getattr(signal, 'data_quality', 'unknown')
                        }
                    else:
                        serialized[symbol] = signal_data
                        
            except Exception as e:
                logger.error(f"Error serializing signal for {symbol}: {e}")
                serialized[symbol] = {
                    "symbol": symbol,
                    "signal_type": "ERROR",
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        return serialized

manager = ProductionConnectionManager()

# ========================================
# SIGNAL GENERATION LOOP
# ========================================
async def production_signal_generation_loop():
    """Production signal generation with comprehensive error handling"""
    logger.info("🚀 Starting production signal generation loop...")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while hyper_state.is_running:
        try:
            loop_start_time = time.time()
            
            # Update performance metrics
            try:
                import psutil
                hyper_state.performance_metrics["memory_usage_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
                hyper_state.performance_metrics["cpu_usage_percent"] = psutil.cpu_percent()
            except:
                pass
            
            # Generate base signals
            if not hyper_state.signal_engine:
                logger.error("❌ Signal engine not initialized")
                await asyncio.sleep(30)
                continue
            
            start_time = time.time()
            base_signals = await hyper_state.signal_engine.generate_all_signals()
            base_generation_time = time.time() - start_time
            
            # Generate ML enhanced signals if available
            enhanced_signals = {}
            ml_generation_time = 0
            
            if hyper_state.ml_enhanced_engine:
                try:
                    start_time = time.time()
                    for symbol in config.TICKERS:
                        if symbol in base_signals:
                            enhanced_signal = await hyper_state.ml_enhanced_engine.enhanced_signal_generation(symbol)
                            enhanced_signals[symbol] = enhanced_signal
                    ml_generation_time = time.time() - start_time
                    hyper_state.stats["ml_enhanced_signals"] += len(enhanced_signals)
                except Exception as e:
                    logger.error(f"❌ ML enhancement failed: {e}")
                    enhanced_signals = {}
            
            # Update state
            hyper_state.current_signals = base_signals
            hyper_state.enhanced_signals = enhanced_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(base_signals)
            hyper_state.stats["successful_cycles"] += 1
            
            # Analyze signal quality
            analyze_signal_quality(base_signals, enhanced_signals)
            
            # Update performance metrics
            total_loop_time = time.time() - loop_start_time
            hyper_state.performance_metrics["avg_generation_time"] = (
                hyper_state.performance_metrics["avg_generation_time"] * 0.9 + base_generation_time * 0.1
            )
            
            # Log signal summary
            log_signal_summary(base_signals, enhanced_signals, base_generation_time, ml_generation_time)
            
            # Broadcast to clients
            await manager.broadcast({
                "type": "signal_update",
                "signals": manager.serialize_signals(base_signals),
                "enhanced_signals": manager.serialize_signals(enhanced_signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(),
                "performance": hyper_state.performance_metrics.copy(),
                "generation_times": {
                    "base_signals": base_generation_time,
                    "ml_enhanced": ml_generation_time,
                    "total_loop": total_loop_time
                }
            })
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Wait for next update
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            consecutive_errors += 1
            hyper_state.stats["errors_encountered"] += 1
            
            logger.error(f"💥 Error in signal generation loop (#{consecutive_errors}): {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Exponential backoff on repeated errors
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"🚨 Too many consecutive errors ({consecutive_errors}). System may be unstable.")
                await asyncio.sleep(300)  # 5 minute pause
            else:
                await asyncio.sleep(30 * consecutive_errors)  # Escalating delay

def analyze_signal_quality(base_signals, enhanced_signals):
    """Analyze signal quality metrics"""
    signals_with_data = 0
    fallback_signals = 0
    high_confidence_count = 0
    confidence_sum = 0
    
    for symbol, signal in base_signals.items():
        if hasattr(signal, 'price') and signal.price > 0:
            signals_with_data += 1
        else:
            fallback_signals += 1
        
        confidence = getattr(signal, 'confidence', 0)
        confidence_sum += confidence
        
        if confidence >= 80:
            high_confidence_count += 1
    
    # Update statistics
    signal_count = len(base_signals) if base_signals else 1
    hyper_state.stats.update({
        "signals_with_data": signals_with_data,
        "fallback_signals": fallback_signals,
        "average_confidence": confidence_sum / signal_count,
        "high_confidence_signals": high_confidence_count
    })

def log_signal_summary(base_signals, enhanced_signals, base_time, ml_time):
    """Log comprehensive signal summary"""
    signal_details = []
    for symbol, signal in base_signals.items():
        if hasattr(signal, '__dict__'):
            signal_type = getattr(signal, 'signal_type', 'HOLD')
            confidence = getattr(signal, 'confidence', 0.0)
            price = getattr(signal, 'price', 0.0)
            signal_details.append(f"{symbol}:{signal_type}({confidence:.0f%)@${price:.2f}")
        else:
            signal_details.append(f"{symbol}:ERROR")
    
    logger.info(f"📊 Signals: {', '.join(signal_details)} (base: {base_time:.2f}s, ml: {ml_time:.2f}s)")

# ========================================
# API ROUTES
# ========================================
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main trading interface"""
    try:
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                content = f.read()
                return HTMLResponse(content=content)
        else:
            # Return embedded HTML if file doesn't exist
            return HTMLResponse(content=get_embedded_html())
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

def get_embedded_html():
    """Get embedded HTML for the frontend"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HYPER Trading System - Production</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #fff; 
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { 
            font-size: 2.5em; 
            background: linear-gradient(45deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .status { 
            display: inline-block; 
            padding: 8px 20px; 
            border-radius: 25px; 
            margin: 10px;
            font-weight: bold;
        }
        .status.connected { background: #4CAF50; }
        .status.disconnected { background: #f44336; }
        .status.demo { background: #ff9800; }
        
        .signals-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 20px; 
            margin-top: 20px; 
        }
        .signal-card { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px; 
            padding: 20px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .signal-card:hover { transform: translateY(-5px); }
        .signal-card.buy { border-left: 5px solid #4CAF50; }
        .signal-card.sell { border-left: 5px solid #f44336; }
        .signal-card.hold { border-left: 5px solid #ff9800; }
        
        .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .symbol { font-size: 1.8em; font-weight: bold; }
        .confidence { 
            font-size: 1.2em; 
            font-weight: bold; 
            padding: 5px 15px; 
            border-radius: 20px; 
            background: rgba(255, 255, 255, 0.2);
        }
        .price { font-size: 1.4em; color: #00d4ff; margin: 10px 0; }
        .details { margin-top: 15px; }
        .detail-row { 
            display: flex; 
            justify-content: space-between; 
            margin: 5px 0; 
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .timestamp { color: #aaa; font-size: 0.9em; margin-top: 10px; }
        
        .stats-panel { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px; 
            padding: 20px; 
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
        }
        .stat-item { text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #00ff88; }
        .stat-label { color: #ccc; margin-top: 5px; }
        
        .warning { 
            background: linear-gradient(45deg, #ff6b35, #f7931e); 
            padding: 15px; 
            border-radius: 10px; 
            margin: 20px 0; 
            text-align: center;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .signals-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 HYPER Trading System</h1>
            <div id="connection-status" class="status disconnected">Connecting...</div>
            <div id="demo-status" class="status demo" style="display: none;">DEMO MODE</div>
        </div>
        
        <div class="warning">
            <strong>⚠️ FOR EDUCATIONAL PURPOSES ONLY</strong><br>
            This system provides trading signals for educational purposes. Not financial advice. Trade at your own risk.
        </div>
        
        <div class="stats-panel">
            <div class="stats-grid">
                <div class="stat-item">
                    <div id="total-signals" class="stat-value">0</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                <div class="stat-item">
                    <div id="avg-confidence" class="stat-value">0%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
                <div class="stat-item">
                    <div id="high-confidence" class="stat-value">0</div>
                    <div class="stat-label">High Confidence</div>
                </div>
                <div class="stat-item">
                    <div id="clients-connected" class="stat-value">0</div>
                    <div class="stat-label">Connected Clients</div>
                </div>
            </div>
        </div>
        
        <div id="signals-container" class="signals-grid">
            <!-- Signals will be populated here -->
        </div>
    </div>
    
    <script>
        class HYPERTradingSystem {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 5000;
                this.connect();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Connected to HYPER Trading System');
                    this.updateConnectionStatus(true);
                    this.reconnectAttempts = 0;
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from HYPER Trading System');
                    this.updateConnectionStatus(false);
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            }
            
            updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connection-status');
                if (connected) {
                    statusEl.textContent = '✅ Connected';
                    statusEl.className = 'status connected';
                } else {
                    statusEl.textContent = '❌ Disconnected';
                    statusEl.className = 'status disconnected';
                }
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'connection_established':
                        console.log('Connection established:', data.message);
                        if (data.demo_mode) {
                            document.getElementById('demo-status').style.display = 'inline-block';
                        }
                        break;
                    case 'signal_update':
                        this.updateSignals(data.signals);
                        this.updateStats(data.stats);
                        break;
                }
            }
            
            updateSignals(signals) {
                const container = document.getElementById('signals-container');
                container.innerHTML = '';
                
                Object.entries(signals).forEach(([symbol, signal]) => {
                    const card = this.createSignalCard(symbol, signal);
                    container.appendChild(card);
                });
            }
            
            createSignalCard(symbol, signal) {
                const card = document.createElement('div');
                card.className = `signal-card ${signal.signal_type.toLowerCase()}`;
                
                const confidenceColor = signal.confidence >= 80 ? '#4CAF50' : 
                                       signal.confidence >= 60 ? '#ff9800' : '#f44336';
                
                card.innerHTML = `
                    <div class="signal-header">
                        <div class="symbol">${symbol}</div>
                        <div class="confidence" style="background-color: ${confidenceColor}">
                            ${signal.confidence.toFixed(1)}%
                        </div>
                    </div>
                    <div class="price">$${signal.price.toFixed(2)}</div>
                    <div class="details">
                        <div class="detail-row">
                            <span>Signal:</span>
                            <span>${signal.signal_type} ${signal.direction}</span>
                        </div>
                        <div class="detail-row">
                            <span>Technical Score:</span>
                            <span>${signal.technical_score.toFixed(1)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Sentiment Score:</span>
                            <span>${signal.sentiment_score.toFixed(1)}</span>
                        </div>
                        ${signal.ml_agreement && signal.ml_agreement !== 'UNKNOWN' ? `
                        <div class="detail-row">
                            <span>ML Agreement:</span>
                            <span>${signal.ml_agreement}</span>
                        </div>
                        ` : ''}
                        <div class="detail-row">
                            <span>Data Quality:</span>
                            <span>${signal.data_quality}</span>
                        </div>
                    </div>
                    <div class="timestamp">${new Date(signal.timestamp).toLocaleString()}</div>
                `;
                
                return card;
            }
            
            updateStats(stats) {
