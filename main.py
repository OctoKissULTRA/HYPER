import os
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
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
# FASTAPI APPLICATION SETUP
# ========================================
app = FastAPI(
    title="⚡ HYPER Trading System - Production",
    description="Production-grade AI-powered trading signals with ML learning and comprehensive testing",
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
# GLOBAL STATE MANAGEMENT
# ========================================
class ProductionHYPERState:
    """Production-grade application state management"""
    
    def __init__(self):
        self.is_running = False
        self.initialization_complete = False
        
        # Core components
        self.data_aggregator = None
        self.signal_engine = None
        self.ml_enhanced_engine = None
        self.model_tester = None
        self.learning_api = None
        self.testing_api = None
        
        # Data storage
        self.current_signals = {}
        self.enhanced_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.startup_time = datetime.now()
        
        # Statistics
        self.stats = {
            "total_signals_generated": 0,
            "ml_enhanced_signals": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "successful_cycles": 0,
            "errors_encountered": 0,
            "average_confidence": 0.0,
            "high_confidence_signals": 0,
            "last_error": None
        }
    
    async def initialize(self):
        """Initialize the production system"""
        logger.info("🚀 Initializing Production HYPER Trading System...")
        logger.info(f"Environment: {config.ENVIRONMENT}")
        logger.info(f"Demo Mode: {config.DEMO_MODE}")
        
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
                logger.warning(f"⚠️ Testing framework failed: {e}")
                self.model_tester = None
                self.testing_api = None
            
            # Initialize ML learning
            try:
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, self.model_tester
                )
                logger.info("✅ ML learning system initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed: {e}")
                self.ml_enhanced_engine = None
                self.learning_api = None
            
            # Validate configuration
            config.validate_config()
            logger.info("✅ Configuration validated")
            
            self.initialization_complete = True
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.stats["errors_encountered"] += 1
            self.stats["last_error"] = str(e)
            return False

hyper_state = ProductionHYPERState()

# ========================================
# WEBSOCKET CONNECTION MANAGER
# ========================================
class ProductionConnectionManager:
    """Production WebSocket manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.total_connections = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.total_connections += 1
            hyper_state.stats["clients_connected"] = len(self.active_connections)
            
            logger.info(f"Client connected. Active: {len(self.active_connections)}")
            
            # Send welcome message
            await self.send_personal_message(websocket, {
                "type": "connection_established",
                "message": "Connected to HYPER Trading System",
                "version": "3.0.0-PRODUCTION",
                "demo_mode": config.DEMO_MODE,
                "timestamp": datetime.now().isoformat()
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
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def serialize_signals(self, signals):
        """Convert signals to JSON format"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal_data in signals.items():
            try:
                if isinstance(signal_data, dict) and 'base_signal' in signal_data:
                    # Enhanced signal
                    base = signal_data['base_signal']
                    serialized[symbol] = {
                        "symbol": base.get('symbol', symbol),
                        "signal_type": base.get('signal_type', 'HOLD'),
                        "confidence": float(signal_data.get('final_confidence', 50)),
                        "direction": base.get('direction', 'NEUTRAL'),
                        "price": float(base.get('price', 0)),
                        "timestamp": base.get('timestamp', datetime.now().isoformat()),
                        "ml_agreement": signal_data.get('ml_agreement', 'UNKNOWN'),
                        "technical_score": float(base.get('technical_score', 50)),
                        "sentiment_score": float(base.get('sentiment_score', 50)),
                        "data_quality": base.get('data_quality', 'unknown')
                    }
                else:
                    # Regular signal
                    if hasattr(signal_data, '__dict__'):
                        signal = signal_data
                        serialized[symbol] = {
                            "symbol": signal.symbol,
                            "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                            "confidence": float(getattr(signal, 'confidence', 0)),
                            "direction": getattr(signal, 'direction', 'NEUTRAL'),
                            "price": float(getattr(signal, 'price', 0)),
                            "timestamp": getattr(signal, 'timestamp', datetime.now().isoformat()),
                            "technical_score": float(getattr(signal, 'technical_score', 50)),
                            "sentiment_score": float(getattr(signal, 'sentiment_score', 50)),
                            "data_quality": getattr(signal, 'data_quality', 'unknown')
                        }
                    else:
                        serialized[symbol] = signal_data
            except Exception as e:
                logger.error(f"Error serializing {symbol}: {e}")
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
    """Production signal generation loop"""
    logger.info("🚀 Starting production signal generation loop...")
    
    consecutive_errors = 0
    cycle_count = 0
    
    while hyper_state.is_running:
        try:
            cycle_count += 1
            start_time = time.time()
            
            # Check initialization
            if not hyper_state.initialization_complete:
                await asyncio.sleep(10)
                continue
            
            # Generate base signals
            base_signals = await hyper_state.signal_engine.generate_all_signals()
            
            # Generate enhanced signals if available
            enhanced_signals = {}
            if hyper_state.ml_enhanced_engine:
                try:
                    for symbol in config.TICKERS:
                        if symbol in base_signals:
                            enhanced_signal = await hyper_state.ml_enhanced_engine.enhanced_signal_generation(symbol)
                            enhanced_signals[symbol] = enhanced_signal
                except Exception as e:
                    logger.error(f"ML enhancement failed: {e}")
            
            # Update state
            hyper_state.current_signals = base_signals
            hyper_state.enhanced_signals = enhanced_signals
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(base_signals)
            hyper_state.stats["successful_cycles"] += 1
            
            # Calculate metrics
            confidence_sum = sum(getattr(s, 'confidence', 0) for s in base_signals.values())
            signal_count = len(base_signals)
            hyper_state.stats["average_confidence"] = confidence_sum / signal_count if signal_count > 0 else 0
            hyper_state.stats["high_confidence_signals"] = sum(
                1 for s in base_signals.values() if getattr(s, 'confidence', 0) >= 80
            )
            
            # Log summary
            generation_time = time.time() - start_time
            signal_summary = []
            for symbol, signal in base_signals.items():
                signal_type = getattr(signal, 'signal_type', 'HOLD')
                confidence = getattr(signal, 'confidence', 0)
                signal_summary.append(f"{symbol}:{signal_type}({confidence:.0f}%)")
            
            logger.info(f"📊 Cycle #{cycle_count}: {', '.join(signal_summary)} ({generation_time:.2f}s)")
            
            # Broadcast to clients
            await manager.broadcast({
                "type": "signal_update",
                "signals": manager.serialize_signals(base_signals),
                "enhanced_signals": manager.serialize_signals(enhanced_signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(),
                "cycle_number": cycle_count,
                "generation_time": generation_time
            })
            
            consecutive_errors = 0
            await asyncio.sleep(config.UPDATE_INTERVALS["signal_generation"])
            
        except Exception as e:
            consecutive_errors += 1
            hyper_state.stats["errors_encountered"] += 1
            hyper_state.stats["last_error"] = str(e)
            
            logger.error(f"💥 Signal generation error #{consecutive_errors}: {e}")
            
            if consecutive_errors >= 5:
                logger.critical("🚨 Too many consecutive errors - extended pause")
                await asyncio.sleep(300)
            else:
                await asyncio.sleep(30 * consecutive_errors)

# ========================================
# API ROUTES
# ========================================

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve main interface"""
    html_content = '''
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
            padding: 8px 16px; 
            margin: 5px; 
            border-radius: 20px; 
            font-weight: bold; 
        }
        .connected { background: #4CAF50; }
        .disconnected { background: #f44336; }
        .demo { background: #ff9800; }
        .production { background: #2196F3; }
        
        .stats-panel { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px; 
            padding: 20px; 
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 15px; 
            text-align: center;
        }
        .stat-item h3 { color: #00ff88; font-size: 1.8em; margin-bottom: 5px; }
        .stat-item p { color: #ccc; font-size: 0.9em; }
        
        .signals-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .signal-card { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px; 
            padding: 20px; 
            backdrop-filter: blur(10px);
            border-left: 5px solid #ccc;
            transition: transform 0.3s ease;
        }
        .signal-card:hover { transform: translateY(-5px); }
        .signal-card.hyper_buy, .signal-card.soft_buy { border-left-color: #4CAF50; }
        .signal-card.hyper_sell, .signal-card.soft_sell { border-left-color: #f44336; }
        .signal-card.hold { border-left-color: #ff9800; }
        
        .signal-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 15px; 
        }
        .symbol { font-size: 1.5em; font-weight: bold; }
        .confidence { 
            padding: 5px 12px; 
            border-radius: 15px; 
            font-weight: bold;
            font-size: 0.9em;
        }
        .confidence.high { background: #4CAF50; }
        .confidence.medium { background: #ff9800; }
        .confidence.low { background: #f44336; }
        
        .price { font-size: 1.3em; color: #00d4ff; margin: 10px 0; }
        .signal-type { 
            font-size: 1.1em; 
            font-weight: bold; 
            margin: 10px 0;
            text-transform: capitalize;
        }
        .details { margin-top: 15px; }
        .detail-row { 
            display: flex; 
            justify-content: space-between; 
            margin: 5px 0; 
            padding: 3px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9em;
        }
        .timestamp { 
            color: #888; 
            font-size: 0.8em; 
            margin-top: 15px; 
        }
        
        .warning { 
            background: linear-gradient(45deg, #ff6b35, #f7931e); 
            padding: 15px; 
            border-radius: 10px; 
            margin: 20px 0; 
            text-align: center;
            font-weight: bold;
        }
        
        .error-message {
            background: #f44336;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .signals-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 HYPER Trading System</h1>
            <div id="connection-status" class="status disconnected">Connecting...</div>
            <div id="mode-status" class="status demo">Loading...</div>
            <div class="status production">Production v3.0</div>
        </div>
        
        <div class="warning">
            <strong>⚠️ EDUCATIONAL & RESEARCH PURPOSES ONLY</strong><br>
            This system provides trading signals for educational and research purposes. Not financial advice. Always do your own research.
        </div>
        
        <div class="stats-panel">
            <div class="stats-grid">
                <div class="stat-item">
                    <h3 id="total-signals">0</h3>
                    <p>Total Signals</p>
                </div>
                <div class="stat-item">
                    <h3 id="avg-confidence">0%</h3>
                    <p>Avg Confidence</p>
                </div>
                <div class="stat-item">
                    <h3 id="high-confidence">0</h3>
                    <p>High Confidence</p>
                </div>
                <div class="stat-item">
                    <h3 id="successful-cycles">0</h3>
                    <p>Successful Cycles</p>
                </div>
                <div class="stat-item">
                    <h3 id="clients-connected">0</h3>
                    <p>Connected Clients</p>
                </div>
                <div class="stat-item">
                    <h3 id="last-update">Never</h3>
                    <p>Last Update</p>
                </div>
            </div>
        </div>
        
        <div id="error-container"></div>
        <div id="signals-container" class="signals-grid">
            <div class="signal-card">
                <div class="signal-header">
                    <div class="symbol">Loading...</div>
                </div>
                <p>Connecting to HYPER Trading System...</p>
            </div>
        </div>
    </div>
    
    <script>
        class HYPERTradingSystem {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 5000;
                this.lastUpdate = null;
                this.connect();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                console.log('Connecting to:', wsUrl);
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
                        this.showError('Error parsing server message');
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from HYPER Trading System');
                    this.updateConnectionStatus(false);
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.showError('Connection error occurred');
                };
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
                } else {
                    this.showError('Failed to reconnect. Please refresh the page.');
                }
            }
            
            updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connection-status');
                if (connected) {
                    statusEl.textContent = '✅ Connected';
                    statusEl.className = 'status connected';
                    this.clearError();
                } else {
                    statusEl.textContent = '❌ Disconnected';
                    statusEl.className = 'status disconnected';
                }
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'connection_established':
                        console.log('Connection established:', data.message);
                        const modeEl = document.getElementById('mode-status');
                        if (data.demo_mode) {
                            modeEl.textContent = '🧪 Demo Mode';
                            modeEl.className = 'status demo';
                        } else {
                            modeEl.textContent = '⚡ Live Mode';
                            modeEl.className = 'status connected';
                        }
                        break;
                    case 'signal_update':
                        this.updateSignals(data.signals);
                        this.updateStats(data.stats);
                        this.lastUpdate = data.timestamp;
                        break;
                }
            }
            
            updateSignals(signals) {
                const container = document.getElementById('signals-container');
                container.innerHTML = '';
                
                if (!signals || Object.keys(signals).length === 0) {
                    container.innerHTML = '<div class="signal-card"><p>No signals available</p></div>';
                    return;
                }
                
                Object.entries(signals).forEach(([symbol, signal]) => {
                    const card = this.createSignalCard(symbol, signal);
                    container.appendChild(card);
                });
            }
            
            createSignalCard(symbol, signal) {
                const card = document.createElement('div');
                card.className = `signal-card ${signal.signal_type.toLowerCase().replace('_', '')}`;
                
                const confidenceClass = signal.confidence >= 80 ? 'high' : 
                                       signal.confidence >= 60 ? 'medium' : 'low';
                
                card.innerHTML = `
                    <div class="signal-header">
                        <div class="symbol">${symbol}</div>
                        <div class="confidence ${confidenceClass}">
                            ${signal.confidence.toFixed(1)}%
                        </div>
                    </div>
                    <div class="price">$${signal.price.toFixed(2)}</div>
                    <div class="signal-type">${signal.signal_type.replace('_', ' ')} ${signal.direction}</div>
                    <div class="details">
                        <div class="detail-row">
                            <span>Technical Score:</span>
                            <span>${signal.technical_score.toFixed(1)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Sentiment Score:</span>
                            <span>${signal.sentiment_score.toFixed(1)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Data Quality:</span>
                            <span>${signal.data_quality}</span>
                        </div>
                        ${signal.ml_agreement && signal.ml_agreement !== 'UNKNOWN' ? `
                        <div class="detail-row">
                            <span>ML Agreement:</span>
                            <span>${signal.ml_agreement}</span>
                        </div>
                        ` : ''}
                    </div>
                    <div class="timestamp">${new Date(signal.timestamp).toLocaleString()}</div>
                `;
                
                return card;
            }
            
            updateStats(stats) {
                document.getElementById('total-signals').textContent = stats.total_signals_generated || 0;
                document.getElementById('avg-confidence').textContent = `${(stats.average_confidence || 0).toFixed(1)}%`;
                document.getElementById('high-confidence').textContent = stats.high_confidence_signals || 0;
                document.getElementById('successful-cycles').textContent = stats.successful_cycles || 0;
                document.getElementById('clients-connected').textContent = stats.clients_connected || 0;
                
                if (this.lastUpdate) {
                    const updateTime = new Date(this.lastUpdate);
                    document.getElementById('last-update').textContent = updateTime.toLocaleTimeString();
                }
            }
            
            showError(message) {
                const container = document.getElementById('error-container');
                container.innerHTML = `<div class="error-message">${message}</div>`;
            }
            
            clearError() {
                const container = document.getElementById('error-container');
                container.innerHTML = '';
            }
        }
        
        // Initialize the system when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new HYPERTradingSystem();
        });
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    uptime = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds()
    
    return {
        "status": "healthy",
        "version": "3.0.0-PRODUCTION",
        "environment": config.ENVIRONMENT,
        "demo_mode": config.DEMO_MODE,
        "is_running": hyper_state.is_running,
        "initialization_complete": hyper_state.initialization_complete,
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.0f}s",
        "last_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "tickers": config.TICKERS,
        "connected_clients": len(manager.active_connections),
        "capabilities": {
            "signal_engine": hyper_state.signal_engine is not None,
            "ml_enhanced": hyper_state.ml_enhanced_engine is not None,
            "model_testing": hyper_state.model_tester is not None,
            "data_aggregator": hyper_state.data_aggregator is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current signals"""
    return {
        "signals": manager.serialize_signals(hyper_state.current_signals),
        "enhanced_signals": manager.serialize_signals(hyper_state.enhanced_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": hyper_state.stats,
        "count": len(hyper_state.current_signals)
    }

@app.get("/api/signals/{symbol}")
async def get
