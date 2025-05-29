# main.py - HYPERtrends v4.0 Modular Application - COMPLETE VERSION
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
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import configuration and modular components
import config # Assuming you have a config.py file
# Ensure these files exist in your project structure
from data_sources import HYPERDataAggregator
from signal_engine import HYPERSignalEngine, HYPERSignal
from model_testing import ModelTester, TestingAPI
from ml_learning import integrate_ml_learning, MLEnhancedSignalEngine, LearningAPI

# ========================================
# LOGGING SETUP
# ========================================
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG.get("level", "INFO")),
    format=config.LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="🌟 HYPERtrends v4.0 - Modular Edition",
    description="AI-powered trading signals with advanced modular architecture",
    version="4.0.0-MODULAR"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving index.html and assets)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("Static files directory 'static' not found. Frontend might not serve correctly if index.html is not in root.")
    if not os.path.exists("index.html"):
        logger.warning("index.html not found in root either. Fallback HTML will be used for '/' route.")


# ========================================
# GLOBAL STATE - Enhanced for Modular Architecture
# ========================================
class ModularHYPERState:
    def __init__(self):
        self.is_running = False
        self.data_aggregator = None
        self.signal_engine = None  # Now modular!
        self.ml_enhanced_engine = None
        self.model_tester = None
        self.learning_api = None
        self.testing_api = None
        
        self.current_signals = {}
        self.enhanced_signals = {}
        self.connected_clients = []
        self.last_update = None
        self.startup_time = datetime.now()
        
        self.stats = {
            "total_signals_generated": 0,
            "clients_connected": 0,
            "uptime_start": datetime.now(),
            "successful_cycles": 0,
            "errors_encountered": 0,
            "average_confidence": 0.0,
            "last_error": None,
            "data_source_status": config.get_data_source_status(),
            "robinhood_available": config.has_robinhood_credentials(),
            "modular_components": [],
            "component_performance": {}
        }
    
    async def initialize(self):
        logger.info("🚀 Initializing HYPERtrends v4.0 Modular System...")
        
        try:
            # Initialize data aggregator
            self.data_aggregator = HYPERDataAggregator()
            if hasattr(self.data_aggregator, 'initialize'):
                await self.data_aggregator.initialize()
            
            # Initialize modular signal engine
            self.signal_engine = HYPERSignalEngine()
            
            # Track active components
            if hasattr(self.signal_engine, '_get_active_components'):
                self.stats["modular_components"] = self.signal_engine._get_active_components()
            
            # Initialize testing framework
            try:
                self.model_tester = ModelTester(self.signal_engine)
                self.testing_api = TestingAPI(self.model_tester)
                logger.info("✅ Testing framework initialized")
            except Exception as e:
                logger.warning(f"⚠️ Testing framework failed: {e}")
            
            # Initialize ML learning
            try:
                self.ml_enhanced_engine, self.learning_api = integrate_ml_learning(
                    self.signal_engine, self.model_tester
                )
                logger.info("✅ ML learning integration successful")
            except Exception as e:
                logger.warning(f"⚠️ ML learning failed: {e}")
            
            # Validate configuration
            config.validate_config()
            
            # Log initialization status
            component_count = len(self.stats.get("modular_components", []))
            logger.info(f"✅ Modular system initialized with {component_count} active components")
            
            if config.has_robinhood_credentials():
                logger.info("✅ Robinhood credentials available - real data mode")
            else:
                logger.info("ℹ️ Using dynamic simulation mode")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Modular initialization failed: {e}")
            return False

hyper_state = ModularHYPERState()

# ========================================
# WEBSOCKET MANAGER - Enhanced for Modular Data
# ========================================
class ModularConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        hyper_state.stats["clients_connected"] = len(self.active_connections)
        logger.info(f"Client connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            hyper_state.stats["clients_connected"] = len(self.active_connections)
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        # Ensure all datetime objects are converted to ISO format strings for JSON serialization
        message_json = json.dumps(message, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e: # Catch more general exceptions for disconnections
                logger.warning(f"Failed to send to client, disconnecting: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def serialize_modular_signals(self, signals):
        """Enhanced serialization for modular signals"""
        if not signals:
            return {}
        
        serialized = {}
        for symbol, signal_data in signals.items():
            try:
                if hasattr(signal_data, '__dict__'): # Check if it's likely a HYPERSignal object
                    signal = signal_data
                    
                    signal_dict = {
                        "symbol": signal.symbol,
                        "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                        "confidence": float(getattr(signal, 'confidence', 0)),
                        "direction": getattr(signal, 'direction', 'NEUTRAL'),
                        "price": float(getattr(signal, 'price', 0)),
                        "timestamp": getattr(signal, 'timestamp', datetime.now()).isoformat(),
                        
                        "technical_score": float(getattr(signal, 'technical_score', 50)),
                        "sentiment_score": float(getattr(signal, 'sentiment_score', 50)),
                        "momentum_score": float(getattr(signal, 'momentum_score', 50)),
                        "ml_score": float(getattr(signal, 'ml_score', 50)), # If using ML enhancement
                        "vix_score": float(getattr(signal, 'vix_score', 50)),
                        "market_structure_score": float(getattr(signal, 'market_structure_score', 50)),
                        "risk_score": float(getattr(signal, 'risk_score', 50)),
                        
                        "williams_r": float(getattr(signal, 'williams_r', -50)),
                        "stochastic_k": float(getattr(signal, 'stochastic_k', 50)),
                        "stochastic_d": float(getattr(signal, 'stochastic_d', 50)),
                        "vix_sentiment": getattr(signal, 'vix_sentiment', 'NEUTRAL'),
                        
                        "data_quality": getattr(signal, 'data_quality', 'unknown'),
                        "reasons": getattr(signal, 'reasons', [])[:3],
                        "warnings": getattr(signal, 'warnings', [])[:2],
                        
                        "enhanced_features": getattr(signal, 'enhanced_features', {}),
                        "components_used": getattr(signal, 'enhanced_features', {}).get('components_used', [])
                    }
                    
                    if hasattr(signal, 'technical_analysis') and signal.technical_analysis:
                        signal_dict["technical_summary"] = {
                            "overall_score": getattr(signal.technical_analysis, 'overall_score', 0),
                            "direction": getattr(signal.technical_analysis, 'direction', 'NEUTRAL'),
                            "key_levels": getattr(signal.technical_analysis, 'key_levels', {})
                        }
                    
                    if hasattr(signal, 'sentiment_analysis') and signal.sentiment_analysis:
                        signal_dict["sentiment_summary"] = {
                            "overall_sentiment": getattr(signal.sentiment_analysis, 'overall_sentiment', 'NEUTRAL'),
                            "retail_sentiment": getattr(signal.sentiment_analysis, 'retail_sentiment', 'NEUTRAL'),
                            "fear_greed_indicator": getattr(signal.sentiment_analysis, 'fear_greed_indicator', 0)
                        }
                    
                    if hasattr(signal, 'vix_analysis') and signal.vix_analysis and hasattr(signal.vix_analysis, 'current_signal'):
                        signal_dict["vix_summary"] = {
                            "vix_level": getattr(signal.vix_analysis.current_signal, 'vix_level', 0),
                            "sentiment": getattr(signal.vix_analysis.current_signal, 'sentiment', 'NEUTRAL'),
                            "contrarian_signal": getattr(signal.vix_analysis.current_signal, 'contrarian_signal', 'NONE')
                        }
                    
                    if hasattr(signal, 'market_structure_analysis') and signal.market_structure_analysis and hasattr(signal.market_structure_analysis, 'current_signal'):
                        signal_dict["market_structure_summary"] = {
                            "market_regime": getattr(signal.market_structure_analysis.current_signal, 'market_regime', 'UNKNOWN'),
                            "structure_score": getattr(signal.market_structure_analysis.current_signal, 'structure_score', 0),
                            "rotation_theme": getattr(signal.market_structure_analysis.current_signal, 'rotation_theme', 'NONE')
                        }
                    
                    if hasattr(signal, 'risk_analysis') and signal.risk_analysis:
                        risk_summary = {
                            "risk_level": getattr(signal.risk_analysis, 'risk_level', 'MODERATE'),
                            "overall_risk_score": getattr(signal.risk_analysis, 'overall_risk_score', 50)
                        }
                        if hasattr(signal.risk_analysis, 'position_risk'):
                           risk_summary["position_size_recommendation"] = getattr(signal.risk_analysis.position_risk, 'position_size_recommendation', 0)
                        signal_dict["risk_summary"] = risk_summary

                    serialized[symbol] = signal_dict
                else: # Handle cases where signal_data might not be a full HYPERSignal object
                    serialized[symbol] = {"symbol": symbol, "error": "Signal data is not in expected format", "raw_data": str(signal_data)}
            except Exception as e:
                logger.error(f"Error serializing modular signal for {symbol}: {e}\n{traceback.format_exc()}")
                serialized[symbol] = {"symbol": symbol, "error": str(e)}
        
        return serialized

manager = ModularConnectionManager()

# ========================================
# ENHANCED SIGNAL GENERATION LOOP
# ========================================
async def modular_signal_generation_loop():
    logger.info("🚀 Starting modular signal generation loop...")
    
    while hyper_state.is_running:
        try:
            start_time = time.time()
            
            all_signals = {}
            component_performance = {}
            
            for symbol in config.TICKERS:
                try:
                    symbol_data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
                    
                    signal_start = time.time()
                    signal = await hyper_state.signal_engine.generate_signal(
                        symbol, 
                        symbol_data.get('quote', {}),
                        symbol_data.get('trends', {}),
                        symbol_data.get('historical_data', []) # Pass historical_data if available and needed
                    )
                    signal_time = time.time() - signal_start
                    
                    all_signals[symbol] = signal
                    component_performance[symbol] = {
                        'generation_time': signal_time,
                        'components_used': getattr(signal, 'enhanced_features', {}).get('components_used', []),
                        'data_quality': getattr(signal, 'data_quality', 'unknown')
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Signal generation failed for {symbol}: {e}\n{traceback.format_exc()}")
                    # Ensure fallback signal has minimal required attributes for serialization
                    fallback_quote = {'price': 100.0, 'symbol': symbol, 'timestamp': datetime.now()}
                    all_signals[symbol] = hyper_state.signal_engine._generate_fallback_signal(
                        symbol, fallback_quote
                    )
            
            enhanced_signals = {}
            if hyper_state.ml_enhanced_engine:
                try:
                    for symbol, base_signal in all_signals.items():
                        # Assuming enhanced_signal_generation needs the base signal or symbol data
                        # This part might need adjustment based on MLEnhancedSignalEngine's interface
                        symbol_data = await hyper_state.data_aggregator.get_comprehensive_data(symbol) # Or use cached data
                        enhanced_signal = await hyper_state.ml_enhanced_engine.enhanced_signal_generation(symbol, base_signal, symbol_data)
                        enhanced_signals[symbol] = enhanced_signal
                except Exception as e:
                    logger.error(f"ML enhancement failed: {e}\n{traceback.format_exc()}")
            
            hyper_state.current_signals = all_signals
            hyper_state.enhanced_signals = enhanced_signals if enhanced_signals else all_signals # Fallback if ML fails
            hyper_state.last_update = datetime.now()
            hyper_state.stats["total_signals_generated"] += len(all_signals)
            hyper_state.stats["successful_cycles"] += 1
            hyper_state.stats["component_performance"] = component_performance
            
            confidences = [s.confidence for s in all_signals.values() if hasattr(s, 'confidence') and s.confidence is not None]
            hyper_state.stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            
            generation_time = time.time() - start_time
            
            signal_summary = []
            for symbol, signal in all_signals.items():
                signal_type = getattr(signal, 'signal_type', 'HOLD')
                confidence = getattr(signal, 'confidence', 0)
                components = len(getattr(signal, 'enhanced_features', {}).get('components_used', []))
                signal_summary.append(f"{symbol}:{signal_type}({confidence:.0f}%,{components}c)")
            
            logger.info(f"📊 Generated modular signals: {', '.join(signal_summary)} ({generation_time:.2f}s)")
            
            await manager.broadcast({
                "type": "modular_signal_update",
                "signals": manager.serialize_modular_signals(hyper_state.current_signals),
                "enhanced_signals": manager.serialize_modular_signals(hyper_state.enhanced_signals),
                "timestamp": hyper_state.last_update.isoformat(),
                "stats": hyper_state.stats.copy(), # Send a copy to avoid issues with concurrent modification
                "generation_time": generation_time,
                "modular_info": {
                    "active_components": hyper_state.stats.get("modular_components", []),
                    "component_performance": component_performance,
                    "data_source_status": config.get_data_source_status(),
                    "version": "4.0.0-MODULAR"
                }
            })
            
            await asyncio.sleep(config.UPDATE_INTERVALS.get("signal_generation", 60)) # Use .get for safety
            
        except Exception as e:
            hyper_state.stats["errors_encountered"] += 1
            hyper_state.stats["last_error"] = str(e)
            logger.error(f"💥 Modular signal generation error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(config.UPDATE_INTERVALS.get("error_retry", 30)) # Use .get for safety

# ========================================
# ENHANCED API ROUTES
# ========================================

@app.get("/health/modular")
async def get_modular_health():
    """Enhanced health check for modular system"""
    symbol = "QQQ" if "QQQ" in config.TICKERS else (config.TICKERS[0] if config.TICKERS else "SPY")
    logger.info(f"🔍 /health/modular check triggered for symbol: {symbol}")
    try:
        if not hyper_state.data_aggregator or not hyper_state.signal_engine:
            await hyper_state.initialize() # Attempt to initialize if not already
            if not hyper_state.data_aggregator or not hyper_state.signal_engine:
                 raise HTTPException(status_code=503, detail="Core components not initialized")

        data = await hyper_state.data_aggregator.get_comprehensive_data(symbol)
        
        signal = await hyper_state.signal_engine.generate_signal(
            symbol, data.get("quote", {}), data.get("trends", {}), data.get("historical_data", [])
        )
        
        return {
            "status": "online",
            "version": "4.0.0-MODULAR",
            "symbol_checked": symbol,
            "modular_components": hyper_state.stats.get("modular_components", []),
            "component_count": len(hyper_state.stats.get("modular_components", [])),
            "signal_generated": {
                "signal_type": getattr(signal, 'signal_type', "N/A"),
                "confidence": getattr(signal, 'confidence', 0),
                "components_used": getattr(signal, 'enhanced_features', {}).get('components_used', []),
                "data_quality": getattr(signal, 'data_quality', "unknown")
            },
            "data_source": {
                "status": config.get_data_source_status(),
                "robinhood_available": config.has_robinhood_credentials(),
                "source_for_quote": data.get("quote", {}).get("data_source", "unknown")
            },
            "performance": {
                "generation_time_sample": getattr(signal, 'enhanced_features', {}).get('generation_time', 0), # This might be from signal object itself if tracked
                "cache_enabled": True, # Assuming caching is generally used
                "last_system_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None
            }
        }
    except Exception as e:
        logger.error(f"❌ Modular health check failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=503, content={
            "status": "error",
            "version": "4.0.0-MODULAR",
            "error_message": str(e),
            "details": "Failed during modular health check execution."
        })

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HYPERtrends v4.0 dashboard"""
    index_path = Path("index.html")
    static_index_path = Path("static/index.html")

    content_to_serve = None
    
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                content_to_serve = f.read()
        except Exception as e:
            logger.error(f"Error reading root index.html: {e}")
    elif static_index_path.exists():
        try:
            with open(static_index_path, "r", encoding="utf-8") as f:
                content_to_serve = f.read()
            logger.info("Served index.html from /static directory.")
        except Exception as e:
            logger.error(f"Error reading static/index.html: {e}")

    if content_to_serve:
        return HTMLResponse(content=content_to_serve)
    
    # Enhanced fallback HTML for v4.0
    logger.warning("index.html not found in root or static folder. Serving fallback HTML.")
    fallback_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HYPERtrends v4.0 - Modular Edition</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #000428, #004e92); color: #00ffff; text-align: center; padding: 50px; margin: 0; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            h1 {{ font-size: 3.5em; margin-bottom: 10px; text-shadow: 0 0 20px #00ffff; }}
            h2 {{ font-size: 1.8em; margin-bottom: 30px; opacity: 0.9; }}
            .status {{ background: rgba(0, 255, 255, 0.1); padding: 25px; border-radius: 15px; margin: 25px 0; border: 2px solid #00ffff; box-shadow: 0 0 30px rgba(0, 255, 255, 0.3); }}
            .components {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .component {{ background: rgba(0, 100, 255, 0.2); padding: 15px; border-radius: 10px; border: 1px solid #0066ff; }}
            .version {{ font-size: 0.9em; opacity: 0.7; margin-top: 20px; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }}
            .pulse {{ animation: pulse 2s infinite; }}
            a {{ color: #00ffff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="pulse">🌟 HYPERtrends</h1>
            <h2>v4.0 Modular Edition</h2>
            <div class="status">
                <h3>🚀 Modular Architecture Active</h3>
                <p>Advanced AI-powered trading signals with modular components</p>
                <p><strong>Data Source:</strong> Robinhood + Dynamic Simulation (check config)</p>
                <p><strong>Robinhood Available:</strong> <span id="robinhoodStatus">Checking...</span></p>
                <p><strong>System Status:</strong> <span id="systemStatus" class="pulse">Initializing...</span></p>
                
                <div class="components" id="componentList">
                    <div class="component">📊 Technical Analysis</div>
                    <div class="component">💭 Sentiment Analysis</div>
                    <div class="component">😱 VIX Fear/Greed</div>
                    <div class="component">🏗️ Market Structure</div>
                    <div class="component">⚠️ Risk Analysis</div>
                </div>
                
                <div class="version">
                    <p>index.html not found. This is a fallback page.</p>
                    <p>API Health: <a href="/health/modular">/health/modular</a></p>
                    <p>API Docs: <a href="/docs">/docs</a> | <a href="/redoc">/redoc</a></p>
                </div>
            </div>
        </div>
        <script>
            function updateStatus() {{
                fetch('/health/modular')
                    .then(r => r.json())
                    .then(data => {{
                        document.getElementById('systemStatus').textContent = data.status + ' (' + (data.component_count || 0) + ' components active)';
                        document.getElementById('systemStatus').className = data.status === 'online' ? 'pulse' : '';
                        document.getElementById('robinhoodStatus').textContent = data.data_source && data.data_source.robinhood_available ? '✅ Available' : '❌ Not Configured';
                        
                        const compList = document.getElementById('componentList');
                        compList.innerHTML = ''; // Clear defaults
                        if (data.modular_components && data.modular_components.length > 0) {{
                            data.modular_components.forEach(compName => {{
                                const div = document.createElement('div');
                                div.className = 'component';
                                div.textContent = compName.replace(/_/g, ' ').replace(/\\b(\\w)/g, c => c.toUpperCase()); // Format name
                                compList.appendChild(div);
                            }});
                        }} else {{
                            compList.innerHTML = '<p>No active components reported by health check.</p>';
                        }}
                    }})
                    .catch(e => {{
                        document.getElementById('systemStatus').textContent = 'Error fetching status';
                        document.getElementById('systemStatus').className = '';
                        document.getElementById('robinhoodStatus').textContent = 'Error';
                        console.error("Fallback page status update error:", e);
                    }});
            }}
            updateStatus(); // Initial call
            setInterval(updateStatus, 15000); // Refresh every 15 seconds
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=fallback_html)

@app.get("/health")
async def health_check():
    """Standard health check with modular info"""
    uptime_seconds = (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds() if hyper_state.stats["uptime_start"] else 0
    
    # Make a deep copy of stats for thread safety if it contains complex objects that might be modified
    safe_stats = json.loads(json.dumps(hyper_state.stats, default=str))


    return {
        "status": "healthy",
        "version": "4.0.0-MODULAR",
        "environment": config.ENVIRONMENT,
        "demo_mode": config.DEMO_MODE,
        "modular_architecture": {
            "enabled": True, # Assuming always true if this codebase is used
            "active_components": hyper_state.stats.get("modular_components", []),
            "component_count": len(hyper_state.stats.get("modular_components", [])),
            "total_possible_components": 5 # Update if this changes
        },
        "data_source_info": { # Renamed for clarity
            "primary": "robinhood (if configured)",
            "fallback": "dynamic_simulation / other APIs",
            "robinhood_credentials_present": config.has_robinhood_credentials(),
            "current_data_source_status": config.get_data_source_status()
        },
        "system_running": hyper_state.is_running,
        "uptime_seconds": uptime_seconds,
        "last_signal_update": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "current_stats": safe_stats, # Use the safe copy
        "monitored_tickers": config.TICKERS,
        "active_websocket_clients": len(manager.active_connections)
    }

@app.get("/api/signals")
async def get_current_signals():
    """Get current modular signals"""
    return {
        "signals": manager.serialize_modular_signals(hyper_state.current_signals),
        "enhanced_signals": manager.serialize_modular_signals(hyper_state.enhanced_signals),
        "timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None,
        "stats": json.loads(json.dumps(hyper_state.stats, default=str)), # Safe copy
        "modular_info": {
            "version": "4.0.0-MODULAR",
            "active_components": hyper_state.stats.get("modular_components", []),
            "component_performance": hyper_state.stats.get("component_performance", {}),
            "data_source_status": config.get_data_source_status()
        }
    }

@app.get("/api/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get detailed modular signal for specific symbol"""
    symbol_upper = symbol.upper()
    if symbol_upper not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol_upper} not tracked. Tracked: {', '.join(config.TICKERS)}")
    
    if symbol_upper in hyper_state.current_signals:
        signal = hyper_state.current_signals[symbol_upper]
        serialized_signals = manager.serialize_modular_signals({symbol_upper: signal})
        
        if symbol_upper not in serialized_signals or "error" in serialized_signals[symbol_upper]:
             raise HTTPException(status_code=500, detail=f"Error serializing signal for {symbol_upper}: {serialized_signals.get(symbol_upper, {}).get('error', 'Unknown serialization error')}")

        detailed_signal = serialized_signals[symbol_upper].copy()
        
        # Add more detailed component analysis if available on the original signal object
        if hasattr(signal, 'technical_analysis') and signal.technical_analysis and hasattr(signal.technical_analysis, 'signals'):
            detailed_signal["detailed_technical"] = {
                "indicator_signals": [{"name": s.indicator_name, "value": s.value, "direction": s.direction} 
                                      for s in signal.technical_analysis.signals[:10]], # Limit for brevity
                "key_levels": getattr(signal.technical_analysis, 'key_levels', {}),
                "pattern_analysis": getattr(signal.technical_analysis, 'pattern_analysis', {})
            }
        
        if hasattr(signal, 'risk_analysis') and signal.risk_analysis and hasattr(signal.risk_analysis, 'position_risk'):
            detailed_signal["detailed_risk"] = {
                "position_sizing": {
                    "recommended_size_pct": getattr(signal.risk_analysis.position_risk, 'position_size_recommendation', 0),
                    "kelly_criterion_raw": getattr(signal.risk_analysis.position_risk, 'kelly_criterion', 0),
                    "optimal_stop_loss_price": getattr(signal.risk_analysis.position_risk, 'optimal_stop_loss', 0)
                },
                "risk_warnings": getattr(signal.risk_analysis, 'risk_warnings', [])[:3] # Limit for brevity
            }
        
        # Include ML enhanced details if present in enhanced_signals
        if symbol_upper in hyper_state.enhanced_signals and hyper_state.enhanced_signals[symbol_upper] != signal:
            enhanced_signal_obj = hyper_state.enhanced_signals[symbol_upper]
            serialized_enhanced = manager.serialize_modular_signals({symbol_upper: enhanced_signal_obj})
            if symbol_upper in serialized_enhanced and "error" not in serialized_enhanced[symbol_upper]:
                 detailed_signal["ml_enhanced_details"] = serialized_enhanced[symbol_upper]


        return detailed_signal
    else:
        # Check if it's an untracked but valid symbol format for a more specific error later if needed
        # For now, just say no signal available for tracked symbol
        raise HTTPException(status_code=404, detail=f"No signal currently available for tracked symbol {symbol_upper}")

@app.get("/api/components/status")
async def get_component_status():
    """Get status of all modular components and their sub-features if applicable"""
    if not hyper_state.signal_engine:
        # Try to initialize if not already
        initialized = await hyper_state.initialize()
        if not initialized or not hyper_state.signal_engine:
            raise HTTPException(status_code=503, detail="Signal engine not initialized and failed to initialize.")
    
    component_status = {}
    
    # Helper to get component details
    def get_details(analyzer_attr, name, default_features_desc):
        status = {"enabled": False, "status": "disabled", "details": "Not loaded or unavailable"}
        analyzer = getattr(hyper_state.signal_engine, analyzer_attr, None)
        if analyzer:
            status["enabled"] = True
            status["status"] = "active"
            status["details"] = getattr(analyzer, 'get_status_summary', lambda: default_features_desc)() # Call method if exists
        return status

    component_status["technical"] = get_details('technical_analyzer', "Technical Analysis", "25+ Indicators")
    component_status["sentiment"] = get_details('sentiment_analyzer', "Sentiment Analysis", "Multi-source NLP")
    component_status["vix"] = get_details('vix_analyzer', "VIX Analysis", "Fear/Greed + Contrarian Signals")
    component_status["market_structure"] = get_details('market_structure_analyzer', "Market Structure", "Breadth + Sectors Analysis")
    component_status["risk"] = get_details('risk_analyzer', "Risk Analysis", "VaR + Position Sizing")
    
    # If ML Engine is separate and active
    if hyper_state.ml_enhanced_engine:
        ml_status = "active" if hyper_state.ml_enhanced_engine else "disabled"
        ml_details = "Enhances signals using machine learning models."
        if hasattr(hyper_state.ml_enhanced_engine, 'get_status_summary'):
            ml_details = hyper_state.ml_enhanced_engine.get_status_summary()
        component_status["ml_enhancement"] = {"enabled": bool(hyper_state.ml_enhanced_engine), "status": ml_status, "details": ml_details}


    return {
        "components": component_status,
        "total_enabled_from_signal_engine": sum(1 for c_name, c_info in component_status.items() if c_info.get("enabled", False) and c_name != "ml_enhancement"),
        "ml_enhancement_active": component_status.get("ml_enhancement", {}).get("enabled", False),
        "overall_performance_summary": hyper_state.stats.get("component_performance", {}), # This is per-symbol, might need aggregation
        "last_signal_update_timestamp": hyper_state.last_update.isoformat() if hyper_state.last_update else None
    }

@app.post("/api/start")
async def start_system():
    """Start the modular system if not already running, or re-initialize if needed."""
    if hyper_state.is_running:
        return {"status": "already_running", "version": "4.0.0-MODULAR", "message": "System is already generating signals."}
    
    if not hyper_state.signal_engine or not hyper_state.data_aggregator: # Check if core components are missing
        logger.info("Core components not found, attempting full initialization...")
        success = await hyper_state.initialize()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize modular system. Check logs.")
    
    hyper_state.is_running = True
    if not hyper_state.stats["uptime_start"] or (datetime.now() - hyper_state.stats["uptime_start"]).total_seconds() < 1:
         hyper_state.stats["uptime_start"] = datetime.now() # Reset uptime start only if it's a fresh start

    # Ensure only one signal generation loop is running
    # This is a simple check; more robust solutions might use a lock or check task status
    active_tasks = [t for t in asyncio.all_tasks() if t.get_coro().__name__ == 'modular_signal_generation_loop']
    if not active_tasks:
        asyncio.create_task(modular_signal_generation_loop())
        logger.info("Modular signal generation loop task created.")
    else:
        logger.info("Modular signal generation loop task already exists.")

    return {
        "status": "started", 
        "version": "4.0.0-MODULAR",
        "message": "Modular signal generation has been initiated.",
        "active_components": hyper_state.stats.get("modular_components", []),
        "data_source_status": config.get_data_source_status()
    }

@app.post("/api/stop")
async def stop_system():
    """Stop the modular signal generation loop."""
    if not hyper_state.is_running:
        return {"status": "already_stopped", "version": "4.0.0-MODULAR", "message": "System was not running."}
    
    hyper_state.is_running = False
    # Note: The loop will exit on its next check of hyper_state.is_running.
    # For immediate stop, task cancellation would be needed, which is more complex.
    logger.info("Signal generation loop requested to stop. It will halt after the current cycle or sleep interval.")
    return {"status": "stopping_initiated", "version": "4.0.0-MODULAR", "message": "Signal generation will stop."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket for real-time modular data, pings, and component status requests."""
    await manager.connect(websocket)
    client_ip = websocket.client.host if websocket.client else "unknown_client"
    logger.info(f"WebSocket connection established with {client_ip}")
    
    try:
        # Send initial state or welcome message if desired
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to HYPERtrends v4.0 WebSocket",
            "version": "4.0.0-MODULAR",
            "timestamp": datetime.now().isoformat()
        }))

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.debug(f"WebSocket received from {client_ip}: {message}")
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                    "server_time_utc": datetime.utcnow().isoformat(),
                    "version": "4.0.0-MODULAR",
                    "active_components_count": len(hyper_state.stats.get("modular_components", [])),
                    "data_source_status": config.get_data_source_status(),
                    "system_running": hyper_state.is_running
                }))
            elif message.get("type") == "get_component_status":
                component_status_data = await get_component_status() # This is an async route handler
                await websocket.send_text(json.dumps({
                    "type": "component_status_response", # Clearer response type
                    "data": component_status_data, # The data is already a dict
                    "timestamp": datetime.now().isoformat()
                }))
            elif message.get("type") == "get_latest_signals":
                 latest_signals_data = await get_current_signals() # This is an async route handler
                 await websocket.send_text(json.dumps({
                    "type": "latest_signals_response",
                    "data": latest_signals_data,
                    "timestamp": datetime.now().isoformat()
                 }))
            else:
                logger.warning(f"WebSocket received unknown message type from {client_ip}: {message.get('type')}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Unknown request type",
                    "request_type_received": message.get("type")
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_ip} (Reason: client closed connection or timeout)")
    except ConnectionResetError:
        logger.warning(f"WebSocket connection reset by peer: {client_ip}")
    except json.JSONDecodeError:
        logger.error(f"WebSocket received invalid JSON from {client_ip}")
        # Optionally send an error back if the connection is still open, though it might not be.
    except Exception as e:
        logger.error(f"WebSocket error with {client_ip}: {e}\n{traceback.format_exc()}")
    finally:
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed and cleaned up for {client_ip}. Active clients: {len(manager.active_connections)}")


# Model testing endpoints (if available)
@app.get("/api/testing/status", summary="Get Model Testing Framework Status")
async def get_testing_status():
    if hyper_state.testing_api and hasattr(hyper_state.testing_api, 'get_test_status'):
        return await hyper_state.testing_api.get_test_status()
    return {"status": "unavailable", "message": "Testing framework not initialized or get_test_status not available."}

@app.post("/api/testing/backtest", summary="Run a Quick Backtest")
async def run_backtest(days: int = 7):
    if not (isinstance(days, int) and 1 <= days <= 365): # Basic validation
        raise HTTPException(status_code=400, detail="Invalid 'days' parameter. Must be an integer between 1 and 365.")
    if hyper_state.testing_api and hasattr(hyper_state.testing_api, 'run_quick_backtest'):
        try:
            return await hyper_state.testing_api.run_quick_backtest(days)
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            raise HTTPException(status_code=500, detail=f"Error during backtest execution: {str(e)}")
    raise HTTPException(status_code=503, detail="Testing framework not available or run_quick_backtest not implemented.")

# ML learning endpoints (if available)  
@app.get("/api/ml/status", summary="Get ML Learning System Status")
async def get_ml_status():
    if hyper_state.learning_api and hasattr(hyper_state.learning_api, 'get_ml_status'):
        return await hyper_state.learning_api.get_ml_status()
    return {"status": "unavailable", "message": "ML learning not initialized or get_ml_status not available."}

@app.get("/api/ml/performance", summary="Get ML Model Performance Metrics")
async def get_ml_performance():
    if hyper_state.learning_api and hasattr(hyper_state.learning_api, 'get_model_performance'):
        try:
            return await hyper_state.learning_api.get_model_performance()
        except Exception as e:
            logger.error(f"ML performance retrieval error: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching ML performance: {str(e)}")
    raise HTTPException(status_code=503, detail="ML learning not available or get_model_performance not implemented.")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 HYPERtrends v4.0 Modular System Startup Sequence Initiated...")
    
    # Initialize core components first
    success = await hyper_state.initialize()
    if not success:
        logger.critical("❌ CRITICAL: Core system initialization failed during startup. System may not function.")
        # Depending on severity, you might want to prevent FastAPI from fully starting,
        # but FastAPI doesn't have a direct way to halt startup from an event handler.
        # Logging critical failure is important.
        hyper_state.is_running = False # Ensure it's marked as not running
        return 

    try:
        # Attempt to generate initial signals only if initialization was successful
        if hyper_state.signal_engine: # Check if signal_engine was initialized
            logger.info("Attempting to generate initial signals...")
            # This assumes generate_all_signals is a method in your HYPERSignalEngine
            # It might be intensive for startup, consider if this is necessary or can be deferred
            if hasattr(hyper_state.signal_engine, 'generate_all_signals'):
                 initial_signals = await hyper_state.signal_engine.generate_all_signals()
                 hyper_state.current_signals = initial_signals
                 hyper_state.last_update = datetime.now()
                 active_components_count = len(hyper_state.stats.get("modular_components", []))
                 logger.info(f"✅ Initial signals generated with {active_components_count} components. Signals for {len(initial_signals)} tickers.")
            else:
                logger.warning("signal_engine.generate_all_signals method not found. Skipping initial signal generation.")
        else:
            logger.warning("Signal engine not available after initialization. Cannot generate initial signals.")

    except Exception as e:
        logger.error(f"⚠️ Failed to generate initial signals during startup: {e}\n{traceback.format_exc()}")
    
    # Start the main signal generation loop if initialization was successful
    hyper_state.is_running = True # Mark as running *before* starting the loop
    hyper_state.stats["uptime_start"] = datetime.now() # Set/reset uptime start
    
    # Ensure only one loop starts
    active_tasks = [t for t in asyncio.all_tasks() if t.get_coro().__name__ == 'modular_signal_generation_loop']
    if not active_tasks:
        asyncio.create_task(modular_signal_generation_loop())
        logger.info("🔥 HYPERtrends v4.0 Modular System auto-started signal generation loop!")
    else:
        logger.info("Signal generation loop already planned or running.")

    active_components_list = hyper_state.stats.get('modular_components', [])
    logger.info(f"🎯 Active components post-startup: {', '.join(active_components_list) if active_components_list else 'None listed'}")
    
    if config.has_robinhood_credentials():
        logger.info("📱 Robinhood integration: Potentially ACTIVE (credentials found)")
    else:
        logger.info("🔄 Dynamic simulation/other APIs: Primary data source mode")
    
    logger.info("✅ HYPERtrends v4.0 Startup Sequence Complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("⏸️ Shutting down HYPERtrends v4.0 Modular System...")
    hyper_state.is_running = False # Signal loops to stop
    
    # Gracefully close WebSocket connections
    logger.info(f"Closing {len(manager.active_connections)} active WebSocket connections...")
    for connection in list(manager.active_connections): # Iterate over a copy
        try:
            await connection.close(code=1000) # Graceful close
        except Exception as e:
            # Log error but continue, as we want to attempt to close all
            logger.warning(f"Error closing a WebSocket connection: {e}")
    manager.active_connections.clear() # Clear the list
    
    # Close data aggregator if it has a close method
    if hyper_state.data_aggregator and hasattr(hyper_state.data_aggregator, 'close'):
        try:
            logger.info("Closing data aggregator...")
            await hyper_state.data_aggregator.close()
        except Exception as e:
            logger.error(f"Error closing data aggregator: {e}")
    
    # Add any other cleanup tasks here (e.g., saving state, closing database connections)
    logger.info("Performing other cleanup tasks...")
    # (Example: await some_other_cleanup_resource.close())

    # Allow a brief moment for async tasks to complete their current iteration
    await asyncio.sleep(1) # Adjust as needed

    logger.info("👋 HYPERtrends v4.0 Modular System shutdown complete.")

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    # This block is for direct execution (e.g., python main.py)
    # Render typically uses the Uvicorn command specified in its dashboard (e.g., uvicorn main:app --host 0.0.0.0 --port $PORT)
    # So, this direct run configuration is more for local development/testing.
    
    logger.info("🌟 Launching HYPERtrends v4.0 - Modular Edition (Direct Execution Mode)")
    logger.info(f"🔧 Configured Environment: {config.ENVIRONMENT}")
    logger.info(f"📱 Robinhood Credentials: {'✅ Available' if config.has_robinhood_credentials() else '❌ Not Configured'}")
    logger.info(f"🔄 Demo Mode Active: {config.DEMO_MODE}")
    
    # Ensure SERVER_CONFIG is loaded from your config.py
    server_host = config.SERVER_CONFIG.get("host", "0.0.0.0")
    server_port = config.SERVER_CONFIG.get("port", 8000) # Default to 8000 for local if not specified
    server_reload = config.SERVER_CONFIG.get("reload", False) # Typically False for production
    server_access_log = config.SERVER_CONFIG.get("access_log", True)

    logger.info(f"Starting Uvicorn server on {server_host}:{server_port} (Reload: {server_reload})")
    
    uvicorn.run(
        "main:app", # Important: Use "main:app" to refer to the app instance in this file
        host=server_host,
        port=server_port,
        reload=server_reload, # Set to False for production deployments on Render
        access_log=server_access_log # Can be True or False based on preference
        #workers=config.SERVER_CONFIG.get("workers", 1) # For production, you might use multiple workers
                                                      # Render might handle this differently (e.g. via Gunicorn)
                                                      # Check Render's best practices for Python web services.
    )
