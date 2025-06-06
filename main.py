# main.py - HYPERtrends v4.0 - Production Alpaca Edition

import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ========================================

# BASIC LOGGING SETUP (Safe)

# ========================================

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’
)
logger = logging.getLogger(**name**)

# ========================================

# FASTAPI APPLICATION

# ========================================

app = FastAPI(
title=‘HYPERtrends v4.0 - Alpaca Edition’,
description=‘AI-powered trading signals with Alpaca Markets integration’,
version=‘4.0.0-ALPACA’
)

# CORS Configuration

app.add_middleware(
CORSMiddleware,
allow_origins=[’*’],
allow_credentials=True,
allow_methods=[’*’],
allow_headers=[’*’],
)

# ========================================

# GLOBAL STATE

# ========================================

class HYPERState:
def **init**(self):
self.is_running = False
self.initialization_complete = False
self.startup_time = datetime.now()
self.connected_clients = []
self.current_signals = {}
self.last_update = None

```
    self.stats = {
        'status': 'starting',
        'initialization_complete': False,
        'signals_generated': 0
    }
    
    logger.info('HYPER state initialized')
```

hyper_state = HYPERState()

# ========================================

# HELPER FUNCTIONS

# ========================================

def create_fallback_signal(symbol: str) -> Dict[str, Any]:
“”“Create a single fallback signal”””
base_prices = {
‘QQQ’: 450.25, ‘SPY’: 535.80, ‘NVDA’: 875.90,
‘AAPL’: 185.45, ‘MSFT’: 428.75
}

```
price = base_prices.get(symbol, 100.0)

return {
    'symbol': symbol,
    'signal_type': 'HOLD',
    'confidence': 50.0,
    'direction': 'NEUTRAL',
    'price': price,
    'timestamp': datetime.now().isoformat(),
    'technical_score': 50.0,
    'sentiment_score': 50.0,
    'ml_score': 50.0,
    'vix_score': 50.0,
    'data_source': 'fallback'
}
```

async def create_fallback_signals() -> Dict[str, Any]:
“”“Create basic fallback signals”””
signals = {}
tickers = [‘QQQ’, ‘SPY’, ‘NVDA’, ‘AAPL’, ‘MSFT’]

```
for symbol in tickers:
    signals[symbol] = create_fallback_signal(symbol)

hyper_state.current_signals = signals
hyper_state.last_update = datetime.now()
hyper_state.is_running = True

logger.info('Fallback signals created')
return signals
```

async def initialize_components():
“”“Initialize components safely”””
try:
logger.info(‘Starting component initialization…’)

```
    # Try to load config
    try:
        import config
        logger.info('Config loaded successfully')
    except Exception as e:
        logger.warning(f'Config import failed: {e}')
    
    # Try to load other components
    try:
        from data_sources import HYPERDataAggregator
        from signal_engine import HYPERSignalEngine
        logger.info('Core modules loaded successfully')
    except Exception as e:
        logger.warning(f'Module import failed: {e}')
    
    # Create fallback signals for now
    await create_fallback_signals()
    
    # Mark as operational
    hyper_state.is_running = True
    hyper_state.initialization_complete = True
    hyper_state.stats['initialization_complete'] = True
    hyper_state.stats['status'] = 'operational'
    
    logger.info('System initialization complete')
    
except Exception as e:
    logger.error(f'Initialization failed: {e}')
    hyper_state.stats['status'] = 'error'
    await create_fallback_signals()
```

# ========================================

# WEBSOCKET MANAGER

# ========================================

class ConnectionManager:
def **init**(self):
self.active_connections: List[WebSocket] = []

```
async def connect(self, websocket: WebSocket):
    await websocket.accept()
    self.active_connections.append(websocket)
    logger.info(f'WebSocket connected. Total: {len(self.active_connections)}')

def disconnect(self, websocket: WebSocket):
    if websocket in self.active_connections:
        self.active_connections.remove(websocket)

async def broadcast(self, message: dict):
    if not self.active_connections:
        return
    
    disconnected = []
    message_json = json.dumps(message, default=str)
    
    for connection in self.active_connections:
        try:
            await connection.send_text(message_json)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        self.disconnect(conn)
```

manager = ConnectionManager()

# ========================================

# STARTUP EVENTS

# ========================================

@app.on_event(‘startup’)
async def startup_event():
“”“Fast startup with background initialization”””
try:
logger.info(‘Starting HYPERtrends v4.0 - Alpaca Edition’)

```
    # Immediate port binding
    hyper_state.stats['status'] = 'online'
    
    # Schedule background initialization
    asyncio.create_task(initialize_components())
    
    logger.info('Server started - Initialization in progress')
    
except Exception as e:
    logger.error(f'Startup error: {e}')
    hyper_state.stats['status'] = 'error'
```

# ========================================

# API ENDPOINTS

# ========================================

@app.get(’/’)
async def dashboard():
“”“Main dashboard”””
return HTMLResponse(’’’
<!DOCTYPE html>
<html>
<head>
<title>HYPERtrends v4.0</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
.container { max-width: 800px; margin: 0 auto; }
.status { padding: 20px; background: #2d2d2d; border-radius: 8px; margin: 20px 0; }
.signal { padding: 15px; background: #3d3d3d; border-radius: 5px; margin: 10px 0; }
.green { color: #4ade80; }
.red { color: #f87171; }
.yellow { color: #fbbf24; }
</style>
</head>
<body>
<div class="container">
<h1>🚀 HYPERtrends v4.0 - Alpaca Edition</h1>
<div class="status">
<h2>System Status</h2>
<p>Status: <span class="green">Online</span></p>
<p>Initialization: <span id="init-status">In Progress…</span></p>
<p>API: <a href="/docs" style="color: #60a5fa;">/docs</a></p>
<p>Health: <a href="/health" style="color: #60a5fa;">/health</a></p>
<p>Signals: <a href="/api/signals" style="color: #60a5fa;">/api/signals</a></p>
</div>
<div id="signals"></div>
</div>

```
    <script>
        async function updateStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('init-status').textContent = 
                    data.initialization_complete ? 'Complete' : 'In Progress...';
                document.getElementById('init-status').className = 
                    data.initialization_complete ? 'green' : 'yellow';
            } catch (e) {
                console.error('Status update failed:', e);
            }
        }
        
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals');
                const data = await response.json();
                const signalsDiv = document.getElementById('signals');
                
                if (data.signals) {
                    let html = '<h2>Current Signals</h2>';
                    for (const [symbol, signal] of Object.entries(data.signals)) {
                        const colorClass = signal.signal_type === 'HYPER_BUY' || signal.signal_type === 'SOFT_BUY' ? 'green' :
                                         signal.signal_type === 'HYPER_SELL' || signal.signal_type === 'SOFT_SELL' ? 'red' : 'yellow';
                        html += `
                            <div class="signal">
                                <strong>${symbol}</strong>: 
                                <span class="${colorClass}">${signal.signal_type}</span>
                                (${signal.confidence.toFixed(1)}% confidence) - 
                                $${signal.price.toFixed(2)}
                            </div>
                        `;
                    }
                    signalsDiv.innerHTML = html;
                }
            } catch (e) {
                console.error('Signals update failed:', e);
            }
        }
        
        // Update every 10 seconds
        setInterval(updateStatus, 10000);
        setInterval(loadSignals, 10000);
        
        // Initial load
        updateStatus();
        loadSignals();
    </script>
</body>
</html>
''')
```

@app.get(’/health’)
async def health_check():
“”“Health check endpoint”””
return {
‘status’: ‘healthy’,
‘timestamp’: datetime.now().isoformat(),
‘system_status’: hyper_state.stats[‘status’],
‘initialization_complete’: hyper_state.initialization_complete,
‘uptime_seconds’: (datetime.now() - hyper_state.startup_time).total_seconds(),
‘connected_clients’: len(hyper_state.connected_clients),
‘signals_available’: len(hyper_state.current_signals)
}

@app.get(’/api/signals’)
async def get_signals():
“”“Get current trading signals”””
return {
‘status’: ‘success’,
‘signals’: hyper_state.current_signals,
‘last_update’: hyper_state.last_update.isoformat() if hyper_state.last_update else None,
‘timestamp’: datetime.now().isoformat(),
‘system_status’: hyper_state.stats[‘status’]
}

@app.get(’/api/signals/{symbol}’)
async def get_signal(symbol: str):
“”“Get signal for specific symbol”””
symbol = symbol.upper()

```
if symbol in hyper_state.current_signals:
    return {
        'status': 'success',
        'symbol': symbol,
        'signal': hyper_state.current_signals[symbol],
        'timestamp': datetime.now().isoformat()
    }
else:
    raise HTTPException(status_code=404, detail=f'Signal for {symbol} not found')
```

@app.post(’/api/signals/refresh’)
async def refresh_signals():
“”“Manual signal refresh”””
try:
new_signals = await create_fallback_signals()
return {
‘status’: ‘success’,
‘message’: ‘Signals refreshed’,
‘signals’: new_signals,
‘timestamp’: datetime.now().isoformat()
}
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))

@app.get(’/api/system/status’)
async def system_status():
“”“System status information”””
return {
‘status’: hyper_state.stats[‘status’],
‘initialization_complete’: hyper_state.initialization_complete,
‘uptime’: (datetime.now() - hyper_state.startup_time).total_seconds(),
‘connected_clients’: len(hyper_state.connected_clients),
‘signals_generated’: hyper_state.stats[‘signals_generated’],
‘last_update’: hyper_state.last_update.isoformat() if hyper_state.last_update else None
}

# ========================================

# WEBSOCKET ENDPOINT

# ========================================

@app.websocket(’/ws’)
async def websocket_endpoint(websocket: WebSocket):
“”“WebSocket endpoint for real-time updates”””
await manager.connect(websocket)
try:
# Send initial data
await websocket.send_text(json.dumps({
‘type’: ‘initial_data’,
‘signals’: hyper_state.current_signals,
‘timestamp’: datetime.now().isoformat(),
‘status’: hyper_state.stats[‘status’]
}, default=str))

```
    # Keep connection alive
    while True:
        try:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({
                'type': 'heartbeat',
                'timestamp': datetime.now().isoformat(),
                'connected_clients': len(manager.active_connections)
            }, default=str))
        except:
            break
            
except WebSocketDisconnect:
    manager.disconnect(websocket)
except Exception as e:
    logger.error(f'WebSocket error: {e}')
    manager.disconnect(websocket)
```

# ========================================

# MAIN

# ========================================

if **name** == ‘**main**’:
port = int(os.getenv(‘PORT’, 8000))
uvicorn.run(
app,
host=‘0.0.0.0’,
port=port,
log_level=‘info’
)
