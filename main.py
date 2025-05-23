from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import yfinance as yf
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="HYPER Trading Engine", version="2.0")

# Enhanced Trading State
class TradingState:
    def __init__(self):
        self.is_trading = False
        self.portfolio_value = 100000.0
        self.positions = {}
        self.signals = []
        self.connected_clients = []
        self.market_data = {}
        self.historical_data = {}
        self.technical_indicators = {}
        
        # Trading parameters
        self.tickers = ['QQQ', 'SPY', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'META']
        self.timeframes = ['1d', '1h']  # Start with these, expand later
        self.last_update = {}

trading_state = TradingState()

# Connection manager (same as before)
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
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ============================================
# REAL MARKET DATA FUNCTIONS
# ============================================

def fetch_real_market_data(ticker: str, period: str = '5d', interval: str = '1h'):
    """Fetch real market data using yfinance"""
    try:
        print(f"📈 Fetching real data for {ticker}...")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)
        
        if df.empty:
            print(f"❌ No data received for {ticker}")
            return None
            
        # Clean and prepare data
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Rename columns to match our format
        column_mapping = {
            'datetime': 'datetime',
            'date': 'datetime',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            df['datetime'] = df.index
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Keep only required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"✅ Got {len(df)} data points for {ticker}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return None

def calculate_technical_indicators(df: pd.DataFrame):
    """Calculate technical indicators on real market data"""
    try:
        if df is None or df.empty or len(df) < 20:
            return None
            
        df = df.copy()
        
        # Ensure numeric data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 20:  # Need minimum data for indicators
            return None
        
        # Calculate indicators
        print(f"📊 Calculating indicators for {len(df)} data points...")
        
        # EMAs
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        
        # RSI
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ADX
        if len(df) >= 14:
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx.adx()
        
        # Price-based indicators
        df['price_change'] = df['close'].pct_change() * 100
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP (simplified)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Momentum indicators
        df['momentum_3'] = df['close'] - df['close'].shift(3)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['atr'] = ((df['high'] - df['low']).rolling(window=14).mean())
        
        # Support/Resistance levels (simplified)
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"✅ Calculated indicators, {len(df)} valid rows remaining")
        return df
        
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

def generate_trading_signal(df: pd.DataFrame, ticker: str):
    """Generate trading signals based on real technical analysis"""
    try:
        if df is None or df.empty:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Initialize signal components
        signals = []
        confidence_factors = []
        
        # 1. RSI Signal
        rsi = latest['rsi_14']
        if rsi < 30:  # Oversold
            signals.append('BUY')
            confidence_factors.append(0.7)
        elif rsi > 70:  # Overbought
            signals.append('SELL')
            confidence_factors.append(0.6)
        
        # 2. EMA Crossover
        if latest['ema_9'] > latest['ema_20'] and prev['ema_9'] <= prev['ema_20']:
            signals.append('BUY')
            confidence_factors.append(0.8)
        elif latest['ema_9'] < latest['ema_20'] and prev['ema_9'] >= prev['ema_20']:
            signals.append('SELL')
            confidence_factors.append(0.7)
        
        # 3. MACD Signal
        if latest['macd_diff'] > 0 and prev['macd_diff'] <= 0:
            signals.append('BUY')
            confidence_factors.append(0.6)
        elif latest['macd_diff'] < 0 and prev['macd_diff'] >= 0:
            signals.append('SELL')
            confidence_factors.append(0.6)
        
        # 4. Bollinger Bands
        if latest['close'] < latest['bb_lower']:
            signals.append('BUY')
            confidence_factors.append(0.5)
        elif latest['close'] > latest['bb_upper']:
            signals.append('SELL')
            confidence_factors.append(0.5)
        
        # 5. Volume confirmation
        volume_factor = 1.0
        if latest['volume_ratio'] > 1.5:  # High volume
            volume_factor = 1.2
        elif latest['volume_ratio'] < 0.7:  # Low volume
            volume_factor = 0.8
        
        # 6. ADX trend strength
        adx_factor = 1.0
        if 'adx' in latest and not pd.isna(latest['adx']):
            if latest['adx'] > 25:  # Strong trend
                adx_factor = 1.1
            elif latest['adx'] < 15:  # Weak trend
                adx_factor = 0.9
        
        # Determine final signal
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if buy_signals > sell_signals:
            action = 'BUY'
            base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'BUY'])
        elif sell_signals > buy_signals:
            action = 'SELL'
            base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'SELL'])
        else:
            action = 'HOLD'
            base_confidence = 0.5
        
        # Apply volume and trend factors
        final_confidence = min(0.95, base_confidence * volume_factor * adx_factor)
        
        # Create detailed signal
        signal = {
            "ticker": ticker,
            "action": action,
            "confidence": round(final_confidence * 100, 1),
            "price": round(float(latest['close']), 2),
            "timestamp": latest['datetime'].isoformat() if hasattr(latest['datetime'], 'isoformat') else str(latest['datetime']),
            "indicators": {
                "rsi": round(float(rsi), 1),
                "ema_9": round(float(latest['ema_9']), 2),
                "ema_20": round(float(latest['ema_20']), 2),
                "macd_diff": round(float(latest['macd_diff']), 4),
                "bb_position": "above" if latest['close'] > latest['bb_upper'] else "below" if latest['close'] < latest['bb_lower'] else "middle",
                "volume_ratio": round(float(latest['volume_ratio']), 2),
                "adx": round(float(latest['adx']), 1) if 'adx' in latest and not pd.isna(latest['adx']) else None
            },
            "models": {
                "technical": round(final_confidence * 100, 1),
                "volume": round(volume_factor * 50, 1),
                "trend": round(adx_factor * 50, 1)
            }
        }
        
        return signal
        
    except Exception as e:
        print(f"❌ Error generating signal for {ticker}: {e}")
        return None

# ============================================
# ENHANCED API ENDPOINTS
# ============================================

@app.get("/")
async def get_interface():
    return FileResponse("index.html")

@app.get("/api/portfolio")
async def get_portfolio():
    return {
        "value": trading_state.portfolio_value,
        "positions": trading_state.positions,
        "is_trading": trading_state.is_trading,
        "timestamp": datetime.now().isoformat(),
        "market_data_status": len(trading_state.market_data),
        "signals_count": len(trading_state.signals)
    }

@app.get("/api/market-data/{ticker}")
async def get_market_data(ticker: str):
    """Get latest market data for a specific ticker"""
    if ticker in trading_state.market_data:
        return trading_state.market_data[ticker]
    return {"error": f"No data available for {ticker}"}

@app.get("/api/indicators/{ticker}")
async def get_indicators(ticker: str):
    """Get technical indicators for a specific ticker"""
    if ticker in trading_state.technical_indicators:
        latest = trading_state.technical_indicators[ticker].iloc[-1].to_dict()
        return {ticker: latest}
    return {"error": f"No indicators available for {ticker}"}

@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    if not trading_state.is_trading:
        trading_state.is_trading = True
        background_tasks.add_task(trading_loop)
        await manager.broadcast({
            "type": "trading_status",
            "status": "started",
            "message": "Live trading with REAL market data started!"
        })
        return {"status": "started", "message": "Real market data trading active"}
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

@app.post("/api/update-data")
async def manual_data_update():
    """Manually trigger data update"""
    try:
        await update_market_data()
        return {"status": "success", "message": "Market data updated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================
# ENHANCED WEBSOCKET
# ============================================

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
            elif message["type"] == "request_market_data":
                ticker = message.get("ticker", "SPY")
                if ticker in trading_state.market_data:
                    await websocket.send_text(json.dumps({
                        "type": "market_data",
                        "ticker": ticker,
                        "data": trading_state.market_data[ticker]
                    }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================
# ENHANCED TRADING LOGIC
# ============================================

async def trading_loop():
    """Main trading loop with real market data"""
    print("🚀 Starting REAL market data trading loop...")
    
    while trading_state.is_trading:
        try:
            # Update market data
            await update_market_data()
            
            # Generate signals based on real data
            await process_real_signals()
            
            # Wait before next update (adjust based on your needs)
            await asyncio.sleep(60)  # Update every minute for real data
            
        except Exception as e:
            print(f"❌ Trading loop error: {e}")
            await asyncio.sleep(30)  # Wait 30 seconds on error

async def update_market_data():
    """Update market data for all tickers"""
    print("📈 Updating market data...")
    
    for ticker in trading_state.tickers:
        try:
            # Fetch real market data
            df = fetch_real_market_data(ticker, period='5d', interval='1h')
            
            if df is not None and not df.empty:
                # Calculate technical indicators
                df_with_indicators = calculate_technical_indicators(df)
                
                if df_with_indicators is not None:
                    # Store market data
                    trading_state.market_data[ticker] = {
                        "latest_price": float(df_with_indicators.iloc[-1]['close']),
                        "change": float(df_with_indicators.iloc[-1]['price_change']) if 'price_change' in df_with_indicators.columns else 0.0,
                        "volume": float(df_with_indicators.iloc[-1]['volume']),
                        "timestamp": str(df_with_indicators.iloc[-1]['datetime'])
                    }
                    
                    # Store technical indicators
                    trading_state.technical_indicators[ticker] = df_with_indicators
                    trading_state.last_update[ticker] = datetime.now()
                    
                    print(f"✅ Updated {ticker}: ${trading_state.market_data[ticker]['latest_price']:.2f}")
                    
            await asyncio.sleep(1)  # Small delay between API calls
            
        except Exception as e:
            print(f"❌ Error updating {ticker}: {e}")
            continue

async def process_real_signals():
    """Generate trading signals from real market data"""
    new_signals = []
    
    for ticker in trading_state.tickers:
        try:
            if ticker in trading_state.technical_indicators:
                df = trading_state.technical_indicators[ticker]
                signal = generate_trading_signal(df, ticker)
                
                if signal:
                    new_signals.append(signal)
                    print(f"🎯 {ticker}: {signal['action']} at ${signal['price']} ({signal['confidence']}% confidence)")
        
        except Exception as e:
            print(f"❌ Error processing signal for {ticker}: {e}")
            continue
    
    # Update signals
    trading_state.signals = new_signals
    
    # Broadcast to connected clients
    await manager.broadcast({
        "type": "signals_update",
        "signals": new_signals,
        "timestamp": datetime.now().isoformat(),
        "data_source": "REAL_MARKET"
    })

async def send_full_update(websocket: WebSocket):
    """Send complete state update to client"""
    update = {
        "type": "full_update",
        "portfolio": {
            "value": trading_state.portfolio_value,
            "positions": trading_state.positions
        },
        "signals": trading_state.signals,
        "is_trading": trading_state.is_trading,
        "market_data": trading_state.market_data,
        "timestamp": datetime.now().isoformat(),
        "data_source": "REAL_MARKET"
    }
    await websocket.send_text(json.dumps(update))

# ============================================
# STARTUP INITIALIZATION
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize with real market data"""
    print("🚀 HYPER Engine starting with REAL market data...")
    
    # Pre-load market data
    print("📈 Pre-loading market data...")
    await update_market_data()
    
    print("✅ HYPER Engine ready with REAL market data!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
