import logging
import time
import random
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
import requests
import aiohttp

from config import ALPACA_CONFIG, TICKERS

logger = logging.getLogger(__name__)

class AlpacaDataClient:
    """Optimized Alpaca API client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key", "")
        self.secret_key = config.get("secret_key", "")
        self.base_url = config.get("base_url", "")
        self.data_url = config.get("data_url", "")
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        # Session for connection pooling
        self.session = None
        self._init_session()
        
        logger.info(f"✅ AlpacaDataClient initialized: {self.base_url}")

    def _init_session(self):
        """Initialize requests session"""
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Connection pooling and timeouts
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for symbol"""
        try:
            if not self.api_key or not self.secret_key:
                return None
                
            endpoint = f"{self.data_url}/stocks/{symbol}/quotes/latest"
            response = self.session.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                quote = data.get('quote', {})
                
                if quote:
                    return {
                        'symbol': symbol,
                        'price': quote.get('ap', quote.get('bp', 0)),  # Ask price or bid price
                        'bid': quote.get('bp', 0),
                        'ask': quote.get('ap', 0),
                        'bid_size': quote.get('bs', 0),
                        'ask_size': quote.get('as', 0),
                        'timestamp': quote.get('t', datetime.now().isoformat()),
                        'data_source': 'alpaca_quote',
                        'enhanced_features': {
                            'data_freshness': 'real_time',
                            'market_hours': self._determine_market_hours(),
                            'spread_bps': self._calculate_spread_bps(quote.get('bp', 0), quote.get('ap', 0))
                        }
                    }
            else:
                logger.warning(f"Alpaca quote API error for {symbol}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting Alpaca quote for {symbol}: {e}")
        
        return None

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price bar for symbol"""
        try:
            if not self.api_key or not self.secret_key:
                return None
                
            endpoint = f"{self.data_url}/stocks/{symbol}/bars"
            params = {
                'timeframe': '1Min',
                'limit': 1,
                'asof': datetime.now().isoformat()
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                
                if bars:
                    bar = bars[0]
                    return {
                        'symbol': symbol,
                        'open': bar.get('o', 0),
                        'high': bar.get('h', 0),
                        'low': bar.get('l', 0),
                        'close': bar.get('c', 0),
                        'price': bar.get('c', 0),
                        'volume': bar.get('v', 0),
                        'timestamp': bar.get('t', datetime.now().isoformat()),
                        'data_source': 'alpaca_bar'
                    }
            else:
                logger.warning(f"Alpaca bar API error for {symbol}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting Alpaca bar for {symbol}: {e}")
        
        return None

    def get_historical_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical bars for symbol"""
        try:
            if not self.api_key or not self.secret_key:
                return []
                
            endpoint = f"{self.data_url}/stocks/{symbol}/bars"
            params = {
                'timeframe': timeframe,
                'limit': limit,
                'adjustment': 'raw'
            }
            
            response = self.session.get(endpoint, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                
                formatted_bars = []
                for bar in bars:
                    formatted_bars.append({
                        'open': bar.get('o', 0),
                        'high': bar.get('h', 0),
                        'low': bar.get('l', 0),
                        'close': bar.get('c', 0),
                        'price': bar.get('c', 0),
                        'volume': bar.get('v', 0),
                        'timestamp': bar.get('t', ''),
                        'date': bar.get('t', '')[:10] if bar.get('t') else ''
                    })
                
                return formatted_bars
            else:
                logger.warning(f"Alpaca historical API error for {symbol}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting Alpaca historical data for {symbol}: {e}")
        
        return []

    def _determine_market_hours(self) -> str:
        """Determine current market hours"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Weekend
        if weekday >= 5:
            return "CLOSED"
        
        # Market hours (9:30 AM - 4:00 PM ET, simplified)
        if 9 <= hour < 16:
            return "REGULAR_HOURS"
        elif 4 <= hour < 9:
            return "PRE_MARKET"
        elif 16 <= hour <= 20:
            return "AFTER_HOURS"
        else:
            return "CLOSED"

    def _calculate_spread_bps(self, bid: float, ask: float) -> float:
        """Calculate bid-ask spread in basis points"""
        if bid > 0 and ask > 0 and ask > bid:
            mid_price = (bid + ask) / 2
            spread = ask - bid
            return (spread / mid_price) * 10000  # Convert to basis points
        return 0.0

    def cleanup(self):
        """Cleanup resources"""
        if self.session:
            self.session.close()

class EnhancedMarketSimulator:
    """Enhanced market data simulator with realistic features"""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.price_history = {}
        self.last_update = {}
        
        # Initialize with realistic base prices
        self.base_prices = {
            "QQQ": 450.25,
            "SPY": 535.80, 
            "NVDA": 875.90,
            "AAPL": 185.45,
            "MSFT": 428.75
        }
        
        # Symbol-specific parameters
        self.symbol_params = {
            "QQQ": {"volatility": 0.025, "volume_base": 45000000},
            "SPY": {"volatility": 0.020, "volume_base": 85000000},
            "NVDA": {"volatility": 0.035, "volume_base": 35000000},
            "AAPL": {"volatility": 0.025, "volume_base": 50000000},
            "MSFT": {"volatility": 0.022, "volume_base": 30000000}
        }
        
        logger.info(f"📊 Enhanced Market Simulator initialized for {len(tickers)} symbols")

    def get_quote_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic quote data"""
        try:
            base_price = self.base_prices.get(symbol, 100.0)
            params = self.symbol_params.get(symbol, {"volatility": 0.025, "volume_base": 25000000})
            
            # Generate realistic price movement
            last_price = self.price_history.get(symbol, base_price)
            
            # Market factors
            time_factor = self._get_time_factor()
            trend_factor = self._get_trend_factor(symbol)
            volatility = params["volatility"] * time_factor
            
            # Price change
            random_change = random.gauss(0, volatility)
            price_change = (trend_factor * 0.1 + random_change) * last_price
            current_price = max(0.01, last_price + price_change)
            
            # Update history
            self.price_history[symbol] = current_price
            self.last_update[symbol] = datetime.now()
            
            # Calculate derived values
            change_percent = ((current_price - base_price) / base_price) * 100
            
            # Generate volume
            base_volume = params["volume_base"]
            volume_multiplier = 1 + abs(change_percent) * 0.1 + random.uniform(-0.3, 0.3)
            volume = int(base_volume * volume_multiplier)
            
            # Generate OHLC
            daily_range = current_price * volatility * 2
            high = current_price + random.uniform(0, daily_range * 0.5)
            low = current_price - random.uniform(0, daily_range * 0.5)
            open_price = current_price + random.uniform(-daily_range * 0.3, daily_range * 0.3)
            
            # Bid/Ask
            spread_pct = random.uniform(0.001, 0.005)  # 0.1% to 0.5% spread
            spread = current_price * spread_pct
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            return {
                'symbol': symbol,
                'price': round(current_price, 2),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'bid_size': random.randint(100, 1000),
                'ask_size': random.randint(100, 1000),
                'change_percent': round(change_percent, 2),
                'timestamp': datetime.now().isoformat(),
                'data_source': 'enhanced_simulation',
                'enhanced_features': {
                    'data_freshness': 'simulated_real_time',
                    'market_hours': self._determine_market_hours(),
                    'spread_bps': round((spread / current_price) * 10000, 1),
                    'volatility_regime': self._get_volatility_regime(volatility),
                    'popularity_rank': random.randint(1, 100)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating quote for {symbol}: {e}")
            return self._get_fallback_quote(symbol)

    def get_historical_data(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic historical data"""
        try:
            current_price = self.base_prices.get(symbol, 100.0)
            params = self.symbol_params.get(symbol, {"volatility": 0.025, "volume_base": 25000000})
            
            history = []
            price = current_price
            
            for i in range(limit):
                # Generate price movement
                volatility = params["volatility"]
                change = random.gauss(0, volatility) + random.uniform(-0.001, 0.001)  # Small trend
                price = max(0.01, price * (1 + change))
                
                # Generate OHLC
                daily_range = price * volatility * random.uniform(1.5, 2.5)
                open_price = price + random.uniform(-daily_range * 0.2, daily_range * 0.2)
                high = max(price, open_price) + random.uniform(0, daily_range * 0.4)
                low = min(price, open_price) - random.uniform(0, daily_range * 0.4)
                
                # Generate volume
                base_volume = params["volume_base"]
                volume_multiplier = random.uniform(0.7, 1.5)
                volume = int(base_volume * volume_multiplier)
                
                date = (datetime.now() - timedelta(days=limit-i)).strftime('%Y-%m-%d')
                
                history.append({
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(price, 2),
                    'price': round(price, 2),
                    'volume': volume,
                    'timestamp': f"{date}T16:00:00Z",
                    'date': date
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error generating historical data for {symbol}: {e}")
            return []

    def _get_time_factor(self) -> float:
        """Get volatility factor based on time of day"""
        hour = datetime.now().hour
        
        # Higher volatility at market open/close
        if hour in [9, 15, 16]:
            return 1.5
        elif 10 <= hour <= 14:
            return 0.8
        else:
            return 1.0

    def _get_trend_factor(self, symbol: str) -> float:
        """Get trend factor for symbol"""
        # Simple trending logic
        if symbol in ["QQQ", "NVDA"]:
            return random.uniform(-0.5, 1.0)  # Slight bullish bias
        elif symbol in ["SPY", "AAPL", "MSFT"]:
            return random.uniform(-0.3, 0.5)   # Neutral
        else:
            return random.uniform(-0.5, 0.5)

    def _determine_market_hours(self) -> str:
        """Determine current market hours"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        if weekday >= 5:
            return "CLOSED"
        elif 9 <= hour < 16:
            return "REGULAR_HOURS"
        elif 4 <= hour < 9:
            return "PRE_MARKET"
        elif 16 <= hour <= 20:
            return "AFTER_HOURS"
        else:
            return "CLOSED"

    def _get_volatility_regime(self, volatility: float) -> str:
        """Determine volatility regime"""
        if volatility > 0.04:
            return "HIGH"
        elif volatility > 0.025:
            return "NORMAL"
        else:
            return "LOW"

    def _get_fallback_quote(self, symbol: str) -> Dict[str, Any]:
        """Get fallback quote data"""
        base_price = self.base_prices.get(symbol, 100.0)
        return {
            'symbol': symbol,
            'price': base_price,
            'volume': 25000000,
            'change_percent': 0.0,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'fallback',
            'enhanced_features': {
                'data_freshness': 'fallback',
                'market_hours': 'UNKNOWN'
            }
        }

class HYPERDataAggregator:
    """Main data aggregation class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ALPACA_CONFIG
        self.alpaca_client = AlpacaDataClient(self.config)
        self.simulator = EnhancedMarketSimulator(TICKERS)
        self.tickers = TICKERS
        
        # Determine primary data source
        self.has_alpaca = bool(self.config.get("api_key") and self.config.get("secret_key"))
        
        logger.info(f"📊 Data Aggregator initialized")
        logger.info(f"🔌 Alpaca API: {'Available' if self.has_alpaca else 'Not Available'}")
        logger.info(f"📈 Tracking: {', '.join(self.tickers)}")

    async def initialize(self):
        """Initialize the data aggregator"""
        try:
            if self.has_alpaca:
                logger.info("✅ Using Alpaca Markets as primary data source")
            else:
                logger.info("📊 Using enhanced simulation as primary data source")
                
        except Exception as e:
            logger.error(f"❌ Data aggregator initialization error: {e}")

    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data for symbol"""
        try:
            # Try Alpaca first if available
            if self.has_alpaca:
                quote_data = await self._get_alpaca_data(symbol)
                if quote_data:
                    historical_data = self.alpaca_client.get_historical_bars(symbol, limit=100)
                    return {
                        "quote": quote_data,
                        "historical": historical_data or self.simulator.get_historical_data(symbol),
                        "trends": {}  # Placeholder for trends data
                    }
            
            # Fall back to enhanced simulation
            quote_data = self.simulator.get_quote_data(symbol)
            historical_data = self.simulator.get_historical_data(symbol)
            
            return {
                "quote": quote_data,
                "historical": historical_data,
                "trends": {}
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive data for {symbol}: {e}")
            return {
                "quote": self.simulator.get_quote_data(symbol),
                "historical": self.simulator.get_historical_data(symbol),
                "trends": {}
            }

    async def _get_alpaca_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Alpaca API"""
        try:
            # Try quote first, then bar data
            quote_data = self.alpaca_client.get_latest_quote(symbol)
            if quote_data and quote_data.get('price', 0) > 0:
                return quote_data
            
            # Fall back to bar data
            bar_data = self.alpaca_client.get_latest_bar(symbol)
            if bar_data and bar_data.get('price', 0) > 0:
                # Enhance bar data with quote-like features
                bar_data.update({
                    'bid': bar_data['price'] * 0.9995,  # Estimate bid
                    'ask': bar_data['price'] * 1.0005,  # Estimate ask
                    'enhanced_features': {
                        'data_freshness': 'real_time',
                        'market_hours': self.alpaca_client._determine_market_hours(),
                        'spread_bps': 5.0  # Estimated spread
                    }
                })
                return bar_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Alpaca data error for {symbol}: {e}")
            return None

    async def get_historical_data_api(self, symbol: str, timeframe: str = "1Day", 
                                     start: str = None, end: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data via API"""
        try:
            if self.has_alpaca:
                historical = self.alpaca_client.get_historical_bars(symbol, timeframe, limit)
                if historical:
                    return historical
            
            # Fall back to simulation
            return self.simulator.get_historical_data(symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"❌ Historical data API error for {symbol}: {e}")
            return self.simulator.get_historical_data(symbol, timeframe, limit)

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.alpaca_client:
                self.alpaca_client.cleanup()
            logger.info("✅ Data aggregator cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Export main class
__all__ = ['HYPERDataAggregator', 'AlpacaDataClient', 'EnhancedMarketSimulator']

logger.info("📊 Optimized Data Sources module loaded successfully")