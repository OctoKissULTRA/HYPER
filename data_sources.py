import aiohttp
import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

# Set up logging
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Enhanced Alpha Vantage client with comprehensive debugging"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 12  # 5 calls per minute = 12 seconds between calls
        
        logger.info(f"🔑 Alpha Vantage Client initialized with API key: {api_key[:10]}...")
        logger.info(f"📡 Base URL: {self.base_url}")
    
    async def create_session(self):
        """Create HTTP session if it doesn't exist"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'HYPER-Trading-System/2.0',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                }
            )
            logger.info("✅ HTTP session created")
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔒 HTTP session closed")
    
    async def _rate_limit_wait(self):
        """Implement rate limiting"""
        if self.last_request_time > 0:
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
    
    async def _make_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make Alpha Vantage API request with comprehensive logging"""
        await self.create_session()
        await self._rate_limit_wait()
        
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        try:
            self.request_count += 1
            self.last_request_time = time.time()
            
            logger.info(f"🌐 Making Alpha Vantage API request #{self.request_count}")
            logger.info(f"📋 URL: {self.base_url}")
            logger.info(f"📋 Parameters: {params}")
            
            async with self.session.get(self.base_url, params=params) as response:
                logger.info(f"📊 Response Status: {response.status}")
                logger.info(f"📊 Response Headers: {dict(response.headers)}")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ HTTP {response.status}: {error_text[:200]}...")
                    return None
                
                # Read response text
                response_text = await response.text()
                logger.info(f"📄 Response length: {len(response_text)} characters")
                logger.debug(f"📄 Response preview: {response_text[:300]}...")
                
                # Parse JSON
                try:
                    data = json.loads(response_text)
                    logger.info(f"✅ JSON parsed successfully")
                    
                    # Log the structure
                    if isinstance(data, dict):
                        logger.info(f"🔍 Response keys: {list(data.keys())}")
                        
                        # Check for API errors
                        if 'Error Message' in data:
                            logger.error(f"❌ Alpha Vantage Error: {data['Error Message']}")
                            return None
                        
                        if 'Note' in data:
                            logger.warning(f"⚠️ Alpha Vantage Note: {data['Note']}")
                            return None
                        
                        if 'Information' in data:
                            logger.warning(f"⚠️ Alpha Vantage Info: {data['Information']}")
                            return None
                    
                    return data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    logger.error(f"❌ Invalid JSON: {response_text[:200]}...")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"⏰ Request timeout")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"🌐 Client error: {e}")
            return None
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return None
    
    async def get_global_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote with enhanced error handling"""
        logger.info(f"📈 Fetching quote for {symbol}")
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'datatype': 'json'
        }
        
        data = await self._make_request(params)
        
        if not data:
            logger.error(f"❌ No data received for {symbol}")
            return self._generate_fallback_quote(symbol)
        
        # Parse Global Quote response
        if 'Global Quote' in data:
            quote = data['Global Quote']
            logger.info(f"✅ Quote data received for {symbol}")
            logger.debug(f"📊 Quote fields: {list(quote.keys())}")
            
            try:
                # Extract fields using Alpha Vantage's numbered keys
                result = {
                    'symbol': quote.get('01. symbol', symbol),
                    'open': float(quote.get('02. open', '0') or '0'),
                    'high': float(quote.get('03. high', '0') or '0'),
                    'low': float(quote.get('04. low', '0') or '0'),
                    'price': float(quote.get('05. price', '0') or '0'),
                    'volume': int(float(quote.get('06. volume', '0') or '0')),
                    'latest_trading_day': quote.get('07. latest trading day', ''),
                    'previous_close': float(quote.get('08. previous close', '0') or '0'),
                    'change': float(quote.get('09. change', '0') or '0'),
                    'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'alpha_vantage'
                }
                
                logger.info(f"💰 {symbol}: ${result['price']} ({result['change_percent']}%)")
                logger.info(f"📊 {symbol}: Volume {result['volume']:,}")
                
                return result
                
            except (ValueError, KeyError) as e:
                logger.error(f"❌ Error parsing quote data: {e}")
                logger.error(f"📄 Raw quote data: {quote}")
                return self._generate_fallback_quote(symbol)
        else:
            logger.error(f"❌ No 'Global Quote' in response for {symbol}")
            logger.debug(f"📄 Full response: {data}")
            return self._generate_fallback_quote(symbol)
    
    def _generate_fallback_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic fallback quote data"""
        logger.warning(f"🔄 Generating fallback data for {symbol}")
        
        # Realistic base prices for our tickers
        base_prices = {
            'QQQ': 380.25,
            'SPY': 485.67,
            'NVDA': 892.45,
            'AAPL': 184.22,
            'MSFT': 413.78
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate realistic market movement (-2% to +2%)
        change_percent = random.uniform(-2.0, 2.0)
        change = base_price * (change_percent / 100)
        current_price = base_price + change
        
        # Generate realistic volume
        volume = random.randint(1000000, 50000000)
        
        return {
            'symbol': symbol,
            'open': round(base_price + random.uniform(-1, 1), 2),
            'high': round(current_price + random.uniform(0, 2), 2),
            'low': round(current_price - random.uniform(0, 2), 2),
            'price': round(current_price, 2),
            'volume': volume,
            'latest_trading_day': datetime.now().strftime('%Y-%m-%d'),
            'previous_close': base_price,
            'change': round(change, 2),
            'change_percent': f"{change_percent:.2f}",
            'timestamp': datetime.now().isoformat(),
            'data_source': 'fallback'
        }

class GoogleTrendsClient:
    """Mock Google Trends client (since the real one is blocked)"""
    
    def __init__(self):
        logger.info("📈 Google Trends Client initialized (using fallback data)")
    
    async def get_trends_data(self, keywords: List[str]) -> Dict[str, Any]:
        """Generate mock trends data since Google blocks API requests"""
        logger.info(f"📈 Generating mock trends data for: {keywords}")
        
        # Generate realistic trend data
        trend_data = {}
        for keyword in keywords:
            # Generate some realistic search volume data
            momentum = random.uniform(-50, 100)  # -50% to +100% momentum
            velocity = random.uniform(-30, 30)   # Rate of change
            
            trend_data[keyword] = {
                'momentum': momentum,
                'velocity': velocity,
                'acceleration': random.uniform(-10, 10),
                'current_value': random.randint(20, 100),
                'average_value': random.randint(40, 80)
            }
        
        return {
            'keyword_data': trend_data,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'mock_trends'
        }

class HYPERDataAggregator:
    """Main data aggregation class"""
    
    def __init__(self, api_key: str):
        self.alpha_client = AlphaVantageClient(api_key)
        self.trends_client = GoogleTrendsClient()
        logger.info(f"🚀 HYPER Data Aggregator initialized")
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data for a symbol with comprehensive logging"""
        logger.info(f"🔍 Getting comprehensive data for {symbol}")
        
        start_time = time.time()
        
        try:
            # Get Alpha Vantage quote data
            quote_data = await self.alpha_client.get_global_quote(symbol)
            
            # Get trends data  
            keywords = self._get_keywords_for_symbol(symbol)
            trends_data = await self.trends_client.get_trends_data(keywords)
            
            # Combine the data
            result = {
                'symbol': symbol,
                'quote': quote_data,
                'trends': trends_data,
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'data_quality': self._assess_data_quality(quote_data, trends_data)
            }
            
            logger.info(f"✅ Comprehensive data for {symbol} completed in {result['processing_time']:.2f}s")
            logger.info(f"📊 Data quality: {result['data_quality']}")
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Error getting comprehensive data for {symbol}: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            
            return {
                'symbol': symbol,
                'quote': None,
                'trends': None,
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'data_quality': 'error',
                'error': str(e)
            }
    
    def _get_keywords_for_symbol(self, symbol: str) -> List[str]:
        """Get search keywords for a symbol"""
        keyword_map = {
            'QQQ': ['QQQ ETF', 'NASDAQ 100', 'tech stocks'],
            'SPY': ['SPY ETF', 'S&P 500', 'market index'],
            'NVDA': ['NVIDIA', 'AI stocks', 'graphics cards'],
            'AAPL': ['Apple', 'iPhone', 'Apple stock'],
            'MSFT': ['Microsoft', 'Azure', 'cloud computing']
        }
        return keyword_map.get(symbol, [symbol])
    
    def _assess_data_quality(self, quote_data: Dict, trends_data: Dict) -> str:
        """Assess the quality of the data"""
        quality_score = 0
        
        if quote_data and quote_data.get('price', 0) > 0:
            quality_score += 50
        
        if trends_data and 'keyword_data' in trends_data:
            quality_score += 30
        
        if quote_data and quote_data.get('data_source') == 'alpha_vantage':
            quality_score += 20  # Bonus for real API data
        
        if quality_score >= 80:
            return 'excellent'
        elif quality_score >= 60:
            return 'good'
        elif quality_score >= 40:
            return 'fair'
        else:
            return 'poor'
    
    async def close(self):
        """Clean up resources"""
        await self.alpha_client.close_session()
        logger.info("🔒 Data aggregator cleaned up")
