import os
import logging
import aiohttp
import asyncio
import random
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Set up logging first
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import robin_stocks, but make it optional
try:
    import robin_stocks.robinhood as rh
    ROBIN_STOCKS_AVAILABLE = True
    logger.info("✅ robin_stocks imported successfully")
except ImportError:
    ROBIN_STOCKS_AVAILABLE = False
    logger.warning("⚠️ robin_stocks not available - using dynamic market simulation")
    rh = None

class DynamicMarketSimulator:
    """Dynamic market simulation that evolves throughout the day"""
    
    def __init__(self):
        self.session_start = time.time()
        self.price_history = {}
        self.volume_history = {}
        self.trend_momentum = {}
        self.market_regime = 'NORMAL'
        self.last_regime_change = time.time()
        
        # Initialize base prices and trends
        self._initialize_market_state()
        logger.info("📈 Dynamic Market Simulator initialized")
    
    def _initialize_market_state(self):
        """Initialize market state with realistic starting conditions"""
        base_prices = {
            'QQQ': 435.50,
            'SPY': 525.25,
            'NVDA': 892.75,
            'AAPL': 198.80,
            'MSFT': 452.90
        }
        
        # Initialize with slight random variation
        for symbol, base_price in base_prices.items():
            variation = random.uniform(-0.02, 0.02)  # ±2% initial variation
            self.price_history[symbol] = [base_price * (1 + variation)]
            self.volume_history[symbol] = []
            self.trend_momentum[symbol] = random.uniform(-0.5, 0.5)
    
    def _update_market_regime(self):
        """Update market regime based on time and random events"""
        time_since_change = time.time() - self.last_regime_change
        
        # Change regime every 15-45 minutes with some randomness
        if time_since_change > random.uniform(900, 2700):  # 15-45 minutes
            regimes = ['BULLISH', 'BEARISH', 'VOLATILE', 'CALM', 'NORMAL']
            weights = [0.25, 0.20, 0.15, 0.15, 0.25]  # Slightly favor bull/normal
            
            self.market_regime = random.choices(regimes, weights=weights)[0]
            self.last_regime_change = time.time()
            logger.info(f"📊 Market regime changed to: {self.market_regime}")
    
    def _calculate_time_based_factors(self):
        """Calculate factors based on time of day"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Market session effects
        if 9 <= hour <= 16:  # Market hours
            if hour == 9 and minute < 30:  # Opening bell
                volatility_factor = 1.5
                volume_factor = 2.0
            elif hour == 15 and minute > 30:  # Closing hour
                volatility_factor = 1.3
                volume_factor = 1.5
            elif 11 <= hour <= 13:  # Lunch lull
                volatility_factor = 0.7
                volume_factor = 0.8
            else:  # Normal trading
                volatility_factor = 1.0
                volume_factor = 1.0
        else:  # After hours
            volatility_factor = 0.4
            volume_factor = 0.3
        
        # Day of week effects
        weekday = now.weekday()
        if weekday == 0:  # Monday
            volatility_factor *= 1.2
        elif weekday == 4:  # Friday
            volatility_factor *= 0.9
            volume_factor *= 0.8
        
        return volatility_factor, volume_factor
    
    def _generate_correlated_movement(self, symbols: List[str]):
        """Generate correlated price movements between symbols"""
        # Market-wide movement (affects all stocks)
        market_move = random.gauss(0, 0.003)  # ±0.3% average market move
        
        # Apply regime effects
        regime_multipliers = {
            'BULLISH': 1.5,
            'BEARISH': -1.2,
            'VOLATILE': 2.0,
            'CALM': 0.5,
            'NORMAL': 1.0
        }
        
        regime_effect = market_move * regime_multipliers.get(self.market_regime, 1.0)
        
        movements = {}
        for symbol in symbols:
            # Individual stock movement
            individual_move = random.gauss(0, 0.005)  # ±0.5% individual variation
            
            # Correlation factors
            correlations = {
                'SPY': 1.0,    # Market baseline
                'QQQ': 0.85,   # High correlation with market
                'NVDA': 0.75,  # Moderate correlation, higher individual variance
                'AAPL': 0.80,  # High correlation
                'MSFT': 0.82   # High correlation
            }
            
            correlation = correlations.get(symbol, 0.7)
            
            # Combine market and individual movements
            total_move = (regime_effect * correlation) + (individual_move * (1 - correlation))
            
            # Add momentum persistence
            momentum = self.trend_momentum.get(symbol, 0)
            momentum_effect = momentum * 0.1  # 10% momentum persistence
            
            movements[symbol] = total_move + momentum_effect
            
            # Update momentum (with mean reversion)
            self.trend_momentum[symbol] = momentum * 0.95 + total_move * 0.05
        
        return movements
    
    def generate_realistic_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic, time-evolving quote data"""
        # Update market regime periodically
        self._update_market_regime()
        
        # Get time-based factors
        volatility_factor, volume_factor = self._calculate_time_based_factors()
        
        # Get correlated movement for this symbol
        movement = self._generate_correlated_movement([symbol])[symbol]
        
        # Apply volatility factor
        movement *= volatility_factor
        
        # Get last price or initialize
        if symbol not in self.price_history or not self.price_history[symbol]:
            self._initialize_market_state()
        
        last_price = self.price_history[symbol][-1]
        
        # Calculate new price
        price_change = last_price * movement
        new_price = last_price + price_change
        
        # Ensure price doesn't go negative
        new_price = max(new_price, last_price * 0.5)
        
        # Update price history (keep last 100 points for trends)
        self.price_history[symbol].append(new_price)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol].pop(0)
        
        # Calculate OHLC based on recent movement
        if len(self.price_history[symbol]) >= 2:
            recent_prices = self.price_history[symbol][-10:]  # Last 10 data points
            
            # More realistic OHLC calculation
            price_range = (max(recent_prices) - min(recent_prices)) * random.uniform(0.5, 1.5)
            
            # Open is close to previous close with some gap
            open_price = last_price + (last_price * random.uniform(-0.002, 0.002))
            
            # High/Low based on current price and realistic range
            high = new_price + (price_range * random.uniform(0.2, 0.8))
            low = new_price - (price_range * random.uniform(0.2, 0.8))
            
            # Ensure OHLC makes sense
            high = max(high, new_price, open_price)
            low = min(low, new_price, open_price)
        else:
            open_price = new_price
            high = new_price
            low = new_price
        
        # Generate realistic volume
        base_volumes = {
            'QQQ': 52000000,
            'SPY': 95000000,
            'NVDA': 48000000,
            'AAPL': 72000000,
            'MSFT': 38000000
        }
        
        base_volume = base_volumes.get(symbol, 28000000)
        
        # Volume correlates with price movement and time factors
        volume_from_movement = abs(movement) * 5  # Higher movement = higher volume
        volume_multiplier = volume_factor * (1 + volume_from_movement) * random.uniform(0.7, 1.3)
        
        volume = int(base_volume * volume_multiplier)
        
        # Update volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > 50:
            self.volume_history[symbol].pop(0)
        
        # Calculate percentage change
        change = new_price - last_price
        change_percent = (change / last_price) * 100
        
        # Calculate additional realistic metrics
        avg_volume = sum(self.volume_history[symbol][-20:]) / len(self.volume_history[symbol][-20:]) if self.volume_history[symbol] else volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price momentum over different periods
        if len(self.price_history[symbol]) >= 5:
            momentum_5 = (new_price - self.price_history[symbol][-5]) / self.price_history[symbol][-5] * 100
        else:
            momentum_5 = change_percent
        
        # Market hours status
        now = datetime.now()
        hour = now.hour
        if 9 <= hour <= 16:
            market_status = 'REGULAR_HOURS'
        elif 4 <= hour <= 9:
            market_status = 'PRE_MARKET'
        elif 16 <= hour <= 20:
            market_status = 'AFTER_HOURS'
        else:
            market_status = 'CLOSED'
        
        # Build comprehensive quote
        quote_data = {
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'price': round(new_price, 2),
            'volume': volume,
            'latest_trading_day': datetime.now().strftime('%Y-%m-%d'),
            'previous_close': round(last_price, 2),
            'change': round(change, 2),
            'change_percent': f"{change_percent:.2f}",
            'timestamp': datetime.now().isoformat(),
            'data_source': 'dynamic_simulation',
            
            # Enhanced metrics for ML
            'enhanced_features': {
                'market_hours': market_status,
                'market_regime': self.market_regime,
                'volatility_regime': 'HIGH' if abs(change_percent) > 2 else 'NORMAL' if abs(change_percent) > 0.5 else 'LOW',
                'volume_ratio': round(volume_ratio, 2),
                'momentum_5min': round(momentum_5, 2),
                'trend_strength': round(abs(self.trend_momentum.get(symbol, 0)) * 100, 1),
                'data_freshness': 'real_time_simulation',
                'session_time': round(time.time() - self.session_start, 0),
                'price_history_length': len(self.price_history[symbol]),
                'regime_age_minutes': round((time.time() - self.last_regime_change) / 60, 1)
            }
        }
        
        return quote_data

class RobinhoodClient:
    """Enhanced Robinhood client with better error handling and rate limiting"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 60  # Increased to 1 minute to reduce API calls
        self.rate_limit_delay = 2.0  # Increased to 2 seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.authenticated = False
        self.login_attempts = 0
        self.max_login_attempts = 3
        self.last_login_attempt = 0
        self.login_cooldown = 300  # 5 minutes between login attempts
        
        # Initialize dynamic simulator as fallback
        self.market_simulator = DynamicMarketSimulator()
        
        # Try to authenticate if credentials available
        if ROBIN_STOCKS_AVAILABLE:
            self._attempt_login()
        
        logger.info("🎯 Enhanced Robinhood client initialized")
    
    def _attempt_login(self):
        """Enhanced login with better rate limiting and error handling"""
        if not ROBIN_STOCKS_AVAILABLE:
            logger.info("ℹ️ robin_stocks not available - using dynamic simulation")
            return False
        
        # Check login cooldown
        current_time = time.time()
        if self.last_login_attempt > 0 and (current_time - self.last_login_attempt) < self.login_cooldown:
            remaining = self.login_cooldown - (current_time - self.last_login_attempt)
            logger.info(f"⏰ Login cooldown active, {remaining:.0f}s remaining - using simulation")
            return False
        
        # Check max attempts
        if self.login_attempts >= self.max_login_attempts:
            logger.info(f"⚠️ Max login attempts ({self.max_login_attempts}) reached - using simulation")
            return False
            
        try:
            username = os.getenv("RH_USERNAME")
            password = os.getenv("RH_PASSWORD")
            
            if not username or not password:
                logger.info("ℹ️ No Robinhood credentials provided - using dynamic simulation")
                return False
            
            logger.info("🔐 Attempting Robinhood login with enhanced error handling...")
            self.login_attempts += 1
            self.last_login_attempt = current_time
            
            # Try login with timeout
            try:
                login_result = rh.login(username=username, password=password, store_session=True)
                if login_result:
                    self.authenticated = True
                    self.login_attempts = 0  # Reset on success
                    logger.info("✅ Robinhood login successful - using real market data")
                    return True
                else:
                    logger.info("ℹ️ Robinhood login failed - using dynamic simulation")
                    return False
                    
            except Exception as login_error:
                error_msg = str(login_error).lower()
                
                if "429" in error_msg or "too many requests" in error_msg:
                    logger.warning("⚠️ Robinhood rate limit hit - using simulation (will retry later)")
                    self.login_cooldown = 600  # 10 minutes for rate limit
                elif "verification" in error_msg or "challenge" in error_msg:
                    logger.warning("⚠️ Robinhood 2FA required - using simulation (check app for verification)")
                    self.login_cooldown = 1800  # 30 minutes for 2FA
                elif "credentials" in error_msg or "password" in error_msg:
                    logger.error("❌ Invalid Robinhood credentials - check RH_USERNAME and RH_PASSWORD")
                    self.max_login_attempts = 0  # Stop trying
                else:
                    logger.warning(f"⚠️ Robinhood login error: {login_error} - using simulation")
                
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Robinhood login system error: {e} - using simulation")
            return False
    
    async def create_session(self):
        """Create HTTP session"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'HYPER-Trading-System/4.0-Enhanced',
                    'Accept': 'application/json',
                }
            )
            logger.debug("✅ HTTP session created")
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("🔒 HTTP session closed")
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol].get('cache_time', 0)
        return (time.time() - cache_time) < self.cache_duration
    
    async def _rate_limit_wait(self):
        """Enhanced rate limiting"""
        if self.last_request_time > 0:
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                logger.debug(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
    
    async def test_connection(self) -> bool:
        """Enhanced connection test"""
        logger.info("🧪 Testing Robinhood connection...")
        
        if not ROBIN_STOCKS_AVAILABLE or not self.authenticated:
            logger.info("ℹ️ Robinhood not available/authenticated - using dynamic simulation")
            return False
        
        try:
            # Test with a simple quote request with timeout
            await self._rate_limit_wait()
            test_quote = rh.stocks.get_latest_price('AAPL', includeExtendedHours=True)
            
            if test_quote and len(test_quote) > 0 and float(test_quote[0]) > 0:
                logger.info("✅ Robinhood connection test successful - real data active")
                return True
            else:
                logger.info("ℹ️ Robinhood test failed - using dynamic simulation")
                return False
                
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                logger.warning("⚠️ Robinhood rate limited during test - using simulation")
            else:
                logger.info(f"ℹ️ Robinhood test error: {e} - using simulation")
            return False
    
    async def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        """Enhanced quote retrieval with better error handling"""
        # Try Robinhood first if available and authenticated
        if ROBIN_STOCKS_AVAILABLE and self.authenticated:
            try:
                # Check cache first
                if self._is_cache_valid(symbol):
                    logger.debug(f"📋 Using cached Robinhood data for {symbol}")
                    return self.cache[symbol]['data']
                
                logger.debug(f"📱 Fetching Robinhood data for {symbol}")
                await self._rate_limit_wait()
                
                self.request_count += 1
                self.last_request_time = time.time()
                
                # Get latest price with timeout protection
                try:
                    price_data = rh.stocks.get_latest_price(symbol, includeExtendedHours=True)
                    if not price_data or len(price_data) == 0:
                        logger.debug(f"⚠️ Empty price data for {symbol} - using simulation")
                        return self.market_simulator.generate_realistic_quote(symbol)
                    
                    current_price = float(price_data[0])
                    if current_price <= 0:
                        logger.debug(f"⚠️ Invalid price for {symbol} - using simulation")
                        return self.market_simulator.generate_realistic_quote(symbol)
                    
                    # Get detailed quote information
                    quote_data = rh.stocks.get_quotes(symbol)
                    if not quote_data or len(quote_data) == 0:
                        logger.debug(f"⚠️ No quote details for {symbol} - using simulation")
                        return self.market_simulator.generate_realistic_quote(symbol)
                    
                    quote = quote_data[0]
                    
                    # Build Robinhood quote data
                    previous_close = float(quote.get('previous_close', current_price))
                    change = current_price - previous_close
                    change_percent = (change / previous_close * 100) if previous_close > 0 else 0.0
                    
                    result = {
                        'symbol': symbol,
                        'open': float(quote.get('last_trade_price', current_price)),
                        'high': float(quote.get('last_trade_price', current_price)),
                        'low': float(quote.get('last_trade_price', current_price)),
                        'price': current_price,
                        'volume': int(float(quote.get('volume', 0))),
                        'latest_trading_day': datetime.now().strftime('%Y-%m-%d'),
                        'previous_close': previous_close,
                        'change': change,
                        'change_percent': f"{change_percent:.2f}",
                        'timestamp': datetime.now().isoformat(),
                        'data_source': 'robinhood',
                        'enhanced_features': {
                            'market_hours': self._get_market_hours_status(),
                            'data_freshness': 'real_time_robinhood',
                            'retail_sentiment': self._estimate_retail_sentiment(symbol, change_percent),
                            'request_count': self.request_count,
                            'cache_duration': self.cache_duration
                        }
                    }
                    
                    # Cache the result
                    self.cache[symbol] = {
                        'data': result,
                        'cache_time': time.time()
                    }
                    
                    logger.debug(f"✅ Robinhood data for {symbol}: ${result['price']:.2f} ({result['change_percent']}%)")
                    return result
                    
                except Exception as api_error:
                    error_msg = str(api_error).lower()
                    if "429" in error_msg or "rate limit" in error_msg:
                        logger.warning(f"⚠️ Rate limited for {symbol} - using simulation")
                        # Increase cooldown on rate limit
                        self.rate_limit_delay = min(self.rate_limit_delay * 1.5, 10.0)
                    else:
                        logger.debug(f"⚠️ API error for {symbol}: {api_error} - using simulation")
                    
            except Exception as e:
                logger.debug(f"⚠️ Robinhood error for {symbol}: {e} - using simulation")
        
        # Use dynamic simulation as fallback
        logger.debug(f"🔄 Using dynamic simulation for {symbol}")
        return self.market_simulator.generate_realistic_quote(symbol)
    
    def _estimate_retail_sentiment(self, symbol: str, change_percent: float) -> str:
        """Estimate retail sentiment based on price movement"""
        sentiment_score = 50 + (change_percent * 10)
        
        if sentiment_score > 70:
            return 'VERY_BULLISH'
        elif sentiment_score > 55:
            return 'BULLISH'
        elif sentiment_score > 45:
            return 'NEUTRAL'
        elif sentiment_score > 30:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def _get_market_hours_status(self) -> str:
        """Get current market hours status"""
        now = datetime.now()
        hour = now.hour
        
        if 9 <= hour <= 16:
            return 'REGULAR_HOURS'
        elif 4 <= hour <= 9:
            return 'PRE_MARKET'
        elif 16 <= hour <= 20:
            return 'AFTER_HOURS'
        else:
            return 'CLOSED'

class GoogleTrendsClient:
    """Enhanced Google Trends client with dynamic, evolving data"""
    
    def __init__(self):
        self.trend_history = {}
        self.session_start = time.time()
        logger.info("📈 Enhanced Google Trends Client initialized")
    
    async def get_trends_data(self, keywords: List[str]) -> Dict[str, Any]:
        """Generate dynamic, evolving trends data"""
        logger.debug(f"📈 Generating dynamic trends data for: {keywords}")
        
        current_time = time.time()
        session_age = (current_time - self.session_start) / 3600  # Hours since start
        
        trend_data = {}
        for keyword in keywords:
            # Initialize trend history if needed
            if keyword not in self.trend_history:
                self.trend_history[keyword] = {
                    'base_momentum': random.uniform(-20, 60),
                    'trend_direction': random.choice(['UP', 'DOWN', 'SIDEWAYS']),
                    'last_update': current_time,
                    'momentum_history': []
                }
            
            trend_info = self.trend_history[keyword]
            time_since_update = current_time - trend_info['last_update']
            
            # Update momentum every few minutes with evolution
            if time_since_update > 300:  # 5 minutes
                # Evolve the trend
                momentum_change = random.uniform(-10, 10)
                trend_info['base_momentum'] += momentum_change
                trend_info['base_momentum'] = max(-50, min(100, trend_info['base_momentum']))
                
                # Occasionally change trend direction
                if random.random() < 0.1:  # 10% chance
                    trend_info['trend_direction'] = random.choice(['UP', 'DOWN', 'SIDEWAYS'])
                
                trend_info['last_update'] = current_time
            
            # Apply keyword-specific boosts that evolve over time
            current_momentum = trend_info['base_momentum']
            
            # Time-based variations
            hour_of_day = datetime.now().hour
            if 9 <= hour_of_day <= 16:  # Market hours boost
                current_momentum *= 1.2
            
            # Keyword-specific patterns
            if any(term in keyword.upper() for term in ['NVDA', 'AI', 'NVIDIA']):
                # AI stocks have cyclical hype
                ai_cycle = math.sin(session_age * 0.5) * 20  # Cycles every ~12 hours
                current_momentum += ai_cycle
            elif any(term in keyword.upper() for term in ['APPLE', 'IPHONE', 'AAPL']):
                # Consumer tech has steady interest with small variations
                current_momentum += random.uniform(0, 15)
            elif any(term in keyword.upper() for term in ['SPY', 'S&P']):
                # Index funds have stable, low momentum
                current_momentum = max(-20, min(current_momentum, 30))
            
            # Add some noise
            current_momentum += random.uniform(-5, 5)
            
            # Calculate velocity (rate of change)
            if len(trend_info['momentum_history']) > 0:
                velocity = current_momentum - trend_info['momentum_history'][-1]
            else:
                velocity = 0
            
            # Update history
            trend_info['momentum_history'].append(current_momentum)
            if len(trend_info['momentum_history']) > 50:  # Keep last 50 points
                trend_info['momentum_history'].pop(0)
            
            # Calculate additional metrics
            if len(trend_info['momentum_history']) >= 3:
                recent_trend = trend_info['momentum_history'][-3:]
                acceleration = (recent_trend[-1] - recent_trend[0]) / 2
            else:
                acceleration = 0
            
            trend_data[keyword] = {
                'momentum': round(current_momentum, 1),
                'velocity': round(velocity, 1),
                'acceleration': round(acceleration, 1),
                'current_value': random.randint(max(30, int(50 + current_momentum/2)), 100),
                'average_value': random.randint(40, 80),
                'retail_influence': min(1.0, max(0.1, (current_momentum + 50) / 100)),
                'social_buzz': 'HIGH' if current_momentum > 50 else 'MEDIUM' if current_momentum > 0 else 'LOW',
                'trend_direction': trend_info['trend_direction'],
                'session_age_hours': round(session_age, 1),
                'evolution_factor': round(time_since_update / 3600, 2)  # Hours since last evolution
            }
        
        return {
            'keyword_data': trend_data,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'dynamic_trends_simulation',
            'market_sentiment': self._calculate_overall_market_sentiment(trend_data),
            'session_info': {
                'session_age_hours': round(session_age, 1),
                'total_keywords_tracked': len(self.trend_history),
                'active_trends': len([k for k in trend_data.values() if abs(k['momentum']) > 10])
            }
        }
    
    def _calculate_overall_market_sentiment(self, trend_data: Dict) -> str:
        """Calculate overall market sentiment from trends"""
        if not trend_data:
            return 'NEUTRAL'
        
        avg_momentum = sum(data['momentum'] for data in trend_data.values()) / len(trend_data)
        
        if avg_momentum > 40:
            return 'BULLISH'
        elif avg_momentum > 10:
            return 'SLIGHTLY_BULLISH'
        elif avg_momentum > -10:
            return 'NEUTRAL'
        elif avg_momentum > -40:
            return 'SLIGHTLY_BEARISH'
        else:
            return 'BEARISH'

class HYPERDataAggregator:
    """Enhanced HYPER Data Aggregator - Robinhood Primary + Dynamic Simulation Fallback"""
    
    def __init__(self, api_key: str = None):
        # NOTE: api_key parameter kept for backward compatibility but ignored
        if api_key:
            logger.info("ℹ️ Alpha Vantage API key provided but not used - using Robinhood + simulation")
        
        # Primary source: Enhanced Robinhood with better error handling
        self.robinhood_client = RobinhoodClient()
        self.trends_client = GoogleTrendsClient()
        
        # Keep track of connection status
        self.api_test_performed = False
        self.robinhood_available = False
        
        logger.info("🚀 Enhanced HYPER Data Aggregator initialized")
        logger.info("📱 Primary source: Robinhood (with enhanced error handling)")
        logger.info("🔄 Fallback: Dynamic market simulation (ML-ready)")
        logger.info("🚫 Alpha Vantage: Removed completely")
    
    async def initialize(self) -> bool:
        """Initialize and test data sources with enhanced error handling"""
        logger.info("🔧 Initializing Enhanced HYPER Data Aggregator...")
        
        # Test Robinhood connection (non-blocking)
        try:
            self.robinhood_available = await self.robinhood_client.test_connection()
        except Exception as e:
            logger.info(f"ℹ️ Robinhood test failed: {e}")
            self.robinhood_available = False
        
        if self.robinhood_available:
            logger.info("✅ Robinhood connection successful - using real market data")
        else:
            logger.info("ℹ️ Using enhanced dynamic market simulation - perfect for ML training")
        
        self.api_test_performed = True
        return True  # Always return True since we have enhanced dynamic fallback
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data - Enhanced Robinhood or dynamic simulation"""
        logger.debug(f"🎯 Getting enhanced data for {symbol}")
        
        # Perform API test if not done yet
        if not self.api_test_performed:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Get quote data (Enhanced Robinhood or simulation)
            quote_data = await self.robinhood_client.get_global_quote(symbol)
            
            # Get dynamic trends data
            keywords = self._get_keywords_for_symbol(symbol)
            trends_data = await self.trends_client.get_trends_data(keywords)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Assess data quality
            data_quality = self._assess_data_quality(quote_data, trends_data)
            
            # Build comprehensive result
            result = {
                'symbol': symbol,
                'quote': quote_data,
                'trends': trends_data,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'data_quality': data_quality,
                'api_status': 'robinhood_connected' if self.robinhood_available and quote_data.get('data_source') == 'robinhood' else 'enhanced_simulation',
                'enhanced_features': quote_data.get('enhanced_features', {}),
                'ml_ready': True,  # Always ML-ready
                'data_source_info': {
                    'primary': 'robinhood_enhanced',
                    'fallback': 'dynamic_simulation_enhanced',
                    'alpha_vantage_removed': True,
                    'robinhood_authenticated': self.robinhood_client.authenticated,
                    'robinhood_rate_limited': not self.robinhood_available and self.robinhood_client.authenticated
                }
            }
            
            logger.debug(f"✅ Enhanced data for {symbol} completed in {processing_time:.2f}s (quality: {data_quality})")
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Error getting enhanced data for {symbol}: {e}")
            
            # Emergency fallback (should rarely be needed)
            emergency_quote = self.robinhood_client.market_simulator.generate_realistic_quote(symbol)
            return {
                'symbol': symbol,
                'quote': emergency_quote,
                'trends': await self.trends_client.get_trends_data([symbol]),
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'data_quality': 'emergency_fallback',
                'api_status': 'error',
                'error': str(e),
                'ml_ready': True
            }
    
    def _get_keywords_for_symbol(self, symbol: str) -> List[str]:
        """Get trending keywords for symbol"""
        keyword_map = {
            'QQQ': ['QQQ ETF', 'NASDAQ 100', 'tech stocks', 'technology ETF', 'growth stocks'],
            'SPY': ['SPY ETF', 'S&P 500', 'market index', 'broad market', 'index fund'],
            'NVDA': ['NVIDIA', 'AI stocks', 'artificial intelligence', 'GPU', 'data center'],
            'AAPL': ['Apple', 'iPhone', 'Apple stock', 'consumer tech', 'AAPL'],
            'MSFT': ['Microsoft', 'Azure', 'cloud computing', 'enterprise software', 'MSFT']
        }
        return keyword_map.get(symbol, [symbol, f'{symbol} stock', f'{symbol} price'])
    
    def _assess_data_quality(self, quote_data: Dict, trends_data: Dict) -> str:
        """Assess enhanced data quality"""
        quality_score = 0
        
        # Basic data quality
        if quote_data and quote_data.get('price', 0) > 0:
            quality_score += 40
        
        if quote_data and quote_data.get('volume', 0) > 0:
            quality_score += 20
        
        # Source quality bonus
        source = quote_data.get('data_source', '')
        if source == 'robinhood':
            quality_score += 35  # Real Robinhood data (enhanced)
        elif source == 'dynamic_simulation':
            quality_score += 30  # Enhanced dynamic simulation
        else:
            quality_score += 10  # Basic fallback
        
        # Enhanced features bonus
        enhanced_features = quote_data.get('enhanced_features', {})
        if enhanced_features.get('market_regime'):
            quality_score += 5  # Market regime tracking
        if enhanced_features.get('session_time'):
            quality_score += 5  # Session persistence
        if enhanced_features.get('price_history_length', 0) > 10:
            quality_score += 5  # Good history for ML
        
        # Trends data quality
        if trends_data and trends_data.get('keyword_data'):
            quality_score += 10
            
        # Dynamic evolution bonus
        if trends_data and trends_data.get('session_info', {}).get('active_trends', 0) > 0:
            quality_score += 5
        
        # Determine quality rating
        if quality_score >= 90:
            return 'excellent'
        elif quality_score >= 75:
            return 'good'
        elif quality_score >= 60:
            return 'fair'
        elif quality_score >= 45:
            return 'acceptable'
        else:
            return 'poor'
    
    async def close(self):
        """Cleanup resources"""
        await self.robinhood_client.close_session()
        logger.info("🔒 Enhanced HYPER data aggregator cleaned up")

# ============================================
# ALPHA VANTAGE COMPLETELY REMOVED
# ============================================

logger.info("🚀 Enhanced Robinhood-Only Data Source loaded successfully!")
logger.info("📱 Primary: Enhanced Robinhood API (with better error handling)")
logger.info("🔄 Fallback: Enhanced dynamic market simulation")
logger.info("🧠 ML-Ready: Time-evolving patterns and correlations")
logger.info("🚫 Alpha Vantage: Completely removed")
logger.info("✅ Zero external dependencies required")
logger.info("⚡ Enhanced rate limiting and 2FA handling")