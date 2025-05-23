import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import asyncio
import aiohttp
from pytrends.request import TrendReq
import time
import logging

from config import config

logger = logging.getLogger(__name__)

# ========================================
# ALPHA VANTAGE CLIENT
# ========================================

class AlphaVantageClient:
    """Professional Alpha Vantage API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request_time = 0
        self.request_count = 0
        
    async def _make_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make rate-limited API request"""
        # Rate limiting: 5 calls per minute
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 12:  # 12 seconds between calls (5 per minute)
            wait_time = 12 - time_since_last
            await asyncio.sleep(wait_time)
        
        params['apikey'] = self.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    self.last_request_time = time.time()
                    self.request_count += 1
                    
                    # Check for errors
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage error: {data['Error Message']}")
                        return None
                    elif 'Note' in data:
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        return None
                    
                    return data
                    
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None
    
    async def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for symbol"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = await self._make_request(params)
        if not data or 'Global Quote' not in data:
            return None
        
        quote = data['Global Quote']
        
        try:
            return {
                'symbol': quote['01. symbol'],
                'price': float(quote['05. price']),
                'change': float(quote['09. change']),
                'change_percent': float(quote['10. change percent'].replace('%', '')),
                'volume': int(quote['06. volume']),
                'timestamp': quote['07. latest trading day'],
                'open': float(quote['02. open']),
                'high': float(quote['03. high']),
                'low': float(quote['04. low']),
                'previous_close': float(quote['08. previous close'])
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing quote data for {symbol}: {e}")
            return None
    
    async def get_intraday_data(self, symbol: str, interval: str = '5min') -> Optional[pd.DataFrame]:
        """Get intraday OHLCV data"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'compact'
        }
        
        data = await self._make_request(params)
        if not data or f'Time Series ({interval})' not in data:
            return None
        
        try:
            time_series = data[f'Time Series ({interval})']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            
            # Add derived columns
            df['price_change'] = df['close'].pct_change() * 100
            df['range_percent'] = (df['high'] - df['low']) / df['close'] * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing intraday data for {symbol}: {e}")
            return None
    
    async def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> Optional[pd.DataFrame]:
        """Get pre-calculated technical indicator"""
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': kwargs.get('interval', 'daily'),
            'time_period': kwargs.get('time_period', 14),
            'series_type': kwargs.get('series_type', 'close')
        }
        
        data = await self._make_request(params)
        if not data:
            return None
        
        # Find the technical analysis key
        tech_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                tech_key = key
                break
        
        if not tech_key:
            return None
        
        try:
            df = pd.DataFrame.from_dict(data[tech_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            return df.sort_index()
        except Exception as e:
            logger.error(f"Error processing {indicator} for {symbol}: {e}")
            return None

# ========================================
# GOOGLE TRENDS CLIENT
# ========================================

class GoogleTrendsClient:
    """Google Trends data client"""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.last_request_time = 0
        
    async def get_trends_data(self, keywords: List[str], timeframe: str = "now 7-d") -> Optional[pd.DataFrame]:
        """Get Google Trends data for keywords"""
        try:
            # Rate limiting: Don't overwhelm Google
            current_time = time.time()
            if current_time - self.last_request_time < 5:
                await asyncio.sleep(5)
            
            # Build payload
            self.pytrends.build_payload(
                keywords, 
                cat=0, 
                timeframe=timeframe, 
                geo='US', 
                gprop=''
            )
            
            # Get interest over time
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df.empty:
                return None
            
            # Remove the 'isPartial' column if it exists
            if 'isPartial' in interest_df.columns:
                interest_df = interest_df.drop(columns=['isPartial'])
            
            # Resample to hourly data and interpolate
            interest_df = interest_df.resample('1H').mean().interpolate()
            
            self.last_request_time = time.time()
            
            return interest_df
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends data: {e}")
            return None
    
    async def analyze_trend_momentum(self, trends_df: pd.DataFrame, keyword: str) -> Dict[str, float]:
        """Analyze trend momentum for a keyword"""
        if trends_df is None or keyword not in trends_df.columns:
            return {"momentum": 0.0, "velocity": 0.0, "acceleration": 0.0}
        
        try:
            series = trends_df[keyword].dropna()
            if len(series) < 3:
                return {"momentum": 0.0, "velocity": 0.0, "acceleration": 0.0}
            
            # Calculate momentum indicators
            current_value = series.iloc[-1]
            previous_value = series.iloc[-2] if len(series) > 1 else current_value
            avg_value = series.mean()
            
            # Momentum: current vs average
            momentum = (current_value / avg_value - 1) * 100 if avg_value > 0 else 0
            
            # Velocity: rate of change
            velocity = (current_value - previous_value) / previous_value * 100 if previous_value > 0 else 0
            
            # Acceleration: change in velocity
            if len(series) >= 3:
                prev_velocity = (previous_value - series.iloc[-3]) / series.iloc[-3] * 100 if series.iloc[-3] > 0 else 0
                acceleration = velocity - prev_velocity
            else:
                acceleration = 0
            
            return {
                "momentum": momentum,
                "velocity": velocity, 
                "acceleration": acceleration,
                "current_value": current_value,
                "average_value": avg_value
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend momentum: {e}")
            return {"momentum": 0.0, "velocity": 0.0, "acceleration": 0.0}

# ========================================
# DATA AGGREGATOR
# ========================================

class HYPERDataAggregator:
    """Combines data from multiple sources"""
    
    def __init__(self):
        self.alpha_client = AlphaVantageClient(config.ALPHA_VANTAGE_API_KEY)
        self.trends_client = GoogleTrendsClient()
        self.cache = {}
        self.cache_expiry = {}
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data for a symbol"""
        cache_key = f"{symbol}_comprehensive"
        
        # Check cache (5-minute expiry)
        if (cache_key in self.cache and 
            self.cache_expiry.get(cache_key, 0) > time.time() - 300):
            return self.cache[cache_key]
        
        try:
            # Get market data
            quote_task = self.alpha_client.get_real_time_quote(symbol)
            intraday_task = self.alpha_client.get_intraday_data(symbol, '5min')
            
            # Get trends data
            keywords = config.get_ticker_keywords(symbol)
            trends_task = self.trends_client.get_trends_data(keywords)
            
            # Execute concurrently
            quote, intraday, trends = await asyncio.gather(
                quote_task, intraday_task, trends_task,
                return_exceptions=True
            )
            
            # Process trends analysis
            trend_analysis = {}
            if isinstance(trends, pd.DataFrame) and not trends.empty:
                for keyword in keywords:
                    if keyword in trends.columns:
                        analysis = await self.trends_client.analyze_trend_momentum(trends, keyword)
                        trend_analysis[keyword] = analysis
            
            # Combine all data
            comprehensive_data = {
                'symbol': symbol,
                'quote': quote if not isinstance(quote, Exception) else None,
                'intraday': intraday if not isinstance(intraday, Exception) else None,
                'trends': trends if not isinstance(trends, Exception) else None,
                'trend_analysis': trend_analysis,
                'timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(quote, intraday, trends)
            }
            
            # Cache the result
            self.cache[cache_key] = comprehensive_data
            self.cache_expiry[cache_key] = time.time()
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'quote': None,
                'intraday': None, 
                'trends': None,
                'trend_analysis': {},
                'timestamp': datetime.now().isoformat(),
                'data_quality': 'poor'
            }
    
    def _assess_data_quality(self, quote, intraday, trends) -> str:
        """Assess overall data quality"""
        quality_score = 0
        
        if quote is not None:
            quality_score += 1
        if intraday is not None and len(intraday) > 10:
            quality_score += 1
        if trends is not None and not trends.empty:
            quality_score += 1
        
        if quality_score >= 3:
            return 'excellent'
        elif quality_score >= 2:
            return 'good'
        elif quality_score >= 1:
            return 'fair'
        else:
            return 'poor'
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overview data for all tickers"""
        tasks = [self.get_comprehensive_data(ticker) for ticker in config.TICKERS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        overview = {}
        for i, ticker in enumerate(config.TICKERS):
            if not isinstance(results[i], Exception):
                overview[ticker] = results[i]
            else:
                logger.error(f"Error getting data for {ticker}: {results[i]}")
                overview[ticker] = None
        
        return {
            'tickers': overview,
            'timestamp': datetime.now().isoformat(),
            'market_status': 'open'  # TODO: Add market hours detection
        }
