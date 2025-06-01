#!/usr/bin/env python3
# test_robinhood.py - Test Robinhood Connection
import os
import logging
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import robin_stocks
try:
    import robin_stocks.robinhood as rh
    ROBIN_STOCKS_AVAILABLE = True
    logger.info("✅ robin_stocks imported successfully")
except ImportError:
    ROBIN_STOCKS_AVAILABLE = False
    logger.error("❌ robin_stocks not available - install with: pip install robin_stocks")

def test_robinhood_connection():
    """Test Robinhood connection"""
    
    if not ROBIN_STOCKS_AVAILABLE:
        logger.error("❌ Cannot test Robinhood - robin_stocks not installed")
        return False
    
    # Get credentials from environment
    username = os.getenv("RH_USERNAME")
    password = os.getenv("RH_PASSWORD")
    
    if not username or not password:
        logger.warning("⚠️ No Robinhood credentials found in environment")
        logger.info("ℹ️ Set RH_USERNAME and RH_PASSWORD environment variables")
        return False
    
    logger.info(f"🔑 Found credentials for: {username}")
    
    try:
        logger.info("🔌 Attempting Robinhood login...")
        login_result = rh.login(username=username, password=password)
        
        if login_result:
            logger.info("✅ Robinhood login successful!")
            
            # Test data retrieval
            logger.info("📊 Testing data retrieval...")
            
            # Test getting a quote
            try:
                aapl_quote = rh.stocks.get_latest_price('AAPL')
                if aapl_quote and len(aapl_quote) > 0:
                    price = float(aapl_quote[0])
                    logger.info(f"✅ AAPL price retrieved: ${price:.2f}")
                else:
                    logger.warning("⚠️ AAPL quote returned empty")
                
                # Test getting detailed quote
                detailed_quote = rh.stocks.get_quotes('AAPL')
                if detailed_quote and len(detailed_quote) > 0:
                    quote = detailed_quote[0]
                    volume = quote.get('volume', 'N/A')
                    logger.info(f"✅ AAPL detailed quote retrieved - Volume: {volume}")
                else:
                    logger.warning("⚠️ AAPL detailed quote returned empty")
                
                return True
                
            except Exception as data_error:
                logger.error(f"❌ Data retrieval failed: {data_error}")
                return False
                
        else:
            logger.error("❌ Robinhood login failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Robinhood connection error: {e}")
        return False

async def test_data_sources():
    """Test data sources integration"""
    try:
        # Import our data sources
        from data_sources import HYPERDataAggregator
        
        logger.info("📡 Testing HYPERDataAggregator...")
        aggregator = HYPERDataAggregator()
        
        # Initialize
        await aggregator.initialize()
        
        # Test getting data for AAPL
        logger.info("📊 Testing data retrieval for AAPL...")
        data = await aggregator.get_comprehensive_data('AAPL')
        
        if data and 'quote' in data:
            quote = data['quote']
            price = quote.get('price', 0)
            data_source = quote.get('data_source', 'unknown')
            
            logger.info(f"✅ AAPL data retrieved: ${price:.2f} (source: {data_source})")
            
            if data_source == 'robinhood':
                logger.info("🎉 Real Robinhood data is working!")
            else:
                logger.info("🔄 Using simulation data (this is fine)")
            
            return True
        else:
            logger.error("❌ Failed to get comprehensive data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data sources test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting Robinhood connection tests...")
    logger.info("=" * 50)
    
    # Test 1: Direct Robinhood connection
    logger.info("🔍 Test 1: Direct Robinhood Connection")
    rh_success = test_robinhood_connection()
    
    # Test 2: Data sources integration
    logger.info("\n🔍 Test 2: Data Sources Integration")
    try:
        ds_success = asyncio.run(test_data_sources())
    except Exception as e:
        logger.error(f"❌ Data sources test crashed: {e}")
        ds_success = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 TEST SUMMARY:")
    logger.info(f"   Direct Robinhood: {'✅ PASS' if rh_success else '❌ FAIL'}")
    logger.info(f"   Data Integration: {'✅ PASS' if ds_success else '❌ FAIL'}")
    
    if rh_success and ds_success:
        logger.info("🎉 ALL TESTS PASSED - Robinhood integration is working!")
    elif ds_success:
        logger.info("✅ System will work with simulation data")
    else:
        logger.error("❌ System may have issues - check configuration")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
