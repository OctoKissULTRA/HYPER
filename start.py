#!/usr/bin/env python3
"""
HYPERtrends v4.0 - Production Startup Script
Alpaca Markets Integration Edition
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add current directory to path

sys.path.insert(0, os.path.dirname(os.path.abspath(**file**)))

def setup_logging():
"""Setup production logging"""
log_level = os.getenv("LOG_LEVEL", "INFO")

```
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hypertrends.log") if os.getenv("LOG_TO_FILE") else logging.NullHandler()
    ]
)

# Suppress noisy third-party logs in production
if log_level != "DEBUG":
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
```

def check_dependencies():
"""Check critical dependencies"""
critical_imports = [
("fastapi", "FastAPI web framework"),
("uvicorn", "ASGI server"),
("pandas", "Data processing"),
("numpy", "Numerical computing"),
("aiohttp", "Async HTTP client")
]

```
optional_imports = [
    ("alpaca", "Alpaca Markets SDK"),
    ("textblob", "Sentiment analysis"),
    ("sklearn", "Machine learning")
]

print("🔍 Checking dependencies...")

missing_critical = []
for module, description in critical_imports:
    try:
        __import__(module)
        print(f"✅ {description}")
    except ImportError:
        print(f"❌ {description} - MISSING")
        missing_critical.append(module)

for module, description in optional_imports:
    try:
        __import__(module)
        print(f"✅ {description}")
    except ImportError:
        print(f"⚠️ {description} - Optional (using fallback)")

if missing_critical:
    print(f"\n❌ Critical dependencies missing: {', '.join(missing_critical)}")
    print("Run: pip install -r requirements.txt")
    return False

print("✅ All critical dependencies available")
return True
```

def check_configuration():
"""Check system configuration"""
print("\n🔧 Checking configuration…")

```
try:
    import config
    
    # Check Alpaca credentials
    if config.has_alpaca_credentials():
        print("✅ Alpaca credentials configured")
        print(f"📊 Data source: {config.get_data_source_status()}")
    else:
        print("⚠️ No Alpaca credentials - using simulation mode")
    
    # Check tickers
    print(f"📈 Tracking {len(config.TICKERS)} symbols: {', '.join(config.TICKERS)}")
    
    # Check feature flags
    enabled_features = [name for name, enabled in config.ENABLED_MODULES.items() if enabled]
    print(f"🎯 Enabled features: {', '.join(enabled_features)}")
    
    # Validate configuration
    if config.validate_config():
        print("✅ Configuration validation passed")
        return True
    else:
        print("❌ Configuration validation failed")
        return False
        
except Exception as e:
    print(f"❌ Configuration error: {e}")
    return False
```

def print_startup_banner():
"""Print startup banner"""
banner = """
⚡ ████████████████████████████████████████ ⚡
⚡                                          ⚡
⚡        HYPERtrends v4.0 - ALPACA         ⚡
⚡     AI-Powered Trading Signal Engine     ⚡
⚡                                          ⚡
⚡ ████████████████████████████████████████ ⚡

```
🚀 Production-Grade Features:
📈 Alpaca Markets Live Data Integration
🧠 Advanced Machine Learning Predictions  
📊 25+ Professional Technical Indicators
💭 Multi-Source Sentiment Analysis
😱 VIX Fear/Greed Contrarian Signals
🏗️ Market Structure & Breadth Analysis
⚠️ Advanced Risk Management & Position Sizing
🎯 Real-Time WebSocket Signal Broadcasting

"""

print(banner)
print(f"    🕐 Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"    🌍 Environment: {os.getenv('ENVIRONMENT', 'production')}")
print(f"    🔧 Python: {sys.version.split()[0]}")
print(f"    📍 Port: {os.getenv('PORT', '8000')}")
print()
```

async def test_system_components():
"""Test critical system components"""
print("🧪 Testing system components…")

```
try:
    # Test data aggregator
    from data_sources import HYPERDataAggregator
    
    data_aggregator = HYPERDataAggregator()
    await data_aggregator.initialize()
    
    # Test signal engine
    from signal_engine import HYPERSignalEngine
    
    signal_engine = HYPERSignalEngine()
    
    print("✅ Core components initialized successfully")
    
    # Cleanup
    if hasattr(data_aggregator, 'cleanup'):
        await data_aggregator.cleanup()
    
    return True
    
except Exception as e:
    print(f"❌ Component test failed: {e}")
    return False
```

def main():
"""Main startup function"""
print_startup_banner()

```
# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("🚀 Starting HYPERtrends v4.0 Production System")

# Check dependencies
if not check_dependencies():
    sys.exit(1)

# Check configuration
if not check_configuration():
    sys.exit(1)

# Test components
print("\n🧪 Testing system components...")
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    component_test = loop.run_until_complete(test_system_components())
    loop.close()
    
    if not component_test:
        print("❌ Component testing failed")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Component testing error: {e}")
    sys.exit(1)

print("\n✅ All systems ready - Starting web server...")
print("🌐 Access dashboard at: http://localhost:8000")
print("📊 API documentation: http://localhost:8000/docs")
print("🔧 Health check: http://localhost:8000/health")
print("\n" + "="*60 + "\n")

# Start the main application
try:
    import uvicorn
    from main import app
    
    # Get server configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=65,
        loop="asyncio"
    )
    
except KeyboardInterrupt:
    print("\n👋 Graceful shutdown initiated...")
    logger.info("🛑 Server stopped by user")
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    logger.error(f"Server startup error: {e}")
    sys.exit(1)
```

if **name** == "**main**":
main()
