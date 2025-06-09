#!/usr/bin/env python3
"""
HYPERtrends v4.0 - Optimized Production Startup Script
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup optimized logging"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress noisy third-party logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def check_critical_dependencies():
    """Check only critical dependencies"""
    critical_imports = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("aiohttp", "Async HTTP client"),
        ("requests", "HTTP client")
    ]
    
    print("🔍 Checking critical dependencies...")
    
    missing = []
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Critical dependencies missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All critical dependencies available")
    return True

def check_configuration():
    """Check system configuration"""
    print("\n🔧 Checking configuration...")
    
    try:
        import config
        
        # Check basic configuration
        print(f"📈 Tracking {len(config.TICKERS)} symbols: {', '.join(config.TICKERS)}")
        print(f"🔧 Environment: {config.ENVIRONMENT}")
        print(f"📊 Data source: {config.get_data_source_status()}")
        
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

def print_startup_banner():
    """Print optimized startup banner"""
    banner = """
⚡ ████████████████████████████████████████ ⚡
⚡                                          ⚡
⚡     HYPERtrends v4.0 - OPTIMIZED        ⚡
⚡   AI-Powered Trading Signal Engine      ⚡
⚡         Production Ready Edition         ⚡
⚡                                          ⚡
⚡ ████████████████████████████████████████ ⚡

🚀 Core Features:
📈 Alpaca Markets Live Data Integration
🧠 Advanced Signal Generation Engine
📊 25+ Technical Indicators
💭 Multi-Source Sentiment Analysis
😱 VIX Fear/Greed Analysis
🏗️ Market Structure Analysis
⚠️ Risk Management System
🎯 Real-Time WebSocket Broadcasting

"""
    
    print(banner)
    print(f"    🕐 Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    🌍 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"    🔧 Python: {sys.version.split()[0]}")
    print(f"    📍 Port: {os.getenv('PORT', '8000')}")
    print()

async def test_core_components():
    """Test core system components"""
    print("🧪 Testing core components...")
    
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

def main():
    """Main startup function"""
    print_startup_banner()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting HYPERtrends v4.0 Optimized Production System")
    
    # Check dependencies
    if not check_critical_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_configuration():
        sys.exit(1)
    
    # Test components
    print("\n🧪 Testing core components...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        component_test = loop.run_until_complete(test_core_components())
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

if __name__ == "__main__":
    main()