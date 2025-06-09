import os
import sys
from typing import Dict, List

# ========================================
# HYPERTRENDS v4.0 - OPTIMIZED CONFIG
# ========================================

# Environment Detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG_MODE = ENVIRONMENT == "development"

# Alpaca API Configuration
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
ALPACA_CONFIG = {
    "api_key": os.getenv("APCA_API_KEY_ID", ""),
    "secret_key": os.getenv("APCA_API_SECRET_KEY", ""),
    "base_url": "https://api.alpaca.markets/v2" if TRADING_MODE == "live" else "https://paper-api.alpaca.markets/v2",
    "data_url": "https://data.alpaca.markets/v2",
    "stream_url": "wss://stream.data.alpaca.markets/v2/sip",
    "trading_mode": TRADING_MODE,
}

# List of tracked tickers
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# Confidence thresholds for signal types
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,
    "SOFT_BUY": 65,
    "HOLD": 40,
    "SOFT_SELL": 35,
    "HYPER_SELL": 15,
}

# Signal component weights (must sum to ~1.0)
SIGNAL_WEIGHTS = {
    "technical": 0.30,
    "sentiment": 0.20,
    "momentum": 0.15,
    "ml_prediction": 0.15,
    "vix_sentiment": 0.10,
    "market_structure": 0.08,
    "risk_adjusted": 0.02,
}

# Update intervals (seconds)
UPDATE_INTERVALS = {
    "signal_generation": 30,
    "data_refresh": 15,
    "ml_training": 3600,
    "risk_analysis": 300,
}

# Technical Analysis Parameters
TECHNICAL_PARAMS: Dict = {
    "rsi_period": 14,
    "williams_r_period": 14,
    "stochastic_k_period": 14,
    "stochastic_d_period": 3,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "adx_period": 14,
    "cci_period": 20,
    "volume_ma_period": 20,
    "vwap_period": 20,
    "ema_periods": [9, 21, 50, 200],
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG: Dict = {
    "news_weight": 0.4,
    "social_weight": 0.35,
    "trends_weight": 0.25,
    "sentiment_weights": {
        "news": 0.4,
        "reddit": 0.35,
        "twitter": 0.25,
    },
    "use_vader": True,
    "use_textblob": True,
    "normalize_scores": True,
}

# VIX Analysis Configuration
VIX_CONFIG: Dict = {
    "extreme_fear_threshold": 30,
    "fear_threshold": 20,
    "complacency_threshold": 12,
    "use_sentiment_adjustment": True,
}

# Market Structure Configuration
MARKET_STRUCTURE_CONFIG: Dict = {
    "breadth_very_bullish": 0.9,
    "breadth_bullish": 0.6,
    "breadth_bearish": 0.4,
    "breadth_very_bearish": 0.1,
    "sector_rotation_weights": {
        "Technology": 1.0,
        "Healthcare": 0.8,
        "Financials": 0.9,
        "Consumer Discretionary": 0.8,
        "Communication Services": 0.7,
        "Industrials": 0.7,
        "Consumer Staples": 0.6,
        "Energy": 0.8,
        "Utilities": 0.5,
        "Real Estate": 0.6,
        "Materials": 0.7,
    }
}

# Risk Analysis Configuration
RISK_CONFIG: Dict = {
    "var_confidence_level": 0.05,
    "max_drawdown_warning": 15.0,
    "max_portfolio_risk": 0.02,
    "stop_loss_percent": 0.05,
    "risk_weights": {
        "var": 0.3,
        "volatility": 0.25,
        "correlation": 0.2,
        "drawdown": 0.15,
        "position_size": 0.1,
    }
}

# Machine Learning Configuration
ML_CONFIG: Dict = {
    "model_types": ["random_forest", "xgboost", "neural_network"],
    "feature_selection": True,
    "ensemble_voting": True,
    "retrain_frequency": 24,
    "prediction_horizons": [1, 3, 7],
    "confidence_threshold": 0.6,
}

# Feature Flags (optimized for production)
ENABLED_MODULES = {
    "technical_indicators": True,
    "sentiment_analysis": True,
    "vix_analysis": True,
    "market_structure": True,
    "risk_analysis": True,
    "ml_learning": False,  # Disabled for stability
}

# Security Configuration
SECURITY_CONFIG = {
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "require_https": ENVIRONMENT == "production",
    "rate_limit_enabled": True,
    "max_requests_per_minute": 60,
}

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0" if ENVIRONMENT == "production" else "127.0.0.1",
    "port": int(os.getenv("PORT", 8000)),
    "reload": ENVIRONMENT == "development",
    "workers": 1,  # Single worker for WebSocket compatibility
    "timeout": 120,
    "keepalive": 65,
}

# ========================================
# UTILITY FUNCTIONS
# ========================================

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return ENABLED_MODULES.get(feature_name, False)

def is_development() -> bool:
    """Check if running in development mode"""
    return ENVIRONMENT == "development"

def is_production() -> bool:
    """Check if running in production mode"""
    return ENVIRONMENT == "production"

def has_alpaca_credentials() -> bool:
    """Check if Alpaca credentials are available"""
    return bool(ALPACA_CONFIG.get("api_key") and ALPACA_CONFIG.get("secret_key"))

def get_data_source_status() -> str:
    """Get current data source status"""
    if has_alpaca_credentials():
        mode = ALPACA_CONFIG.get("trading_mode", "paper")
        return f"Alpaca Markets ({mode.capitalize()} Mode)"
    else:
        return "Simulation Mode"

def validate_config() -> bool:
    """Validate configuration"""
    try:
        # Check critical settings
        if not TICKERS:
            raise ValueError("No tickers configured")
        
        # Validate signal weights sum to approximately 1.0
        total_weight = sum(SIGNAL_WEIGHTS.values())
        if not (0.95 <= total_weight <= 1.05):
            raise ValueError(f"Signal weights sum to {total_weight}, should be ~1.0")
        
        # Validate confidence thresholds
        thresholds = list(CONFIDENCE_THRESHOLDS.values())
        if not all(0 <= t <= 100 for t in thresholds):
            raise ValueError("Confidence thresholds must be 0-100")
        
        # Validate port
        port = SERVER_CONFIG.get("port", 8000)
        if not (1000 <= port <= 65535):
            raise ValueError(f"Invalid port: {port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def get_environment_info() -> Dict:
    """Get environment information"""
    return {
        "environment": ENVIRONMENT,
        "debug_mode": DEBUG_MODE,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "alpaca_available": has_alpaca_credentials(),
        "trading_mode": TRADING_MODE,
        "data_source": get_data_source_status(),
        "enabled_features": [name for name, enabled in ENABLED_MODULES.items() if enabled],
        "tracked_symbols": TICKERS,
        "server_config": SERVER_CONFIG
    }

def print_config_summary():
    """Print configuration summary"""
    print("🔧 Configuration Summary:")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Data Source: {get_data_source_status()}")
    print(f"   Tracked Symbols: {', '.join(TICKERS)}")
    print(f"   Enabled Features: {len([f for f in ENABLED_MODULES.values() if f])}/{len(ENABLED_MODULES)}")
    print(f"   Server: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")

# Auto-validate on import
if __name__ != "__main__":
    try:
        if not validate_config():
            print("⚠️ Configuration validation failed - check settings")
    except Exception as e:
        print(f"⚠️ Configuration error: {e}")

# Export key functions and configs
__all__ = [
    'TICKERS', 'CONFIDENCE_THRESHOLDS', 'SIGNAL_WEIGHTS', 'TECHNICAL_PARAMS',
    'SENTIMENT_CONFIG', 'VIX_CONFIG', 'MARKET_STRUCTURE_CONFIG', 'RISK_CONFIG',
    'ENABLED_MODULES', 'ALPACA_CONFIG', 'SERVER_CONFIG', 'SECURITY_CONFIG',
    'is_feature_enabled', 'is_development', 'is_production', 'has_alpaca_credentials',
    'get_data_source_status', 'validate_config', 'get_environment_info'
]
        