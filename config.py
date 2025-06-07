import os
from typing import Dict, List
import logging

# ========================================
# HYPERTRENDS v4.0 - ALPACA CONFIGURATION
# ========================================

# Environment Detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG_MODE = ENVIRONMENT == "development"

# Alpaca API Configuration
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()  # 'paper' or 'live'
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

# Feature Flags for Module Enable/Disable
ENABLED_MODULES = {
    "technical_indicators": True,
    "sentiment_analysis": True,
    "vix_analysis": True,
    "market_structure": True,
    "risk_analysis": True,
    "ml_learning": True,
}

def is_feature_enabled(feature_name: str) -> bool:
    return ENABLED_MODULES.get(feature_name, False)

def is_development():
    return ENVIRONMENT == "development"

def is_production():
    return ENVIRONMENT == "production"

def has_alpaca_credentials():
    return bool(ALPACA_CONFIG.get("api_key") and ALPACA_CONFIG.get("secret_key"))

SECURITY_CONFIG = {
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "require_https": ENVIRONMENT == "production",
    "rate_limit_enabled": True,
    "max_requests_per_minute": 60,
}

def get_data_source_status():
    """
    Returns a human-readable string describing current data source status.
    """
    if has_alpaca_credentials():
        mode = ALPACA_CONFIG.get("trading_mode", "paper")
        return f"Alpaca Markets ({mode.capitalize()} Mode)"
    else:
        return "Simulation Mode"
