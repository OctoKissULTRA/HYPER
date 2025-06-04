
import os

# ========================================
# HYPER CONFIGURATION (ALPACA DEPLOYMENT)
# ========================================

# ENVIRONMENT SETTINGS
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # development or production
USE_SANDBOX = os.getenv("USE_SANDBOX", "True") == "True"
DATA_SOURCE = os.getenv("DATA_SOURCE", "alpaca")  # should now reflect Alpaca

# API CREDENTIALS
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

# TICKERS TO TRACK
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,
    "SOFT_BUY": 65,
    "HOLD": 35,
    "SELL": 0
}

# ENABLED MODULES
ENABLED_MODULES = {
    "technical_indicators": True,
    "sentiment_analysis": True,
    "vix_analysis": True,
    "market_structure": True,
    "risk_analysis": True,
    "ml_models": True,
}

# DEBUG / VERBOSE LOGGING
VERBOSE = os.getenv("VERBOSE", "False") == "True"
