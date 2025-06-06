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
ALPACA_CONFIG = {
    "api_key": os.getenv("APCA_API_KEY_ID", "PK2AML2QK9VUI5J1G1BC"),
    "secret_key": os.getenv("APCA_API_SECRET_KEY", ""),
    "base_url": "https://paper-api.alpaca.markets" if os.getenv("USE_SANDBOX", "True").lower() == "true" else "https://api.alpaca.markets",
    "data_url": "https://data.alpaca.markets",
    "stream_url": "wss://stream.data.alpaca.markets",
    "use_sandbox": os.getenv("USE_SANDBOX", "True").lower() == "true"
}

# List of tracked tickers
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,
    "SOFT_BUY": 65,
    "HOLD": 40,
    "SOFT_SELL": 35,
    "HYPER_SELL": 15,
}

# Signal component weights (must sum to ~1.0)
SIGNAL_WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.20,
    "momentum": 0.15,
    "ml_prediction": 0.15,
    "vix_sentiment": 0.10,
    "market_structure": 0.10,
    "risk_adjusted": 0.05,
}

# Update intervals (seconds)
UPDATE_INTERVALS = {
    "signal_generation": 30,
    "data_refresh": 15,
    "ml_training": 3600,  # 1 hour
    "risk_analysis": 300,  # 5 minutes
}

# Enabled module flags
ENABLED_MODULES = {
    "technical_indicators": True,
    "sentiment_analysis": True,
    "vix_analysis": True,
    "market_structure": True,
    "risk_analysis": True,
    "ml_learning": True,
}

# Check if a feature/module is enabled
def is_feature_enabled(feature_name: str) -> bool:
    return ENABLED_MODULES.get(feature_name, False)

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
    "var_confidence_level": 0.05,  # 95% VaR
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

# ML Model Configuration
ML_CONFIG: Dict = {
    "model_types": ["random_forest", "xgboost", "neural_network"],
    "feature_selection": True,
    "ensemble_voting": True,
    "retrain_frequency": 24,  # hours
    "prediction_horizons": [1, 3, 7],  # days
    "confidence_threshold": 0.6,
}

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "reload": DEBUG_MODE,
    "workers": 1,  # WebSocket compatibility
    "timeout": 120,
    "keepalive": 65,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None if ENVIRONMENT == "production" else "logs/hyper.log"
}

# Security Configuration
SECURITY_CONFIG = {
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "require_https": ENVIRONMENT == "production",
    "rate_limit_enabled": True,
    "max_requests_per_minute": 60,
}

# Cache Configuration
CACHE_CONFIG = {
    "redis_url": os.getenv("REDIS_URL"),
    "default_ttl": 300,  # 5 minutes
    "signal_ttl": 30,    # 30 seconds
    "data_ttl": 60,      # 1 minute
}

# Data Quality Thresholds
DATA_QUALITY_CONFIG = {
    "min_volume": 1000,
    "max_spread_bps": 50,
    "max_price_change": 0.15,  # 15% max single-bar change
    "required_history_days": 30,
}

# Feature Flags for Gradual Rollout
FEATURE_FLAGS = {
    "enable_ml_predictions": True,
    "enable_live_trading": False,  # Disabled for safety
    "enable_advanced_risk": True,
    "enable_real_time_news": True,
    "enable_options_analysis": True,
    "enable_crypto_signals": False,
}

# Performance Monitoring
MONITORING_CONFIG = {
    "track_latency": True,
    "track_accuracy": True,
    "alert_on_errors": True,
    "performance_window": 3600,  # 1 hour
}

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check required Alpaca credentials
        if not ALPACA_CONFIG["api_key"]:
            raise ValueError("APCA_API_KEY_ID is required")

        if not ALPACA_CONFIG["secret_key"] and ENVIRONMENT == "production":
            logging.warning("APCA_API_SECRET_KEY not set - using paper trading")
        
        # Validate tickers
        if not TICKERS:
            raise ValueError("No tickers configured")
        
        # Validate weights sum to ~1.0
        weight_sum = sum(SIGNAL_WEIGHTS.values())
        if not (0.95 <= weight_sum <= 1.05):
            logging.warning(f"Signal weights sum to {weight_sum:.2f}, should be ~1.0")
        
        # Check update intervals
        if UPDATE_INTERVALS["signal_generation"] < 10:
            logging.warning("Signal generation interval may be too aggressive")
        
        logging.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False

def get_alpaca_credentials() -> Dict[str, str]:
    """Get Alpaca API credentials"""
    return {
        "api_key": ALPACA_CONFIG["api_key"],
        "secret_key": ALPACA_CONFIG["secret_key"],
        "base_url": ALPACA_CONFIG["base_url"],
        "data_url": ALPACA_CONFIG["data_url"],
    }

def has_alpaca_credentials() -> bool:
    """Check if Alpaca credentials are configured"""
    return bool(ALPACA_CONFIG["api_key"] and (
        ALPACA_CONFIG["secret_key"] or ALPACA_CONFIG["use_sandbox"]
    ))

def get_data_source_status() -> str:
    """Get current data source status"""
    if has_alpaca_credentials():
        env_type = "Paper Trading" if ALPACA_CONFIG["use_sandbox"] else "Live Trading"
        return f"Alpaca Markets ({env_type})"
    else:
        return "Simulation Mode"

def is_production() -> bool:
    """Check if running in production"""
    return ENVIRONMENT == "production"

def is_development() -> bool:
    """Check if running in development"""
    return ENVIRONMENT == "development"

# Initialize logging
if LOGGING_CONFIG.get("file") and not os.path.exists(os.path.dirname(LOGGING_CONFIG["file"])):
    os.makedirs(os.path.dirname(LOGGING_CONFIG["file"]), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    filename=LOGGING_CONFIG.get("file")
)

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print(f"Environment: {ENVIRONMENT}")
    print(f"Data Source: {get_data_source_status()}")
    print(f"Tracking {len(TICKERS)} symbols")
    print(f"ML Models: {len(ML_CONFIG['model_types'])} types")
else:
    validate_config()

print("HYPERtrends v4.0 configuration loaded successfully!")
