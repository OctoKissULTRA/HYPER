import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG_MODE = ENVIRONMENT == "development"

TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
ALPACA_CONFIG = {
    "api_key": os.getenv("APCA_API_KEY_ID", ""),
    "secret_key": os.getenv("APCA_API_SECRET_KEY", ""),
    "base_url": "https://api.alpaca.markets/v2" if TRADING_MODE == "live" else "https://paper-api.alpaca.markets/v2",
    "data_url": "https://data.alpaca.markets/v2",
    "stream_url": "wss://stream.data.alpaca.markets/v2/sip",
    "trading_mode": TRADING_MODE,
}

TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,
    "SOFT_BUY": 65,
    "HOLD": 40,
    "SOFT_SELL": 35,
    "HYPER_SELL": 15,
}

SIGNAL_WEIGHTS = {
    "technical": 0.40,  # Increased due to disabled modules
    "sentiment": 0.30,
    "momentum": 0.20,
    "ml_prediction": 0.10,
    "vix_sentiment": 0.0,
    "market_structure": 0.0,
    "risk_adjusted": 0.0,
}

UPDATE_INTERVALS = {
    "signal_generation": 30,
    "data_refresh": 15,
    "ml_training": 3600,
    "risk_analysis": 300,
}

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

SENTIMENT_CONFIG: Dict = {
    "news_weight": 0.4,
    "social_weight": 0.35,
    "trends_weight": 0.25,
    "sentiment_weights": {
        "news": 0.4,
        "reddit": 0.35,
        "twitter": 0.25,
    },
    "use_vader": False,  # Disabled due to optional import
    "use_textblob": True,
    "normalize_scores": True,
}

VIX_CONFIG: Dict = {
    "extreme_fear_threshold": 30,
    "fear_threshold": 20,
    "complacency_threshold": 12,
    "use_sentiment_adjustment": True,
}

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

ML_CONFIG: Dict = {
    "model_types": ["random_forest", "xgboost", "neural_network"],
    "feature_selection": True,
    "ensemble_voting": True,
    "retrain_frequency": 24,
    "prediction_horizons": [1, 3, 7],
    "confidence_threshold": 0.6,
}

ENABLED_MODULES = {
    "technical_indicators": True,
    "sentiment_analysis": True,
    "vix_analysis": False,  # Disabled due to missing module
    "market_structure": False,  # Disabled due to missing module
    "risk_analysis": False,  # Disabled due to missing module
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
    if has_alpaca_credentials():
        mode = ALPACA_CONFIG.get("trading_mode", "paper")
        return f"Alpaca Markets ({mode.capitalize()} Mode)"
    else:
        return "Simulation Mode"
