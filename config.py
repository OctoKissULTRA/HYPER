
import os
from typing import Dict

# ========================================
# CONFIGURATION FOR HYPERtrends v4.0
# ========================================

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
    "technical": 0.2,
    "sentiment": 0.15,
    "momentum": 0.1,
    "ml_prediction": 0.2,
    "vix_sentiment": 0.15,
    "market_structure": 0.1,
    "risk_adjusted": 0.1,
}

# Enabled module flags
ENABLED_MODULES = {
    "enable_advanced_technical": True,
    "enable_sentiment_analysis": True,
    "enable_vix_analysis": True,
    "enable_market_structure": True,
    "enable_risk_metrics": True,
}

# Check if a feature/module is enabled
def is_feature_enabled(feature_name: str) -> bool:
    return ENABLED_MODULES.get(feature_name, False)

# Technical Analysis Params
TECHNICAL_PARAMS: Dict = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "stochastic_k": 14,
    "stochastic_d": 3,
    "ema_periods": [9, 20, 50, 200],
    "atr_period": 14,
    "adx_period": 14
}

# Sentiment Analysis Config
SENTIMENT_CONFIG: Dict = {
    "use_vader": True,
    "use_textblob": True,
    "use_reddit": True,
    "use_twitter": False,
    "normalize_scores": True,
}

# VIX Analysis Config
VIX_CONFIG: Dict = {
    "vix_thresholds": {
        "fear": 25,
        "greed": 15
    },
    "use_sentiment_adjustment": True
}

# Market Structure Config
MARKET_STRUCTURE_CONFIG: Dict = {
    "breadth_thresholds": {
        "bullish": 0.6,
        "bearish": 0.4
    },
    "sector_rotation_weights": {
        "tech": 1.0,
        "healthcare": 0.8,
        "finance": 0.6
    }
}

# Risk Analysis Config
RISK_CONFIG: Dict = {
    "max_position_size_pct": 0.05,
    "use_var_model": True,
    "volatility_window": 10
}
