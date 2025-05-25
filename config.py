import os
from typing import Dict, List

# ========================================
# ULTRA-ENHANCED HYPER CONFIGURATION
# ========================================

# API CREDENTIALS
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")

# Enhanced API Keys (Optional - system uses fallbacks)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# TARGET TICKERS (The Big 5)
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# ULTRA-ENHANCED SIGNAL THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,     # 85-100% confidence UP (increased from 90)
    "SOFT_BUY": 65,      # 65-84% confidence UP (decreased from 70)
    "HOLD": 35,          # 35-64% confidence (decreased from 50)
    "SOFT_SELL": 65,     # 65-84% confidence DOWN
    "HYPER_SELL": 85,    # 85-100% confidence DOWN
}

# ULTRA-ENHANCED SIGNAL WEIGHTS
SIGNAL_WEIGHTS = {
    "technical": 0.20,          # Traditional + advanced technical indicators
    "ml_prediction": 0.18,      # LSTM + ensemble ML predictions
    "sentiment": 0.15,          # Multi-source sentiment analysis
    "pattern": 0.12,            # Chart pattern recognition
    "market_structure": 0.12,   # Market breadth + sector rotation
    "economic": 0.10,           # Economic indicators
    "momentum": 0.08,           # Price momentum
    "risk_adjusted": 0.05       # Risk metrics (VaR, drawdown)
}

# ADVANCED TECHNICAL INDICATOR SETTINGS
TECHNICAL_PARAMS = {
    # Original indicators
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "ema_short": 9,
    "ema_long": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "volume_ma_period": 20,
    
    # NEW: Advanced indicators
    "williams_r_period": 14,
    "williams_r_oversold": -80,
    "williams_r_overbought": -20,
    "stochastic_k_period": 14,
    "stochastic_d_period": 3,
    "stochastic_oversold": 20,
    "stochastic_overbought": 80,
    "fibonacci_lookback": 50,
    "atr_period": 14,
    "cci_period": 20
}

# VIX SENTIMENT CONFIGURATION
VIX_CONFIG = {
    "extreme_fear_threshold": 30,      # VIX > 30 = Extreme Fear
    "fear_threshold": 20,              # VIX 20-30 = Fear
    "complacency_threshold": 12,       # VIX < 12 = Complacency
    "contrarian_signal": True,         # Use VIX as contrarian indicator
    "weight_adjustment": 0.1           # How much VIX affects other signals
}

# MARKET STRUCTURE ANALYSIS
MARKET_STRUCTURE_CONFIG = {
    "breadth_very_bullish": 0.9,      # 90%+ advancing stocks
    "breadth_bullish": 0.6,           # 60%+ advancing stocks
    "breadth_bearish": 0.4,           # <40% advancing stocks
    "breadth_very_bearish": 0.1,      # <10% advancing stocks
    "sector_rotation_lookback": 5,     # Days to analyze rotation
    "volume_spike_threshold": 2.0,     # 2x normal volume = spike
    "dark_pool_threshold": 0.4         # 40%+ dark pool ratio = institutional
}

# LSTM & ML CONFIGURATION
ML_CONFIG = {
    "lstm_sequence_length": 60,        # 60-day input sequences
    "lstm_prediction_days": [1, 3, 5, 7, 14],  # Forecast horizons
    "ensemble_models": ["RandomForest", "GradientBoost", "SVM", "Linear", "LSTM"],
    "anomaly_contamination": 0.1,      # 10% expected anomalies
    "pattern_confidence_threshold": 0.7,
    "model_retrain_interval": 86400,   # 24 hours
    "feature_importance_top_n": 10
}

# ECONOMIC INDICATORS CONFIGURATION
ECONOMIC_CONFIG = {
    "indicators": {
        "gdp_growth": {"weight": 0.25, "bullish_threshold": 3.0, "bearish_threshold": 1.5},
        "unemployment": {"weight": 0.20, "bullish_threshold": 4.0, "bearish_threshold": 5.5},
        "inflation": {"weight": 0.20, "optimal_min": 2.0, "optimal_max": 3.0, "danger_threshold": 4.0},
        "interest_rate": {"weight": 0.15, "neutral_range": [1.0, 3.0]},
        "retail_sales": {"weight": 0.10, "bullish_threshold": 2.0, "bearish_threshold": -1.0},
        "manufacturing_pmi": {"weight": 0.10, "expansion_threshold": 50, "strong_threshold": 55}
    },
    "update_frequency": 3600,          # Update every hour
    "cache_duration": 14400            # Cache for 4 hours
}

# RISK MANAGEMENT CONFIGURATION
RISK_CONFIG = {
    "var_confidence_level": 0.05,      # 95% VaR
    "max_drawdown_warning": 15.0,      # Warn if >15% drawdown risk
    "correlation_warning": 0.9,        # Warn if correlation >90%
    "volatility_percentile_high": 80,  # High vol = 80th percentile
    "volatility_percentile_low": 20,   # Low vol = 20th percentile
    "stress_test_scenarios": ["market_crash", "sector_rotation", "volatility_spike"]
}

# SENTIMENT ANALYSIS CONFIGURATION
SENTIMENT_CONFIG = {
    "news_sources": ["NewsAPI", "AlphaVantage", "Yahoo"],
    "social_sources": ["Reddit", "Twitter", "StockTwits"],
    "sentiment_lookback_hours": 24,
    "sentiment_weights": {
        "news": 0.4,
        "reddit": 0.35,
        "twitter": 0.25
    },
    "extreme_sentiment_threshold": 80,  # >80 or <20 = extreme
    "sentiment_momentum_periods": [1, 3, 7]  # Days
}

# REAL-TIME UPDATE SETTINGS (Enhanced)
UPDATE_INTERVALS = {
    "market_data": 60,              # Market data every 60 seconds
    "advanced_technical": 120,      # Advanced indicators every 2 minutes
    "sentiment_analysis": 300,      # Sentiment every 5 minutes
    "market_structure": 180,        # Market breadth every 3 minutes
    "economic_data": 3600,          # Economic data every hour
    "ml_predictions": 600,          # ML predictions every 10 minutes
    "signal_generation": 30,        # Generate signals every 30 seconds
    "websocket_ping": 30,           # WebSocket keepalive
    "risk_calculations": 300        # Risk metrics every 5 minutes
}

# API RATE LIMITS (Enhanced)
RATE_LIMITS = {
    "alpha_vantage_calls_
