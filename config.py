import os
from typing import Dict, List

# ========================================
# COMBINED ENHANCED HYPER CONFIGURATION
# ========================================

# API CREDENTIALS
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")

# Optional Enhanced API Keys (system will use fallbacks if not provided)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# TARGET TICKERS (The Big 5)
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# ENHANCED SIGNAL CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,     # 85-100% confidence UP
    "SOFT_BUY": 65,      # 65-84% confidence UP  
    "HOLD": 35,          # 35-64% confidence (unclear)
    "SOFT_SELL": 65,     # 65-84% confidence DOWN
    "HYPER_SELL": 85,    # 85-100% confidence DOWN
}

# COMBINED ENHANCED SIGNAL WEIGHTS
SIGNAL_WEIGHTS = {
    "technical": 0.25,          # Technical analysis + enhanced indicators
    "sentiment": 0.20,          # Multi-source sentiment analysis
    "momentum": 0.15,           # Price momentum
    "ml_prediction": 0.15,      # ML predictions + patterns
    "market_structure": 0.10,   # Market breadth + sector rotation
    "vix_sentiment": 0.08,      # VIX fear/greed
    "economic": 0.05,           # Economic indicators
    "risk_adjusted": 0.02       # Risk penalty
}

# ENHANCED TECHNICAL INDICATOR SETTINGS
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
    
    # Enhanced indicators
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

# ML & PATTERN RECOGNITION CONFIGURATION
ML_CONFIG = {
    "lstm_sequence_length": 60,        # 60-day input sequences
    "lstm_prediction_days": [1, 3, 5, 7, 14],  # Forecast horizons
    "ensemble_models": ["RandomForest", "GradientBoost", "SVM", "Linear", "LSTM"],
    "anomaly_contamination": 0.1,      # 10% expected anomalies
    "pattern_confidence_threshold": 0.7,
    "feature_importance_top_n": 10,
    "model_retrain_interval": 86400    # 24 hours
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

# GOOGLE TRENDS CONFIGURATION (Enhanced)
TRENDS_CONFIG = {
    "timeframe": "now 7-d",
    "geo": "US",
    "keywords": {
        "QQQ": ["QQQ ETF", "NASDAQ 100", "tech stocks", "technology sector"],
        "SPY": ["SPY ETF", "S&P 500", "market index", "broad market"],
        "NVDA": ["NVIDIA", "AI stocks", "graphics cards", "semiconductor"],
        "AAPL": ["Apple", "iPhone", "Apple stock", "consumer tech"],
        "MSFT": ["Microsoft", "Azure", "cloud computing", "enterprise software"],
    },
    "related_queries": True,
    "sentiment_multiplier": 1.2
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
    "risk_calculations": 300,       # Risk metrics every 5 minutes
    "google_trends": 300,           # Google Trends every 5 minutes
    "vix_analysis": 180             # VIX analysis every 3 minutes
}

# API RATE LIMITS (Enhanced)
RATE_LIMITS = {
    "alpha_vantage_calls_per_minute": 5,
    "alpha_vantage_calls_per_day": 500,
    "news_api_calls_per_hour": 100,
    "reddit_api_calls_per_minute": 60,
    "twitter_api_calls_per_hour": 300,
    "vix_data_calls_per_hour": 100,
    "economic_data_calls_per_hour": 50,
    "ml_model_calls_per_minute": 30,
    "google_trends_requests_per_hour": 100
}

# SERVER CONFIGURATION
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "reload": os.getenv("RELOAD", "false").lower() == "true",
    "workers": 1,
    "max_connections": 1000
}

# LOGGING CONFIGURATION (Enhanced)
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "hyper_combined.log",
    "max_size_mb": 20,
    "backup_count": 10,
    "enable_performance_logging": True,
    "log_signal_details": True
}

# FEATURE FLAGS (For easy enable/disable of enhanced features)
FEATURE_FLAGS = {
    "enable_enhanced_signals": True,
    "enable_advanced_technical": True,
    "enable_williams_r": True,
    "enable_stochastic": True,
    "enable_vix_analysis": True,
    "enable_fibonacci_levels": True,
    "enable_lstm_predictions": True,
    "enable_ensemble_voting": True,
    "enable_sentiment_analysis": True,
    "enable_market_structure": True,
    "enable_economic_indicators": True,
    "enable_risk_metrics": True,
    "enable_anomaly_detection": True,
    "enable_pattern_recognition": True
}

# PERFORMANCE THRESHOLDS
PERFORMANCE_THRESHOLDS = {
    "signal_generation_max_time": 10.0,    # Max 10 seconds per signal
    "api_response_max_time": 5.0,          # Max 5 seconds per API call
    "ml_prediction_max_time": 3.0,         # Max 3 seconds for ML
    "total_update_cycle_max_time": 30.0,   # Max 30 seconds for full cycle
    "memory_usage_warning": 500,           # Warn at 500MB memory
    "cpu_usage_warning": 80                # Warn at 80% CPU
}

# ========================================
# HELPER FUNCTIONS (Enhanced)
# ========================================

def get_signal_threshold(signal_type: str) -> int:
    """Get confidence threshold for signal type"""
    return CONFIDENCE_THRESHOLDS.get(signal_type, 35)

def get_ticker_keywords(ticker: str) -> List[str]:
    """Get Google Trends keywords for ticker"""
    return TRENDS_CONFIG["keywords"].get(ticker, [ticker])

def is_high_confidence_signal(confidence: float, direction: str) -> str:
    """Determine signal type based on confidence and direction"""
    if direction.upper() == "UP":
        if confidence >= CONFIDENCE_THRESHOLDS["HYPER_BUY"]:
            return "HYPER_BUY"
        elif confidence >= CONFIDENCE_THRESHOLDS["SOFT_BUY"]:
            return "SOFT_BUY"
    elif direction.upper() == "DOWN":
        if confidence >= CONFIDENCE_THRESHOLDS["HYPER_SELL"]:
            return "HYPER_SELL"
        elif confidence >= CONFIDENCE_THRESHOLDS["SOFT_SELL"]:
            return "SOFT_SELL"
    
    return "HOLD"

def get_enhanced_config() -> Dict:
    """Get complete enhanced configuration"""
    return {
        "api_keys": {
            "alpha_vantage": ALPHA_VANTAGE_API_KEY,
            "news_api": NEWS_API_KEY,
            "reddit_client_id": REDDIT_CLIENT_ID,
            "reddit_secret": REDDIT_SECRET,
            "twitter_bearer": TWITTER_BEARER_TOKEN
        },
        "signal_weights": SIGNAL_WEIGHTS,
        "technical_params": TECHNICAL_PARAMS,
        "vix_config": VIX_CONFIG,
        "ml_config": ML_CONFIG,
        "economic_config": ECONOMIC_CONFIG,
        "risk_config": RISK_CONFIG,
        "sentiment_config": SENTIMENT_CONFIG,
        "feature_flags": FEATURE_FLAGS,
        "rate_limits": RATE_LIMITS,
        "update_intervals": UPDATE_INTERVALS,
        "performance_thresholds": PERFORMANCE_THRESHOLDS
    }

def validate_config() -> bool:
    """Validate enhanced configuration settings - FIXED VERSION"""
    try:
        # Check API key exists
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("Alpha Vantage API key not configured")
        
        # Check tickers are defined
        if not TICKERS:
            raise ValueError("No tickers configured")
        
        # Check signal weights sum to approximately 1.0
        total_weight = sum(SIGNAL_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.02:
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight}")
        
        # Validate confidence thresholds
        for signal_type, threshold in CONFIDENCE_THRESHOLDS.items():
            if not (0 <= threshold <= 100):
                raise ValueError(f"Invalid threshold for {signal_type}: {threshold}")
        
        # Validate feature flags
        if not isinstance(FEATURE_FLAGS, dict):
            raise ValueError("Feature flags must be a dictionary")
        
        # Validate technical parameters
        for param, value in TECHNICAL_PARAMS.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid technical parameter {param}: {value}")
            
            # FIXED: Special handling for Williams %R - only check oversold/overbought values
            if 'williams_r' in param and ('oversold' in param or 'overbought' in param):
                if not (-100 <= value <= 0):
                    raise ValueError(f"Williams %R {param} must be between -100 and 0: {value}")
            # All other parameters (including williams_r_period) should be positive
            elif 'williams_r' not in param or 'period' in param:
                if value <= 0:
                    raise ValueError(f"Invalid technical parameter {param}: {value}")
        
        # Validate VIX configuration
        if VIX_CONFIG["extreme_fear_threshold"] <= VIX_CONFIG["fear_threshold"]:
            raise ValueError("VIX extreme fear threshold must be higher than fear threshold")
        
        # Validate update intervals
        for interval_name, interval_value in UPDATE_INTERVALS.items():
            if not isinstance(interval_value, (int, float)) or interval_value <= 0:
                raise ValueError(f"Invalid update interval {interval_name}: {interval_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise

def get_enabled_features() -> List[str]:
    """Get list of enabled enhanced features"""
    return [feature for feature, enabled in FEATURE_FLAGS.items() if enabled]

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific enhanced feature is enabled"""
    return FEATURE_FLAGS.get(feature_name, False)

def get_vix_sentiment_thresholds() -> Dict[str, float]:
    """Get VIX sentiment interpretation thresholds"""
    return {
        "extreme_fear": VIX_CONFIG["extreme_fear_threshold"],
        "fear": VIX_CONFIG["fear_threshold"],
        "complacency": VIX_CONFIG["complacency_threshold"]
    }

def get_technical_indicator_params(indicator: str) -> Dict:
    """Get parameters for specific technical indicator"""
    indicator_params = {
        "williams_r": {
            "period": TECHNICAL_PARAMS["williams_r_period"],
            "oversold": TECHNICAL_PARAMS["williams_r_oversold"],
            "overbought": TECHNICAL_PARAMS["williams_r_overbought"]
        },
        "stochastic": {
            "k_period": TECHNICAL_PARAMS["stochastic_k_period"],
            "d_period": TECHNICAL_PARAMS["stochastic_d_period"],
            "oversold": TECHNICAL_PARAMS["stochastic_oversold"],
            "overbought": TECHNICAL_PARAMS["stochastic_overbought"]
        },
        "rsi": {
            "period": TECHNICAL_PARAMS["rsi_period"],
            "oversold": TECHNICAL_PARAMS["rsi_oversold"],
            "overbought": TECHNICAL_PARAMS["rsi_overbought"]
        }
    }
    return indicator_params.get(indicator, {})

def get_market_structure_thresholds() -> Dict[str, float]:
    """Get market structure analysis thresholds"""
    return {
        "very_bullish": MARKET_STRUCTURE_CONFIG["breadth_very_bullish"],
        "bullish": MARKET_STRUCTURE_CONFIG["breadth_bullish"],
        "bearish": MARKET_STRUCTURE_CONFIG["breadth_bearish"],
        "very_bearish": MARKET_STRUCTURE_CONFIG["breadth_very_bearish"]
    }

def get_risk_management_params() -> Dict:
    """Get risk management configuration"""
    return {
        "var_confidence": RISK_CONFIG["var_confidence_level"],
        "max_drawdown_warning": RISK_CONFIG["max_drawdown_warning"],
        "correlation_warning": RISK_CONFIG["correlation_warning"],
        "stress_scenarios": RISK_CONFIG["stress_test_scenarios"]
    }

# ========================================
# AUTO-VALIDATE ON IMPORT
# ========================================
try:
    validate_config()
    enabled_features = get_enabled_features()
    print("✅ Combined Enhanced HYPER configuration validated successfully")
    print(f"🔥 Enhanced features enabled: {len(enabled_features)}/{len(FEATURE_FLAGS)}")
    print(f"📊 Signal components: {len(SIGNAL_WEIGHTS)} weighted factors")
    print(f"🎯 Tracking: {', '.join(TICKERS)}")
    print(f"⚡ Key features: Williams %R, VIX Analysis, ML Predictions, Fibonacci Levels")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    raise
