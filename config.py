import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time
import json
import logging

# ========================================
# HYPER CONFIGURATION v3.1 - ROBINHOOD ENHANCED
# Clean, functional config for maximum performance
# ========================================

# ENVIRONMENT SETTINGS
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# API CREDENTIALS
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")  # Backup only
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ROBINHOOD CONFIGURATION (NEW)
ROBINHOOD_CONFIG = {
    "rate_limit_delay": 2,              # Seconds between requests
    "cache_duration": 30,               # Cache data for 30 seconds
    "timeout": 15,                      # Request timeout
    "retry_attempts": 3,                # Number of retries
    "use_enhanced_features": True,      # Enable sentiment estimation
    "respect_rate_limits": True,        # Be respectful to Robinhood
    "log_requests": DEBUG_MODE          # Log requests in debug mode
}

# DATA SOURCE CONFIGURATION (NEW)
DATA_SOURCE_CONFIG = {
    "primary_source": "robinhood",           # Use Robinhood as primary
    "fallback_enabled": True,                # Keep fallback system
    "cache_enabled": True,                   # Enable caching for performance
    "quality_threshold": "fair",             # Minimum acceptable data quality
    "source_timeout": 15,                    # Timeout for data requests
    "fallback_quality": "enhanced"           # Use enhanced fallback data
}

# TARGET TICKERS
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# TRADING SAFETY
PAPER_TRADING_ONLY = True
MAX_POSITION_SIZE = 10000
RISK_TOLERANCE = "MODERATE"

# SIGNAL CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,     # 85-100% confidence UP
    "SOFT_BUY": 65,      # 65-84% confidence UP  
    "HOLD": 35,          # 35-64% confidence (unclear)
    "SOFT_SELL": 65,     # 65-84% confidence DOWN
    "HYPER_SELL": 85,    # 85-100% confidence DOWN
}

# ENHANCED SIGNAL WEIGHTS
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

# TECHNICAL INDICATOR SETTINGS
TECHNICAL_PARAMS = {
    # Core indicators
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
    "cci_period": 20,
    "adx_period": 14,
    "obv_period": 10
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

# ML CONFIGURATION
ML_CONFIG = {
    "enabled": True,
    "lstm_sequence_length": 60,        # 60-day input sequences
    "lstm_prediction_days": [1, 3, 5, 7, 14],  # Forecast horizons
    "ensemble_models": ["RandomForest", "GradientBoost", "XGBoost", "Linear", "LSTM"],
    "anomaly_contamination": 0.1,      # 10% expected anomalies
    "pattern_confidence_threshold": 0.7,
    "feature_importance_top_n": 10,
    "model_retrain_interval": 86400,   # 24 hours
    "min_training_samples": 100,       # Minimum samples before training
    "cross_validation_folds": 5,
    "early_stopping_patience": 10
}

# RISK MANAGEMENT CONFIGURATION
RISK_CONFIG = {
    "var_confidence_level": 0.05,      # 95% VaR
    "max_drawdown_warning": 15.0,      # Warn if >15% drawdown risk
    "correlation_warning": 0.9,        # Warn if correlation >90%
    "volatility_percentile_high": 80,  # High vol = 80th percentile
    "volatility_percentile_low": 20,   # Low vol = 20th percentile
    "stress_test_scenarios": ["market_crash", "sector_rotation", "volatility_spike"],
    "position_sizing_method": "kelly",  # kelly, fixed, percent_volatility
    "max_portfolio_risk": 0.02,        # 2% max portfolio risk per trade
    "stop_loss_percent": 0.05,         # 5% stop loss
    "take_profit_ratio": 2.0           # 2:1 reward:risk ratio
}

# SENTIMENT ANALYSIS CONFIGURATION
SENTIMENT_CONFIG = {
    "enabled": True,
    "news_sources": ["NewsAPI", "AlphaVantage", "Yahoo"],
    "social_sources": ["Reddit", "Twitter", "StockTwits"],
    "sentiment_lookback_hours": 24,
    "sentiment_weights": {
        "news": 0.4,
        "reddit": 0.35,
        "twitter": 0.25
    },
    "extreme_sentiment_threshold": 80,  # >80 or <20 = extreme
    "sentiment_momentum_periods": [1, 3, 7],  # Days
    "language_models": ["vader", "textblob", "finbert"],
    "sentiment_decay_factor": 0.9,       # How quickly sentiment fades
    # NEW: Robinhood retail sentiment
    "robinhood_sentiment": {
        "enabled": True,
        "weight": 0.3,                   # Weight for retail sentiment
        "popularity_threshold": 50,      # Top 50 = popular
        "sentiment_boost": 1.2           # Boost factor for popular stocks
    }
}

# GOOGLE TRENDS CONFIGURATION
TRENDS_CONFIG = {
    "enabled": True,
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
    "sentiment_multiplier": 1.2,
    "trend_momentum_weight": 0.3,
    # NEW: Enhanced with retail behavior
    "retail_influence": {
        "enabled": True,
        "social_buzz_weight": 0.4,
        "momentum_amplification": 1.5
    }
}

# UPDATE INTERVALS (Optimized for Robinhood)
UPDATE_INTERVALS = {
    "market_data": 30 if ENVIRONMENT == "production" else 60,
    "advanced_technical": 120,      # Advanced indicators every 2 minutes
    "sentiment_analysis": 300,      # Sentiment every 5 minutes
    "market_structure": 180,        # Market breadth every 3 minutes
    "economic_data": 3600,          # Economic data every hour
    "ml_predictions": 600,          # ML predictions every 10 minutes
    "signal_generation": 15 if ENVIRONMENT == "production" else 30,
    "websocket_ping": 30,           # WebSocket keepalive
    "risk_calculations": 300,       # Risk metrics every 5 minutes
    "google_trends": 300,           # Google Trends every 5 minutes
    "vix_analysis": 180,            # VIX analysis every 3 minutes
    "database_cleanup": 86400,      # Daily cleanup
    "model_training": 3600,         # Hourly ML training check
    "performance_monitoring": 60,   # Performance metrics every minute
    # NEW: Robinhood specific
    "robinhood_data": 30,           # Robinhood data every 30 seconds
    "robinhood_sentiment": 300,     # Robinhood sentiment every 5 minutes
    "fallback_check": 600           # Check if API is back every 10 minutes
}

# RATE LIMITS (Robinhood Optimized)
RATE_LIMITS = {
    "robinhood_requests_per_minute": 30,    # Conservative rate limiting
    "alpha_vantage_calls_per_minute": 5,    # Backup only
    "news_api_calls_per_hour": 100,
    "reddit_api_calls_per_minute": 60,
    "twitter_api_calls_per_hour": 300,
    "google_trends_requests_per_hour": 100,
    "websocket_connections_max": 100,
    "api_requests_per_user_per_minute": 60
}

# SERVER CONFIGURATION
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "debug": DEBUG_MODE,
    "reload": DEBUG_MODE and ENVIRONMENT != "production",
    "workers": int(os.getenv("WORKERS", "1")),
    "max_connections": int(os.getenv("MAX_CONNECTIONS", "1000")),
    "keepalive_timeout": int(os.getenv("KEEPALIVE_TIMEOUT", "65")),
    "access_log": ENVIRONMENT == "production",
    "proxy_headers": True,
    "forwarded_allow_ips": "*"
}

# LOGGING CONFIGURATION
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO" if ENVIRONMENT == "production" else "DEBUG"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": f"logs/hyper_{ENVIRONMENT}.log",
    "max_size_mb": 50,
    "backup_count": 30,
    "enable_performance_logging": True,
    "log_signal_details": DEBUG_MODE,
    "log_api_calls": DEBUG_MODE,
    "log_ml_training": True,
    "structured_logging": ENVIRONMENT == "production",
    # NEW: Data source logging
    "log_data_quality": True,
    "log_fallback_usage": True,
    "log_cache_performance": DEBUG_MODE
}

# FEATURE FLAGS
FEATURE_FLAGS = {
    "enable_enhanced_signals": True,
    "enable_advanced_technical": True,
    "enable_williams_r": True,
    "enable_stochastic": True,
    "enable_vix_analysis": True,
    "enable_fibonacci_levels": True,
    "enable_lstm_predictions": ML_CONFIG["enabled"],
    "enable_ensemble_voting": ML_CONFIG["enabled"],
    "enable_sentiment_analysis": SENTIMENT_CONFIG["enabled"],
    "enable_market_structure": True,
    "enable_risk_metrics": True,
    "enable_anomaly_detection": ML_CONFIG["enabled"],
    "enable_pattern_recognition": ML_CONFIG["enabled"],
    "enable_backtesting": True,
    "enable_paper_trading": True,
    "enable_real_trading": False,       # NEVER enable for safety
    "enable_caching": True,
    "enable_rate_limiting": True,
    "enable_circuit_breaker": True,
    # NEW: Robinhood features
    "enable_robinhood_primary": True,
    "enable_retail_sentiment": True,
    "enable_popularity_tracking": True,
    "enable_enhanced_fallback": True
}

# PERFORMANCE THRESHOLDS
PERFORMANCE_THRESHOLDS = {
    "signal_generation_max_time": 5.0,     # 5 seconds max
    "api_response_max_time": 3.0,          # 3 seconds max
    "ml_prediction_max_time": 2.0,         # 2 seconds max
    "total_update_cycle_max_time": 15.0,   # 15 seconds max
    "memory_usage_warning": 512,           # 512MB warning
    "memory_usage_critical": 1024,         # 1GB critical
    "cpu_usage_warning": 70,               # 70% CPU warning
    "cpu_usage_critical": 90,              # 90% CPU critical
    "disk_usage_warning": 80,              # 80% disk warning
    "websocket_response_time": 1.0,        # 1s WebSocket response
    "database_query_time": 0.5,            # 500ms DB query limit
    "cache_hit_ratio_minimum": 0.8,        # 80% cache hit ratio
    # NEW: Data source performance
    "robinhood_response_time": 5.0,        # 5s max for Robinhood
    "fallback_generation_time": 1.0,       # 1s max for fallback
    "data_quality_minimum": "fair"         # Minimum acceptable quality
}

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_database_url() -> str:
    """Get database connection URL"""
    return f"sqlite:///hyper_{ENVIRONMENT}.db"

def is_production() -> bool:
    """Check if running in production"""
    return ENVIRONMENT == "production"

def is_demo_mode() -> bool:
    """Check if running in demo mode"""
    return DEMO_MODE

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

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled"""
    return FEATURE_FLAGS.get(feature_name, False)

def validate_config() -> bool:
    """Enhanced configuration validation"""
    try:
        # Environment validation
        if ENVIRONMENT not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {ENVIRONMENT}")
        
        # Production safety checks
        if is_production():
            if FEATURE_FLAGS["enable_real_trading"]:
                raise ValueError("Real trading must never be enabled")
        
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
        
        # Validate technical parameters
        for param, value in TECHNICAL_PARAMS.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid technical parameter {param}: {value}")
            
            if 'williams_r' in param and ('oversold' in param or 'overbought' in param):
                if not (-100 <= value <= 0):
                    raise ValueError(f"Williams %R {param} must be between -100 and 0: {value}")
            elif 'period' in param and value <= 0:
                raise ValueError(f"Invalid technical parameter {param}: {value}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Configuration validation error: {e}")
        raise

def get_enabled_features() -> List[str]:
    """Get list of enabled features"""
    return [feature for feature, enabled in FEATURE_FLAGS.items() if enabled]

# ========================================
# AUTO-VALIDATE ON IMPORT
# ========================================
try:
    validate_config()
    enabled_features = get_enabled_features()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Enhanced HYPER configuration validated successfully")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"🔧 Demo mode: {DEMO_MODE}")
    print(f"📱 Primary data source: Robinhood")
    print(f"🔥 Enhanced features enabled: {len(enabled_features)}/{len(FEATURE_FLAGS)}")
    print(f"📊 Signal components: {len(SIGNAL_WEIGHTS)} weighted factors")
    print(f"🎯 Tracking: {', '.join(TICKERS)}")
    print(f"⚡ Key features: Williams %R, VIX Analysis, ML Predictions, Retail Sentiment")
    print(f"🛡️ Security: Rate limiting, input validation, CORS protection")
    print(f"📈 Performance: Optimized for Robinhood + enhanced fallback")
    
    if is_production():
        print("🚀 PRODUCTION MODE - Enhanced monitoring and security active")
    elif DEMO_MODE:
        print("🧪 DEMO MODE - Using enhanced fallback data when needed")
    else:
        print("🛠️ DEVELOPMENT MODE - Debug features enabled")
        
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    raise
