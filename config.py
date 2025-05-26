import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time
import json
import logging

# ========================================
# PRODUCTION HYPER CONFIGURATION v3.0
# ========================================

# ENVIRONMENT SETTINGS
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# API CREDENTIALS - Personal Use
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")
if not ALPHA_VANTAGE_API_KEY:
    logging.warning("⚠️ No Alpha Vantage API key - running in demo mode")
    DEMO_MODE = True

# Optional Enhanced API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")

# DATABASE CONFIGURATION
DATABASE_CONFIG = {
    "type": os.getenv("DB_TYPE", "sqlite"),  # sqlite, postgresql
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "hyper_trading.db"),
    "username": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "ssl_mode": os.getenv("DB_SSL_MODE", "prefer")
}

# REDIS CONFIGURATION (for caching and rate limiting)
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "db": int(os.getenv("REDIS_DB", "0")),
    "ssl": os.getenv("REDIS_SSL", "false").lower() == "true"
}

# TARGET TICKERS
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

# TRADING SAFETY
PAPER_TRADING_ONLY = True  # Never change this - system is for signals only
MAX_POSITION_SIZE = 10000  # Maximum theoretical position size for risk calculations
RISK_TOLERANCE = "MODERATE"  # CONSERVATIVE, MODERATE, AGGRESSIVE

# ENHANCED SIGNAL CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,     # 85-100% confidence UP
    "SOFT_BUY": 65,      # 65-84% confidence UP  
    "HOLD": 35,          # 35-64% confidence (unclear)
    "SOFT_SELL": 65,     # 65-84% confidence DOWN
    "HYPER_SELL": 85,    # 85-100% confidence DOWN
}

# SIGNAL WEIGHTS (Enhanced for Production)
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

# ML & PATTERN RECOGNITION CONFIGURATION
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

# ECONOMIC INDICATORS CONFIGURATION
ECONOMIC_CONFIG = {
    "enabled": True,
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
    "sentiment_decay_factor": 0.9       # How quickly sentiment fades
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
    "trend_momentum_weight": 0.3
}

# ENHANCED UPDATE INTERVALS (Production Optimized)
UPDATE_INTERVALS = {
    "market_data": 30 if ENVIRONMENT == "production" else 60,              # More frequent in prod
    "advanced_technical": 120,      # Advanced indicators every 2 minutes
    "sentiment_analysis": 300,      # Sentiment every 5 minutes
    "market_structure": 180,        # Market breadth every 3 minutes
    "economic_data": 3600,          # Economic data every hour
    "ml_predictions": 600,          # ML predictions every 10 minutes
    "signal_generation": 15 if ENVIRONMENT == "production" else 30,        # Faster signals in prod
    "websocket_ping": 30,           # WebSocket keepalive
    "risk_calculations": 300,       # Risk metrics every 5 minutes
    "google_trends": 300,           # Google Trends every 5 minutes
    "vix_analysis": 180,            # VIX analysis every 3 minutes
    "database_cleanup": 86400,      # Daily cleanup
    "model_training": 3600,         # Hourly ML training check
    "performance_monitoring": 60    # Performance metrics every minute
}

# ENHANCED RATE LIMITS (Production)
RATE_LIMITS = {
    "alpha_vantage_calls_per_minute": 5,
    "alpha_vantage_calls_per_day": 500,
    "news_api_calls_per_hour": 100,
    "reddit_api_calls_per_minute": 60,
    "twitter_api_calls_per_hour": 300,
    "vix_data_calls_per_hour": 100,
    "economic_data_calls_per_hour": 50,
    "ml_model_calls_per_minute": 30,
    "google_trends_requests_per_hour": 100,
    "websocket_connections_max": 100,
    "api_requests_per_user_per_minute": 60
}

# SERVER CONFIGURATION (Production Ready)
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

# ENHANCED LOGGING CONFIGURATION
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
    "structured_logging": ENVIRONMENT == "production"
}

# MONITORING & ALERTING
MONITORING_CONFIG = {
    "enabled": ENVIRONMENT == "production",
    "metrics_endpoint": "/metrics",
    "health_check_interval": 30,
    "alert_email": os.getenv("ALERT_EMAIL", ""),
    "slack_webhook": os.getenv("SLACK_WEBHOOK", ""),
    "error_threshold": 10,              # Alert after 10 errors
    "latency_threshold": 5.0,           # Alert if response > 5s
    "memory_threshold": 1024,           # Alert at 1GB memory usage
    "disk_threshold": 85                # Alert at 85% disk usage
}

# FEATURE FLAGS (Production Safe)
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
    "enable_economic_indicators": ECONOMIC_CONFIG["enabled"],
    "enable_risk_metrics": True,
    "enable_anomaly_detection": ML_CONFIG["enabled"],
    "enable_pattern_recognition": ML_CONFIG["enabled"],
    "enable_backtesting": True,
    "enable_paper_trading": True,
    "enable_real_trading": False,       # NEVER enable for safety
    "enable_caching": True,
    "enable_rate_limiting": True,
    "enable_circuit_breaker": True
}

# PERFORMANCE THRESHOLDS (Production)
PERFORMANCE_THRESHOLDS = {
    "signal_generation_max_time": 5.0,     # Reduced for production
    "api_response_max_time": 3.0,          # Stricter timeout
    "ml_prediction_max_time": 2.0,         # Faster ML inference
    "total_update_cycle_max_time": 15.0,   # Tighter cycle time
    "memory_usage_warning": 512,           # 512MB warning
    "memory_usage_critical": 1024,         # 1GB critical
    "cpu_usage_warning": 70,               # 70% CPU warning
    "cpu_usage_critical": 90,              # 90% CPU critical
    "disk_usage_warning": 80,              # 80% disk warning
    "websocket_response_time": 1.0,        # 1s WebSocket response
    "database_query_time": 0.5,            # 500ms DB query limit
    "cache_hit_ratio_minimum": 0.8         # 80% cache hit ratio
}

# SECURITY CONFIGURATION
SECURITY_CONFIG = {
    "api_key_rotation_days": 90,
    "session_timeout_minutes": 60,
    "max_failed_attempts": 5,
    "lockout_duration_minutes": 15,
    "require_https": ENVIRONMENT == "production",
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "rate_limit_enabled": True,
    "input_validation": True,
    "sql_injection_protection": True,
    "xss_protection": True
}

# ========================================
# ENHANCED HELPER FUNCTIONS
# ========================================

def get_database_url() -> str:
    """Get database connection URL"""
    if DATABASE_CONFIG["type"] == "sqlite":
        return f"sqlite:///{DATABASE_CONFIG['database']}"
    elif DATABASE_CONFIG["type"] == "postgresql":
        return (f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
                f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
    else:
        raise ValueError(f"Unsupported database type: {DATABASE_CONFIG['type']}")

def get_redis_url() -> str:
    """Get Redis connection URL"""
    protocol = "rediss" if REDIS_CONFIG["ssl"] else "redis"
    auth = f":{REDIS_CONFIG['password']}@" if REDIS_CONFIG["password"] else ""
    return f"{protocol}://{auth}{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}"

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

def get_enhanced_config() -> Dict:
    """Get complete enhanced configuration"""
    return {
        "environment": ENVIRONMENT,
        "demo_mode": DEMO_MODE,
        "api_keys_configured": {
            "alpha_vantage": bool(ALPHA_VANTAGE_API_KEY),
            "news_api": bool(NEWS_API_KEY),
            "reddit": bool(REDDIT_CLIENT_ID and REDDIT_SECRET),
            "twitter": bool(TWITTER_BEARER_TOKEN)
        },
        "database": DATABASE_CONFIG,
        "redis": REDIS_CONFIG,
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
        "performance_thresholds": PERFORMANCE_THRESHOLDS,
        "security_config": SECURITY_CONFIG,
        "monitoring": MONITORING_CONFIG
    }

def validate_config() -> bool:
    """Enhanced configuration validation"""
    try:
        # Environment validation
        if ENVIRONMENT not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {ENVIRONMENT}")
        
        # Production safety checks
        if is_production():
            if not ALPHA_VANTAGE_API_KEY:
                raise ValueError("Alpha Vantage API key required in production")
            if FEATURE_FLAGS["enable_real_trading"]:
                raise ValueError("Real trading must never be enabled")
            if not MONITORING_CONFIG["enabled"]:
                raise ValueError("Monitoring must be enabled in production")
        
        # Check API key exists (unless demo mode)
        if not DEMO_MODE and not ALPHA_VANTAGE_API_KEY:
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
        
        # Validate technical parameters
        for param, value in TECHNICAL_PARAMS.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid technical parameter {param}: {value}")
            
            # Williams %R special handling
            if 'williams_r' in param and ('oversold' in param or 'overbought' in param):
                if not (-100 <= value <= 0):
                    raise ValueError(f"Williams %R {param} must be between -100 and 0: {value}")
            elif 'period' in param or param in ['rsi_oversold', 'rsi_overbought', 'stochastic_oversold', 'stochastic_overbought']:
                if value <= 0:
                    raise ValueError(f"Invalid technical parameter {param}: {value}")
        
        # Validate VIX configuration
        if VIX_CONFIG["extreme_fear_threshold"] <= VIX_CONFIG["fear_threshold"]:
            raise ValueError("VIX extreme fear threshold must be higher than fear threshold")
        
        # Validate update intervals
        for interval_name, interval_value in UPDATE_INTERVALS.items():
            if not isinstance(interval_value, (int, float)) or interval_value <= 0:
                raise ValueError(f"Invalid update interval {interval_name}: {interval_value}")
        
        # Validate database configuration
        if DATABASE_CONFIG["type"] not in ["sqlite", "postgresql"]:
            raise ValueError(f"Unsupported database type: {DATABASE_CONFIG['type']}")
        
        # Validate ML configuration
        if ML_CONFIG["enabled"]:
            if ML_CONFIG["min_training_samples"] < 50:
                raise ValueError("Minimum training samples should be at least 50")
            if not (0 < ML_CONFIG["anomaly_contamination"] < 0.5):
                raise ValueError("Anomaly contamination must be between 0 and 0.5")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Configuration validation error: {e}")
        raise

def get_enabled_features() -> List[str]:
    """Get list of enabled enhanced features"""
    return [feature for feature, enabled in FEATURE_FLAGS.items() if enabled]

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific enhanced feature is enabled"""
    return FEATURE_FLAGS.get(feature_name, False)

# ========================================
# AUTO-VALIDATE ON IMPORT
# ========================================
try:
    validate_config()
    enabled_features = get_enabled_features()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Production HYPER configuration validated successfully")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"🔧 Demo mode: {DEMO_MODE}")
    print(f"🔥 Enhanced features enabled: {len(enabled_features)}/{len(FEATURE_FLAGS)}")
    print(f"📊 Signal components: {len(SIGNAL_WEIGHTS)} weighted factors")
    print(f"🎯 Tracking: {', '.join(TICKERS)}")
    print(f"⚡ Key features: Williams %R, VIX Analysis, ML Predictions, Risk Management")
    print(f"🛡️ Security: Rate limiting, input validation, CORS protection")
    print(f"📈 Performance: Optimized intervals, caching, monitoring")
    
    if is_production():
        print("🚀 PRODUCTION MODE - Enhanced monitoring and security active")
    elif DEMO_MODE:
        print("🧪 DEMO MODE - Using fallback data sources")
    else:
        print("🛠️ DEVELOPMENT MODE - Debug features enabled")
        
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    raise
