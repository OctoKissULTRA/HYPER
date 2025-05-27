import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time
import json
import logging

# ========================================
# ENHANCED HYPER CONFIGURATION - MAXIMUM DATA EDITION
# Professional-grade multi-source data configuration
# ========================================

# ENVIRONMENT SETTINGS
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ========================================
# ENHANCED API CREDENTIALS - MULTI-SOURCE
# ========================================

# Primary Data Sources
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Robinhood (Retail Sentiment)
ROBINHOOD_CONFIG = {
    "username": os.getenv("ROBINHOOD_USERNAME", ""),
    "password": os.getenv("ROBINHOOD_PASSWORD", ""),
    "enable_login": os.getenv("ROBINHOOD_LOGIN", "false").lower() == "true"
}

# Social Media & News APIs
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
    "client_secret": os.getenv("REDDIT_SECRET", ""),
    "user_agent": "HYPER:1.0.0 (by /u/hypertrader)"
}
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Economic Data APIs
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")

# Alternative Data
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "")
SEEKING_ALPHA_API_KEY = os.getenv("SEEKING_ALPHA_API_KEY", "")

# ========================================
# DATA SOURCE CONFIGURATION
# ========================================

DATA_SOURCES_CONFIG = {
    "primary_sources": {
        "robinhood": {
            "enabled": True,
            "priority": 1,
            "features": ["quotes", "sentiment", "popularity", "options_flow"],
            "rate_limit": 100,  # requests per minute
            "timeout": 15,
            "retry_attempts": 3
        },
        "alpha_vantage": {
            "enabled": bool(ALPHA_VANTAGE_API_KEY),
            "priority": 2,
            "features": ["quotes", "fundamentals", "technical"],
            "rate_limit": 5,
            "timeout": 20,
            "retry_attempts": 2
        },
        "finnhub": {
            "enabled": bool(FINNHUB_API_KEY),
            "priority": 3,
            "features": ["quotes", "news", "sentiment", "fundamentals"],
            "rate_limit": 60,
            "timeout": 10,
            "retry_attempts": 3
        },
        "polygon": {
            "enabled": bool(POLYGON_API_KEY),
            "priority": 4,
            "features": ["quotes", "options", "crypto", "forex"],
            "rate_limit": 200,
            "timeout": 10,
            "retry_attempts": 2
        },
        "yfinance": {
            "enabled": True,
            "priority": 5,
            "features": ["quotes", "fundamentals", "historical"],
            "rate_limit": 2000,
            "timeout": 15,
            "retry_attempts": 3
        }
    },
    "news_sources": {
        "newsapi": {
            "enabled": bool(NEWS_API_KEY),
            "sentiment_weight": 0.25,
            "rate_limit": 1000
        },
        "finnhub_news": {
            "enabled": bool(FINNHUB_API_KEY),
            "sentiment_weight": 0.25,
            "rate_limit": 60
        },
        "reddit": {
            "enabled": bool(REDDIT_CONFIG["client_id"]),
            "sentiment_weight": 0.30,
            "subreddits": ["stocks", "investing", "SecurityAnalysis", "ValueInvesting"],
            "rate_limit": 60
        },
        "twitter": {
            "enabled": bool(TWITTER_BEARER_TOKEN),
            "sentiment_weight": 0.20,
            "rate_limit": 300
        }
    },
    "fallback_chain": ["robinhood", "alpha_vantage", "yfinance", "finnhub", "polygon"],
    "data_quality_thresholds": {
        "minimum_sources": 2,
        "price_deviation_limit": 0.01,  # 1% max deviation between sources
        "volume_threshold": 1000,
        "staleness_limit": 300  # 5 minutes max age
    }
}

# ========================================
# ENHANCED ML CONFIGURATION
# ========================================

ENHANCED_ML_CONFIG = {
    "enabled": True,
    "models": {
        "ensemble_voting": {
            "enabled": True,
            "models": ["random_forest", "xgboost", "lightgbm", "neural_network"],
            "voting_strategy": "soft",  # soft or hard voting
            "weight_optimization": True
        },
        "deep_learning": {
            "enabled": True,
            "lstm_enabled": True,
            "sequence_length": 60,
            "prediction_horizons": [1, 3, 5, 7, 14],
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.2,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 15
        },
        "traditional_ml": {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "feature_importance_threshold": 0.01
            },
            "xgboost": {
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "lightgbm": {
                "n_estimators": 150,
                "max_depth": 10,
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8
            }
        }
    },
    "feature_engineering": {
        "technical_indicators": 25,  # Number of technical indicators
        "sentiment_features": 15,    # Sentiment-derived features
        "fundamental_features": 10,  # Fundamental analysis features
        "macro_features": 8,         # Macroeconomic features
        "alternative_features": 12,  # Alternative data features
        "lag_features": [1, 2, 3, 5, 7, 14],  # Lag periods
        "rolling_windows": [5, 10, 20, 50],   # Rolling window sizes
        "volatility_features": True,
        "momentum_features": True,
        "mean_reversion_features": True
    },
    "training": {
        "update_frequency": 3600,  # Retrain every hour
        "validation_split": 0.2,
        "test_split": 0.1,
        "cross_validation_folds": 5,
        "hyperparameter_optimization": True,
        "feature_selection": True,
        "ensemble_optimization": True
    },
    "performance_monitoring": {
        "accuracy_threshold": 0.65,
        "precision_threshold": 0.60,
        "recall_threshold": 0.60,
        "f1_threshold": 0.60,
        "sharpe_threshold": 1.0,
        "max_drawdown_threshold": 0.15
    }
}

# ========================================
# ENHANCED TECHNICAL ANALYSIS
# ========================================

ENHANCED_TECHNICAL_CONFIG = {
    "indicators": {
        # Trend Indicators
        "moving_averages": {
            "sma": [5, 10, 20, 50, 100, 200],
            "ema": [9, 12, 20, 26, 50],
            "wma": [10, 20],
            "hull_ma": [9, 16, 21]
        },
        "trend_lines": {
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "adx": {"period": 14, "threshold": 25},
            "aroon": {"period": 14},
            "parabolic_sar": {"acceleration": 0.02, "maximum": 0.2}
        },
        
        # Momentum Indicators
        "momentum": {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "stochastic": {
                "k_period": 14, "d_period": 3,
                "oversold": 20, "overbought": 80
            },
            "williams_r": {
                "period": 14, "oversold": -80, "overbought": -20
            },
            "cci": {"period": 20, "oversold": -100, "overbought": 100},
            "roc": {"period": 12},
            "momentum": {"period": 10}
        },
        
        # Volatility Indicators
        "volatility": {
            "bollinger_bands": {"period": 20, "std": 2},
            "atr": {"period": 14},
            "keltner_channels": {"period": 20, "multiplier": 2},
            "donchian_channels": {"period": 20},
            "vix_correlation": True
        },
        
        # Volume Indicators
        "volume": {
            "obv": True,
            "volume_ma": {"period": 20},
            "money_flow": {"period": 14},
            "volume_profile": True,
            "accumulation_distribution": True,
            "chaikin_oscillator": {"fast": 3, "slow": 10}
        },
        
        # Support/Resistance
        "levels": {
            "fibonacci_retracements": True,
            "pivot_points": True,
            "support_resistance": {"lookback": 50},
            "key_levels": True
        }
    },
    "pattern_recognition": {
        "enabled": True,
        "patterns": [
            "head_and_shoulders", "inverse_head_and_shoulders",
            "double_top", "double_bottom",
            "triangles", "flags", "pennants",
            "cup_and_handle", "wedges"
        ],
        "confidence_threshold": 0.7
    }
}

# ========================================
# ENHANCED SENTIMENT ANALYSIS
# ========================================

ENHANCED_SENTIMENT_CONFIG = {
    "enabled": True,
    "sources": {
        "news": {
            "weight": 0.25,
            "lookback_hours": 24,
            "sentiment_models": ["vader", "textblob", "finbert"],
            "keywords_tracking": True,
            "entity_recognition": True
        },
        "social_media": {
            "reddit": {
                "weight": 0.30,
                "subreddits": ["stocks", "investing", "SecurityAnalysis", "ValueInvesting", "wallstreetbets"],
                "sentiment_threshold": 50,
                "post_score_threshold": 10,
                "comment_analysis": True
            },
            "twitter": {
                "weight": 0.20,
                "keywords": ["$AAPL", "$MSFT", "$NVDA", "$SPY", "$QQQ"],
                "influencer_tracking": True,
                "hashtag_analysis": True,
                "retweet_weighting": True
            }
        },
        "retail_sentiment": {
            "robinhood_popularity": {
                "weight": 0.15,
                "ranking_impact": True,
                "options_flow": True,
                "holdings_analysis": True
            }
        },
        "analyst_sentiment": {
            "weight": 0.10,
            "rating_changes": True,
            "price_target_changes": True,
            "upgrade_downgrade_impact": True
        }
    },
    "aggregation": {
        "method": "weighted_average",
        "normalization": "z_score",
        "decay_factor": 0.9,  # How quickly old sentiment decays
        "momentum_tracking": True,
        "contrarian_signals": True
    }
}

# ========================================
# MARKET STRUCTURE & MACRO ANALYSIS
# ========================================

MARKET_STRUCTURE_CONFIG = {
    "enabled": True,
    "breadth_indicators": {
        "advance_decline": True,
        "new_highs_lows": True,
        "up_down_volume": True,
        "breadth_thrust": {"threshold": 0.9},
        "arms_index": True,
        "mcclellan_oscillator": True
    },
    "sector_analysis": {
        "sector_rotation": True,
        "relative_strength": True,
        "sector_etfs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
        "leadership_tracking": True
    },
    "vix_analysis": {
        "enabled": True,
        "fear_greed_levels": {
            "extreme_fear": 30,
            "fear": 20,
            "neutral": [12, 20],
            "complacency": 12
        },
        "term_structure": True,
        "put_call_ratio": True
    },
    "economic_indicators": {
        "enabled": True,
        "indicators": {
            "gdp": {"weight": 0.20},
            "unemployment": {"weight": 0.15},
            "inflation": {"weight": 0.15},
            "interest_rates": {"weight": 0.15},
            "retail_sales": {"weight": 0.10},
            "manufacturing_pmi": {"weight": 0.10},
            "consumer_confidence": {"weight": 0.10},
            "yield_curve": {"weight": 0.05}
        }
    }
}

# ========================================
# ENHANCED SIGNAL WEIGHTS
# Professional-grade multi-factor model
# ========================================

ENHANCED_SIGNAL_WEIGHTS = {
    # Core Technical Analysis (40%)
    "technical_trend": 0.15,          # Moving averages, MACD, ADX
    "technical_momentum": 0.12,       # RSI, Stochastic, Williams %R
    "technical_volatility": 0.08,     # Bollinger Bands, ATR
    "technical_volume": 0.05,         # OBV, Volume indicators
    
    # Advanced ML Predictions (25%)
    "ml_ensemble": 0.15,              # XGBoost, LightGBM, Random Forest
    "deep_learning": 0.10,            # LSTM, Neural Networks
    
    # Multi-Source Sentiment (20%)
    "news_sentiment": 0.05,           # News analysis
    "social_sentiment": 0.06,         # Reddit, Twitter
    "retail_sentiment": 0.05,         # Robinhood popularity
    "analyst_sentiment": 0.04,        # Professional ratings
    
    # Market Structure (10%)
    "market_breadth": 0.04,           # Advance/decline, breadth indicators
    "sector_rotation": 0.03,          # Sector strength analysis
    "vix_sentiment": 0.03,            # Fear/greed indicators
    
    # Risk & Macro (5%)
    "economic_indicators": 0.02,      # GDP, inflation, employment
    "risk_metrics": 0.02,             # VaR, correlation, volatility
    "liquidity_analysis": 0.01        # Volume, spread analysis
}

# Validate weights sum to 1.0
assert abs(sum(ENHANCED_SIGNAL_WEIGHTS.values()) - 1.0) < 0.001, "Signal weights must sum to 1.0"

# ========================================
# ENHANCED CONFIDENCE THRESHOLDS
# ========================================

ENHANCED_CONFIDENCE_THRESHOLDS = {
    "HYPER_BUY": 85,      # 85-100% confidence UP
    "STRONG_BUY": 75,     # 75-84% confidence UP
    "SOFT_BUY": 65,       # 65-74% confidence UP
    "HOLD": 35,           # 35-64% confidence (unclear)
    "SOFT_SELL": 65,      # 65-74% confidence DOWN
    "STRONG_SELL": 75,    # 75-84% confidence DOWN
    "HYPER_SELL": 85,     # 85-100% confidence DOWN
}

# ========================================
# ENHANCED UPDATE INTERVALS
# ========================================

ENHANCED_UPDATE_INTERVALS = {
    "real_time_data": 5,              # Every 5 seconds for price data
    "technical_indicators": 30,        # Every 30 seconds
    "sentiment_analysis": 60,          # Every minute
    "ml_predictions": 300,             # Every 5 minutes
    "market_structure": 180,           # Every 3 minutes
    "economic_data": 3600,             # Every hour
    "news_sentiment": 300,             # Every 5 minutes
    "social_sentiment": 180,           # Every 3 minutes
    "pattern_recognition": 60,         # Every minute
    "risk_calculations": 300,          # Every 5 minutes
    "model_retraining": 3600,          # Every hour
    "performance_monitoring": 60       # Every minute
}

# ========================================
# TARGET TICKERS - EXPANDED
# ========================================

ENHANCED_TICKERS = {
    "primary": ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"],
    "sector_etfs": ["XLK", "XLF", "XLE", "XLV", "XLI"],
    "volatility": ["VIX", "UVXY", "SVXY"],
    "crypto_proxies": ["MSTR", "COIN"],
    "watchlist": ["TSLA", "GOOGL", "AMZN", "META", "NFLX"]
}

# Main tickers (backward compatibility)
TICKERS = ENHANCED_TICKERS["primary"]

# ========================================
# ENHANCED FEATURE FLAGS
# ========================================

ENHANCED_FEATURE_FLAGS = {
    # Data Sources
    "enable_multi_source_data": True,
    "enable_robinhood_sentiment": True,
    "enable_news_sentiment": bool(NEWS_API_KEY),
    "enable_social_sentiment": bool(REDDIT_CONFIG["client_id"]),
    "enable_fundamental_analysis": True,
    "enable_options_flow": True,
    
    # ML Features
    "enable_ensemble_ml": True,
    "enable_deep_learning": True,
    "enable_automated_feature_engineering": True,
    "enable_hyperparameter_optimization": True,
    "enable_model_interpretation": True,
    
    # Technical Analysis
    "enable_advanced_technical": True,
    "enable_pattern_recognition": True,
    "enable_support_resistance": True,
    "enable_fibonacci_analysis": True,
    
    # Market Analysis
    "enable_market_structure": True,
    "enable_sector_analysis": True,
    "enable_vix_analysis": True,
    "enable_economic_indicators": True,
    
    # Risk & Performance
    "enable_risk_management": True,
    "enable_performance_attribution": True,
    "enable_backtesting": True,
    "enable_walk_forward_analysis": True,
    
    # Production Features
    "enable_real_time_alerts": True,
    "enable_performance_monitoring": True,
    "enable_model_drift_detection": True,
    "enable_automated_reporting": True
}

# ========================================
# ENHANCED PERFORMANCE TARGETS
# ========================================

PERFORMANCE_TARGETS = {
    "signal_accuracy": 0.72,          # 72% accuracy target
    "precision": 0.68,                # 68% precision
    "recall": 0.65,                   # 65% recall
    "f1_score": 0.66,                 # F1 score target
    "sharpe_ratio": 1.5,              # Risk-adjusted returns
    "max_drawdown": 0.12,             # Maximum 12% drawdown
    "win_rate": 0.60,                 # 60% winning trades
    "profit_factor": 1.8,             # Profit factor target
    "calmar_ratio": 1.2,              # Risk-adjusted performance
    "information_ratio": 0.8          # Active return vs tracking error
}

# ========================================
# SERVER & DEPLOYMENT CONFIG
# ========================================

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "debug": DEBUG_MODE,
    "reload": False,  # Disabled for production
    "workers": 1,
    "max_connections": 1000,
    "keepalive_timeout": 65,
    "access_log": True,
    "proxy_headers": True,
    "forwarded_allow_ips": "*"
}

# ========================================
# LOGGING CONFIGURATION
# ========================================

ENHANCED_LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "structured_logging": ENVIRONMENT == "production",
    "performance_logging": True,
    "ml_training_logs": True,
    "data_quality_logs": True,
    "signal_generation_logs": DEBUG_MODE,
    "api_request_logs": DEBUG_MODE
}

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_enabled_data_sources() -> List[str]:
    """Get list of enabled data sources"""
    return [
        source for source, config in DATA_SOURCES_CONFIG["primary_sources"].items()
        if config["enabled"]
    ]

def get_ml_model_config(model_name: str) -> Dict:
    """Get configuration for specific ML model"""
    return ENHANCED_ML_CONFIG["models"].get(model_name, {})

def is_feature_enabled(feature_name: str) -> bool:
    """Check if enhanced feature is enabled"""
    return ENHANCED_FEATURE_FLAGS.get(feature_name, False)

def validate_enhanced_config():
    """Validate enhanced configuration"""
    # Validate signal weights
    total_weight = sum(ENHANCED_SIGNAL_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(f"Enhanced signal weights must sum to 1.0, got {total_weight}")
    
    # Validate data sources
    enabled_sources = get_enabled_data_sources()
    if len(enabled_sources) < 2:
        raise ValueError("At least 2 data sources must be enabled for redundancy")
    
    # Validate ML configuration
    if ENHANCED_ML_CONFIG["enabled"] and not any(
        model["enabled"] for model in ENHANCED_ML_CONFIG["models"].values()
    ):
        raise ValueError("At least one ML model must be enabled")
    
    return True

# ========================================
# INITIALIZATION
# ========================================

# Validate configuration on import
try:
    validate_enhanced_config()
    enabled_sources = get_enabled_data_sources()
    
    print("🚀 ENHANCED HYPER CONFIGURATION LOADED")
    print(f"📊 Data Sources: {len(enabled_sources)} enabled ({', '.join(enabled_sources)})")
    print(f"🧠 ML Models: {len([m for m in ENHANCED_ML_CONFIG['models'] if ENHANCED_ML_CONFIG['models'][m].get('enabled')])} enabled")
    print(f"📈 Technical Indicators: {sum(len(group) if isinstance(group, list) else len(group) if isinstance(group, dict) else 1 for group in ENHANCED_TECHNICAL_CONFIG['indicators'].values())}")
    print(f"🎯 Performance Target: {PERFORMANCE_TARGETS['signal_accuracy']:.1%} accuracy")
    print(f"⚡ Enhanced Features: {sum(ENHANCED_FEATURE_FLAGS.values())}/{len(ENHANCED_FEATURE_FLAGS)} enabled")
    
except Exception as e:
    print(f"❌ Enhanced configuration validation failed: {e}")
    raise
