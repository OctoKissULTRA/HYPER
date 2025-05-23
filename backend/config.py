import os
from typing import Dict, List

class HYPERConfig:
    """Central configuration for HYPER trading system"""
    
    # ========================================
    # API CREDENTIALS
    # ========================================
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "RXV371OVWXK6BQ17")
    
    # ========================================
    # TARGET TICKERS (The Big 5)
    # ========================================
    TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]
    
    # ========================================
    # SIGNAL CONFIDENCE THRESHOLDS
    # ========================================
    CONFIDENCE_THRESHOLDS = {
        "HYPER_BUY": 90,     # 90-100% confidence UP
        "SOFT_BUY": 70,      # 70-89% confidence UP  
        "HOLD": 50,          # 50-69% confidence (unclear)
        "SOFT_SELL": 70,     # 70-89% confidence DOWN
        "HYPER_SELL": 90,    # 90-100% confidence DOWN
    }
    
    # ========================================
    # SIGNAL WEIGHT CONFIGURATION
    # ========================================
    SIGNAL_WEIGHTS = {
        "technical_analysis": 0.30,    # RSI, MACD, EMA analysis
        "alpha_vantage_momentum": 0.25, # Real-time price momentum
        "google_trends": 0.15,         # Search trend analysis
        "volume_analysis": 0.15,       # Volume patterns
        "ml_ensemble": 0.15,           # Machine learning predictions
    }
    
    # ========================================
    # TECHNICAL INDICATOR SETTINGS
    # ========================================
    TECHNICAL_PARAMS = {
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
    }
    
    # ========================================
    # GOOGLE TRENDS CONFIGURATION
    # ========================================
    TRENDS_CONFIG = {
        "timeframe": "now 7-d",
        "geo": "US",
        "keywords": {
            "QQQ": ["QQQ ETF", "tech stocks", "NASDAQ"],
            "SPY": ["SPY ETF", "S&P 500", "market index"],
            "NVDA": ["NVDA stock", "NVIDIA", "AI stocks"],
            "AAPL": ["AAPL stock", "Apple stock", "iPhone"],
            "MSFT": ["MSFT stock", "Microsoft", "cloud stocks"],
        }
    }
    
    # ========================================
    # FAKE-OUT DETECTION SETTINGS
    # ========================================
    FAKE_OUT_FILTERS = {
        "volume_threshold": 1.5,       # Volume must be 1.5x average
        "trend_duration_min": 2,       # Trend must last 2+ hours
        "pattern_success_rate": 0.60,  # Historical success rate threshold
        "sentiment_extreme": 0.90,     # Extreme sentiment warning level
    }
    
    # ========================================
    # REAL-TIME UPDATE SETTINGS
    # ========================================
    UPDATE_INTERVALS = {
        "market_data": 60,        # Update market data every 60 seconds
        "google_trends": 300,     # Update trends every 5 minutes
        "signal_generation": 30,  # Generate signals every 30 seconds
        "websocket_ping": 30,     # WebSocket keepalive
    }
    
    # ========================================
    # API RATE LIMITS
    # ========================================
    RATE_LIMITS = {
        "alpha_vantage_calls_per_minute": 5,
        "alpha_vantage_calls_per_day": 500,
        "google_trends_requests_per_hour": 100,
    }
    
    # ========================================
    # CONFIDENCE CALIBRATION
    # ========================================
    CONFIDENCE_ADJUSTMENTS = {
        "low_volume_penalty": -0.20,      # Reduce confidence for low volume
        "extreme_sentiment_penalty": -0.15, # Reduce for extreme sentiment
        "pattern_failure_penalty": -0.25,   # Reduce for failed patterns
        "multi_signal_boost": 0.10,        # Boost when signals align
    }
    
    # ========================================
    # UI/DISPLAY SETTINGS
    # ========================================
    UI_CONFIG = {
        "update_frequency_ms": 1000,     # UI updates every second
        "animation_duration_ms": 500,    # Pulse animation duration
        "max_signals_displayed": 10,     # Max signals in history
        "price_decimal_places": 2,       # Price display precision
    }
    
    # ========================================
    # LOGGING CONFIGURATION
    # ========================================
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "hyper.log",
        "max_size_mb": 10,
        "backup_count": 5,
    }
    
    # ========================================
    # DEPLOYMENT SETTINGS
    # ========================================
    SERVER_CONFIG = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "reload": os.getenv("RELOAD", "false").lower() == "true",
    }
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    @classmethod
    def get_signal_threshold(cls, signal_type: str) -> int:
        """Get confidence threshold for signal type"""
        return cls.CONFIDENCE_THRESHOLDS.get(signal_type, 50)
    
    @classmethod
    def get_ticker_keywords(cls, ticker: str) -> List[str]:
        """Get Google Trends keywords for ticker"""
        return cls.TRENDS_CONFIG["keywords"].get(ticker, [ticker])
    
    @classmethod
    def is_high_confidence_signal(cls, confidence: float, direction: str) -> str:
        """Determine signal type based on confidence and direction"""
        if direction.upper() == "UP":
            if confidence >= cls.CONFIDENCE_THRESHOLDS["HYPER_BUY"]:
                return "HYPER_BUY"
            elif confidence >= cls.CONFIDENCE_THRESHOLDS["SOFT_BUY"]:
                return "SOFT_BUY"
        elif direction.upper() == "DOWN":
            if confidence >= cls.CONFIDENCE_THRESHOLDS["HYPER_SELL"]:
                return "HYPER_SELL"
            elif confidence >= cls.CONFIDENCE_THRESHOLDS["SOFT_SELL"]:
                return "SOFT_SELL"
        
        return "HOLD"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        # Check API key exists
        if not cls.ALPHA_VANTAGE_API_KEY:
            raise ValueError("Alpha Vantage API key not configured")
        
        # Check tickers are defined
        if not cls.TICKERS:
            raise ValueError("No tickers configured")
        
        # Check signal weights sum to 1.0
        total_weight = sum(cls.SIGNAL_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight}")
        
        return True

# ========================================
# GLOBAL CONFIG INSTANCE
# ========================================
config = HYPERConfig()

# Validate config on import
config.validate_config()
