# ========================================
# SMART ENVIRONMENT DETECTION & CONFIG
# ========================================

# FILE: environment.py
""""
Smart environment detection and configuration management
""""
import os
import platform
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class EnvironmentDetector:
    """Automatically detect deployment environment and configure accordingly""""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.platform_info = self._get_platform_info()
        self._setup_logging()
    
    def _detect_environment(self) -> str:
        """Automatically detect the deployment environment""""
        
        # Check for explicit environment variable first
        explicit_env = os.getenv("ENVIRONMENT", "").lower()
        if explicit_env in ["development", "staging", "production"]:
            return explicit_env
        
        # Auto-detection based on platform indicators
        
        # Render.com detection
        if self._is_render():
            return "production""
        
        # Heroku detection
        if self._is_heroku():
            return "production""
        
        # Railway detection
        if self._is_railway():
            return "production""
        
        # Vercel detection
        if self._is_vercel():
            return "production""
        
        # Docker detection
        if self._is_docker():
            return "staging""
        
        # GitHub Actions / CI detection
        if self._is_ci():
            return "testing""
        
        # Local development (default)
        return "development""
    
    def _is_render(self) -> bool:
        """Detect Render.com deployment""""
        return any([
            os.getenv("RENDER"),
            os.getenv("RENDER_SERVICE_ID"),
            os.getenv("RENDER_SERVICE_NAME"),
            "render.com" in os.getenv("RENDER_EXTERNAL_URL", "")
        ])
    
    def _is_heroku(self) -> bool:
        """Detect Heroku deployment""""
        return any([
            os.getenv("DYNO"),
            os.getenv("HEROKU_APP_NAME"),
            os.getenv("HEROKU_SLUG_COMMIT")
        ])
    
    def _is_railway(self) -> bool:
        """Detect Railway deployment""""
        return any([
            os.getenv("RAILWAY_ENVIRONMENT"),
            os.getenv("RAILWAY_PROJECT_ID"),
            os.getenv("RAILWAY_SERVICE_NAME")
        ])
    
    def _is_vercel(self) -> bool:
        """Detect Vercel deployment""""
        return any([
            os.getenv("VERCEL"),
            os.getenv("VERCEL_ENV"),
            os.getenv("VERCEL_URL")
        ])
    
    def _is_docker(self) -> bool:
        """Detect Docker environment""""
        return any([
            os.path.exists("/.dockerenv"),
            os.getenv("DOCKER_CONTAINER"),
            "docker" in os.getenv("HOSTNAME", "").lower()
        ])
    
    def _is_ci(self) -> bool:
        """Detect CI/CD environment""""
        return any([
            os.getenv("CI"),
            os.getenv("GITHUB_ACTIONS"),
            os.getenv("GITLAB_CI"),
            os.getenv("JENKINS_URL"),
            os.getenv("TRAVIS"),
            os.getenv("CIRCLECI")
        ])
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get detailed platform information""""
        return {
            "system": platform.system(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "is_render": self._is_render(),
            "is_heroku": self._is_heroku(),
            "is_railway": self._is_railway(),
            "is_docker": self._is_docker(),
            "is_ci": self._is_ci(),
            "has_gpu": self._has_gpu(),
            "memory_gb": self._get_memory_gb(),
            "cpu_count": os.cpu_count()
        }
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available""""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except:
            return False
    
    def _get_memory_gb(self) -> float:
        """Get available memory in GB""""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 0.0
    
    def _setup_logging(self):
        """Setup logging based on environment""""
        log_level = self.get_log_level()
        logging.basicConfig(
            level=log_level,
            format=self.get_log_format(),
            force=True
        )
    
    def get_log_level(self) -> int:
        """Get appropriate log level for environment""""
        env_levels = {
            "development": logging.DEBUG,
            "testing": logging.INFO,
            "staging": logging.INFO,
            "production": logging.WARNING
        }
        
        # Allow override via environment variable
        override = os.getenv("LOG_LEVEL", "").upper()
        if override in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            return getattr(logging, override)
        
        return env_levels.get(self.environment, logging.INFO)
    
    def get_log_format(self) -> str:
        """Get appropriate log format for environment""""
        if self.environment == "production":
            # Structured logging for production
            return "%(asctime)s - %(name)s - %(levelname)s - %(message)s""
        else:
            # Detailed logging for development
            return "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s""
    
    def is_production(self) -> bool:
        """Check if running in production""""
        return self.environment == "production""
    
    def is_development(self) -> bool:
        """Check if running in development""""
        return self.environment == "development""
    
    def should_use_demo_mode(self) -> bool:
        """Determine if demo mode should be used""""
        # Force demo mode if no API key available
        if not os.getenv("ALPHA_VANTAGE_API_KEY"):
            return True
        
        # Use demo mode in non-production by default
        demo_override = os.getenv("DEMO_MODE", "").lower()
        if demo_override in ["true", "false"]:
            return demo_override == "true""
        
        # Auto-determine based on environment
        return self.environment in ["development", "testing"]
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration based on environment""""
        base_config = {
            "host": "127.0.0.1" if self.environment == "development" else "0.0.0.0",
            "port": int(os.getenv("PORT", "8000")),
            "reload": self.environment == "development",
            "debug": self.environment == "development",
            "workers": 1,  # WebSocket compatibility
            "access_log": self.environment in ["production", "staging"],
            "proxy_headers": self.environment != "development",
            "forwarded_allow_ips": "*" if self.environment != "development" else "127.0.0.1""
        }
        
        # Platform-specific adjustments
        if self._is_render():
            base_config.update({
                "host": "0.0.0.0",
                "proxy_headers": True,
                "forwarded_allow_ips": "*",
                "keepalive_timeout": 65
            })
        elif self._is_heroku():
            base_config.update({
                "host": "0.0.0.0",
                "proxy_headers": True,
                "timeout_keep_alive": 30
            })
        
        return base_config
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration based on environment""""
        # Check for managed database URL (Render, Heroku, etc.)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return {
                "type": "postgresql",
                "url": database_url,
                "ssl_mode": "require" if self.is_production() else "prefer""
            }
        
        # Default to SQLite
        db_name = f"hyper_{self.environment}.db""
        return {
            "type": "sqlite",
            "database": db_name,
            "path": Path("data") / db_name
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration based on environment""""
        redis_url = os.getenv("REDIS_URL") or os.getenv("REDISCLOUD_URL")
        
        if redis_url and self.is_production():
            return {
                "type": "redis",
                "url": redis_url,
                "ssl": True
            }
        else:
            return {
                "type": "memory",
                "max_size": 1000
            }
    
    def get_api_rate_limits(self) -> Dict[str, int]:
        """Get API rate limits based on environment and platform""""
        if self.environment == "development":
            return {
                "requests_per_minute": 1000,
                "alpha_vantage_calls_per_minute": 10
            }
        elif self.platform_info["memory_gb"] < 1:  # Free tier
            return {
                "requests_per_minute": 100,
                "alpha_vantage_calls_per_minute": 3
            }
        else:  # Paid tier
            return {
                "requests_per_minute": 500,
                "alpha_vantage_calls_per_minute": 5
            }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags based on environment and resources""""
        base_flags = {
            "enable_ml_learning": True,
            "enable_model_testing": True,
            "enable_advanced_technical": True,
            "enable_sentiment_analysis": True,
            "enable_caching": True,
            "enable_rate_limiting": True
        }
        
        # Adjust based on environment
        if self.environment == "development":
            base_flags.update({
                "enable_debug_endpoints": True,
                "enable_hot_reload": True,
                "enable_profiling": True
            })
        elif self.platform_info["memory_gb"] < 1:  # Resource constrained
            base_flags.update({
                "enable_ml_learning": False,
                "enable_advanced_technical": False,
                "enable_sentiment_analysis": False
            })
        
        return base_flags
    
    def print_environment_info(self):
        """Print comprehensive environment information""""
        print("=" * 60)
        print("🌍 HYPER TRADING SYSTEM - ENVIRONMENT DETECTION")
        print("=" * 60)
        print(f"🔍 Detected Environment: {self.environment.upper()}")
        print(f"🖥️  Platform: {self.platform_info['platform']}")
        print(f"🐍 Python: {self.platform_info['python_version']}")
        print(f"💾 Memory: {self.platform_info['memory_gb']} GB")
        print(f"🔧 CPUs: {self.platform_info['cpu_count']}")
        print(f"🏠 Hostname: {self.platform_info['hostname']}")
        
        print("\n📡 Platform Detection:")
        platforms = ["render", "heroku", "railway", "docker", "ci"]
        for platform in platforms:
            detected = self.platform_info.get(f"is_{platform}", False)
            print(f"   {platform.title()}: {'✅' if detected else '❌'}")
        
        print(f"\n⚙️  Configuration:")
        print(f"   Demo Mode: {'✅' if self.should_use_demo_mode() else '❌'}")
        print(f"   Debug Mode: {'✅' if self.environment == 'development' else '❌'}")
        print(f"   Log Level: {logging.getLevelName(self.get_log_level())}")
        
        server_config = self.get_server_config()
        print(f"   Server: {server_config['host']}:{server_config['port']}")
        
        print("=" * 60)

# Initialize global environment detector
env = EnvironmentDetector()

# FILE: .env.development
""""
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
DEMO_MODE=true
LOG_LEVEL=DEBUG

# API Keys (for testing - use your own)
ALPHA_VANTAGE_API_KEY=demo
NEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_SECRET=
TWITTER_BEARER_TOKEN=

# Database
DB_TYPE=sqlite
DB_NAME=hyper_development.db

# Server
HOST=127.0.0.1
PORT=8000

# Features (all enabled for development)
ENABLE_ML_LEARNING=true
ENABLE_MODEL_TESTING=true
ENABLE_ADVANCED_TECHNICAL=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_HOT_RELOAD=true
ENABLE_PROFILING=true

# Security (relaxed for development)
CORS_ORIGINS=*
REQUIRE_HTTPS=false
RATE_LIMIT_ENABLED=false
""""

# FILE: .env.production
""""
# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false
DEMO_MODE=false
LOG_LEVEL=INFO

# API Keys (set in deployment platform)
ALPHA_VANTAGE_API_KEY=
NEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_SECRET=
TWITTER_BEARER_TOKEN=

# Database (managed database URL set by platform)
DATABASE_URL=

# Cache (managed Redis URL set by platform)
REDIS_URL=

# Security (strict for production)
CORS_ORIGINS=https://yourdomain.com
REQUIRE_HTTPS=true
RATE_LIMIT_ENABLED=true

# Features (optimized for production)
ENABLE_ML_LEARNING=true
ENABLE_MODEL_TESTING=true
ENABLE_ADVANCED_TECHNICAL=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_CACHING=true
ENABLE_DEBUG_ENDPOINTS=false
ENABLE_HOT_RELOAD=false
ENABLE_PROFILING=false
""""

# FILE: .env.example
""""
# HYPER Trading System Environment Configuration
# Copy this file to .env and customize for your environment

# ===========================================
# ENVIRONMENT DETECTION (optional - auto-detected)
# ===========================================
# ENVIRONMENT=development  # development, staging, production
# DEBUG=false
# DEMO_MODE=true

# ===========================================
# API CREDENTIALS
# ===========================================
# Get free API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Optional API keys for enhanced features
NEWS_API_KEY=your_news_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
# Automatically configured for most platforms
# DATABASE_URL=postgresql://user:pass@host:port/db  # Set by platform
DB_TYPE=sqlite
DB_NAME=hyper_trading.db

# ===========================================
# CACHE CONFIGURATION
# ===========================================
# Automatically configured for most platforms
# REDIS_URL=redis://localhost:6379  # Set by platform

# ===========================================
# SERVER CONFIGURATION
# ===========================================
# Automatically configured for most platforms
# PORT=8000  # Set by platform
# HOST=0.0.0.0  # Set automatically

# ===========================================
# LOGGING
# ===========================================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# ===========================================
# SECURITY
# ===========================================
CORS_ORIGINS=*  # Set to your domain in production
REQUIRE_HTTPS=false  # Set to true in production

# ===========================================
# FEATURE FLAGS
# ===========================================
ENABLE_ML_LEARNING=true
ENABLE_MODEL_TESTING=true
ENABLE_ADVANCED_TECHNICAL=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true

# ===========================================
# PERFORMANCE TUNING
# ===========================================
# UPDATE_INTERVAL_SIGNALS=30  # seconds
# UPDATE_INTERVAL_ML=600       # seconds
# MAX_CONNECTIONS=1000
# WORKER_COUNT=1

# ===========================================
# MONITORING & ALERTS
# ===========================================
ALERT_EMAIL=your-email@example.com
SLACK_WEBHOOK=your_slack_webhook_url

# ===========================================
# NOTES
# ===========================================
# - Most settings are auto-detected based on your platform
# - Only set what you need to override
# - API keys are the most important settings
# - The system works great with just ALPHA_VANTAGE_API_KEY
""""

# Updated config.py to use environment detection
import os
from environment import env

# ========================================
# SMART CONFIGURATION USING AUTO-DETECTION
# ========================================

# Environment detection (automatic)
ENVIRONMENT = env.environment
DEBUG_MODE = env.environment == "development""
DEMO_MODE = env.should_use_demo_mode()

# Platform detection
IS_RENDER = env._is_render()
IS_HEROKU = env._is_heroku()
IS_RAILWAY = env._is_railway()
IS_DOCKER = env._is_docker()

# API Credentials - Personal Use
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "OKUH0GNJE410ONTC")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Smart configuration based on environment
SERVER_CONFIG = env.get_server_config()
DATABASE_CONFIG = env.get_database_config()
CACHE_CONFIG = env.get_cache_config()
RATE_LIMITS = env.get_api_rate_limits()
FEATURE_FLAGS = env.get_feature_flags()

# Rest of configuration (tickers, weights, etc.) remains the same...
TICKERS = ["QQQ", "SPY", "NVDA", "AAPL", "MSFT"]

SIGNAL_WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.20,
    "momentum": 0.15,
    "ml_prediction": 0.15,
    "market_structure": 0.10,
    "vix_sentiment": 0.08,
    "economic": 0.05,
    "risk_adjusted": 0.02
}

# Smart logging configuration
LOGGING_CONFIG = {
    "level": logging.getLevelName(env.get_log_level()),
    "format": env.get_log_format(),
    "file": None if IS_RENDER else "logs/hyper.log""
}

# Smart security configuration
SECURITY_CONFIG = {
    "require_https": env.is_production(),
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "rate_limit_enabled": FEATURE_FLAGS["enable_rate_limiting"]
}

def validate_config():
    """Enhanced validation with environment awareness""""
    env.print_environment_info()
    
    if env.is_production() and not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Alpha Vantage API key required in production")
    
    if not TICKERS:
        raise ValueError("No tickers configured")
    
    return True

# Helper functions
def is_production() -> bool:
    return env.is_production()

def is_development() -> bool:
    return env.is_development()

def get_platform_info() -> dict:
    return env.platform_info

print("✅ Smart environment configuration loaded!")
print(f"🌍 Environment: {ENVIRONMENT}")
print(f"🔧 Demo Mode: {DEMO_MODE}")
print(f"📡 Platform: {env.platform_info['platform']}")