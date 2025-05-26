# ============================================
# HYPER Trading System - Environment Variables
# ============================================

# Alpha Vantage API Configuration
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=false
RELOAD=false

# Logging Configuration
LOG_LEVEL=INFO

# Rate Limiting
ALPHA_VANTAGE_CALLS_PER_MINUTE=5
ALPHA_VANTAGE_CALLS_PER_DAY=500

# Update Intervals (seconds)
SIGNAL_GENERATION_INTERVAL=30
MARKET_DATA_INTERVAL=60
GOOGLE_TRENDS_INTERVAL=300

# Security (for production)
SECRET_KEY=your_secret_key_here

# Database (if needed later)
# DATABASE_URL=postgresql://user:pass@localhost/hyper_db
