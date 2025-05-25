# ============================================
# COMBINED ENHANCED HYPER TRADING SYSTEM
# Requirements for Production Deployment
# ============================================

# Core FastAPI and Server Components (Required)
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0

# HTTP Client and Async Support (Required)
aiohttp==3.9.1

# Data Processing and Analysis (Required)
pandas==2.1.4
numpy==1.26.2

# Google Trends Analysis (Required)
pytrends==4.9.2

# Basic Utilities (Required)
pathlib
python-dateutil==2.8.2

# ============================================
# ENHANCED FEATURES (Optional)
# System will use fallback/simulation if not available
# Uncomment any of these for full enhanced functionality
# ============================================

# Machine Learning and AI
# scikit-learn==1.3.2
# tensorflow==2.15.0
# torch==2.1.0

# Sentiment Analysis
# textblob==0.17.1
# vaderSentiment==3.3.2
# requests==2.31.0
# beautifulsoup4==4.12.2

# Technical Analysis
# TA-Lib==0.4.28
# pandas-ta==0.3.14b0

# Financial Data APIs
# yfinance==0.2.28
# alpha-vantage==2.3.1

# News and Social Media APIs
# newsapi-python==0.2.7
# praw==7.7.1
# tweepy==4.14.0

# Statistical Analysis
# scipy==1.11.4
# statsmodels==0.14.1

# Environment Management
# python-dotenv==1.0.0

# Model Persistence
# joblib==1.3.2

# Performance Monitoring
# psutil==5.9.6

# API Rate Limiting
# ratelimit==2.2.1
# backoff==2.2.1

# ============================================
# DEPLOYMENT NOTES
# ============================================

# CORE SYSTEM: Works with just the required packages above
# - Your original HYPER signals will work perfectly
# - Basic technical analysis and Google Trends
# - Real-time WebSocket streaming
# - Professional FastAPI backend

# ENHANCED FEATURES: Uncomment optional packages for:
# - Williams %R and Stochastic oscillators
# - VIX fear/greed sentiment analysis  
# - ML predictions and pattern recognition
# - Fibonacci level calculations
# - Multi-source sentiment analysis
# - Advanced risk metrics
# - Economic indicator integration

# RENDER DEPLOYMENT: 
# - Uses only core requirements by default
# - Enhanced features use simulation/fallback data
# - No build failures due to complex dependencies
# - Instant deployment with basic functionality

# PRODUCTION ENHANCEMENT:
# - Add enhanced packages gradually in production
# - Test each addition individually
# - Monitor build times and memory usage
# - Some packages may require system dependencies

# SYSTEM COMPATIBILITY:
# - Core system: Works on all platforms
# - TA-Lib: May need system libraries
# - TensorFlow: May need specific Python versions
# - All enhanced features have graceful fallbacks
