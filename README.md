# 🚀 HYPERtrends v4.0 - Alpaca Edition

## Production-Grade AI Trading Signal Engine

**HYPERtrends v4.0** is a breakthrough AI-powered market prediction system that combines cutting-edge machine learning, advanced technical indicators, multi-source data fusion, and real-time market analysis to predict stock movements with unprecedented accuracy.

### 🌟 Key Features

- **📈 Live Alpaca Markets Integration** - Real-time market data with microsecond precision
- **🧠 Advanced ML Predictions** - Neural networks, ensemble methods, deep learning
- **📊 25+ Technical Indicators** - RSI, MACD, Williams %R, Stochastic, Bollinger Bands, and more
- **💭 Multi-Source Sentiment** - News, social media, Google Trends analysis
- **😱 VIX Fear/Greed Signals** - Contrarian market sentiment detection
- **🏗️ Market Structure Analysis** - Breadth, sector rotation, institutional flow
- **⚠️ Advanced Risk Management** - VaR, position sizing, portfolio optimization
- **🎯 Real-Time WebSocket** - Live signal broadcasting to connected clients

-----

## 🚀 Quick Deploy to Render

### 1. One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yourusername/hypertrends-v4)

### 2. Manual Deploy Steps

1. **Fork/Clone Repository**
   
   ```bash
   git clone https://github.com/yourusername/hypertrends-v4.git
   cd hypertrends-v4
   ```
1. **Create Render Account**
- Sign up at [render.com](https://render.com)
- Connect your GitHub repository
1. **Deploy Web Service**
- Create new “Web Service”
- Connect to your repository
- Use these settings:
  - **Environment**: Python 3
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn main:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120`
  - **Plan**: Starter or higher
1. **Configure Environment Variables**
   
   ```bash
   # Required
   APCA_API_KEY_ID=PK2AML2QK9VUI5J1G1BC
   APCA_API_SECRET_KEY=your_secret_key_here
   USE_SANDBOX=True
   ENVIRONMENT=production
   
   # Optional for enhanced features
   NEWS_API_KEY=your_news_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   ```
1. **Deploy!**
- Click “Create Web Service”
- Render will automatically deploy your application
- Access at your assigned Render URL

-----

## 🏠 Local Development Setup

### Prerequisites

- Python 3.9+
- pip package manager
- Alpaca Markets account (free paper trading)

### Installation

1. **Clone Repository**
   
   ```bash
   git clone https://github.com/yourusername/hypertrends-v4.git
   cd hypertrends-v4
   ```
1. **Create Virtual Environment**
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```
1. **Install Dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
1. **Configure Environment**
   
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
1. **Run Application**
   
   ```bash
   python start.py
   # or
   python main.py
   ```
1. **Access Dashboard**
- Open http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

-----

## 🔧 Configuration

### Required API Keys

1. **Alpaca Markets** (Primary Data Source)
- Sign up at [alpaca.markets](https://alpaca.markets)
- Get API key and secret from dashboard
- Free paper trading account available

### Optional API Keys (Enhanced Features)

1. **News API** - For enhanced sentiment analysis
- Get key at [newsapi.org](https://newsapi.org)
1. **Alpha Vantage** - For additional market data
- Get key at [alphavantage.co](https://www.alphavantage.co)

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
USE_SANDBOX=True  # Use paper trading for safety

# Performance
LOG_LEVEL=INFO
MAX_REQUESTS_PER_MINUTE=100
SIGNAL_UPDATE_INTERVAL=30

# Features
ENABLE_ML_PREDICTIONS=True
ENABLE_ADVANCED_RISK=True
ENABLE_REAL_TIME_NEWS=True

# Security
CORS_ORIGINS=*  # Configure for production
```

-----

## 📊 API Endpoints

### Core Endpoints

- `GET /` - Main dashboard
- `GET /health` - System health check
- `GET /api/signals` - Current trading signals
- `GET /api/signals/{symbol}` - Signal for specific symbol
- `POST /api/signals/refresh` - Manual signal refresh

### ML & Testing

- `GET /api/ml/status` - ML system status
- `GET /api/testing/status` - Model testing status
- `GET /api/testing/backtest?days=7` - Run backtest

### WebSocket

- `WS /ws` - Real-time signal updates

### Example API Usage

```javascript
// Get current signals
fetch('/api/signals')
  .then(response => response.json())
  .then(data => {
    console.log('Current signals:', data.signals);
  });

// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'signal_update') {
    console.log('New signals:', data.signals);
  }
};
```

-----

## 🧠 Architecture Overview

### Core Components

1. **Data Sources** (`data_sources.py`)
- Alpaca Markets API integration
- Google Trends analysis
- Enhanced market simulation fallback
1. **Signal Engine** (`signal_engine.py`)
- Orchestrates all analysis components
- Weighted signal generation
- Real-time processing
1. **Technical Analysis** (`technical_indicators.py`)
- 25+ professional indicators
- Pattern recognition
- Volume analysis
1. **Sentiment Analysis** (`sentiment_analysis.py`)
- Multi-source NLP processing
- News and social media analysis
- Contrarian signal detection
1. **VIX Analysis** (`vix_analysis.py`)
- Fear/greed sentiment tracking
- Market regime detection
- Contrarian opportunities
1. **Market Structure** (`market_structure.py`)
- Breadth analysis
- Sector rotation tracking
- Institutional flow analysis
1. **Risk Analysis** (`risk_analysis.py`)
- Value at Risk (VaR) calculations
- Position sizing recommendations
- Portfolio risk management
1. **ML Enhancement** (`ml_learning.py`)
- Neural network predictions
- Ensemble model voting
- Continuous learning

### Data Flow

```
Alpaca Markets API → Data Aggregator → Signal Engine → Analysis Components → ML Enhancement → WebSocket Broadcast
```

-----

## 🎯 Signal Types & Confidence Levels

### Signal Types

- **HYPER_BUY** (85-100% confidence) - Strong buy with high conviction
- **SOFT_BUY** (65-84% confidence) - Moderate buy signal
- **HOLD** (35-64% confidence) - Neutral/unclear signals
- **SOFT_SELL** (65-84% confidence) - Moderate sell signal
- **HYPER_SELL** (85-100% confidence) - Strong sell with high conviction

### Signal Components

Each signal includes:

- Technical analysis score (25+ indicators)
- Sentiment analysis score (multi-source)
- VIX fear/greed score
- Market structure score
- Risk-adjusted score
- ML prediction confidence
- Position sizing recommendations

-----

## 📈 Performance & Monitoring

### Health Monitoring

The system includes comprehensive health monitoring:

- Real-time system status
- Component health checks
- Performance metrics tracking
- Error rate monitoring
- Data quality assessment

### Performance Optimizations

- Async/await for concurrent processing
- Intelligent caching strategies
- Rate limiting and request optimization
- Memory-efficient data processing
- Graceful fallback mechanisms

-----

## 🛡️ Security & Risk Management

### Security Features

- Environment-based configuration
- API key protection
- CORS configuration
- Request rate limiting
- Input validation

### Risk Management

- Paper trading mode by default
- Position sizing recommendations
- Risk-adjusted signals
- Maximum drawdown monitoring
- Portfolio correlation analysis

-----

## 🔧 Troubleshooting

### Common Issues

1. **“No Alpaca credentials”**
- Set `APCA_API_KEY_ID` environment variable
- Ensure API key is valid
- Check Render dashboard environment variables
1. **“Signal generation failed”**
- Check Alpaca API status
- Verify internet connection
- Review application logs
1. **“WebSocket connection failed”**
- Ensure WebSocket support in hosting environment
- Check firewall settings
- Verify correct WebSocket URL
1. **“Low data quality warnings”**
- Check Alpaca API rate limits
- Verify market hours
- Review data source configuration

### Debugging

Enable debug mode:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=True
```

Check logs:

```bash
# View real-time logs on Render
render logs follow your-service-name

# Local development
tail -f hypertrends.log
```

-----

## 📚 Advanced Usage

### Custom Indicators

Add custom technical indicators in `technical_indicators.py`:

```python
def custom_indicator(self, prices: np.ndarray) -> np.ndarray:
    """Custom indicator implementation"""
    # Your logic here
    return result
```

### ML Model Enhancement

Extend ML capabilities in `ml_learning.py`:

```python
class CustomMLModel:
    """Custom ML model for specific use cases"""
    
    def predict(self, features):
        # Your model logic
        return predictions
```

### Custom Data Sources

Add new data sources in `data_sources.py`:

```python
class CustomDataSource:
    """Custom data source integration"""
    
    async def get_data(self, symbol):
        # Your data retrieval logic
        return data
```

-----

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Alpaca API credentials configured
- [ ] Environment variables set
- [ ] Dependencies installed
- [ ] Configuration validated
- [ ] Local testing completed

### Post-Deployment

- [ ] Health check endpoint responding
- [ ] WebSocket connections working
- [ ] Signal generation active
- [ ] Logs showing no errors
- [ ] Performance metrics normal

### Production Checklist

- [ ] Use live Alpaca account (if desired)
- [ ] Configure proper CORS origins
- [ ] Set up monitoring alerts
- [ ] Enable SSL/TLS
- [ ] Configure rate limiting
- [ ] Set up backup strategies

-----

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
1. Create a feature branch
1. Add tests for new functionality
1. Ensure all tests pass
1. Submit a pull request

-----

## 📄 License

This project is licensed under the MIT License - see the <LICENSE> file for details.

-----

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **API Support**: Refer to Alpaca Markets documentation

-----

## 🚀 What’s Next?

### Planned Features

- [ ] Cryptocurrency signal support
- [ ] Options flow analysis
- [ ] Enhanced backtesting framework
- [ ] Mobile app integration
- [ ] Advanced portfolio optimization
- [ ] Real-time news sentiment
- [ ] Earnings prediction models

### Contribution Areas

- ML model improvements
- Additional technical indicators
- Enhanced data sources
- Performance optimizations
- Documentation improvements

-----

**⚡ Ready to deploy your AI trading signal engine? Get started now!**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

-----

*HYPERtrends v4.0 - Where AI meets Wall Street* 🚀📈
