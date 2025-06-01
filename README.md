# ⚡ HYPERtrends v4.0 – Neural Trading Matrix

A modular, production-ready real-time trading intelligence engine.  
Combines market data from Robinhood, AI-driven signal generation, and a cyberpunk web dashboard with VIX/risk/sentiment modules.

---

## 🔧 Features
- ✅ Real-time Robinhood + simulated fallback data feed
- 📈 ML-based prediction using RandomForest, XGBoost, LSTM
- 🔁 Modular signal engine with confidence scoring
- 🎨 Cyberpunk-themed dashboard UI (`index.html`)
- 📊 VIX, sentiment, options flow, risk scoring
- 🧠 Streamed neural signals via WebSocket or polling
- 🚀 Deploy-ready via `render.yaml`

---

## 🛠️ Setup

### 1. Environment Variables (set via Render or `.env`)
RH_USERNAME=your@email.com
RH_PASSWORD=yourSecurePassword
ENVIRONMENT=production
DEMO_MODE=tru
### 2. Install Requirements
```bash
pip install -r requirements.txt
uvicorn main:app --reload
├── main.py                # FastAPI backend
├── signal_engine.py       # Modular signal generation
├── data_sources.py        # Robinhood + fallback feed
├── technical_indicators.py
├── sentiment_analysis.py
├── vix_analysis.py
├── market_structure.py
├── risk_analysis.py
├── ml_learning.py
├── model_testing.py
├── config.py              # Unified config + env parsing
├── environment.py         # Optional local fallback
├── render.yaml            # Render deployment spec
├── requirements.txt
├── index.html             # Frontend dashboard
