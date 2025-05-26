# ============================================
# HYPER PREDICTIVE MODEL TESTING FRAMEWORK
# Comprehensive backtesting and validation system
# ============================================

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from signal_engine import HYPERSignalEngine, HYPERSignal

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Single prediction result for tracking"""
    timestamp: str
    symbol: str
    signal_type: str
    confidence: float
    direction: str
    predicted_price: float
    actual_price: Optional[float] = None
    actual_change: Optional[float] = None
    prediction_horizon: int = 1  # days
    accuracy: Optional[float] = None
    profit_loss: Optional[float] = None
    signal_id: str = ""

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    total_predictions: int
    correct_predictions: int
    accuracy_percentage: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    best_signal_type: str
    worst_signal_type: str
    confidence_correlation: float

class PredictionTracker:
    """Track and evaluate prediction accuracy over time"""
    
    def __init__(self, db_path: str = "hyper_predictions.db"):
        self.db_path = db_path
        self.predictions = []
        self.init_database()
        logger.info(f"🧪 Prediction Tracker initialized with DB: {db_path}")
    
    def init_database(self):
        """Initialize SQLite database for prediction storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    direction TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    actual_change REAL,
                    prediction_horizon INTEGER DEFAULT 1,
                    accuracy REAL,
                    profit_loss REAL,
                    signal_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON predictions(symbol, timestamp)
            ''')
            logger.info("✅ Prediction database initialized")
    
    def record_prediction(self, signal: HYPERSignal, prediction_horizon: int = 1) -> str:
        """Record a new prediction for future validation"""
        prediction_id = f"{signal.symbol}_{signal.timestamp}_{prediction_horizon}"
        
        prediction = PredictionResult(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            direction=signal.direction,
            predicted_price=signal.price,
            prediction_horizon=prediction_horizon,
            signal_id=prediction_id
        )
        
        # Store in memory
        self.predictions.append(prediction)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO predictions 
                (timestamp, symbol, signal_type, confidence, direction, 
                 predicted_price, prediction_horizon, signal_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.timestamp, prediction.symbol, prediction.signal_type,
                prediction.confidence, prediction.direction, prediction.predicted_price,
                prediction.prediction_horizon, prediction.signal_id
            ))
        
        logger.info(f"📝 Recorded prediction: {signal.symbol} {signal.signal_type} ({signal.confidence:.1f}%)")
        return prediction_id
    
    def update_prediction_outcome(self, signal_id: str, actual_price: float) -> bool:
        """Update prediction with actual outcome"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get original prediction
                cursor = conn.execute(
                    'SELECT * FROM predictions WHERE signal_id = ?', (signal_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"⚠️ Prediction not found: {signal_id}")
                    return False
                
                predicted_price = row[6]  # predicted_price column
                direction = row[5]        # direction column
                
                # Calculate outcomes
                actual_change = ((actual_price - predicted_price) / predicted_price) * 100
                
                # Determine accuracy based on direction
                if direction == "UP":
                    accuracy = 1.0 if actual_price > predicted_price else 0.0
                elif direction == "DOWN":
                    accuracy = 1.0 if actual_price < predicted_price else 0.0
                else:  # NEUTRAL
                    accuracy = 1.0 if abs(actual_change) < 2.0 else 0.0
                
                # Calculate profit/loss (assuming $1000 position)
                position_size = 1000.0
                if direction == "UP":
                    profit_loss = position_size * (actual_change / 100)
                elif direction == "DOWN":
                    profit_loss = position_size * (-actual_change / 100)  # Short position
                else:
                    profit_loss = 0.0
                
                # Update database
                conn.execute('''
                    UPDATE predictions 
                    SET actual_price = ?, actual_change = ?, accuracy = ?, profit_loss = ?
                    WHERE signal_id = ?
                ''', (actual_price, actual_change, accuracy, profit_loss, signal_id))
                
                logger.info(f"✅ Updated prediction {signal_id}: accuracy={accuracy:.1f}, P&L=${profit_loss:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error updating prediction {signal_id}: {e}")
            return False
    
    def get_prediction_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """Get prediction statistics for recent period"""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    COUNT(accuracy) as evaluated,
                    AVG(CASE WHEN accuracy IS NOT NULL THEN accuracy END) as avg_accuracy,
                    AVG(CASE WHEN profit_loss IS NOT NULL THEN profit_loss END) as avg_pnl,
                    MAX(profit_loss) as best_trade,
                    MIN(profit_loss) as worst_trade
                FROM predictions 
                WHERE timestamp >= ?
            ''', (cutoff_date,))
            
            overall = cursor.fetchone()
            
            # Stats by signal type
            cursor = conn.execute('''
                SELECT 
                    signal_type,
                    COUNT(*) as count,
                    AVG(CASE WHEN accuracy IS NOT NULL THEN accuracy END) as accuracy,
                    AVG(CASE WHEN profit_loss IS NOT NULL THEN profit_loss END) as avg_pnl
                FROM predictions 
                WHERE timestamp >= ? AND accuracy IS NOT NULL
                GROUP BY signal_type
                ORDER BY accuracy DESC
            ''', (cutoff_date,))
            
            by_signal = cursor.fetchall()
            
            # Stats by symbol
            cursor = conn.execute('''
                SELECT 
                    symbol,
                    COUNT(*) as count,
                    AVG(CASE WHEN accuracy IS NOT NULL THEN accuracy END) as accuracy,
                    AVG(CASE WHEN profit_loss IS NOT NULL THEN profit_loss END) as avg_pnl
                FROM predictions 
                WHERE timestamp >= ? AND accuracy IS NOT NULL
                GROUP BY symbol
                ORDER BY accuracy DESC
            ''', (cutoff_date,))
            
            by_symbol = cursor.fetchall()
        
        return {
            'period_days': days_back,
            'total_predictions': overall[0],
            'evaluated_predictions': overall[1],
            'overall_accuracy': round(overall[2] * 100, 1) if overall[2] else 0,
            'average_pnl': round(overall[3], 2) if overall[3] else 0,
            'best_trade': round(overall[4], 2) if overall[4] else 0,
            'worst_trade': round(overall[5], 2) if overall[5] else 0,
            'by_signal_type': [
                {
                    'signal_type': row[0],
                    'count': row[1], 
                    'accuracy': round(row[2] * 100, 1) if row[2] else 0,
                    'avg_pnl': round(row[3], 2) if row[3] else 0
                } for row in by_signal
            ],
            'by_symbol': [
                {
                    'symbol': row[0],
                    'count': row[1],
                    'accuracy': round(row[2] * 100, 1) if row[2] else 0,
                    'avg_pnl': round(row[3], 2) if row[3] else 0
                } for row in by_symbol
            ]
        }

class BacktestEngine:
    """Comprehensive backtesting engine for HYPER signals"""
    
    def __init__(self, signal_engine: HYPERSignalEngine):
        self.signal_engine = signal_engine
        self.results = []
        logger.info("🧪 Backtest Engine initialized")
    
    def generate_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate realistic historical price data for backtesting"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='1D'
        )
        
        # Base prices (same as your current system)
        base_prices = {
            'QQQ': 485.20,
            'SPY': 525.75,
            'NVDA': 135.50,
            'AAPL': 190.80,
            'MSFT': 425.90
        }
        
        start_price = base_prices.get(symbol, 150.0)
        prices = []
        volumes = []
        
        current_price = start_price
        
        for date in dates:
            # Generate realistic daily movements
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% mean, 2% volatility
            
            # Add some trend and mean reversion
            if len(prices) > 5:
                recent_trend = (current_price / prices[-5] - 1)
                if recent_trend > 0.1:  # Mean revert if up >10%
                    daily_return -= 0.005
                elif recent_trend < -0.1:  # Bounce if down >10%
                    daily_return += 0.005
            
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            daily_range = abs(daily_return) * np.random.uniform(1.5, 3.0)
            high = current_price * (1 + daily_range/2)
            low = current_price * (1 - daily_range/2)
            open_price = prices[-1] if prices else current_price
            
            prices.append(current_price)
            
            # Generate volume
            base_volume = {'QQQ': 45000000, 'SPY': 85000000, 'NVDA': 350000000, 
                          'AAPL': 65000000, 'MSFT': 35000000}.get(symbol, 25000000)
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            volumes.append(volume)
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': [start_price] + prices[:-1],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    async def run_backtest(self, symbols: List[str], days: int = 30) -> BacktestMetrics:
        """Run comprehensive backtest across symbols and time period"""
        logger.info(f"🚀 Starting backtest: {symbols} over {days} days")
        
        all_predictions = []
        all_trades = []
        
        for symbol in symbols:
            logger.info(f"📊 Backtesting {symbol}...")
            
            # Generate historical data
            hist_data = self.generate_historical_data(symbol, days)
            
            # Run signals on each day
            for i in range(len(hist_data) - 1):
                current_day = hist_data.iloc[i]
                next_day = hist_data.iloc[i + 1]
                
                # Create mock quote data for signal generation
                quote_data = {
                    'symbol': symbol,
                    'price': current_day['close'],
                    'open': current_day['open'],
                    'high': current_day['high'],
                    'low': current_day['low'],
                    'volume': current_day['volume'],
                    'change': current_day['close'] - current_day['open'],
                    'change_percent': f"{((current_day['close'] / current_day['open'] - 1) * 100):.2f}",
                    'timestamp': current_day['date'].isoformat(),
                    'data_source': 'backtest'
                }
                
                try:
                    # Generate signal (mock the full signal generation)
                    signal = await self._generate_backtest_signal(symbol, quote_data)
                    
                    # Calculate actual outcome
                    actual_return = (next_day['close'] / current_day['close'] - 1) * 100
                    
                    # Determine accuracy
                    if signal.direction == "UP":
                        accuracy = 1.0 if actual_return > 0 else 0.0
                        trade_return = actual_return
                    elif signal.direction == "DOWN":
                        accuracy = 1.0 if actual_return < 0 else 0.0
                        trade_return = -actual_return  # Short position
                    else:  # NEUTRAL/HOLD
                        accuracy = 1.0 if abs(actual_return) < 1.0 else 0.0
                        trade_return = 0.0
                    
                    prediction = {
                        'date': current_day['date'],
                        'symbol': symbol,
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'direction': signal.direction,
                        'predicted_direction': signal.direction,
                        'actual_return': actual_return,
                        'trade_return': trade_return,
                        'accuracy': accuracy,
                        'technical_score': signal.technical_score,
                        'sentiment_score': signal.sentiment_score,
                        'ml_score': signal.ml_score
                    }
                    
                    all_predictions.append(prediction)
                    
                    if signal.signal_type != "HOLD":
                        all_trades.append(prediction)
                    
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol} on {current_day['date']}: {e}")
                    continue
        
        # Calculate comprehensive metrics
        metrics = self._calculate_backtest_metrics(all_predictions, all_trades)
        
        logger.info(f"✅ Backtest complete: {metrics.accuracy_percentage:.1f}% accuracy")
        return metrics
    
    async def _generate_backtest_signal(self, symbol: str, quote_data: Dict) -> HYPERSignal:
        """Generate a signal for backtesting (simplified version)"""
        # This is a simplified version - in practice you'd call your full signal engine
        
        # Mock technical analysis
        price = quote_data['price']
        change_pct = float(quote_data['change_percent'])
        
        technical_score = 50 + (change_pct * 10)  # Simple momentum
        technical_score = max(0, min(100, technical_score))
        
        # Mock sentiment (random but realistic)
        sentiment_score = np.random.normal(50, 15)
        sentiment_score = max(0, min(100, sentiment_score))
        
        # Mock ML score
        ml_score = np.random.normal(55, 20)
        ml_score = max(0, min(100, ml_score))
        
        # Calculate combined confidence
        confidence = (technical_score * 0.4 + sentiment_score * 0.3 + ml_score * 0.3)
        
        # Determine signal type and direction
        if confidence > 75:
            signal_type = "HYPER_BUY" if change_pct > 0 else "HYPER_SELL"
            direction = "UP" if change_pct > 0 else "DOWN"
        elif confidence > 60:
            signal_type = "SOFT_BUY" if change_pct > 0 else "SOFT_SELL"
            direction = "UP" if change_pct > 0 else "DOWN"
        else:
            signal_type = "HOLD"
            direction = "NEUTRAL"
        
        return HYPERSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            direction=direction,
            price=price,
            timestamp=quote_data['timestamp'],
            technical_score=technical_score,
            momentum_score=technical_score,  # Simplified
            trends_score=sentiment_score,
            volume_score=50.0,
            ml_score=ml_score,
            sentiment_score=sentiment_score,
            reasons=[f"Technical: {technical_score:.1f}", f"ML: {ml_score:.1f}"],
            warnings=[],
            data_quality="backtest"
        )
    
    def _calculate_backtest_metrics(self, predictions: List[Dict], trades: List[Dict]) -> BacktestMetrics:
        """Calculate comprehensive backtest performance metrics"""
        if not predictions:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NONE", "NONE", 0)
        
        df = pd.DataFrame(predictions)
        
        # Basic accuracy metrics
        total_predictions = len(predictions)
        correct_predictions = df['accuracy'].sum()
        accuracy_percentage = (correct_predictions / total_predictions) * 100
        
        # Signal type performance
        signal_performance = df.groupby('signal_type')['accuracy'].mean()
        best_signal = signal_performance.idxmax() if len(signal_performance) > 0 else "NONE"
        worst_signal = signal_performance.idxmin() if len(signal_performance) > 0 else "NONE"
        
        # Trading metrics
        if trades:
            trade_df = pd.DataFrame(trades)
            returns = trade_df['trade_return'].values
            
            win_rate = (returns > 0).mean() * 100
            avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
            
            total_return = returns.sum()
            
            # Sharpe ratio (simplified)
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max)
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        else:
            win_rate = avg_win = avg_loss = total_return = 0
            sharpe_ratio = max_drawdown = profit_factor = 0
        
        # Confidence correlation
        confidence_correlation = df['confidence'].corr(df['accuracy']) if len(df) > 1 else 0
        
        return BacktestMetrics(
            total_predictions=total_predictions,
            correct_predictions=int(correct_predictions),
            accuracy_percentage=round(accuracy_percentage, 2),
            precision=round(accuracy_percentage / 100, 3),  # Simplified
            recall=round(accuracy_percentage / 100, 3),     # Simplified
            f1_score=round(accuracy_percentage / 100, 3),   # Simplified
            sharpe_ratio=round(sharpe_ratio, 3),
            max_drawdown=round(max_drawdown, 2),
            total_return=round(total_return, 2),
            win_rate=round(win_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            best_signal_type=best_signal,
            worst_signal_type=worst_signal,
            confidence_correlation=round(confidence_correlation, 3)
        )

class ModelTester:
    """Main testing interface for HYPER predictive models"""
    
    def __init__(self, signal_engine: HYPERSignalEngine):
        self.signal_engine = signal_engine
        self.tracker = PredictionTracker()
        self.backtest_engine = BacktestEngine(signal_engine)
        logger.info("🧪 Model Tester initialized")
    
    async def start_live_testing(self):
        """Start live prediction tracking"""
        logger.info("🚀 Starting live prediction testing...")
        
        while True:
            try:
                # Generate signals for all tickers
                signals = await self.signal_engine.generate_all_signals()
                
                # Record predictions
                for symbol, signal in signals.items():
                    if signal.confidence > 60:  # Only track confident predictions
                        prediction_id = self.tracker.record_prediction(signal)
                        logger.info(f"📝 Tracking prediction: {prediction_id}")
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in live testing: {e}")
                await asyncio.sleep(60)
    
    async def run_backtest_suite(self) -> Dict[str, Any]:
        """Run comprehensive backtest suite"""
        logger.info("🧪 Running comprehensive backtest suite...")
        
        # Test different time periods
        results = {}
        
        for days in [7, 14, 30]:
            logger.info(f"📊 Testing {days}-day backtest...")
            metrics = await self.backtest_engine.run_backtest(
                symbols=['QQQ', 'SPY', 'NVDA', 'AAPL', 'MSFT'],
                days=days
            )
            results[f'{days}_days'] = asdict(metrics)
        
        # Get live prediction stats
        live_stats = self.tracker.get_prediction_stats(days_back=30)
        results['live_predictions'] = live_stats
        
        return results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report = f"""
# 🧪 HYPER Predictive Model Test Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Backtest Results Summary

"""
        
        for period, metrics in results.items():
            if period.endswith('_days'):
                days = period.replace('_days', '')
                report += f"""
### {days}-Day Backtest
- **Accuracy**: {metrics['accuracy_percentage']:.1f}%
- **Total Predictions**: {metrics['total_predictions']}
- **Win Rate**: {metrics['win_rate']:.1f}%
- **Total Return**: {metrics['total_return']:.2f}%
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}
- **Max Drawdown**: {metrics['max_drawdown']:.2f}%
- **Best Signal Type**: {metrics['best_signal_type']}
- **Confidence Correlation**: {metrics['confidence_correlation']:.3f}

"""
        
        if 'live_predictions' in results:
            live = results['live_predictions']
            report += f"""
## 🔴 Live Prediction Performance (30 days)
- **Total Predictions**: {live['total_predictions']}
- **Evaluated**: {live['evaluated_predictions']}
- **Overall Accuracy**: {live['overall_accuracy']:.1f}%
- **Average P&L**: ${live['average_pnl']:.2f}
- **Best Trade**: ${live['best_trade']:.2f}
- **Worst Trade**: ${live['worst_trade']:.2f}

### Performance by Signal Type
"""
            
            for signal_data in live['by_signal_type']:
                report += f"- **{signal_data['signal_type']}**: {signal_data['accuracy']:.1f}% accuracy, ${signal_data['avg_pnl']:.2f} avg P&L\n"
        
        return report

# ============================================
# TESTING API ENDPOINTS
# ============================================

class TestingAPI:
    """API endpoints for model testing"""
    
    def __init__(self, model_tester: ModelTester):
        self.tester = model_tester
    
    async def get_test_status(self) -> Dict[str, Any]:
        """Get current testing status"""
        stats = self.tester.tracker.get_prediction_stats(days_back=7)
        return {
            'status': 'active',
            'recent_stats': stats,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_quick_backtest(self, days: int = 7) -> Dict[str, Any]:
        """Run quick backtest for immediate results"""
        metrics = await self.tester.backtest_engine.run_backtest(
            symbols=['QQQ', 'NVDA'], 
            days=days
        )
        return asdict(metrics)
    
    async def get_prediction_history(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """Get prediction history for analysis"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.tester.tracker.db_path) as conn:
            if symbol:
                cursor = conn.execute('''
                    SELECT * FROM predictions 
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (symbol, cutoff_date))
            else:
                cursor = conn.execute('''
                    SELECT * FROM predictions 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (cutoff_date,))
            
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return results

# ============================================
# USAGE EXAMPLE
# ============================================

async def main():
    """Example usage of the testing framework"""
    
    # Initialize testing system
    from signal_engine import HYPERSignalEngine
    signal_engine = HYPERSignalEngine()
    model_tester = ModelTester(signal_engine)
    
    # Run comprehensive backtest
    results = await model_tester.run_backtest_suite()
    
    # Generate report
    report = model_tester.generate_test_report(results)
    print(report)
    
    # Save results
    with open(f'hyper_test_report_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
