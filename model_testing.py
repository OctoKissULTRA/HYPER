import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# ========================================
# MODEL TESTING FRAMEWORK
# ========================================

@dataclass
class PredictionRecord:
    """Record of a prediction for testing purposes""""
    prediction_id: str
    timestamp: datetime
    symbol: str
    signal_type: str
    confidence: float
    direction: str
    price: float
    technical_score: float
    sentiment_score: float
    prediction_horizon: int = 1  # Days
    actual_outcome: Optional[str] = None
    actual_price_change: Optional[float] = None
    actual_direction: Optional[str] = None
    was_correct: Optional[bool] = None
    evaluation_timestamp: Optional[datetime] = None

@dataclass
class BacktestResult:
    """Results from backtesting""""
    start_date: datetime
    end_date: datetime
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision_by_signal: Dict[str, float]
    confidence_accuracy: Dict[str, float]
    symbol_performance: Dict[str, Dict]
    avg_confidence: float
    profitable_signals_percent: float
    sharpe_ratio: float
    max_drawdown: float

class PredictionTracker:
    """Track and evaluate predictions""""
    
    def __init__(self, db_path: str = "hyper_predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for predictions""""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(""""
                    CREATE TABLE IF NOT EXISTS predictions (
                        prediction_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        direction TEXT NOT NULL,
                        price REAL NOT NULL,
                        technical_score REAL,
                        sentiment_score REAL,
                        prediction_horizon INTEGER DEFAULT 1,
                        actual_outcome TEXT,
                        actual_price_change REAL,
                        actual_direction TEXT,
                        was_correct INTEGER,
                        evaluation_timestamp TEXT
                    )
                """)
                
                conn.execute(""""
                    CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                    ON predictions(symbol, timestamp)
                """)
                
                conn.execute(""""
                    CREATE INDEX IF NOT EXISTS idx_evaluation 
                    ON predictions(evaluation_timestamp)
                """)
                
            logger.info("✅ Prediction database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize prediction database: {e}")
    
    def record_prediction(self, signal) -> str:
        """Record a new prediction""""
        try:
            prediction_id = f"{signal.symbol}_{signal.timestamp}_{signal.signal_type}""
            
            prediction = PredictionRecord(
                prediction_id=prediction_id,
                timestamp=datetime.fromisoformat(signal.timestamp) if isinstance(signal.timestamp, str) else signal.timestamp,
                symbol=signal.symbol,
                signal_type=getattr(signal, 'signal_type', 'HOLD'),
                confidence=float(getattr(signal, 'confidence', 0.0)),
                direction=getattr(signal, 'direction', 'NEUTRAL'),
                price=float(getattr(signal, 'price', 0.0)),
                technical_score=float(getattr(signal, 'technical_score', 50.0)),
                sentiment_score=float(getattr(signal, 'sentiment_score', 50.0))
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(""""
                    INSERT OR REPLACE INTO predictions 
                    (prediction_id, timestamp, symbol, signal_type, confidence, direction, 
                     price, technical_score, sentiment_score, prediction_horizon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_id,
                    prediction.timestamp.isoformat(),
                    prediction.symbol,
                    prediction.signal_type,
                    prediction.confidence,
                    prediction.direction,
                    prediction.price,
                    prediction.technical_score,
                    prediction.sentiment_score,
                    prediction.prediction_horizon
                ))
            
            logger.debug(f"📝 Recorded prediction: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record prediction: {e}")
            return """
    
    def evaluate_prediction(self, prediction_id: str, actual_price: float, actual_direction: str):
        """Evaluate a prediction against actual outcome""""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get original prediction
                cursor = conn.execute(""""
                    SELECT * FROM predictions WHERE prediction_id = ?
                """, (prediction_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Prediction {prediction_id} not found")
                    return
                
                # Calculate outcome
                original_price = row[6]  # price column
                predicted_direction = row[5]  # direction column
                predicted_signal = row[3]  # signal_type column
                
                price_change = (actual_price - original_price) / original_price
                
                # Determine if prediction was correct
                was_correct = False
                if predicted_direction == actual_direction:
                    was_correct = True
                elif predicted_direction == "UP" and price_change > 0.01:  # >1% gain
                    was_correct = True
                elif predicted_direction == "DOWN" and price_change < -0.01:  # >1% loss
                    was_correct = True
                elif predicted_direction == "NEUTRAL" and abs(price_change) < 0.01:  # <1% change
                    was_correct = True
                
                # Update database
                conn.execute(""""
                    UPDATE predictions 
                    SET actual_price_change = ?, actual_direction = ?, 
                        was_correct = ?, evaluation_timestamp = ?
                    WHERE prediction_id = ?
                """, (
                    price_change,
                    actual_direction,
                    1 if was_correct else 0,
                    datetime.now().isoformat(),
                    prediction_id
                ))
                
            logger.debug(f"✅ Evaluated prediction {prediction_id}: {'✓' if was_correct else '✗'}")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate prediction {prediction_id}: {e}")
    
    def get_recent_predictions(self, days: int = 30) -> List[PredictionRecord]:
        """Get recent predictions""""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(""""
                    SELECT * FROM predictions 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, (since_date.isoformat(),))
                
                predictions = []
                for row in cursor.fetchall():
                    prediction = PredictionRecord(
                        prediction_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        symbol=row[2],
                        signal_type=row[3],
                        confidence=row[4],
                        direction=row[5],
                        price=row[6],
                        technical_score=row[7] or 50.0,
                        sentiment_score=row[8] or 50.0,
                        prediction_horizon=row[9] or 1,
                        actual_outcome=row[10],
                        actual_price_change=row[11],
                        actual_direction=row[12],
                        was_correct=bool(row[13]) if row[13] is not None else None,
                        evaluation_timestamp=datetime.fromisoformat(row[14]) if row[14] else None
                    )
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"❌ Failed to get recent predictions: {e}")
            return []

class ModelTester:
    """Main model testing class""""
    
    def __init__(self, signal_engine):
        self.signal_engine = signal_engine
        self.tracker = PredictionTracker()
        logger.info("✅ Model tester initialized")
    
    async def run_backtest_suite(self, days: int = 30) -> Dict[str, Any]:
        """Run comprehensive backtest""""
        logger.info(f"🧪 Running backtest suite for {days} days...")
        
        try:
            # Get predictions to evaluate
            predictions = self.tracker.get_recent_predictions(days)
            
            if not predictions:
                return {
                    "status": "no_data",
                    "message": f"No predictions found for the last {days} days""
                }
            
            # Calculate basic metrics
            evaluated_predictions = [p for p in predictions if p.was_correct is not None]
            
            if not evaluated_predictions:
                return {
                    "status": "no_evaluated_data",
                    "message": "No evaluated predictions found",
                    "total_predictions": len(predictions)
                }
            
            # Calculate accuracy
            correct_count = sum(1 for p in evaluated_predictions if p.was_correct)
            total_count = len(evaluated_predictions)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            # Accuracy by signal type
            signal_accuracy = {}
            for signal_type in ['HYPER_BUY', 'SOFT_BUY', 'HOLD', 'SOFT_SELL', 'HYPER_SELL']:
                signal_preds = [p for p in evaluated_predictions if p.signal_type == signal_type]
                if signal_preds:
                    correct_signal = sum(1 for p in signal_preds if p.was_correct)
                    signal_accuracy[signal_type] = correct_signal / len(signal_preds)
            
            # Accuracy by confidence level
            confidence_accuracy = {}
            confidence_ranges = [(0, 50), (50, 70), (70, 85), (85, 100)]
            for low, high in confidence_ranges:
                range_preds = [p for p in evaluated_predictions if low <= p.confidence < high]
                if range_preds:
                    correct_range = sum(1 for p in range_preds if p.was_correct)
                    confidence_accuracy[f"{low}-{high}%"] = correct_range / len(range_preds)
            
            # Symbol performance
            symbol_performance = {}
            for symbol in set(p.symbol for p in evaluated_predictions):
                symbol_preds = [p for p in evaluated_predictions if p.symbol == symbol]
                correct_symbol = sum(1 for p in symbol_preds if p.was_correct)
                
                # Calculate average return for correct predictions
                correct_symbol_preds = [p for p in symbol_preds if p.was_correct and p.actual_price_change]
                avg_return = np.mean([p.actual_price_change for p in correct_symbol_preds]) if correct_symbol_preds else 0
                
                symbol_performance[symbol] = {
                    "accuracy": correct_symbol / len(symbol_preds),
                    "total_predictions": len(symbol_preds),
                    "correct_predictions": correct_symbol,
                    "average_return": avg_return
                }
            
            # Calculate financial metrics
            returns = [p.actual_price_change for p in evaluated_predictions if p.actual_price_change and p.was_correct]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
            
            # Max drawdown calculation
            cumulative_returns = np.cumsum(returns) if returns else [0]
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            result = BacktestResult(
                start_date=min(p.timestamp for p in evaluated_predictions),
                end_date=max(p.timestamp for p in evaluated_predictions),
                total_predictions=total_count,
                correct_predictions=correct_count,
                accuracy=accuracy,
                precision_by_signal=signal_accuracy,
                confidence_accuracy=confidence_accuracy,
                symbol_performance=symbol_performance,
                avg_confidence=np.mean([p.confidence for p in evaluated_predictions]),
                profitable_signals_percent=len(returns) / total_count * 100 if total_count > 0 else 0,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            logger.info(f"✅ Backtest completed: {accuracy:.1%} accuracy over {total_count} predictions")
            
            return {
                "status": "success",
                "backtest_result": asdict(result),
                "summary": {
                    "accuracy": f"{accuracy:.1%}",
                    "total_predictions": total_count,
                    "correct_predictions": correct_count,
                    "best_performing_symbol": max(symbol_performance.items(), key=lambda x: x[1]["accuracy"])[0] if symbol_performance else None,
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "max_drawdown": f"{max_drawdown:.1%}""
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

class TestingAPI:
    """API interface for testing functionality""""
    
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        logger.info("✅ Testing API initialized")
    
    async def get_test_status(self) -> Dict[str, Any]:
        """Get current testing status""""
        try:
            recent_predictions = self.model_tester.tracker.get_recent_predictions(7)
            evaluated_count = sum(1 for p in recent_predictions if p.was_correct is not None)
            
            return {
                "status": "active",
                "recent_predictions": len(recent_predictions),
                "evaluated_predictions": evaluated_count,
                "evaluation_rate": evaluated_count / len(recent_predictions) if recent_predictions else 0,
                "database_path": self.model_tester.tracker.db_path,
                "last_prediction": recent_predictions[0].timestamp.isoformat() if recent_predictions else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get test status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_quick_backtest(self, days: int = 7) -> Dict[str, Any]:
        """Run quick backtest""""
        return await self.model_tester.run_backtest_suite(days)
    
    async def get_prediction_history(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get prediction history""""
        try:
            predictions = self.model_tester.tracker.get_recent_predictions(days)
            
            if symbol:
                predictions = [p for p in predictions if p.symbol == symbol]
            
            prediction_data = []
            for p in predictions:
                prediction_data.append({
                    "prediction_id": p.prediction_id,
                    "timestamp": p.timestamp.isoformat(),
                    "symbol": p.symbol,
                    "signal_type": p.signal_type,
                    "confidence": p.confidence,
                    "direction": p.direction,
                    "price": p.price,
                    "was_correct": p.was_correct,
                    "actual_price_change": p.actual_price_change,
                    "evaluated": p.was_correct is not None
                })
            
            return {
                "status": "success",
                "predictions": prediction_data,
                "total_count": len(prediction_data),
                "symbol_filter": symbol,
                "days": days
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction history: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup testing resources""""
        logger.info("🧹 Cleaning up testing resources...")
        # Add any cleanup logic here if needed