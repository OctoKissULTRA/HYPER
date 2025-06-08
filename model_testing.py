import logging
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    prediction_time = Column(DateTime)
    model_version = Column(String)

class PredictionTracker:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL", "sqlite:///predictions.db")
        if db_url.startswith("postgres"):
            db_url = db_url.replace("postgres://", "postgresql://")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_prediction(self, symbol: str, predicted_price: float, prediction_time: datetime, model_version: str):
        prediction = Prediction(
            symbol=symbol,
            predicted_price=predicted_price,
            prediction_time=prediction_time,
            model_version=model_version
        )
        self.session.add(prediction)
        self.session.commit()

    def update_actual_price(self, prediction_id: int, actual_price: float):
        prediction = self.session.query(Prediction).filter_by(id=prediction_id).first()
        if prediction:
            prediction.actual_price = actual_price
            self.session.commit()

    def get_all_predictions(self) -> List[Prediction]:
        return self.session.query(Prediction).all()

class ModelTester:
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }

    def evaluate_prediction(self, prediction: Prediction) -> Dict[str, Any]:
        if prediction.actual_price is None:
            return {"status": "pending"}
        predicted = prediction.predicted_price
        actual = prediction.actual_price
        direction_correct = (actual >= predicted) == (actual >= prediction.predicted_price)
        return {
            "symbol": prediction.symbol,
            "direction_correct": direction_correct,
            "percentage_error": abs(actual - predicted) / actual * 100
        }

    def run_backtest(self, symbol: str, predictions: List[Prediction]) -> Dict[str, Any]:
        correct = 0
        total = 0
        returns = []
        for pred in predictions:
            eval_result = self.evaluate_prediction(pred)
            if eval_result["status"] == "pending":
                continue
            if eval_result["direction_correct"]:
                correct += 1
            total += 1
            if pred.actual_price and pred.predicted_price:
                returns.append((pred.actual_price - pred.predicted_price) / pred.predicted_price)
        accuracy = correct / total if total > 0 else 0.0
        sharpe_ratio = sum(returns) / len(returns) if returns else 0.0
        self.metrics.update({"accuracy": accuracy, "sharpe_ratio": sharpe_ratio})
        return self.metrics

class TestingAPI:
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester

    async def get_backtest_results(self, symbol: str) -> Dict[str, Any]:
        predictions = self.model_tester.tracker.get_all_predictions()
        return self.model_tester.run_backtest(symbol, predictions)

    async def get_metrics(self) -> Dict[str, Any]:
        return self.model_tester.metrics
