import asyncio
import logging
import numpy as np
import pandas as pd
import sqlite3
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import queue

# ML Libraries with graceful fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ========================================
# ML LEARNING FRAMEWORK
# ========================================

@dataclass
class TrainingExample:
    """Single training example for ML models"""
    timestamp: str
    symbol: str
    
    # Features (inputs)
    technical_indicators: Dict[str, float]
    market_conditions: Dict[str, float]
    sentiment_data: Dict[str, float]
    price_history: List[float]
    volume_data: List[float]
    
    # Labels (outputs) - what we're trying to predict
    actual_direction: str  # UP, DOWN, NEUTRAL
    actual_price_change: float  # percentage change
    actual_volatility: float  # realized volatility
    signal_accuracy: float  # was our signal correct?
    
    # Metadata
    prediction_horizon: int = 1  # days ahead
    confidence_at_time: float = 50.0

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]
    training_samples: int
    last_updated: str
    prediction_confidence: float

class FeatureEngineer:
    """Create features for ML models from market data"""
    
    def __init__(self):
        self.feature_names = []
        logger.info("🔧 Feature Engineer initialized")
    
    def extract_features(self, signal_data: Dict, price_history: List[float] = None, 
                        volume_history: List[int] = None) -> np.ndarray:
        """Extract comprehensive features for ML training"""
        
        features = []
        feature_names = []
        
        # 1. TECHNICAL INDICATORS
        technical = signal_data.get('technical_indicators', {})
        
        # Basic technical features
        features.extend([
            technical.get('rsi', 50),
            technical.get('williams_r', -50),
            technical.get('stochastic_k', 50),
            technical.get('stochastic_d', 50),
            technical.get('volume_ratio', 1.0),
            technical.get('range_percent', 0)
        ])
        feature_names.extend([
            'rsi', 'williams_r', 'stochastic_k', 'stochastic_d', 'volume_ratio', 'range_percent'
        ])
        
        # 2. PRICE-BASED FEATURES
        if price_history and len(price_history) >= 5:
            recent_prices = price_history[-10:]
            
            # Price momentum features
            returns_1d = (recent_prices[-1] / recent_prices[-2] - 1) if len(recent_prices) >= 2 else 0
            returns_5d = (recent_prices[-1] / recent_prices[-5] - 1) if len(recent_prices) >= 5 else 0
            
            # Volatility features
            price_changes = [recent_prices[i] / recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
            volatility = np.std(price_changes) if price_changes else 0
            
            # Trend features
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0] if len(recent_prices) > 1 else 0
            
            features.extend([returns_1d, returns_5d, volatility, price_trend])
            feature_names.extend(['returns_1d', 'returns_5d', 'volatility', 'price_trend'])
        else:
            features.extend([0, 0, 0, 0])
            feature_names.extend(['returns_1d', 'returns_5d', 'volatility', 'price_trend'])
        
        # 3. VOLUME FEATURES
        if volume_history and len(volume_history) >= 3:
            volume_trend = np.polyfit(range(len(volume_history)), volume_history, 1)[0]
            volume_ratio = volume_history[-1] / np.mean(volume_history) if volume_history else 1
            
            features.extend([volume_trend, volume_ratio])
            feature_names.extend(['volume_trend', 'volume_ratio_advanced'])
        else:
            features.extend([0, 1])
            feature_names.extend(['volume_trend', 'volume_ratio_advanced'])
        
        # 4. SENTIMENT FEATURES
        sentiment = signal_data.get('sentiment_data', {})
        features.extend([
            sentiment.get('overall_sentiment_score', 50),
            sentiment.get('news_sentiment', 50),
            sentiment.get('social_sentiment', 50),
            sentiment.get('overall_confidence', 0.5) * 100
        ])
        feature_names.extend(['sentiment_overall', 'sentiment_news', 'sentiment_social', 'sentiment_confidence'])
        
        # 5. MARKET STRUCTURE FEATURES
        market = signal_data.get('market_data', {})
        features.extend([
            market.get('vix_sentiment', 20),
            market.get('market_breadth', 50),
            market.get('sector_rotation_score', 50)
        ])
        feature_names.extend(['vix_sentiment', 'market_breadth', 'sector_rotation'])
        
        # 6. TIME-BASED FEATURES
        now = datetime.now()
        features.extend([
            now.hour,  # Hour of day
            now.weekday(),  # Day of week
            now.month,  # Month of year
            1 if 9 <= now.hour <= 16 else 0  # Market hours
        ])
        feature_names.extend(['hour', 'weekday', 'month', 'market_hours'])
        
        # Store feature names for interpretation
        self.feature_names = feature_names
        
        return np.array(features).reshape(1, -1)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for model interpretation"""
        return self.feature_names

class AdaptiveLearningEngine:
    """Main ML learning engine with continuous adaptation"""
    
    def __init__(self, model_save_path: str = "hyper_models"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        self.feature_engineer = FeatureEngineer()
        self.training_queue = queue.Queue()
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        
        # Initialize models
        self._initialize_models()
        
        # Start background training thread
        self.training_active = True
        self.training_thread = threading.Thread(target=self._background_training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"🧠 Adaptive Learning Engine initialized (ML Available: {ML_AVAILABLE})")
    
    def _initialize_models(self):
        """Initialize ML models with fallbacks"""
        if ML_AVAILABLE:
            self.models = {
                'direction_classifier': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ),
                'confidence_predictor': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'volatility_predictor': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42
                ),
                'ensemble_voter': LogisticRegression(
                    random_state=42,
                    class_weight='balanced'
                )
            }
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
        else:
            # Fallback models (statistical)
            self.models = {
                'direction_classifier': self._fallback_direction_model,
                'confidence_predictor': self._fallback_confidence_model,
                'volatility_predictor': self._fallback_volatility_model,
                'ensemble_voter': self._fallback_ensemble_model
            }
        
        logger.info(f"✅ Initialized {len(self.models)} ML models")
    
    def add_training_example(self, example: TrainingExample):
        """Add new training example to the learning queue"""
        try:
            self.training_queue.put(example, timeout=1.0)
            logger.debug(f"📚 Added training example: {example.symbol} {example.actual_direction}")
        except queue.Full:
            logger.warning("⚠️ Training queue full - dropping example")
    
    def _background_training_loop(self):
        """Background thread for continuous model training"""
        examples_buffer = []
        last_training = datetime.now()
        
        while self.training_active:
            try:
                # Collect examples
                while not self.training_queue.empty() and len(examples_buffer) < 100:
                    try:
                        example = self.training_queue.get(timeout=1.0)
                        examples_buffer.append(example)
                    except queue.Empty:
                        break
                
                # Train if we have enough examples or enough time has passed
                should_train = (
                    len(examples_buffer) >= 50 or  # Enough examples
                    (len(examples_buffer) > 10 and 
                     (datetime.now() - last_training).total_seconds() > 3600)  # 1 hour
                )
                
                if should_train and examples_buffer:
                    logger.info(f"🎓 Training models with {len(examples_buffer)} examples...")
                    self._train_models(examples_buffer)
                    examples_buffer = []
                    last_training = datetime.now()
                
                # Sleep before next iteration
                threading.Event().wait(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in background training: {e}")
                threading.Event().wait(60)  # Wait longer on error
    
    def _train_models(self, examples: List[TrainingExample]):
        """Train all models with new examples"""
        if not ML_AVAILABLE:
            logger.info("📊 ML libraries not available - using statistical updates")
            return
        
        try:
            # Prepare training data
            X, y_direction, y_confidence, y_volatility = self._prepare_training_data(examples)
            
            if len(X) < 10:
                logger.warning("⚠️ Not enough training data - skipping training")
                return
            
            # Train direction classifier
            if len(set(y_direction)) > 1:  # Need multiple classes
                X_scaled = self.scalers['direction_classifier'].fit_transform(X)
                self.models['direction_classifier'].fit(X_scaled, y_direction)
                
                # Evaluate
                scores = cross_val_score(self.models['direction_classifier'], X_scaled, y_direction, cv=3)
                direction_accuracy = np.mean(scores)
                
                logger.info(f"📈 Direction classifier accuracy: {direction_accuracy:.3f}")
            
            # Train confidence predictor
            X_scaled = self.scalers['confidence_predictor'].fit_transform(X)
            self.models['confidence_predictor'].fit(X_scaled, y_confidence)
            
            # Train volatility predictor
            volatility_classes = self._discretize_volatility(y_volatility)
            if len(set(volatility_classes)) > 1:
                X_scaled = self.scalers['volatility_predictor'].fit_transform(X)
                self.models['volatility_predictor'].fit(X_scaled, volatility_classes)
            
            # Save models
            self._save_models()
            
            # Update performance tracking
            self._update_performance_metrics(X, y_direction, y_confidence, y_volatility)
            
            logger.info(f"✅ Models trained successfully with {len(examples)} examples")
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
    
    def _prepare_training_data(self, examples: List[TrainingExample]) -> Tuple[np.ndarray, List, List, List]:
        """Convert training examples to ML-ready format"""
        X = []
        y_direction = []
        y_confidence = []
        y_volatility = []
        
        for example in examples:
            try:
                # Create feature vector
                signal_data = {
                    'technical_indicators': example.technical_indicators,
                    'sentiment_data': example.sentiment_data,
                    'market_data': example.market_conditions
                }
                
                features = self.feature_engineer.extract_features(
                    signal_data, 
                    example.price_history, 
                    example.volume
