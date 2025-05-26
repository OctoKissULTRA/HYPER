# ============================================
# HYPER ML LEARNING SYSTEM
# Adaptive machine learning with continuous improvement
# ============================================

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

# ML Libraries (with graceful fallbacks)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("⚠️ ML libraries not available - using fallback models")

logger = logging.getLogger(__name__)

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
                    example.volume_data
                )
                
                X.append(features.flatten())
                
                # Labels
                y_direction.append(example.actual_direction)
                y_confidence.append(example.signal_accuracy)
                y_volatility.append(example.actual_volatility)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing example: {e}")
                continue
        
        return np.array(X), y_direction, y_confidence, y_volatility
    
    def _discretize_volatility(self, volatilities: List[float]) -> List[str]:
        """Convert continuous volatility to discrete classes"""
        classes = []
        for vol in volatilities:
            if vol < 0.01:  # < 1%
                classes.append('LOW')
            elif vol < 0.03:  # < 3%
                classes.append('MEDIUM')
            else:
                classes.append('HIGH')
        return classes
    
    def predict(self, signal_data: Dict, price_history: List[float] = None, 
                volume_history: List[int] = None) -> Dict[str, Any]:
        """Make predictions using trained models"""
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                signal_data, price_history, volume_history
            )
            
            predictions = {}
            
            if ML_AVAILABLE and all(model in self.models for model in ['direction_classifier', 'confidence_predictor']):
                # Direction prediction
                direction_features = self.scalers['direction_classifier'].transform(features)
                direction_probs = self.models['direction_classifier'].predict_proba(direction_features)[0]
                direction_pred = self.models['direction_classifier'].predict(direction_features)[0]
                
                predictions['direction'] = {
                    'prediction': direction_pred,
                    'confidence': float(np.max(direction_probs)),
                    'probabilities': {
                        'UP': float(direction_probs[2]) if len(direction_probs) > 2 else 0.33,
                        'DOWN': float(direction_probs[0]) if len(direction_probs) > 0 else 0.33,
                        'NEUTRAL': float(direction_probs[1]) if len(direction_probs) > 1 else 0.34
                    }
                }
                
                # Confidence prediction
                confidence_features = self.scalers['confidence_predictor'].transform(features)
                confidence_pred = self.models['confidence_predictor'].predict(confidence_features)[0]
                
                predictions['confidence'] = {
                    'predicted_accuracy': float(np.clip(confidence_pred, 0, 1)),
                    'model_confidence': float(0.7)  # Model's confidence in its prediction
                }
                
                # Volatility prediction
                if 'volatility_predictor' in self.scalers:
                    volatility_features = self.scalers['volatility_predictor'].transform(features)
                    volatility_pred = self.models['volatility_predictor'].predict(volatility_features)[0]
                    
                    predictions['volatility'] = {
                        'prediction': volatility_pred,
                        'expected_range': self._volatility_to_range(volatility_pred)
                    }
                
                # Feature importance
                if hasattr(self.models['direction_classifier'], 'feature_importances_'):
                    feature_names = self.feature_engineer.get_feature_names()
                    importance = self.models['direction_classifier'].feature_importances_
                    
                    top_features = sorted(
                        zip(feature_names, importance), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    predictions['feature_importance'] = [
                        {'feature': name, 'importance': float(imp)} 
                        for name, imp in top_features
                    ]
            
            else:
                # Fallback predictions
                predictions = self._generate_fallback_predictions(signal_data)
            
            predictions['model_type'] = 'sklearn_ensemble' if ML_AVAILABLE else 'statistical_fallback'
            predictions['timestamp'] = datetime.now().isoformat()
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return self._generate_fallback_predictions(signal_data)
    
    def _generate_fallback_predictions(self, signal_data: Dict) -> Dict[str, Any]:
        """Generate predictions when ML models are unavailable"""
        
        # Simple statistical approach
        technical = signal_data.get('technical_indicators', {})
        sentiment = signal_data.get('sentiment_data', {})
        
        # Direction based on technical + sentiment
        tech_score = technical.get('rsi', 50)
        sent_score = sentiment.get('overall_sentiment_score', 50)
        combined_score = (tech_score + sent_score) / 2
        
        if combined_score > 60:
            direction = 'UP'
            confidence = 0.6 + (combined_score - 60) / 100
        elif combined_score < 40:
            direction = 'DOWN'
            confidence = 0.6 + (40 - combined_score) / 100
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'direction': {
                'prediction': direction,
                'confidence': confidence,
                'probabilities': {
                    'UP': 0.6 if direction == 'UP' else 0.2,
                    'DOWN': 0.6 if direction == 'DOWN' else 0.2,
                    'NEUTRAL': 0.6 if direction == 'NEUTRAL' else 0.2
                }
            },
            'confidence': {
                'predicted_accuracy': confidence,
                'model_confidence': 0.5
            },
            'volatility': {
                'prediction': 'MEDIUM',
                'expected_range': ['-2%', '+2%']
            },
            'feature_importance': [
                {'feature': 'rsi', 'importance': 0.3},
                {'feature': 'sentiment_overall', 'importance': 0.25}
            ]
        }
    
    def _volatility_to_range(self, volatility_class: str) -> List[str]:
        """Convert volatility class to expected price range"""
        ranges = {
            'LOW': ['-1%', '+1%'],
            'MEDIUM': ['-2%', '+2%'],
            'HIGH': ['-4%', '+4%']
        }
        return ranges.get(volatility_class, ['-2%', '+2%'])
    
    def _save_models(self):
        """Save trained models to disk"""
        if not ML_AVAILABLE:
            return
        
        try:
            for name, model in self.models.items():
                if hasattr(model, 'fit'):  # Sklearn model
                    model_path = self.model_save_path / f"{name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    scaler_path = self.model_save_path / f"{name}_scaler.pkl"
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[name], f)
            
            logger.info(f"💾 Models saved to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        if not ML_AVAILABLE:
            return
        
        try:
            for name in self.models.keys():
                model_path = self.model_save_path / f"{name}.pkl"
                scaler_path = self.model_save_path / f"{name}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
            
            logger.info(f"📂 Models loaded from {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def _update_performance_metrics(self, X: np.ndarray, y_direction: List, 
                                   y_confidence: List, y_volatility: List):
        """Update model performance tracking"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Direction classifier metrics
            X_scaled = self.scalers['direction_classifier'].transform(X)
            direction_pred = self.models['direction_classifier'].predict(X_scaled)
            direction_accuracy = accuracy_score(y_direction, direction_pred)
            
            # Confidence predictor metrics
            X_scaled = self.scalers['confidence_predictor'].transform(X)
            confidence_pred = self.models['confidence_predictor'].predict(X_scaled)
            confidence_mse = np.mean((np.array(y_confidence) - confidence_pred) ** 2)
            
            # Store performance
            self.performance_history[datetime.now().isoformat()] = {
                'direction_accuracy': direction_accuracy,
                'confidence_mse': confidence_mse,
                'training_samples': len(X),
                'feature_count': X.shape[1]
            }
            
            logger.info(f"📊 Performance updated - Direction: {direction_accuracy:.3f}, Confidence MSE: {confidence_mse:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of ML models"""
        
        status = {
            'ml_available': ML_AVAILABLE,
            'models_initialized': len(self.models) > 0,
            'training_active': self.training_active,
            'training_queue_size': self.training_queue.qsize(),
            'models': {}
        }
        
        for name, model in self.models.items():
            if ML_AVAILABLE and hasattr(model, 'fit'):
                # Check if model is trained
                is_trained = hasattr(model, 'classes_') or hasattr(model, 'feature_importances_')
                status['models'][name] = {
                    'type': type(model).__name__,
                    'trained': is_trained,
                    'parameters': len(model.get_params()) if hasattr(model, 'get_params') else 0
                }
            else:
                status['models'][name] = {
                    'type': 'fallback_function',
                    'trained': True,
                    'parameters': 0
                }
        
        # Recent performance
        if self.performance_history:
            latest_perf = list(self.performance_history.values())[-1]
            status['latest_performance'] = latest_perf
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown the learning engine"""
        self.training_active = False
        if hasattr(self, 'training_thread'):
            self.training_thread.join(timeout=5)
        self._save_models()
        logger.info("🛑 Adaptive Learning Engine shutdown complete")

class LearningIntegrator:
    """Integrate ML learning with the existing HYPER signal engine"""
    
    def __init__(self, signal_engine, model_tester=None):
        self.signal_engine = signal_engine
        self.model_tester = model_tester
        self.learning_engine = AdaptiveLearningEngine()
        self.prediction_history = {}
        
        logger.info("🔗 Learning Integrator initialized")
    
    async def enhanced_signal_generation(self, symbol: str) -> Dict[str, Any]:
        """Generate signals with ML enhancement"""
        
        # Get base signal from existing engine
        base_signal = await self.signal_engine.generate_signal(symbol)
        
        # Prepare data for ML prediction
        signal_data = {
            'technical_indicators': {
                'rsi': getattr(base_signal, 'indicators', {}).get('rsi', 50),
                'williams_r': getattr(base_signal, 'williams_r', -50),
                'stochastic_k': getattr(base_signal, 'stochastic_k', 50),
                'stochastic_d': getattr(base_signal, 'stochastic_d', 50),
                'volume_ratio': getattr(base_signal, 'indicators', {}).get('volume_ratio', 1),
                'range_percent': getattr(base_signal, 'indicators', {}).get('range_percent', 0)
            },
            'sentiment_data': {
                'overall_sentiment_score': getattr(base_signal, 'sentiment_score', 50),
                'news_sentiment': 50,  # Would come from enhanced sentiment
                'social_sentiment': 50,
                'overall_confidence': 0.7
            },
            'market_data': {
                'vix_sentiment': 20,
                'market_breadth': getattr(base_signal, 'market_breadth', 50),
                'sector_rotation_score': 50
            }
        }
        
        # Get ML predictions
        ml_predictions = self.learning_engine.predict(signal_data)
        
        # Combine base signal with ML insights
        enhanced_confidence = self._combine_confidences(
            base_signal.confidence,
            ml_predictions.get('confidence', {}).get('predicted_accuracy', 0.5) * 100
        )
        
        ml_direction = ml_predictions.get('direction', {}).get('prediction', 'NEUTRAL')
        
        # Adjust signal based on ML agreement/disagreement
        if ml_direction == base_signal.direction:
            # ML agrees - boost confidence
            final_confidence = min(95, enhanced_confidence * 1.1)
            agreement = 'AGREE'
        elif ml_direction == 'NEUTRAL' or base_signal.direction == 'NEUTRAL':
            # One is neutral - slight confidence reduction
            final_confidence = enhanced_confidence * 0.95
            agreement = 'NEUTRAL'
        else:
            # ML disagrees - reduce confidence
            final_confidence = enhanced_confidence * 0.8
            agreement = 'DISAGREE'
        
        # Create enhanced signal result
        enhanced_result = {
            'base_signal': asdict(base_signal),
            'ml_predictions': ml_predictions,
            'final_confidence': round(final_confidence, 1),
            'ml_agreement': agreement,
            'enhanced_reasoning': self._generate_ml_reasoning(base_signal, ml_predictions),
            'model_status': self.learning_engine.get_model_status()
        }
        
        # Store for learning feedback
        self.prediction_history[f"{symbol}_{base_signal.timestamp}"] = {
            'base_signal': base_signal,
            'ml_prediction': ml_predictions,
            'final_confidence': final_confidence
        }
        
        return enhanced_result
    
    def _combine_confidences(self, base_confidence: float, ml_confidence: float) -> float:
        """Intelligently combine base and ML confidences"""
        # Weighted average with slight preference for base system
        return (base_confidence * 0.6 + ml_confidence * 0.4)
    
    def _generate_ml_reasoning(self, base_signal, ml_predictions: Dict) -> List[str]:
        """Generate human-readable ML reasoning"""
        reasoning = []
        
        # Direction reasoning
        ml_direction = ml_predictions.get('direction', {})
        if ml_direction.get('prediction') == base_signal.direction:
            confidence = ml_direction.get('confidence', 0)
            reasoning.append(f"ML confirms {base_signal.direction} direction ({confidence:.1%} confidence)")
        else:
            reasoning.append(f"ML suggests {ml_direction.get('prediction', 'NEUTRAL')} (differs from technical)")
        
        # Feature importance reasoning
        top_features = ml_predictions.get('feature_importance', [])[:3]
        if top_features:
            feature_text = ", ".join([f"{f['feature']} ({f['importance']:.1%})" for f in top_features])
            reasoning.append(f"Key ML factors: {feature_text}")
        
        # Volatility insight
        volatility = ml_predictions.get('volatility', {})
        if volatility.get('prediction'):
            vol_range = volatility.get('expected_range', ['-2%', '+2%'])
            reasoning.append(f"Expected volatility: {volatility['prediction']} {vol_range[0]} to {vol_range[1]}")
        
        return reasoning
    
    def provide_learning_feedback(self, prediction
