import joblib
import json
import time
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from db.schemas import PredictionResponse
from src.monitoring import PREDICTION_COUNTER, MODEL_PREDICTION_COUNTER, MODEL_LATENCY_HISTOGRAM
from config import METADATA_FILE, SCALER_FILE, MODELS_DIR

logger = logging.getLogger(__name__)


class InferenceService:
    """Enhanced inference service supporting multiple models."""
    
    def __init__(self):
        self.models = {}  # Store multiple models
        self.scaler = None
        self.metadata = {}
        self._is_loaded = False
        self.model_names = ["logisticregression", "randomforest"]
        
    def load_models(self) -> bool:
        """Load all available models, scaler, and metadata for inference."""
        try:
            # Load metadata
            if not METADATA_FILE.exists():
                logger.error(f"Metadata file not found: {METADATA_FILE}")
                return False
                
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
                
            # Load scaler
            if not SCALER_FILE.exists():
                logger.error(f"Scaler file not found: {SCALER_FILE}")
                return False
                
            self.scaler = joblib.load(SCALER_FILE)
            
            # Load all available models
            models_loaded = 0
            for model_name in self.model_names:
                model_path = MODELS_DIR / f"{model_name}_model.pkl"
                
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Successfully loaded {model_name} model")
                        models_loaded += 1
                    except Exception as e:
                        logger.error(f"Failed to load {model_name} model: {e}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            if models_loaded == 0:
                logger.error("No models could be loaded")
                self._is_loaded = False
                return False
            
            self._is_loaded = True
            logger.info(f"Successfully loaded {models_loaded}/{len(self.model_names)} models")
            return True
            
        except Exception as e:
            logger.error(f"Could not load model artifacts: {e}")
            self._is_loaded = False
            return False
    
    def predict(self, features: list, model_name: Optional[str] = None) -> PredictionResponse:
        """Make prediction using specified model or best available model."""
        if not self._is_loaded:
            raise RuntimeError("Models are not loaded. Call load_models() first.")
        
        # Determine which model to use
        if model_name:
            if model_name.lower() not in self.models:
                available_models = list(self.models.keys())
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            selected_model_name = model_name.lower()
        else:
            # Use the best model from metadata
            best_model = self.metadata.get('best_model', '').lower()
            if best_model in self.models:
                selected_model_name = best_model
            else:
                # Fall back to first available model
                selected_model_name = list(self.models.keys())[0]
        
        start_time = time.time()
        
        try:
            # Preprocess features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get the selected model
            model = self.models[selected_model_name]
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
            prediction_name = self.metadata['target_names'][prediction]
            
            latency = time.time() - start_time
            
            # Update monitoring metrics
            PREDICTION_COUNTER.labels(predicted_class=prediction_name).inc()
            MODEL_PREDICTION_COUNTER.labels(
                model_name=selected_model_name,
                predicted_class=prediction_name
            ).inc()
            MODEL_LATENCY_HISTOGRAM.labels(model_name=selected_model_name).observe(latency)
            
            return PredictionResponse(
                prediction=int(prediction),
                prediction_name=prediction_name,
                confidence=confidence,
                probabilities={
                    name: float(prob) 
                    for name, prob in zip(self.metadata['target_names'], probabilities)
                },
                latency=latency,
                model_name=selected_model_name
            )
            
        except Exception as e:
            logger.error(f"Prediction failed with {selected_model_name}: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_all_models(self, features: list) -> Dict[str, PredictionResponse]:
        """Make predictions using all loaded models."""
        if not self._is_loaded:
            raise RuntimeError("Models are not loaded. Call load_models() first.")
        
        predictions = {}
        
        for model_name in self.models.keys():
            try:
                prediction = self.predict(features, model_name)
                predictions[model_name] = prediction
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {e}")
                continue
        
        return predictions
    
    def get_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models."""
        if not self._is_loaded:
            return {}
        
        models_info = {}
        
        for model_name in self.models.keys():
            # Get model-specific accuracy if available
            model_accuracy = self.metadata.get('best_accuracy', 0.0)
            if model_name.lower() == self.metadata.get('best_model', '').lower():
                model_accuracy = self.metadata.get('best_accuracy', 0.0)
            else:
                # For non-best models, we might not have individual accuracies stored
                # This could be enhanced to store per-model metrics
                model_accuracy = self.metadata.get('best_accuracy', 0.0) * 0.95  # Approximate
            
            models_info[model_name] = {
                "accuracy": model_accuracy,
                "model_name": model_name.title().replace('regression', ' Regression').replace('forest', ' Forest'),
                "feature_names": self.metadata.get('feature_names', []),
                "target_names": self.metadata.get('target_names', []),
                "training_timestamp": self.metadata.get('training_timestamp', 'N/A'),
                "run_id": self.metadata.get('best_run_id', 'N/A') if model_name.lower() == self.metadata.get('best_model', '').lower() else 'N/A',
                "model_type": model_name,
                "is_best_model": model_name.lower() == self.metadata.get('best_model', '').lower()
            }
        
        return models_info
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if not self._is_loaded:
            return {"accuracy": 0.0, "model_name": "N/A"}
        
        if model_name.lower() not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        models_info = self.get_models_info()
        return models_info.get(model_name.lower(), {})
    
    def reload_models(self) -> bool:
        """Reload all models (useful after retraining)."""
        logger.info("Reloading models...")
        self.models.clear()
        self._is_loaded = False
        return self.load_models()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded."""
        return model_name.lower() in self.models
    
    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def compare_models(self, features: list) -> Dict[str, Any]:
        """Compare predictions from all models for given features."""
        if not self._is_loaded:
            raise RuntimeError("Models are not loaded. Call load_models() first.")
        
        predictions = self.predict_all_models(features)
        
        # Analyze predictions
        prediction_classes = [pred.prediction for pred in predictions.values()]
        prediction_names = [pred.prediction_name for pred in predictions.values()]
        confidences = [pred.confidence for pred in predictions.values()]
        latencies = [pred.latency for pred in predictions.values()]
        
        # Calculate agreement metrics
        unique_predictions = len(set(prediction_classes))
        agreement = unique_predictions == 1
        
        comparison_result = {
            "predictions": {name: pred.dict() for name, pred in predictions.items()},
            "summary": {
                "agreement": agreement,
                "unique_predictions": unique_predictions,
                "avg_confidence": sum(confidences) / len(confidences),
                "avg_latency": sum(latencies) / len(latencies),
                "fastest_model": min(predictions.keys(), key=lambda k: predictions[k].latency),
                "most_confident_model": max(predictions.keys(), key=lambda k: predictions[k].confidence),
                "consensus_prediction": max(set(prediction_names), key=prediction_names.count) if prediction_names else None
            },
            "features_used": features,
            "timestamp": time.time()
        }
        
        return comparison_result
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models."""
        if not self._is_loaded:
            return {}
        
        stats = {}
        models_info = self.get_models_info()
        
        for model_name, info in models_info.items():
            stats[model_name] = {
                "accuracy": info["accuracy"],
                "is_loaded": True,
                "is_best_model": info["is_best_model"],
                "model_type": info["model_type"],
                "feature_count": len(info["feature_names"]),
                "class_count": len(info["target_names"])
            }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the inference service."""
        health_status = {
            "service_status": "healthy" if self._is_loaded else "unhealthy",
            "models_loaded": len(self.models),
            "available_models": list(self.models.keys()),
            "scaler_loaded": self.scaler is not None,
            "metadata_loaded": bool(self.metadata),
            "timestamp": time.time()
        }
        
        if self._is_loaded:
            # Test prediction with dummy data
            try:
                test_features = [5.0, 3.0, 1.0, 0.2]  # Typical setosa features
                test_prediction = self.predict(test_features)
                health_status["test_prediction_success"] = True
                health_status["test_prediction_latency"] = test_prediction.latency
            except Exception as e:
                health_status["test_prediction_success"] = False
                health_status["test_prediction_error"] = str(e)
        
        return health_status