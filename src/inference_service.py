import joblib
import json
import time
import numpy as np
from typing import Dict, Any, Optional
from db.schemas import PredictionResponse
from src.monitoring import PREDICTION_COUNTER
from config import METADATA_FILE, SCALER_FILE, MODELS_DIR


class InferenceService:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = {}
        self._is_loaded = False
        
    def load_model(self) -> bool:
        try:
            # Load metadata
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
                
            # Load scaler
            self.scaler = joblib.load(SCALER_FILE)
            
            # Load model
            model_path = MODELS_DIR / f"{self.metadata['best_model'].lower()}_model.pkl"
            self.model = joblib.load(model_path)
            
            self._is_loaded = True
            print("[INFO] Model, scaler, and metadata loaded successfully.")
            return True
            
        except Exception as e:
            print(f"[ERROR] Could not load model artifacts: {e}")
            self._is_loaded = False
            return False
            
    def predict(self, features: list) -> PredictionResponse:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
            
        start_time = time.time()
        
        # Preprocess features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        prediction_name = self.metadata['target_names'][prediction]
        
        # Update monitoring metrics
        PREDICTION_COUNTER.labels(predicted_class=prediction_name).inc()
        
        latency = time.time() - start_time
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            confidence=confidence,
            probabilities={
                name: float(prob) 
                for name, prob in zip(self.metadata['target_names'], probabilities)
            },
            latency=latency
        )
        
    def get_model_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {"accuracy": 0.0, "model_name": "N/A"}
            
        return {
            "accuracy": self.metadata.get('best_accuracy', 0.0),
            "model_name": self.metadata.get('best_model', 'N/A'),
            "feature_names": self.metadata.get('feature_names', []),
            "target_names": self.metadata.get('target_names', []),
            "training_timestamp": self.metadata.get('training_timestamp', 'N/A')
        }
        
    def reload_model(self) -> bool:
        print("[INFO] Reloading model...")
        return self.load_model()
