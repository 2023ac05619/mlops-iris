import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from config import MODELS_DIR, SCALER_FILE, METADATA_FILE
from app.schemas import PredictionOutput
from db.database import log_prediction

class InferenceService:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.scaler: object = None
        self.metadata: Dict = {}
        # The _load_artifacts method is now more verbose
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Loads all necessary ML artifacts with detailed logging to diagnose issues.
        """
        print("LOAD: Attempting to load artifacts...")
        
        # 1. Check for metadata file
        if not METADATA_FILE.exists():
            print(f"[ERROR] Metadata file not found at: {METADATA_FILE}")
            print("[ERROR] Please run 'dvc repro' to generate model artifacts.")
            return # Stop loading if metadata is missing

        # 2. Check for scaler file
        if not SCALER_FILE.exists():
            print(f"[ERROR] Scaler file not found at: {SCALER_FILE}")
            print("[ERROR] Please run 'dvc repro' to generate model artifacts.")
            return # Stop loading if scaler is missing

        try:
            # Load metadata and scaler
            with open(METADATA_FILE, "r") as f:
                self.metadata = json.load(f)
            self.scaler = joblib.load(SCALER_FILE)
            print("LOAD: Metadata and scaler loaded successfully.")

            # Load models
            models_to_load = self.metadata.get("models_trained", [])
            if not models_to_load:
                print("[WARNING] 'models_trained' key not found in metadata.json or is empty. No models will be loaded.")
                return

            print(f"LOAD: Found models to load in metadata: {models_to_load}")
            for model_name in models_to_load:
                model_path = MODELS_DIR / f"{model_name.lower()}_model.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"LOAD:   - Successfully loaded model: {model_name}")
                else:
                    print(f"[WARNING] - Model file not found for '{model_name}' at: {model_path}")
            
            if not self.models:
                print("[ERROR] No models were successfully loaded, though metadata was found.")
            else:
                print("SUCCESS: Artifact loading complete.")

        except Exception as e:
            print(f"[CRITICAL ERROR] An unexpected error occurred while loading artifacts: {e}")

    def predict(self, features: List[float], model_name: Optional[str] = None) -> List[PredictionOutput]:
        # This check is now more robust.
        if not self.scaler or not self.models:
            raise RuntimeError("Inference service is not initialized. Check startup logs for errors.")

        features_scaled = self.scaler.transform(np.array(features).reshape(1, -1))
        
        results = []
        models_to_run = [model_name] if model_name else self.models.keys()

        for name in models_to_run:
            if name not in self.models:
                raise ValueError(f"Model '{name}' not found.")

            model = self.models[name]
            prediction_idx = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            target_names = sorted(self.metadata['target_names'])
            prediction_name = target_names[int(prediction_idx)]
            confidence = float(max(probabilities))

            log_prediction(
                model_name=name,
                prediction_name=prediction_name,
                confidence=confidence,
                features=str(features)
            )

            results.append(
                PredictionOutput(
                    model_name=name,
                    prediction=int(prediction_idx),
                    prediction_name=prediction_name,
                    confidence=confidence,
                    probabilities={name: float(prob) for name, prob in zip(target_names, probabilities)}
                )
            )
        return results

_inference_service_instance: Optional[InferenceService] = None

def get_inference_service() -> InferenceService:
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance
