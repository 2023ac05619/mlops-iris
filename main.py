import uvicorn
import joblib
import json
from src.data_pipeline import load_raw_data, split_data
from src.model_pipeline import train_models
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import os
import sys
from pathlib import Path

# Adding project root to Python path
sys.path.append(str(Path(__file__).parent))


# Add root to path
ROOT_DIR_APP = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR_APP))

from app.api import app
from config.config import MODELS_DIR, METADATA_FILE, SCALER_FILE, API_HOST, API_PORT



# --- Prometheus Metrics ---
# This gauge will track predictions for each model
SYSTEM_METRICS = Gauge(
    "system_metrics",
    "Predictions per model",
    ["model_name", "predicted_class"]
)

@app.on_event("startup")
def load_artifacts():
    """Load models, scaler, and metadata on application startup."""
    print("--- Loading application artifacts ---")
    try:
        with open(METADATA_FILE, "r") as f:
            app.state.metadata = json.load(f)

        app.state.scaler = joblib.load(SCALER_FILE)

        app.state.models = {}
        for name, details in app.state.metadata["models"].items():
            model_path = MODELS_DIR / f"{name}.pkl"
            app.state.models[name] = joblib.load(model_path)
            print(f"Successfully loaded model: {name}")

        # Instrument the app with Prometheus metrics
        Instrumentator().instrument(app).expose(app)
        print("--- Artifacts loaded successfully ---")

    except FileNotFoundError as e:
        print(f"ERROR: Could not load artifacts. File not found: {e}")
        print("Please run the training pipeline first.")
        # In a real app, you might want to exit or handle this more gracefully
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")


if __name__ == "__main__":
    
    try:
        # Run complete pipeline
        load_raw_data()
        split_data()
        train_models()
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise

    uvicorn.run(app, host=API_HOST, port=API_PORT)



