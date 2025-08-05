import numpy as np
import requests
import os
from fastapi import FastAPI, Request, HTTPException
from typing import List

# Add root to path
import sys
from pathlib import Path
ROOT_DIR_API = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR_API))

from app.schemas import PredictionInput, PredictionOutput, NewDataInput
from config.config import GITHUB_REPO, GITHUB_TOKEN

# Import the metrics gauge from main
# This is a bit of a workaround to share the metric object
try:
    from app.main import SYSTEM_METRICS
except ImportError:
    # Dummy object for local testing if main is not run first
    class DummyGauge:
        def labels(self, *args, **kwargs): return self
        def inc(self): pass
    SYSTEM_METRICS = DummyGauge()


app = FastAPI(
    title="MLOps Iris Classifier API",
    description="API for predicting Iris flower species.",
    version="1.0.0"
)

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

@app.post("/predict", response_model=List[PredictionOutput], tags=["Prediction"])
def predict(request: Request, payload: PredictionInput):
    """
    Predicts the Iris species for the given features.
    If `model_name` is provided, it uses that model.
    Otherwise, it returns predictions from all loaded models.
    """
    scaler = request.app.state.scaler
    metadata = request.app.state.metadata
    models = request.app.state.models

    features_scaled = scaler.transform(np.array(payload.features).reshape(1, -1))

    results = []
    models_to_run = [payload.model_name] if payload.model_name else models.keys()

    for name in models_to_run:
        if name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")

        model = models[name]
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        prediction_name = metadata['target_names'][prediction]

        # Increment Prometheus counter
        SYSTEM_METRICS.labels(model_name=name, predicted_class=prediction_name).inc()

        results.append(
            PredictionOutput(
                model_name=name,
                prediction=int(prediction),
                prediction_name=prediction_name,
                confidence=float(max(probabilities)),
                probabilities={name: float(prob) for name, prob in zip(metadata['target_names'], probabilities)}
            )
        )
    return results

@app.post("/add_data", tags=["Data"])
def add_new_data(payload: NewDataInput):
    """
    Adds new labeled data and triggers a GitHub Actions workflow to retrain the model.
    **Requires a GitHub Personal Access Token with `repo` scope.**
    """
    # In a real-world scenario, you would append this data to a staging dataset
    # and then trigger the DVC pipeline. For this example, we just trigger the workflow.
    print(f"Received new data: features={payload.features}, target={payload.target}")
    print("Triggering GitHub Actions workflow for retraining...")

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/ci-cd.yml/dispatches"
    data = {"ref": "main"} # Or your primary branch

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 204:
            return {"status": "success", "message": "Retraining workflow triggered successfully."}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to trigger workflow: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting GitHub API: {e}")
