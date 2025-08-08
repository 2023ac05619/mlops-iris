import requests
from fastapi import APIRouter, Depends, HTTPException
from typing import List

# Corrected import paths
from .schemas import PredictionInput, PredictionOutput, NewDataInput
from config import GITHUB_REPO, GITHUB_TOKEN
from src.inference_service import InferenceService, get_inference_service

router = APIRouter()

@router.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

@router.post("/predict", response_model=List[PredictionOutput], tags=["Prediction"])
def predict(
    payload: PredictionInput,
    service: InferenceService = Depends(get_inference_service)
):
    """
    Predicts the Iris species using the injected InferenceService.
    """
    try:
        return service.predict(features=payload.features, model_name=payload.model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/add_data", tags=["Data"])
def add_new_data(payload: NewDataInput):
    """
    Triggers a GitHub Actions workflow to retrain the model.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise HTTPException(status_code=500, detail="GitHub token or repo not configured in .env file.")

    print("Triggering GitHub Actions workflow for retraining...")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/ci-cd.yml/dispatches"
    data = {"ref": "main"}

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
