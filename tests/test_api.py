import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from main import app
from src.inference_service import InferenceService, get_inference_service
from app.schemas import PredictionOutput

# --- Mock Inference Service ---
# Create a mock version of the InferenceService to use in tests.
# This prevents tests from needing actual model files or a database.
mock_prediction_output = PredictionOutput(
    model_name="RandomForest",
    prediction=0,
    prediction_name="setosa",
    confidence=0.95,
    probabilities={"setosa": 0.95, "versicolor": 0.03, "virginica": 0.02}
)

class MockInferenceService:
    def predict(self, features, model_name=None):
        if model_name == "NonExistentModel":
            raise ValueError("Model 'NonExistentModel' not found.")
        
        # If a specific model is requested, return just that one
        if model_name:
            return [mock_prediction_output]
            
        # Otherwise, return results for two mock models
        return [
            mock_prediction_output,
            PredictionOutput(
                model_name="LogisticRegression",
                prediction=0,
                prediction_name="setosa",
                confidence=0.91,
                probabilities={"setosa": 0.91, "versicolor": 0.07, "virginica": 0.02}
            )
        ]

# --- Test Setup ---
@pytest.fixture(scope="module", autouse=True)
def override_inference_service():
    app.dependency_overrides[get_inference_service] = MockInferenceService
    yield
    app.dependency_overrides = {} # Clean up overrides after tests

client = TestClient(app)

# --- Test Cases ---
def test_read_root():
    """Tests the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

@patch('db.database.log_prediction') # Mock the database logging function
def test_predict_all_models(mock_log_prediction: MagicMock):
    """Tests the /predict endpoint without specifying a model."""
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["model_name"] == "RandomForest"
    assert data[1]["model_name"] == "LogisticRegression"
    
    # Verify that the database logging function was called for each prediction
    assert mock_log_prediction.call_count == 2
    mock_log_prediction.assert_any_call(
        model_name="RandomForest",
        prediction_name="setosa",
        confidence=0.95,
        features=str([5.1, 3.5, 1.4, 0.2])
    )

@patch('db.database.log_prediction')
def test_predict_single_model(mock_log_prediction: MagicMock):
    """Tests the /predict endpoint with a specific model."""
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2], "model_name": "RandomForest"})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["model_name"] == "RandomForest"
    
    # Verify that the database logging function was called once
    mock_log_prediction.assert_called_once()
    mock_log_prediction.assert_called_with(
        model_name="RandomForest",
        prediction_name="setosa",
        confidence=0.95,
        features=str([5.1, 3.5, 1.4, 0.2])
    )

def test_predict_model_not_found():
    """Tests that a 404 is returned for a non-existent model."""
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2], "model_name": "NonExistentModel"})
    assert response.status_code == 404
    assert "Model 'NonExistentModel' not found" in response.json()["detail"]
