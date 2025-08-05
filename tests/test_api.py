from fastapi.testclient import TestClient
import sys
from pathlib import Path
import pytest

# Add root to path
ROOT_DIR_TEST = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR_TEST))

from app.api import app

client = TestClient(app)

# A simple fixture to ensure artifacts are "loaded" for tests
@pytest.fixture(scope="module", autouse=True)
def load_test_artifacts():
    # In a real test suite, you'd create mock artifacts here
    # For simplicity, we assume the app startup logic handles it or we mock app.state
    app.state.metadata = {
        "target_names": ["setosa", "versicolor", "virginica"],
        "models": {"RandomForest": {}, "LogisticRegression": {}}
    }
    # Mock models and scaler
    class MockModel:
        def predict(self, X): return [0]
        def predict_proba(self, X): return [[0.9, 0.05, 0.05]]
    class MockScaler:
        def transform(self, X): return X

    app.state.models = {"RandomForest": MockModel(), "LogisticRegression": MockModel()}
    app.state.scaler = MockScaler()
    yield # this is where the testing happens

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

def test_predict_all_models():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["model_name"] in ["RandomForest", "LogisticRegression"]
    assert data[0]["prediction_name"] == "setosa"

def test_predict_single_model():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2], "model_name": "RandomForest"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["model_name"] == "RandomForest"

def test_predict_model_not_found():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2], "model_name": "NonExistentModel"})
    assert response.status_code == 404
