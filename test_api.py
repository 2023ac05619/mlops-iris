import pytest
import json
import tempfile
import os
from pathlib import Path

# Setup test environment
os.environ['TESTING'] = 'true'

from api import create_app
from db.database import DatabaseManager
from src.inference_service import InferenceService
from train.mlops_pipeline import MLOpsPipeline


@pytest.fixture
def setup_test_pipeline():
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Override config paths for testing
        import config
        config.MODELS_DIR = temp_path / "models"
        config.LOGS_DIR = temp_path / "logs"
        config.DATA_DIR = temp_path / "data"
        config.DB_PATH = temp_path / "logs" / "test_predictions.db"
        config.METADATA_FILE = temp_path / "models" / "metadata.json"
        config.SCALER_FILE = temp_path / "models" / "scaler.pkl"
        
        # Create directories
        for directory in [config.MODELS_DIR, config.LOGS_DIR, config.DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline to create artifacts
        pipeline = MLOpsPipeline()
        data = pipeline.run_data_pipeline()
        pipeline.run_model_training(data)
        
        yield pipeline


@pytest.fixture
def client(setup_test_pipeline):
    pipeline = setup_test_pipeline
    
    # Initialize services
    db_manager = DatabaseManager()
    inference_service = InferenceService()
    
    db_manager.init_db()
    inference_service.load_model()
    
    # Create app
    app = create_app(inference_service, db_manager)
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data


def test_home_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'endpoints' in data
    assert data['version'] == '2.0'


def test_predict_valid_input(client):
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'prediction_name' in data
    assert 'confidence' in data
    assert 'probabilities' in data
    assert 'latency' in data
    
    # Validate data types
    assert isinstance(data['prediction'], int)
    assert isinstance(data['confidence'], float)
    assert isinstance(data['probabilities'], dict)
    assert data['confidence'] >= 0 and data['confidence'] <= 1


def test_predict_invalid_input(client):
    # Wrong number of features
    payload = {"features": [5.1, 3.5]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Input validation failed' in data['error']


def test_predict_negative_features(client):
    payload = {"features": [-1.0, 3.5, 1.4, 0.2]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 400


def test_add_training_data_valid(client):
    payload = {"features": [4.9, 3.0, 1.4, 0.2], "target": 0}
    response = client.post('/add_training_data', json=payload)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'message' in data
    assert 'Training data added' in data['message']


def test_add_training_data_invalid_target(client):
    payload = {"features": [4.9, 3.0, 1.4, 0.2], "target": 5}  # Invalid target
    response = client.post('/add_training_data', json=payload)
    assert response.status_code == 400


def test_metrics_endpoint(client):
    response = client.get('/system_metrics')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    required_fields = [
        'total_predictions', 'average_db_latency', 'prediction_distribution',
        'new_samples_pending', 'retrain_threshold', 'model_info', 'timestamp'
    ]
    
    for field in required_fields:
        assert field in data
    
    # Validate model info structure
    model_info = data['model_info']
    assert 'accuracy' in model_info
    assert 'model_name' in model_info
    assert isinstance(model_info['accuracy'], float)


def test_predictions_history(client):
    # Make a prediction first
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    client.post('/predict', json=payload)
    
    # Get history
    response = client.get('/predictions/history?limit=5')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'history' in data
    assert 'count' in data
    assert isinstance(data['history'], list)
    assert data['count'] >= 0


def test_trigger_retrain_endpoint(client):
    payload = {"force_retrain": True}
    response = client.post('/trigger_retrain', json=payload)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'message' in data
    assert 'forced' in data
    assert data['forced'] == True


def test_dashboard_endpoint(client):
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'ML Model Monitoring Dashboard' in response.data
    assert b'System Healthy' in response.data


def test_prometheus_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'text/plain; version=0.0.4; charset=utf-8'


def test_integration_workflow(client):
    # 1. Make prediction
    predict_payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    predict_response = client.post('/predict', json=predict_payload)
    assert predict_response.status_code == 200
    
    # 2. Add training data
    train_payload = {"features": [4.9, 3.0, 1.4, 0.2], "target": 0}
    train_response = client.post('/add_training_data', json=train_payload)
    assert train_response.status_code == 200
    
    # 3. Check metrics updated
    metrics_response = client.get('/system_metrics')
    assert metrics_response.status_code == 200
    metrics_data = json.loads(metrics_response.data)
    assert metrics_data['total_predictions'] >= 1
    assert metrics_data['new_samples_pending'] >= 1
    
    # 4. Check history
    history_response = client.get('/predictions/history')
    assert history_response.status_code == 200
    history_data = json.loads(history_response.data)
    assert history_data['count'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])