from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ModelName(str, Enum):
    """Available model names."""
    LOGISTIC_REGRESSION = "logisticregression"
    RANDOM_FOREST = "randomforest"


class IrisClass(int, Enum):
    """Iris classification targets."""
    SETOSA = 0
    VERSICOLOR = 1
    VIRGINICA = 2


class PredictionRequest(BaseModel):
    """Enhanced prediction request with optional model selection."""
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features: sepal length, sepal width, petal length, petal width",
        min_items=4,
        max_items=4,
        example=[5.1, 3.5, 1.4, 0.2]
    )
    model_name: Optional[ModelName] = Field(
        None,
        description="Specific model to use. If not provided, both models will be used"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that all features are positive numbers within reasonable bounds."""
        if len(v) != 4:
            raise ValueError('Exactly 4 features required for Iris classification')
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        feature_bounds = [(0.1, 20.0), (0.1, 10.0), (0.1, 20.0), (0.1, 10.0)]
        
        for i, (feature, (min_val, max_val)) in enumerate(zip(v, feature_bounds)):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'{feature_names[i]} must be a number')
            if feature < min_val or feature > max_val:
                raise ValueError(
                    f'{feature_names[i]} must be between {min_val} and {max_val}, got {feature}'
                )
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_name": "logisticregression"
            }
        }


class PredictionResponse(BaseModel):
    """Single model prediction response."""
    prediction: int = Field(..., description="Predicted class (0=setosa, 1=versicolor, 2=virginica)")
    prediction_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    latency: float = Field(..., description="Prediction latency in seconds")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_name": "setosa",
                "confidence": 1.0,
                "probabilities": {
                    "setosa": 1.0,
                    "versicolor": 0.0,
                    "virginica": 0.0
                },
                "latency": 0.003,
                "model_name": "logisticregression"
            }
        }


class BulkPredictionResponse(BaseModel):
    """Response for predictions from multiple models."""
    predictions: Dict[str, PredictionResponse] = Field(
        ..., 
        description="Predictions from each model"
    )
    timestamp: str = Field(..., description="Timestamp of prediction")
    features_used: List[float] = Field(..., description="Input features used")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": {
                    "logisticregression": {
                        "prediction": 0,
                        "prediction_name": "setosa",
                        "confidence": 1.0,
                        "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
                        "latency": 0.003,
                        "model_name": "logisticregression"
                    },
                    "randomforest": {
                        "prediction": 0,
                        "prediction_name": "setosa",
                        "confidence": 0.98,
                        "probabilities": {"setosa": 0.98, "versicolor": 0.01, "virginica": 0.01},
                        "latency": 0.005,
                        "model_name": "randomforest"
                    }
                },
                "timestamp": "2024-01-01T12:00:00",
                "features_used": [5.1, 3.5, 1.4, 0.2]
            }
        }


class NewDataSample(BaseModel):
    """Enhanced schema for adding new training data."""
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features",
        min_items=4,
        max_items=4
    )
    target: IrisClass = Field(
        ..., 
        description="True class label (0=setosa, 1=versicolor, 2=virginica)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata about the sample"
    )
    source: Optional[str] = Field(
        None,
        description="Source of the data (e.g., 'manual', 'api', 'batch_upload')"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features with same logic as PredictionRequest."""
        if len(v) != 4:
            raise ValueError('Exactly 4 features required')
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        feature_bounds = [(0.1, 20.0), (0.1, 10.0), (0.1, 20.0), (0.1, 10.0)]
        
        for i, (feature, (min_val, max_val)) in enumerate(zip(v, feature_bounds)):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'{feature_names[i]} must be a number')
            if feature < min_val or feature > max_val:
                raise ValueError(
                    f'{feature_names[i]} must be between {min_val} and {max_val}, got {feature}'
                )
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [4.9, 3.0, 1.4, 0.2],
                "target": 0,
                "metadata": {"quality": "high", "verified": True},
                "source": "manual"
            }
        }


class RetrainingRequest(BaseModel):
    """Enhanced retraining request schema."""
    trigger_threshold: Optional[int] = Field(
        default=10, 
        description="Number of new samples needed to trigger retraining",
        ge=1,
        le=1000
    )
    force_retrain: Optional[bool] = Field(
        default=False, 
        description="Force retraining regardless of threshold"
    )
    retrain_models: Optional[List[ModelName]] = Field(
        None,
        description="Specific models to retrain. If None, all models will be retrained"
    )
    experiment_name: Optional[str] = Field(
        None,
        description="Custom experiment name for MLflow tracking"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "trigger_threshold": 10,
                "force_retrain": False,
                "retrain_models": ["logisticregression", "randomforest"],
                "experiment_name": "custom_retrain_experiment"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    version: Optional[str] = Field(None, description="API version")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00",
                "version": "3.0.0",
                "uptime": 3600.0
            }
        }


class ModelInfo(BaseModel):
    """Model information schema."""
    accuracy: float = Field(..., description="Model accuracy")
    model_name: str = Field(..., description="Model name")
    feature_names: List[str] = Field(..., description="Feature names")
    target_names: List[str] = Field(..., description="Target class names")
    training_timestamp: str = Field(..., description="When the model was trained")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    
    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.967,
                "model_name": "RandomForest",
                "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
                "target_names": ["setosa", "versicolor", "virginica"],
                "training_timestamp": "2024-01-01T10:00:00",
                "run_id": "abc123def456"
            }
        }


class MetricsResponse(BaseModel):
    """Enhanced system metrics response schema."""
    total_predictions: int = Field(..., description="Total number of predictions made")
    predictions_by_model: Dict[str, int] = Field(..., description="Predictions count per model")
    average_latency_by_model: Dict[str, float] = Field(..., description="Average latency per model")
    prediction_distribution: Dict[str, int] = Field(..., description="Distribution of predictions by class")
    new_samples_pending: int = Field(..., description="Number of new samples pending for retraining")
    retrain_threshold: int = Field(..., description="Threshold for triggering retraining")
    model_info: Dict[str, ModelInfo] = Field(..., description="Information about loaded models")
    system_health: Dict[str, Any] = Field(..., description="System health metrics")
    timestamp: str = Field(..., description="Timestamp of metrics collection")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 150,
                "predictions_by_model": {
                    "logisticregression": 80,
                    "randomforest": 70
                },
                "average_latency_by_model": {
                    "logisticregression": 0.003,
                    "randomforest": 0.005
                },
                "prediction_distribution": {
                    "setosa": 50,
                    "versicolor": 50,
                    "virginica": 50
                },
                "new_samples_pending": 3,
                "retrain_threshold": 10,
                "model_info": {
                    "logisticregression": {
                        "accuracy": 0.95,
                        "model_name": "LogisticRegression",
                        "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
                        "target_names": ["setosa", "versicolor", "virginica"],
                        "training_timestamp": "2024-01-01T10:00:00"
                    }
                },
                "system_health": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "disk_usage": 23.1
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class PredictionHistoryItem(BaseModel):
    """Single prediction history item."""
    id: int = Field(..., description="Prediction ID")
    timestamp: str = Field(..., description="Prediction timestamp")
    input_data: List[float] = Field(..., description="Input features")
    model_name: str = Field(..., description="Model used")
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence")
    latency: float = Field(..., description="Prediction latency")


class PredictionHistoryResponse(BaseModel):
    """Prediction history response schema."""
    history: List[PredictionHistoryItem] = Field(..., description="List of prediction history items")
    count: int = Field(..., description="Number of items returned")
    limit: int = Field(..., description="Requested limit")
    model_filter: Optional[str] = Field(None, description="Applied model filter")
    
    class Config:
        schema_extra = {
            "example": {
                "history": [
                    {
                        "id": 1,
                        "timestamp": "2024-01-01T12:00:00",
                        "input_data": [5.1, 3.5, 1.4, 0.2],
                        "model_name": "logisticregression",
                        "prediction": "setosa",
                        "confidence": 1.0,
                        "latency": 0.003
                    }
                ],
                "count": 1,
                "limit": 10,
                "model_filter": None
            }
        }


class GitHubWorkflowTrigger(BaseModel):
    """GitHub workflow trigger request schema."""
    workflow_id: str = Field(..., description="GitHub workflow ID or filename")
    ref: str = Field(default="main", description="Git reference (branch/tag)")
    inputs: Optional[Dict[str, str]] = Field(None, description="Workflow inputs")
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "ci-cd.yml",
                "ref": "main",
                "inputs": {
                    "retrain_trigger": "true"
                }
            }
        }


class BatchDataUpload(BaseModel):
    """Schema for batch data upload."""
    samples: List[NewDataSample] = Field(
        ..., 
        description="List of training samples",
        min_items=1,
        max_items=1000
    )
    batch_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata for the entire batch"
    )
    auto_trigger_retrain: bool = Field(
        default=True,
        description="Whether to automatically trigger retraining if threshold is met"
    )
    
    @validator('samples')
    def validate_batch_size(cls, v):
        """Validate batch size constraints."""
        if len(v) > 1000:
            raise ValueError('Maximum 1000 samples per batch')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "features": [5.1, 3.5, 1.4, 0.2],
                        "target": 0,
                        "source": "batch_upload"
                    },
                    {
                        "features": [6.2, 2.9, 4.3, 1.3],
                        "target": 1,
                        "source": "batch_upload"
                    }
                ],
                "batch_metadata": {
                    "upload_source": "data_scientist",
                    "quality_checked": True
                },
                "auto_trigger_retrain": True
            }
        }


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison."""
    models: Dict[str, ModelInfo] = Field(..., description="Information about each model")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison metrics between models")
    recommendation: Optional[str] = Field(None, description="Recommended model based on metrics")
    timestamp: str = Field(..., description="Timestamp of comparison")
    
    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "logisticregression": {
                        "accuracy": 0.95,
                        "model_name": "LogisticRegression",
                        "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
                        "target_names": ["setosa", "versicolor", "virginica"],
                        "training_timestamp": "2024-01-01T10:00:00"
                    },
                    "randomforest": {
                        "accuracy": 0.97,
                        "model_name": "RandomForest",
                        "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
                        "target_names": ["setosa", "versicolor", "virginica"],
                        "training_timestamp": "2024-01-01T10:00:00"
                    }
                },
                "comparison_metrics": {
                    "accuracy_difference": 0.02,
                    "latency_difference": 0.002,
                    "memory_usage_difference": 50.2
                },
                "recommendation": "randomforest",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class SystemStatus(BaseModel):
    """System status schema."""
    api_status: str = Field(..., description="API service status")
    database_status: str = Field(..., description="Database status")
    models_status: Dict[str, str] = Field(..., description="Status of each model")
    mlflow_status: str = Field(..., description="MLflow tracking status")
    github_integration_status: str = Field(..., description="GitHub Actions integration status")
    last_training: Optional[str] = Field(None, description="Last training timestamp")
    uptime: float = Field(..., description="System uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "api_status": "healthy",
                "database_status": "connected",
                "models_status": {
                    "logisticregression": "loaded",
                    "randomforest": "loaded"
                },
                "mlflow_status": "tracking",
                "github_integration_status": "configured",
                "last_training": "2024-01-01T10:00:00",
                "uptime": 3600.0
            }
        }