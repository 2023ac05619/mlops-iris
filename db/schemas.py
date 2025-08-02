from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features: sepal length, sepal width, petal length, petal width",
        min_length=4,
        max_length=4,
        example=[5.1, 3.5, 1.4, 0.2]
    )
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Exactly 4 features required for Iris classification')
        
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i+1} must be a number')
            if feature < 0:
                raise ValueError(f'Feature {i+1} must be positive')
            if feature > 50:  # Reasonable upper bound for Iris features
                raise ValueError(f'Feature {i+1} seems unreasonably large (>{50})')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0=setosa, 1=versicolor, 2=virginica)")
    prediction_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    latency: float = Field(..., description="Prediction latency in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_name": "setosa",
                "confidence": 1.0,
                "probabilities": {
                    "setosa": 1.0,
                    "versicolor": 0.0,
                    "virginica": 0.0
                },
                "latency": 0.003
            }
        }


class RetrainingRequest(BaseModel):
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "trigger_threshold": 10,
                "force_retrain": False
            }
        }


class NewDataSample(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features",
        min_length=4,
        max_length=4
    )
    target: int = Field(
        ..., 
        description="True class label (0=setosa, 1=versicolor, 2=virginica)",
        ge=0,
        le=2
    )
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Exactly 4 features required')
        
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i+1} must be a number')
            if feature < 0:
                raise ValueError(f'Feature {i+1} must be positive')
            if feature > 50:  # Reasonable upper bound for Iris features
                raise ValueError(f'Feature {i+1} seems unreasonably large (>{50})')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [4.9, 3.0, 1.4, 0.2],
                "target": 0
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class MetricsResponse(BaseModel):
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_db_latency: float = Field(..., description="Average database latency in seconds")
    prediction_distribution: Dict[str, int] = Field(..., description="Distribution of predictions by class")
    new_samples_pending: int = Field(..., description="Number of new samples pending for retraining")
    retrain_threshold: int = Field(..., description="Threshold for triggering retraining")
    model_info: Dict[str, Union[str, float, List[str]]] = Field(..., description="Information about the current model")
    timestamp: str = Field(..., description="Timestamp of metrics collection")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 42,
                "average_db_latency": 0.002,
                "prediction_distribution": {
                    "setosa": 15,
                    "versicolor": 12,
                    "virginica": 15
                },
                "new_samples_pending": 3,
                "retrain_threshold": 10,
                "model_info": {
                    "accuracy": 0.967,
                    "model_name": "RandomForest"
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }