from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PredictionInput(BaseModel):
    """Schema for prediction input data."""
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2])
    model_name: Optional[str] = Field(None, example="RandomForest")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.9, 3.0, 5.1, 1.8],
                "model_name": "RandomForest"
            }
        }

class PredictionOutput(BaseModel):
    """Schema for a single model's prediction output."""
    model_name: str
    prediction: int
    prediction_name: str
    confidence: float
    probabilities: Dict[str, float]

class NewDataInput(BaseModel):
    """Schema for adding new data, triggering retraining."""
    features: List[float] = Field(..., example=[4.9, 3.0, 1.4, 0.2])
    target: int = Field(..., example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "features": [4.9, 3.0, 1.4, 0.2],
                "target": 0
            }
        }

