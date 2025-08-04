import os
import sys
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
sys.path.append(str(Path(__file__).parent))
from src.data_pipeline import DataPipeline
from src.model_pipeline import ModelPipeline
from src.inference_service import InferenceService
from src.mlflow_utils import MLflowManager
from db.database import DatabaseManager
from api.app import create_app
from config import MLRUNS_DIR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLOpsPipeline:
        
    def __init__(self):
        self.data_pipeline = DataPipeline()
        self.model_pipeline = ModelPipeline()
        self.inference_service = InferenceService()
        self.mlflow_manager = MLflowManager()
        self.db_manager = DatabaseManager()
        
    def run_data_pipeline(self):
        """Step 1: Data loading and preprocessing."""
        logger.info("Running data pipeline...")
        return self.data_pipeline.load_and_preprocess()
        
    def run_model_training(self, data):
        """Step 2: Model training and evaluation."""
        logger.info("Running model training pipeline...")
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{MLRUNS_DIR.resolve()}"
        return self.model_pipeline.train_and_evaluate(data)
        
    def demonstrate_mlflow(self):
        """Step 3: MLflow tracking demonstration."""
        logger.info("Running MLflow demonstration...")
        self.mlflow_manager.demonstrate_tracking()
        self.mlflow_manager.show_detailed_run_info()
        
    def initialize_services(self):
        """Step 4: Initialize database and inference service."""
        logger.info("Initializing services...")
        self.db_manager.init_db()
        if not self.inference_service.load_models():  # Load both models
            logger.error("Failed to load models for serving.")
            return False
        return True