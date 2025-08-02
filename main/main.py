import os
import sys
from pathlib import Path
# Adding project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.data_pipeline import DataPipeline
from src.model_pipeline import ModelPipeline
from db.database import DatabaseManager
from src.mlflow_utils import MLflowManager

class MLOpsPipeline:
    
    def __init__(self):
        self.data_pipeline = DataPipeline()
        self.model_pipeline = ModelPipeline()
        
    def run_data_pipeline(self):
        print("\n[STEP 1] Running data pipeline...")
        return self.data_pipeline.load_and_preprocess()

    def initialize_services(self):
        self.db_manager.init_db()
        return True
    
    def demonstrate_mlflow(self):
        print("\n[STEP 2] Running MLflow demonstration...")
        self.mlflow_manager.demonstrate_tracking()
        self.mlflow_manager.show_detailed_run_info()
        
        
def main():
    """Main function to orchestrate the complete MLOps pipeline."""
    pipeline = MLOpsPipeline()
    
    try:
        # Run complete pipeline
        data = pipeline.run_data_pipeline()        
        pipeline.run_model_training(data)
        pipeline.demonstrate_mlflow()
        pipeline.initialize_services()
            
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
