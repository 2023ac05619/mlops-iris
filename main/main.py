import os
import sys
from pathlib import Path
# Adding project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.data_pipeline import DataPipeline
from src.model_pipeline import ModelPipeline
from db.database import DatabaseManager


class MLOpsPipeline:
    """Main MLOps pipeline orchestrator."""
    
    def __init__(self):
        self.data_pipeline = DataPipeline()
        
    def run_data_pipeline(self):
        print("\n[STEP 1] Running data pipeline...")
        return self.data_pipeline.load_and_preprocess()


def main():
    """Main function to orchestrate the complete MLOps pipeline."""
    pipeline = MLOpsPipeline()
    
    try:
        # Run complete pipeline
        data = pipeline.run_data_pipeline()
        
            
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
