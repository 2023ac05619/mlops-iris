
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_pipeline import DataPipeline
from src.model_pipeline import ModelPipeline
from src.inference_service import InferenceService
from src.mlflow_utils import MLflowManager
from db.database import DatabaseManager
from api import create_app
# from config import MLRUNS_DIR
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
MLRUNS_DIR = ROOT_DIR / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

class MLOpsPipeline:
    
    def __init__(self):
        self.data_pipeline = DataPipeline()
        self.model_pipeline = ModelPipeline()
        self.inference_service = InferenceService()
        self.mlflow_manager = MLflowManager()
        self.db_manager = DatabaseManager()
        
    def run_data_pipeline(self):
        print("\n[INFO] Running data pipeline...")
        return self.data_pipeline.load_and_preprocess()
        
    def run_model_training(self, data):
        print("\n[INFO] Running model training pipeline...")
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{MLRUNS_DIR.resolve()}"
        return self.model_pipeline.train_and_evaluate(data)
        
    def demonstrate_mlflow(self):
        print("\n[INFO] Running MLflow demonstration...")
        self.mlflow_manager.demonstrate_tracking()
        self.mlflow_manager.show_detailed_run_info()
        
    def initialize_services(self):
        print("\n[INFO] Initializing services...")
        self.db_manager.init_db()
        if not self.inference_service.load_model():
            print("[ERROR] Failed to load model for serving. Exiting.")
            return False
        return True
        
    def start_api_server(self):
        print("\n[INFO] Starting Flask API server...")
        # print("API is available at http://0.0.0.0:5001")
        app = create_app(self.inference_service, self.db_manager)
        app.run(host='0.0.0.0', port=5001, debug=False)


