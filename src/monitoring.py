import threading
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from db.database import DatabaseManager
from config import RETRAIN_THRESHOLD


# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API Request Latency')
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['predicted_class'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
RETRAIN_TRIGGER_COUNT = Counter('retraining_triggered_total', 'Total times retraining was triggered')


class MonitoringService:
    
    def __init__(self, db_manager: DatabaseManager, inference_service=None):
        self.db_manager = db_manager
        self.inference_service = inference_service
        
    def check_retrain_condition(self, threshold: int = RETRAIN_THRESHOLD) -> bool:
        return self.db_manager.count_new_samples() >= threshold
        
    def trigger_retraining(self):
        def retrain_task():
            print("[INFO] Starting background model retraining...")
            RETRAIN_TRIGGER_COUNT.inc()
            self.db_manager.mark_data_as_used()
            
            try:
                # Import here to avoid circular imports
                from src.data_pipeline import DataPipeline
                from src.model_pipeline import ModelPipeline
                
                # Run retraining pipeline
                data_pipeline = DataPipeline()
                model_pipeline = ModelPipeline()
                
                data = data_pipeline.load_and_preprocess()
                model_pipeline.train_and_evaluate(data)
                
                # Reload model in inference service
                if self.inference_service:
                    self.inference_service.reload_model()
                    
                print("[INFO] Background retraining and model reload complete.")
                
            except Exception as e:
                print(f"[ERROR] Background retraining failed: {e}")
                
        thread = threading.Thread(target=retrain_task, daemon=True)
        thread.start()
        
    def log_and_store_data(self, features: list, target: int) -> tuple[bool, str]:
        try:
            timestamp = datetime.now().isoformat()
            self.db_manager.store_new_data(timestamp, features, target)
            
            if self.check_retrain_condition():
                self.trigger_retraining()
                message = "Training data added. Retraining triggered!"
            else:
                message = "Training data added successfully."
                
            return True, message
            
        except Exception as e:
            return False, str(e)
            
    def get_system_metrics(self) -> dict:
        db_stats = self.db_manager.get_stats()
        model_info = self.inference_service.get_model_info() if self.inference_service else {}
        
        return {
            **db_stats,
            "new_samples_pending": self.db_manager.count_new_samples(),
            "retrain_threshold": RETRAIN_THRESHOLD,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        

