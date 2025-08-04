import threading
import psutil
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from db.database import DatabaseManager
from config import RETRAIN_THRESHOLD
import logging

logger = logging.getLogger(__name__)

# Enhanced Prometheus metrics with model-specific tracking
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API Request Latency', ['endpoint'])
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['predicted_class'])
MODEL_PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Total predictions made by each model', 
    ['model_name', 'predicted_class']
)
MODEL_LATENCY_HISTOGRAM = Histogram(
    'model_prediction_duration_seconds', 
    'Model prediction latency', 
    ['model_name']
)
MODEL_ACCURACY_GAUGE = Gauge('model_accuracy', 'Model accuracy', ['model_name'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
RETRAIN_TRIGGER_COUNT = Counter('retraining_triggered_total', 'Total times retraining was triggered')
TRAINING_DATA_SAMPLES = Gauge('training_data_samples_total', 'Total training data samples')
PENDING_SAMPLES = Gauge('pending_training_samples', 'Samples pending for retraining')

# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')

# Model info
MODEL_INFO = Info('model_info', 'Information about loaded models', ['model_name'])


class MonitoringService:
    """Enhanced centralized monitoring and metrics service."""
    
    def __init__(self, db_manager: DatabaseManager, inference_service=None):
        self.db_manager = db_manager
        self.inference_service = inference_service
        self.retrain_threshold = RETRAIN_THRESHOLD
        self.start_time = time.time()
        
        # Start background metrics collection
        self._start_system_metrics_collection()
        
        # Initialize model metrics if inference service is available
        if self.inference_service and self.inference_service._is_loaded:
            self._initialize_model_metrics()
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    SYSTEM_CPU_USAGE.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    SYSTEM_MEMORY_USAGE.set(memory.percent)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    SYSTEM_DISK_USAGE.set(disk.percent)
                    
                    # Update pending samples
                    pending = self.db_manager.count_new_samples()
                    PENDING_SAMPLES.set(pending)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)  # Wait longer if there's an error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def _initialize_model_metrics(self):
        """Initialize model-specific metrics."""
        if not self.inference_service or not self.inference_service._is_loaded:
            return
        
        try:
            models_info = self.inference_service.get_models_info()
            
            for model_name, info in models_info.items():
                # Set model accuracy gauge
                MODEL_ACCURACY_GAUGE.labels(model_name=model_name).set(info['accuracy'])
                
                # Set model info
                MODEL_INFO.labels(model_name=model_name).info({
                    'model_type': info['model_name'],
                    'training_timestamp': info['training_timestamp'],
                    'is_best_model': str(info.get('is_best_model', False)),
                    'run_id': info.get('run_id', 'N/A')
                })
                
            logger.info("Model metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model metrics: {e}")
    
    def update_model_metrics(self):
        """Update model-specific metrics."""
        self._initialize_model_metrics()
    
    def check_retrain_condition(self, threshold: int = None) -> bool:
        """Check if retraining should be triggered."""
        if threshold is None:
            threshold = self.retrain_threshold
        return self.db_manager.count_new_samples() >= threshold
    
    def trigger_retraining(self, method: str = "local"):
        """Trigger model retraining - local or via GitHub Actions."""
        def retrain_task():
            logger.info(f"Starting background model retraining via {method}...")
            RETRAIN_TRIGGER_COUNT.inc()
            self.db_manager.mark_data_as_used()
            
            try:
                if method == "github":
                    # Trigger GitHub Actions workflow
                    from src.github_trigger import GitHubActionsTrigger
                    github_trigger = GitHubActionsTrigger()
                    success = github_trigger.trigger_retraining_workflow("threshold_reached")
                    
                    if success:
                        logger.info("GitHub Actions retraining workflow triggered successfully")
                    else:
                        logger.error("Failed to trigger GitHub Actions workflow")
                        # Fall back to local retraining
                        method = "local"
                
                if method == "local":
                    # Local retraining
                    from src.data_pipeline import DataPipeline
                    from src.model_pipeline import ModelPipeline
                    
                    # Run retraining pipeline
                    data_pipeline = DataPipeline()
                    model_pipeline = ModelPipeline()
                    
                    data = data_pipeline.load_and_preprocess()
                    model_pipeline.train_and_evaluate(data)
                    
                    # Reload models in inference service
                    if self.inference_service:
                        self.inference_service.reload_models()
                        # Update model metrics
                        self.update_model_metrics()
                    
                    logger.info("Local retraining and model reload complete")
                
            except Exception as e:
                logger.error(f"Background retraining failed: {e}")
                
        thread = threading.Thread(target=retrain_task, daemon=True)
        thread.start()
    
    def log_and_store_data(self, features: list, target: int, metadata: dict = None) -> tuple[bool, str]:
        """Store new training data and check for retraining."""
        try:
            timestamp = datetime.now().isoformat()
            self.db_manager.store_new_data(timestamp, features, target, metadata)
            
            # Update training data samples gauge
            total_samples = self.db_manager.get_total_training_samples()
            TRAINING_DATA_SAMPLES.set(total_samples)
            
            if self.check_retrain_condition():
                # For production, prefer GitHub Actions retraining
                self.trigger_retraining("github")
                message = "Training data added. Automated retraining triggered via GitHub Actions!"
            else:
                message = "Training data added successfully."
                
            return True, message
            
        except Exception as e:
            logger.error(f"Error storing training data: {e}")
            return False, str(e)
    
    def get_system_metrics(self) -> dict:
        """Compile comprehensive system and model metrics."""
        try:
            # Database statistics
            db_stats = self.db_manager.get_stats()
            
            # Model information
            model_info = {}
            if self.inference_service and self.inference_service._is_loaded:
                model_info = self.inference_service.get_models_info()
            
            # System health metrics
            system_health = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "uptime": time.time() - self.start_time
            }
            
            # Model-specific prediction statistics
            predictions_by_model = self.db_manager.get_predictions_by_model()
            average_latency_by_model = self.db_manager.get_average_latency_by_model()
            
            return {
                "total_predictions": db_stats.get("total_predictions", 0),
                "predictions_by_model": predictions_by_model,
                "average_latency_by_model": average_latency_by_model,
                "prediction_distribution": db_stats.get("prediction_distribution", {}),
                "new_samples_pending": self.db_manager.count_new_samples(),
                "retrain_threshold": self.retrain_threshold,
                "model_info": model_info,
                "system_health": system_health,
                "database_stats": db_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error compiling system metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_performance_comparison(self) -> dict:
        """Get detailed model performance comparison."""
        try:
            if not self.inference_service or not self.inference_service._is_loaded:
                return {"error": "Inference service not available"}
            
            models_info = self.inference_service.get_models_info()
            predictions_by_model = self.db_manager.get_predictions_by_model()
            latency_by_model = self.db_manager.get_average_latency_by_model()
            
            comparison = {
                "models": {},
                "comparison_metrics": {},
                "recommendation": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Collect model data
            for model_name, info in models_info.items():
                comparison["models"][model_name] = {
                    **info,
                    "total_predictions": predictions_by_model.get(model_name, 0),
                    "average_latency": latency_by_model.get(model_name, 0.0)
                }
            
            # Calculate comparison metrics
            if len(models_info) >= 2:
                accuracies = [info["accuracy"] for info in models_info.values()]
                latencies = [latency_by_model.get(name, 0.0) for name in models_info.keys()]
                
                comparison["comparison_metrics"] = {
                    "accuracy_range": max(accuracies) - min(accuracies),
                    "latency_range": max(latencies) - min(latencies),
                    "best_accuracy_model": max(models_info.keys(), key=lambda k: models_info[k]["accuracy"]),
                    "fastest_model": min(models_info.keys(), key=lambda k: latency_by_model.get(k, float('inf')))
                }
                
                # Simple recommendation logic
                best_acc_model = comparison["comparison_metrics"]["best_accuracy_model"]
                fastest_model = comparison["comparison_metrics"]["fastest_model"]
                
                if best_acc_model == fastest_model:
                    comparison["recommendation"] = best_acc_model
                else:
                    # Prefer accuracy over speed for this use case
                    comparison["recommendation"] = best_acc_model
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating model comparison: {e}")
            return {"error": str(e)}
    
    def get_training_data_insights(self) -> dict:
        """Get insights about training data distribution and quality."""
        try:
            insights = self.db_manager.get_training_data_insights()
            
            # Add retraining recommendations
            pending_samples = self.db_manager.count_new_samples()
            
            insights["retraining_status"] = {
                "samples_pending": pending_samples,
                "threshold": self.retrain_threshold,
                "ready_for_retraining": pending_samples >= self.retrain_threshold,
                "progress_percentage": min(100, (pending_samples / self.retrain_threshold) * 100)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting training data insights: {e}")
            return {"error": str(e)}
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics in the expected format."""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return ""
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        try:
            # Clear counters (note: this doesn't actually reset Prometheus counters)
            logger.warning("Metrics reset requested - Prometheus counters cannot be reset")
            
            # Re-initialize model metrics
            self.update_model_metrics()
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status of all services."""
        try:
            health = {
                "overall_status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }
            
            # Check database
            try:
                db_stats = self.db_manager.get_stats()
                health["services"]["database"] = {
                    "status": "healthy",
                    "total_predictions": db_stats.get("total_predictions", 0)
                }
            except Exception as e:
                health["services"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["overall_status"] = "degraded"
            
            # Check inference service
            if self.inference_service:
                try:
                    inference_health = self.inference_service.health_check()
                    health["services"]["inference"] = inference_health
                    if not inference_health.get("service_status") == "healthy":
                        health["overall_status"] = "degraded"
                except Exception as e:
                    health["services"]["inference"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health["overall_status"] = "degraded"
            else:
                health["services"]["inference"] = {
                    "status": "not_configured"
                }
                health["overall_status"] = "degraded"
            
            # Check system resources
            try:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                resource_status = "healthy"
                if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                    resource_status = "warning"
                    if health["overall_status"] == "healthy":
                        health["overall_status"] = "degraded"
                
                health["services"]["system_resources"] = {
                    "status": resource_status,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                }
            except Exception as e:
                health["services"]["system_resources"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["overall_status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class MetricsCollector:
    """Background metrics collector for advanced monitoring."""
    
    def __init__(self, monitoring_service: MonitoringService):
        self.monitoring_service = monitoring_service
        self.running = False
        self.collection_interval = 60  # seconds
    
    def start(self):
        """Start background metrics collection."""
        if self.running:
            return
        
        self.running = True
        thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        thread.start()
        logger.info("Background metrics collection started")
    
    def stop(self):
        """Stop background metrics collection."""
        self.running = False
        logger.info("Background metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                # Collect and update various metrics
                self._collect_prediction_metrics()
                self._collect_model_performance_metrics()
                self._collect_data_quality_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval * 2)  # Wait longer on error
    
    def _collect_prediction_metrics(self):
        """Collect prediction-related metrics."""
        try:
            # Get recent prediction statistics
            stats = self.monitoring_service.db_manager.get_recent_prediction_stats()
            
            # Update Prometheus metrics based on recent data
            for model_name, count in stats.get("predictions_by_model", {}).items():
                # These would typically be updated by the actual prediction calls
                pass
                
        except Exception as e:
            logger.error(f"Error collecting prediction metrics: {e}")
    
    def _collect_model_performance_metrics(self):
        """Collect model performance metrics."""
        try:
            if not self.monitoring_service.inference_service:
                return
            
            # Update model accuracy gauges
            models_info = self.monitoring_service.inference_service.get_models_info()
            for model_name, info in models_info.items():
                MODEL_ACCURACY_GAUGE.labels(model_name=model_name).set(info["accuracy"])
                
        except Exception as e:
            logger.error(f"Error collecting model performance metrics: {e}")
    
    def _collect_data_quality_metrics(self):
        """Collect data quality and training metrics."""
        try:
            # Update training data metrics
            total_samples = self.monitoring_service.db_manager.get_total_training_samples()
            TRAINING_DATA_SAMPLES.set(total_samples)
            
            pending_samples = self.monitoring_service.db_manager.count_new_samples()
            PENDING_SAMPLES.set(pending_samples)
            
        except Exception as e:
            logger.error(f"Error collecting data quality metrics: {e}")


# Global metrics collector instance
_metrics_collector = None

def start_metrics_collection(monitoring_service: MonitoringService):
    """Start global metrics collection."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(monitoring_service)
        _metrics_collector.start()

def stop_metrics_collection():
    """Stop global metrics collection."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop()
        _metrics_collector = None