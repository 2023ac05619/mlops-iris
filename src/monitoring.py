from prometheus_client import Gauge, REGISTRY, start_http_server
from prometheus_fastapi_instrumentator import Instrumentator
import time
import threading

# Import database functions
from db.database import get_prediction_counts

# --- System Metrics (from Instrumentator) ---
def setup_system_monitoring(app):
    """Sets up standard system metrics like request latency and counts."""
    Instrumentator().instrument(app).expose(app)

# --- Inference Prediction Metrics ---
INFERENCE_PREDICTIONS = Gauge(
    'inference_predictions_total',
    'Total number of inference predictions',
    ['model_name', 'predicted_class']
)

def update_inference_metrics():
    """Periodically queries the DB and updates the inference predictions gauge."""
    while True:
        try:
            counts = get_prediction_counts()
            for model_name, predicted_class, count in counts:
                INFERENCE_PREDICTIONS.labels(model_name=model_name, predicted_class=predicted_class).set(count)
        except Exception as e:
            print(f"[ERROR] Failed to update inference metrics: {e}")
        time.sleep(15) # Update every 15 seconds

def start_metrics_exporter():
    """Starts the background thread to export inference metrics."""
    metrics_thread = threading.Thread(target=update_inference_metrics, daemon=True)
    metrics_thread.start()
    # print("[INFO] Started background thread for exporting inference metrics.")
