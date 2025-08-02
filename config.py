from pathlib import Path

# Directory Structure
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# File Paths
DB_PATH = LOGS_DIR / "predictions.db"
LOG_FILE = LOGS_DIR / "requests.log"
METADATA_FILE = MODELS_DIR / "metadata.json"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# Model Configuration
EXPERIMENT_NAME = "Iris_Classification_Experiment"
RETRAIN_THRESHOLD = 10
MODEL_CONFIGS = {
    "LogisticRegression": {'C': 1.0, 'max_iter': 200, 'random_state': 42},
    "RandomForest": {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
}

# Create directories
for directory in [MODELS_DIR, LOGS_DIR, DATA_DIR, MLRUNS_DIR]:
    directory.mkdir(exist_ok=True)
