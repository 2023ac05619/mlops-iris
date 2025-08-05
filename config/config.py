from pathlib import Path
import os
from dotenv import load_dotenv

# --- Directories ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# --- Files ---
RAW_DATA_FILE = DATA_DIR / "iris.csv"
TRAIN_DATA_FILE = DATA_DIR / "train.csv"
TEST_DATA_FILE = DATA_DIR / "test.csv"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
METADATA_FILE = MODELS_DIR / "metadata.json"

# --- DVC ---
DVC_REMOTE = "dvc_remote"

# --- Model Training ---
EXPERIMENT_NAME = "Iris_Classification_Experiment"
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_CONFIGS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 200, "random_state": RANDOM_STATE},
    "RandomForest": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE},
}

# --- API & Deployment ---
envValues = load_dotenv()
if envValues:
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_REPO = os.getenv("GITHUB_REPO")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    MLFLOW_UI_PORT = int(os.getenv("MLFLOW_UI_PORT", 5000))
else:
    GITHUB_TOKEN = None
    GITHUB_REPO = None

# Create necessary directories
for directory in [MODELS_DIR, LOGS_DIR, DATA_DIR, MLRUNS_DIR]:
    directory.mkdir(exist_ok=True)


