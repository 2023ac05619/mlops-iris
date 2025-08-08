from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# --- Directories ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Files ---
# Corrected path to point directly to iris_full.csv
RAW_DATA_FILE = DATA_DIR / "iris_full.csv"
TRAIN_DATA_FILE = DATA_DIR / "processed" / "train.csv"
TEST_DATA_FILE = DATA_DIR / "processed" / "test.csv"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
METADATA_FILE = MODELS_DIR / "metadata.json"

# --- API & Deployment ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))