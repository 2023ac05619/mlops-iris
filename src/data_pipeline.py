import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import yaml

# --- Fix for ModuleNotFoundError ---
# Add the project root directory to the Python path
# sys.path.append(str(Path(__file__).parent.parent))
# # -----------------------------------
# from config.config import DATA_DIR, RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "iris_full.csv"
TRAIN_DATA_FILE = DATA_DIR / "processed" / "train.csv"
TEST_DATA_FILE = DATA_DIR / "processed" / "test.csv"


def run_data_pipeline():
    """
    Loads raw data, cleans column names, splits the data, and saves the
    processed files based on parameters from params.yaml.
    """
    print("--- DVC Stage: Running Data Preprocessing ---")
    
    # Load parameters from params.yaml
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
        test_size = params['TEST_SIZE']
        random_state = params['RANDOM_STATE']
        
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the CSV, letting pandas automatically detect the header
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Clean up column names by removing "(cm)" and stripping whitespace
    # df.columns = [col.replace('(cm)', '').strip() for col in df.columns]
    df.columns = [col.replace('(cm)', '').strip().lower() for col in df.columns]
    
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Removed stratify=y to handle cases where a class has only one member
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_DATA_FILE, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_DATA_FILE, index=False)
    
    print("--- DVC Stage: Data Preprocessing Finished ---")

if __name__ == "__main__":
    run_data_pipeline()
