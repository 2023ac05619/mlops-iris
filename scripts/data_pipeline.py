import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add root to path to allow module imports
ROOT_DIR_SCRIPT = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR_SCRIPT))

from config.config import RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN

def load_raw_data():
    """Loads the Iris dataset and saves it to a CSV file."""
    print("--- Starting: Load Raw Data ---")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df[TARGET_COLUMN] = iris.target
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"Raw data saved to {RAW_DATA_FILE}")
    print("--- Finished: Load Raw Data ---\n")

def split_data():
    """Splits the raw data into training and testing sets."""
    print("--- Starting: Split Data ---")
    df = pd.read_csv(RAW_DATA_FILE)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    print(f"Training data saved to {TRAIN_DATA_FILE}")
    print(f"Testing data saved to {TEST_DATA_FILE}")
    print("--- Finished: Split Data ---\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "load":
            load_raw_data()
        elif sys.argv[1] == "split":
            split_data()

