import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_pipeline import DataPipeline
import joblib
from db.database import DatabaseManager
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def preprocess():
    
    try:      
        db_manager = DatabaseManager()
        db_manager.connect()
        pipeline = DataPipeline(db_manager)
        
        processed_data = pipeline.load_and_preprocess()
        print("[INFO] Data preprocessing completed successfully.")
        processedFile = DATA_DIR / 'processed_data.pkl'
        joblib.dump(processed_data, processedFile)
        print(f'[INFO] Processed data saved to {DATA_DIR}/processed_data.pkl')
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    preprocess()