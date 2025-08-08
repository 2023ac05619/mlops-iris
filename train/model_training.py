from mlops_pipeline import MLOpsPipeline
import joblib

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def train():    
    pipeline = MLOpsPipeline()
    try:
        # Load the data processed by the previous stage
        processedFile = DATA_DIR / 'processed_data.pkl'
        data = joblib.load(processedFile)
        pipeline.run_model_training(data)
        pipeline.demonstrate_mlflow()
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    train()