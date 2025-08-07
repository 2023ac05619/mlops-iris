import pandas as pd
import yaml
import joblib
import json
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from datetime import datetime
import sys

# --- Fix for ModuleNotFoundError ---
# sys.path.append(str(Path(__file__).parent.parent))
# -----------------------------------
# from config import MODELS_DIR, TRAIN_DATA_FILE
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
TRAIN_DATA_FILE = DATA_DIR / "processed" / "train.csv"

def run_model_pipeline():
    """
    Trains models, logs to MLflow, and saves artifacts for DVC.
    """
    print("--- DVC Stage: Running Model Training ---")
    
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
        training_params = params['training']
        random_state = params['RANDOM_STATE'] # Get top-level random state
        
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    mlflow.set_experiment(training_params['mlflow_experiment_name'])
    
    models_trained = {}
    
    with mlflow.start_run(run_name="LogisticRegression") as run:
        lr_params = training_params['models']['LogisticRegression']
        mlflow.log_params(lr_params)
        lr = LogisticRegression(**lr_params, random_state=random_state)
        lr.fit(X_train_scaled, y_train)
        mlflow.sklearn.log_model(lr, "model")
        models_trained['LogisticRegression'] = {'run_id': run.info.run_id, 'model': lr}

    with mlflow.start_run(run_name="RandomForest") as run:
        rf_params = training_params['models']['RandomForest']
        mlflow.log_params(rf_params)
        rf = RandomForestClassifier(**rf_params, random_state=random_state)
        rf.fit(X_train_scaled, y_train)
        mlflow.sklearn.log_model(rf, "model")
        models_trained['RandomForest'] = {'run_id': run.info.run_id, 'model': rf}
        
    MODELS_DIR.mkdir(exist_ok=True)
    
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    for name, data in models_trained.items():
        joblib.dump(data['model'], MODELS_DIR / f"{name.lower()}_model.pkl")

    metadata = {
        'models_trained': list(models_trained.keys()),
        'training_timestamp': datetime.now().isoformat(),
        'feature_names': list(X_train.columns),
        'target_names': sorted(list(y_train.unique()))
    }
    with open(MODELS_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("--- DVC Stage: Model Training Finished ---")

if __name__ == "__main__":
    run_model_pipeline()