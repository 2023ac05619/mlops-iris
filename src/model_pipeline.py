import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

# Add root to path
ROOT_DIR_SCRIPT = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR_SCRIPT))

from config.config import (
    TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE, METADATA_FILE, MODELS_DIR,
    EXPERIMENT_NAME, MODEL_CONFIGS, TARGET_COLUMN, MLRUNS_DIR
)

def train_models():
    """Trains models, logs to MLflow, and saves the best artifacts."""
    print("--- Starting: Model Training ---")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_train = train_df.drop(TARGET_COLUMN, axis=1)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(TARGET_COLUMN, axis=1)
    y_test = test_df[TARGET_COLUMN]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    # Train and evaluate models
    models = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
    }

    model_artifacts = {}
    for name, model_class in models.items():
        with mlflow.start_run(run_name=name) as run:
            params = MODEL_CONFIGS[name]
            mlflow.log_params(params)

            model = model_class(**params)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, artifact_path=name)

            # Save model locally
            model_path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)

            model_artifacts[name] = {
                "run_id": run.info.run_id,
                "accuracy": accuracy,
                "path": str(model_path)
            }
            print(f"Trained {name} with Accuracy: {accuracy:.4f}")

    # Save metadata
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "models": model_artifacts,
        "scaler_path": str(SCALER_FILE),
        "training_timestamp": datetime.now().isoformat(),
        "feature_names": list(X_train.columns),
        "target_names": ["setosa", "versicolor", "virginica"] # Iris-specific
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {METADATA_FILE}")
    print("--- Finished: Model Training ---\n")

if __name__ == "__main__":
    train_models()

