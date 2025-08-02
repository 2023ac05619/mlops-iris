import mlflow
import mlflow.sklearn
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import MODELS_DIR, METADATA_FILE, SCALER_FILE, EXPERIMENT_NAME, MODEL_CONFIGS


class ModelPipeline:
    """Handles model training, evaluation, and artifact saving."""
    
    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression,
            "RandomForest": RandomForestClassifier
        }
        
    def train_single_model(self, model_name, model_class, params, X_train, y_train, X_test, y_test):
        """Train a single model with MLflow tracking."""
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"[INFO] {model_name}: Accuracy = {accuracy:.4f}")
            
            return model, accuracy, run.info.run_id
            
    def train_and_evaluate(self, data):
        """Train all models and select the best one."""
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        scaler = data['scaler']
        feature_names = data['feature_names']
        target_names = data['target_names']
        
        # Set up MLflow experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"[INFO] MLflow experiment set to '{EXPERIMENT_NAME}'")
        
        # Train all models
        results = {}
        for model_name, model_class in self.models.items():
            params = MODEL_CONFIGS[model_name]
            model, accuracy, run_id = self.train_single_model(
                model_name, model_class, params, X_train, y_train, X_test, y_test
            )
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'run_id': run_id
            }
            
        # Select best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_model_info = results[best_model_name]
        
        print(f"\n[INFO] Best model: {best_model_name} with accuracy: {best_model_info['accuracy']:.4f}")
        
        # Save artifacts
        self._save_model_artifacts(
            best_model_name, best_model_info, scaler, 
            feature_names, target_names, results
        )
        
        return best_model_info
        
    def _save_model_artifacts(self, best_model_name, best_model_info, scaler, 
                            feature_names, target_names, all_results):
        """Save model artifacts and metadata."""
        # Save best model
        best_model = best_model_info['model']
        model_path = MODELS_DIR / f"{best_model_name.lower()}_model.pkl"
        joblib.dump(best_model, model_path)
        
        # Save scaler
        joblib.dump(scaler, SCALER_FILE)
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'best_accuracy': best_model_info['accuracy'],
            'best_run_id': best_model_info['run_id'],
            'feature_names': feature_names,
            'target_names': target_names,
            'models_evaluated': list(all_results.keys()),
            'training_timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'scaler_path': str(SCALER_FILE)
        }
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"[INFO] Model artifacts saved to {MODELS_DIR}")

