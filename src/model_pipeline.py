import mlflow
import mlflow.sklearn
import joblib
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from typing import Dict, Any, Tuple

from config import MODELS_DIR, METADATA_FILE, SCALER_FILE, EXPERIMENT_NAME, MODEL_CONFIGS
from src.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class ModelPipeline:
    """Enhanced model training pipeline with comprehensive evaluation and dual model support."""
    
    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression,
            "RandomForest": RandomForestClassifier
        }
        self.model_configs = MODEL_CONFIGS
        self.metrics_tracker = MetricsTracker()
        
    def train_single_model(self, model_name: str, model_class, params: Dict[str, Any], 
                          X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          perform_cv: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model with comprehensive evaluation."""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            try:
                # Log parameters
                mlflow.log_params(params)
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X_train.shape[1])
                
                # Train model
                logger.info(f"Training {model_name} model...")
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_proba_test = model.predict_proba(X_test)
                
                # Basic metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                # Cross-validation if requested
                cv_scores = None
                if perform_cv:
                    logger.info(f"Performing cross-validation for {model_name}...")
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    mlflow.log_metric("cv_accuracy_mean", cv_mean)
                    mlflow.log_metric("cv_accuracy_std", cv_std)
                    mlflow.log_metric("cv_accuracy_min", cv_scores.min())
                    mlflow.log_metric("cv_accuracy_max", cv_scores.max())
                
                # Detailed classification metrics
                classification_rep = classification_report(y_test, y_pred_test, output_dict=True)
                
                # Log comprehensive metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("precision_macro", classification_rep['macro avg']['precision'])
                mlflow.log_metric("recall_macro", classification_rep['macro avg']['recall'])
                mlflow.log_metric("f1_macro", classification_rep['macro avg']['f1-score'])
                mlflow.log_metric("precision_weighted", classification_rep['weighted avg']['precision'])
                mlflow.log_metric("recall_weighted", classification_rep['weighted avg']['recall'])
                mlflow.log_metric("f1_weighted", classification_rep['weighted avg']['f1-score'])
                
                # Per-class metrics
                for class_idx, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
                    if str(class_idx) in classification_rep:
                        class_metrics = classification_rep[str(class_idx)]
                        mlflow.log_metric(f"{class_name}_precision", class_metrics['precision'])
                        mlflow.log_metric(f"{class_name}_recall", class_metrics['recall'])
                        mlflow.log_metric(f"{class_name}_f1", class_metrics['f1-score'])
                        mlflow.log_metric(f"{class_name}_support", class_metrics['support'])
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred_test)
                
                # Model complexity metrics
                if hasattr(model, 'n_estimators'):
                    mlflow.log_metric("n_estimators", model.n_estimators)
                if hasattr(model, 'max_depth') and model.max_depth:
                    mlflow.log_metric("max_depth", model.max_depth)
                if hasattr(model, 'C'):
                    mlflow.log_metric("regularization_C", model.C)
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    for i, importance in enumerate(feature_importance):
                        mlflow.log_metric(f"feature_importance_{i}", importance)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifact(METADATA_FILE, "metadata")
                mlflow.log_artifact(SCALER_FILE, "scaler")
                # Save model locally
                model_path = f"{MODELS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Model saved to {model_path}")
                # Log confusion matrix
                mlflow.log_artifact(cm, "confusion_matrix")
                # Log confusion matrix as JSON
                cm_json = json.dumps(cm.tolist())
                mlflow.log_text(cm_json, "confusion_matrix.json")
                logger.info(f"Confusion matrix logged for {model_name}.")
                # Log run metadata
                run_metadata = {
                    "model_name": model_name,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "cv_mean": cv_mean if perform_cv else None,
                    "cv_std": cv_std if perform_cv else None,
                    "classification_report": classification_rep
                }
                self.metrics_tracker.log_run_metadata(run.info.run_id, run_metadata)
                logger.info(f"Run metadata logged for {model_name}.")
                return model, run_metadata
            except Exception as e:
                logger.error(f"Error during training {model_name}: {e}")
                raise e from None
            
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train all models defined in the configuration."""
        results = {}
        for model_name, model_class in self.models.items():
            params = self.model_configs.get(model_name, {})
            logger.info(f"Starting training for {model_name} with parameters: {params}")
            model, metadata = self.train_single_model(
                model_name, model_class, params, 
                X_train, y_train, X_test, y_test
            )
            results[model_name] = {
                "model": model,
                "metadata": metadata
            }
            logger.info(f"Completed training for {model_name}.")
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                        models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Evaluate trained models on the test set."""
        evaluation_results = {}
        for model_name, model_info in models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_results[model_name] = {
                "accuracy": accuracy,
                "classification_report": classification_rep,
                "y_proba": y_proba.tolist()  # Convert to list for JSON serialization
            }
            logger.info(f"Evaluation completed for {model_name}.")
        return evaluation_results   
    
    def save_models(self, models: Dict[str, Any]) -> None:
        """Save trained models to disk."""
        for model_name, model_info in models.items():
            model = model_info['model']
            model_path = f"{MODELS_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model {model_name} saved to {model_path}.")
            mlflow.log_artifact(model_path, "models")
            self.metrics_tracker.save_model_metadata(model_name, model_path)
            logger.info(f"Model metadata saved for {model_name}.")
            mlflow.log_param("model_saved_path", model_path)
            mlflow.log_param("model_name", model_name)
            mlflow.sklearn.log_model(model, "model")
            logger.info(f"Model {model_name} logged to MLflow.")
            mlflow.end_run()
            logger.info(f"MLflow run ended for {model_name}.")
            self.metrics_tracker.log_model(model_name, model_path)
            logger.info(f"Model {model_name} logged to metrics tracker.")
            
    def load_model(self, model_name: str) -> Any:
        """Load a trained model from disk."""
        model_path = f"{MODELS_DIR}/{model_name}.pkl"
        try:
            model = joblib.load(model_path)
            logger.info(f"Model {model_name} loaded from {model_path}.")
            return model
        except FileNotFoundError:
            logger.error(f"Model {model_name} not found at {model_path}.")
            raise
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise e from None
        
    def load_models(self) -> Dict[str, Any]:
        """Load all trained models from disk."""
        models = {}
        for model_name in self.models.keys():
            try:
                model = self.load_model(model_name)
                models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        return models
                
                
                
                