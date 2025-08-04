import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """Track and save metrics for DVC monitoring."""
    
    def __init__(self):
        self.metrics_dir = Path("metrics")
        self.plots_dir = Path("plots")
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
    
    def save_data_metrics(self, data: Dict[str, Any]):
        """Save data processing metrics."""
        metrics = {
            "train_samples": data["X_train"].shape[0],
            "test_samples": data["X_test"].shape[0],
            "features": data["X_train"].shape[1],
            "classes": len(data["target_names"]),
            "train_mean": float(np.mean(data["X_train"])),
            "train_std": float(np.std(data["X_train"])),
            "test_mean": float(np.mean(data["X_test"])),
            "test_std": float(np.std(data["X_test"])),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.metrics_dir / "data_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    def save_model_metrics(self, model_results: Dict[str, Any], data: Dict[str, Any]):
        """Save model training metrics."""
        metrics = {
            "models_trained": len(model_results),
            "best_model": None,
            "best_accuracy": 0.0,
            "model_accuracies": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Find best model and collect accuracies
        for model_name, result in model_results.items():
            accuracy = result["accuracy"]
            metrics["model_accuracies"][model_name] = accuracy
            
            if accuracy > metrics["best_accuracy"]:
                metrics["best_accuracy"] = accuracy
                metrics["best_model"] = model_name
        
        with open(self.metrics_dir / "model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create model comparison plot
        self.create_model_comparison_plot(model_results)
    
    def save_evaluation_metrics(self, models: Dict[str, Any], X_test: np.ndarray, 
                              y_test: np.ndarray, target_names: list):
        """Save detailed model evaluation metrics."""
        evaluation_metrics = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_name, model_info in models.items():
            model = model_info["model"]
            y_pred = model.predict(X_test)
            
            # Classification report
            report = classification_report(y_test, y_pred, 
                                         target_names=target_names, 
                                         output_dict=True)
            
            evaluation_metrics["models"][model_name] = {
                "accuracy": float(report["accuracy"]),
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
                "per_class": {target_names[i]: report[str(i)] for i in range(len(target_names))}
            }
            
            # Create confusion matrix plot
            self.create_confusion_matrix_plot(y_test, y_pred, target_names, model_name)
        
        with open(self.metrics_dir / "evaluation_metrics.json", "w") as f:
            json.dump(evaluation_metrics, f, indent=2)
    
    def create_model_comparison_plot(self, model_results: Dict[str, Any]):
        """Create model comparison plot for DVC."""
        models = list(model_results.keys())
        accuracies = [model_results[model]["accuracy"] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save plot data for DVC
        plot_data = {
            "model_comparison": [
                {"model": model, "accuracy": acc} 
                for model, acc in zip(models, accuracies)
            ]
        }
        
        with open(self.plots_dir / "model_comparison.json", "w") as f:
            json.dump(plot_data, f, indent=2)
    
    def create_confusion_matrix_plot(self, y_test: np.ndarray, y_pred: np.ndarray, 
                                   target_names: list, model_name: str):
        """Create confusion matrix plot."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {model_name.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"confusion_matrix_{model_name.lower()}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
