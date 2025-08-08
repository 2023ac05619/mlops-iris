import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
# from config import METADATA_FILE, EXPERIMENT_NAME
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)   
METADATA_FILE = MODELS_DIR / "metadata.json"
EXPERIMENT_NAME = "Iris_Classification_Experiment"


class MLflowManager:
    
    def __init__(self):
        self.client = MlflowClient()
        
    def demonstrate_tracking(self):
        print("\n--- MLflow Tracking Demonstration ---")
        
        try:
            # List experiments
            experiments = self.client.search_experiments()
            print(f"[INFO] Found {len(experiments)} experiments.")
            
            for exp in experiments:
                print(f"  - Experiment: '{exp.name}' (ID: {exp.experiment_id})")
            
            # Get specific experiment
            exp = self.client.get_experiment_by_name(EXPERIMENT_NAME)
            if not exp:
                print(f"[WARNING] Experiment '{EXPERIMENT_NAME}' not found.")
                return
                
            # List runs
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id], 
                order_by=["metrics.accuracy DESC"]
            )
            
            print(f"\n[INFO] Found {len(runs)} runs in '{exp.name}':")
            
            # Create runs summary
            run_data = []
            for run in runs:
                run_data.append({
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'accuracy': run.data.metrics.get('accuracy', 0.0),
                    'status': run.info.status,
                    'run_id': run.info.run_id[:8] + '...'  # Shortened for display
                })
                
            if run_data:
                df = pd.DataFrame(run_data)
                print(df.to_string(index=False))
            else:
                print("  No runs found.")
                
        except Exception as e:
            print(f"[ERROR] MLflow tracking demonstration failed: {e}")
            
        print("-------------------------------------\n")
        
    def show_detailed_run_info(self):
        print("\n--- Detailed Info for Best Run ---")
        
        try:
            # Load best run ID from metadata
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                
            run_id = metadata['best_run_id']
            run = self.client.get_run(run_id)
            
            print(f"Run ID: {run_id}")
            print(f"Model: {metadata['best_model']}")
            print(f"Accuracy: {metadata['best_accuracy']:.4f}")
            
            print("\nParameters:")
            for key, value in run.data.params.items():
                print(f"  {key}: {value}")
                
            print("\nMetrics:")
            for key, value in run.data.metrics.items():
                print(f"  {key}: {value:.4f}")
                
            print("\nArtifacts:")
            artifacts = self.client.list_artifacts(run_id)
            for artifact in artifacts:
                print(f"  - {artifact.path}")
                
        except Exception as e:
            print(f"[ERROR] Could not retrieve detailed run info: {e}")
            
        print("----------------------------------\n")
        
    def get_experiment_summary(self) -> dict:
        try:
            exp = self.client.get_experiment_by_name(EXPERIMENT_NAME)
            if not exp:
                return {"error": "Experiment not found"}
                
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.accuracy DESC"]
            )
            
            return {
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": exp.experiment_id,
                "total_runs": len(runs),
                "best_accuracy": runs[0].data.metrics.get('accuracy', 0.0) if runs else 0.0,
                "runs": [
                    {
                        "run_id": run.info.run_id,
                        "run_name": run.data.tags.get('mlflow.runName', 'N/A'),
                        "accuracy": run.data.metrics.get('accuracy', 0.0),
                        "status": run.info.status
                    }
                    for run in runs[:5]  # Top 5 runs
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}

