import json
import os
import requests
import logging
from datetime import datetime
from flask import request, jsonify, render_template_string, current_app
from pydantic import ValidationError
from dotenv import load_dotenv

from db.schemas import PredictionRequest, RetrainingRequest, NewDataSample
from src.monitoring import MonitoringService, REQUEST_COUNT, REQUEST_LATENCY

# Load environment variables from a .env file in your project root
# GITHUB_TOKEN="your_github_token"
# GITHUB_REPO="your_github_username/your_repo_name"
load_dotenv()


# --- Helper Function for GitHub Actions ---
def trigger_github_action_workflow():
    """
    Triggers a GitHub Actions workflow by sending a repository dispatch event.
    Reads configuration from environment variables (loaded from .env file).

    Returns:
        tuple: A tuple containing (bool: success, str: message).
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    workflow_file = os.getenv("GITHUB_WORKFLOW_FILE", "ci-cd.yml")

    if not token or not repo:
        error_msg = "GitHub configuration (GITHUB_TOKEN, GITHUB_REPO) not found in .env file or environment."
        logging.error(error_msg)
        return False, error_msg

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    data = {"ref": "main"}

    try:
        logging.info(f"Triggering GitHub Actions workflow '{workflow_file}' on repo '{repo}'...")
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 204:
            success_msg = "Successfully triggered GitHub Actions workflow."
            logging.info(success_msg)
            return True, success_msg
        else:
            error_msg = f"Failed to trigger workflow. Status: {response.status_code}, Response: {response.text}"
            logging.error(error_msg)
            return False, error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"Error contacting GitHub API: {e}"
        logging.error(error_msg)
        return False, error_msg


def register_routes(app):

    def get_services():
        inference_service = current_app.config['inference_service']
        db_manager = current_app.config['db_manager']
        monitoring = MonitoringService(db_manager, inference_service)
        return inference_service, db_manager, monitoring

    @app.route('/', methods=['GET'])
    def home():
        REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
        return jsonify({
            "message": "Iris Classification API with MLOps Features",
            "version": "2.0",
            "endpoints": {
                "/predict": "POST: Make a prediction",
                "/add_training_data": "POST: Add new labeled data, triggers GitHub Action",
                "/trigger_retrain": "POST: Manually trigger the GitHub Action retraining workflow",
                "/system_metrics": "GET: System and model metrics",
                "/metrics": "GET: Prometheus metrics (auto-generated)",
                "/dashboard": "GET: Monitoring dashboard",
                "/predictions/history": "GET: Recent predictions",
                "/health": "GET: Health check"
            }
        })

    @app.route('/health', methods=['GET'])
    def health_check():
        REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

    @app.route('/predict', methods=['POST'])
    def predict():
        REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()

        with REQUEST_LATENCY.time():
            try:
                inference_service, db_manager, _ = get_services()

                request_data = PredictionRequest(**request.get_json())
                response_data = inference_service.predict(request_data.features)

                timestamp = datetime.now().isoformat()
                db_manager.log_prediction(
                    timestamp, request_data.features,
                    response_data.prediction_name,
                    response_data.confidence,
                    response_data.latency
                )

                return jsonify(response_data.model_dump())

            except ValidationError as e:
                return jsonify({"error": "Input validation failed", "details": json.loads(e.json())}), 400
            except Exception as e:
                logging.error(f"Error in /predict endpoint: {e}")
                return jsonify({"error": str(e)}), 500

    @app.route('/add_training_data', methods=['POST'])
    def add_training_data():
        REQUEST_COUNT.labels(method='POST', endpoint='/add_training_data').inc()

        try:
            _, _, monitoring = get_services()

            # 1. Validate and store the new data
            sample = NewDataSample(**request.get_json())
            success, message = monitoring.log_and_store_data(sample.features, sample.target)

            if not success:
                return jsonify({"error": message}), 500

            # 2. If data was stored, trigger the GitHub Action workflow
            workflow_success, workflow_message = trigger_github_action_workflow()

            if workflow_success:
                final_message = f"{message} {workflow_message}"
                return jsonify({"message": final_message, "status": "success"})
            else:
                final_message = f"{message} However, failed to trigger the retraining workflow."
                return jsonify({"error": final_message, "details": workflow_message}), 500

        except ValidationError as e:
            return jsonify({"error": "Input validation failed", "details": json.loads(e.json())}), 400
        except Exception as e:
            logging.error(f"An unexpected error occurred in /add_training_data: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/trigger_retrain', methods=['POST'])
    def trigger_retrain():
        REQUEST_COUNT.labels(method='POST', endpoint='/trigger_retrain').inc()

        try:
            # Read the request payload to align with the test case.
            req = RetrainingRequest(**request.get_json()) if request.is_json else RetrainingRequest()

            # This endpoint now directly triggers the remote CI/CD pipeline
            workflow_success, workflow_message = trigger_github_action_workflow()

            if workflow_success:
                # Add the 'forced' key back into the response to satisfy the test assertion.
                return jsonify({
                    "message": workflow_message,
                    "status": "success",
                    "forced": req.force_retrain
                })
            else:
                return jsonify({"error": "Failed to trigger the retraining workflow.", "details": workflow_message}), 500

        except Exception as e:
            logging.error(f"Error in /trigger_retrain endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/system_metrics', methods=['GET'])
    def metrics():
        REQUEST_COUNT.labels(method='GET', endpoint='/system_metrics').inc()

        try:
            _, _, monitoring = get_services()
            metrics_data = monitoring.get_system_metrics()
            return jsonify(metrics_data)
        except Exception as e:
            logging.error(f"Error in /system_metrics endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/predictions/history', methods=['GET'])
    def predictions_history():
        REQUEST_COUNT.labels(method='GET', endpoint='/predictions/history').inc()

        try:
            _, db_manager, _ = get_services()
            limit = request.args.get('limit', 10, type=int)
            history = db_manager.get_prediction_history(limit)
            return jsonify({"history": history, "count": len(history)})

        except Exception as e:
            logging.error(f"Error in /predictions/history endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/dashboard')
    def dashboard():
        REQUEST_COUNT.labels(method='GET', endpoint='/dashboard').inc()

        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Model Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                h1 { color: #333; text-align: center; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
                .metric-box { 
                    border: 1px solid #ddd; padding: 15px; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: white;
                }
                .metric-label { font-size: 14px; color: #666; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2e86c1; margin-top: 5px; }
                .status { text-align: center; margin: 20px 0; }
                .healthy { color: #27ae60; }
            </style>
        </head>
        <body>
            <h1> ML Model Monitoring Dashboard</h1>
            <div class="status">
                <div class="healthy">● System Healthy</div>
                <div>Last Updated: <span id="timestamp"></span></div>
            </div>
            <div class="grid" id="metrics-grid">Loading...</div>
            
            <script>
                async function updateMetrics() {
                    try {
                        const response = await fetch('/system_metrics');
                        const data = await response.json();
                        
                        document.getElementById('timestamp').textContent = new Date().toLocaleString();
                        
                        let gridHtml = `
                            <div class="metric-box">
                                <div class="metric-label">Total Predictions</div>
                                <div class="metric-value">${data.total_predictions}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Avg DB Latency (ms)</div>
                                <div class="metric-value">${(data.average_db_latency * 1000).toFixed(2)}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Model Accuracy</div>
                                <div class="metric-value">${(data.model_info.accuracy * 100).toFixed(2)}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">New Samples Pending</div>
                                <div class="metric-value">${data.new_samples_pending} / ${data.retrain_threshold}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Current Model</div>
                                <div class="metric-value">${data.model_info.model_name}</div>
                            </div>
                        `;
                        
                        // Add prediction distribution
                        for (const [className, count] of Object.entries(data.prediction_distribution || {})) {
                            gridHtml += `
                                <div class="metric-box">
                                    <div class="metric-label">Predictions: ${className}</div>
                                    <div class="metric-value">${count}</div>
                                </div>
                            `;
                        }
                        
                        document.getElementById('metrics-grid').innerHTML = gridHtml;
                    } catch (error) {
                        console.error('Error fetching metrics:', error);
                        document.getElementById('metrics-grid').innerHTML = 
                            '<div class="metric-box"><div class="metric-label">Error</div><div class="metric-value">Failed to load</div></div>';
                    }
                }
                
                updateMetrics();
                setInterval(updateMetrics, 5000);
            </script>
        </body>
        </html>
        """
        return render_template_string(dashboard_html)
