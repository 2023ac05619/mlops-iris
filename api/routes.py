import time
from datetime import datetime
from flask import request, jsonify, render_template_string, current_app
from pydantic import ValidationError
from db.schemas import PredictionRequest, PredictionResponse, RetrainingRequest, NewDataSample
from src.monitoring import MonitoringService, REQUEST_COUNT, REQUEST_LATENCY


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
                "/add_training_data": "POST: Add new labeled data",
                "/trigger_retrain": "POST: Manually trigger retraining",
                "/metrics": "GET: System and model metrics",
                "/prometheus_metrics": "GET: Prometheus metrics",
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
                
                # Validate input
                request_data = PredictionRequest(**request.get_json())
                
                # Make prediction
                response_data = inference_service.predict(request_data.features)
                
                # Log to database
                timestamp = datetime.now().isoformat()
                db_manager.log_prediction(
                    timestamp, request_data.features,
                    response_data.prediction_name,
                    response_data.confidence,
                    response_data.latency
                )
                
                return jsonify(response_data.model_dump())
                
            except ValidationError as e:
                return jsonify({"error": "Input validation failed", "details": e.errors()}), 400
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    @app.route('/add_training_data', methods=['POST'])
    def add_training_data():
        REQUEST_COUNT.labels(method='POST', endpoint='/add_training_data').inc()
        
        try:
            _, _, monitoring = get_services()
            
            sample = NewDataSample(**request.get_json())
            success, message = monitoring.log_and_store_data(sample.features, sample.target)
            
            if success:
                return jsonify({"message": message})
            else:
                return jsonify({"error": message}), 500
                
        except ValidationError as e:
            return jsonify({"error": "Input validation failed", "details": e.errors()}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/trigger_retrain', methods=['POST'])
    def trigger_retrain():
        REQUEST_COUNT.labels(method='POST', endpoint='/trigger_retrain').inc()
        
        try:
            _, _, monitoring = get_services()
            
            req = RetrainingRequest(**request.get_json()) if request.is_json else RetrainingRequest()
            
            if req.force_retrain or monitoring.check_retrain_condition(req.trigger_threshold):
                monitoring.trigger_retraining()
                message = "Model retraining triggered successfully."
            else:
                message = "Not enough new samples to trigger retraining."
                
            return jsonify({"message": message, "forced": req.force_retrain})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        REQUEST_COUNT.labels(method='GET', endpoint='/metrics').inc()
        
        try:
            _, _, monitoring = get_services()
            metrics_data = monitoring.get_system_metrics()
            return jsonify(metrics_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/prometheus_metrics', methods=['GET'])
    def prometheus_metrics():
        return MonitoringService.get_prometheus_metrics()
    
    @app.route('/predictions/history', methods=['GET'])
    def predictions_history():
        REQUEST_COUNT.labels(method='GET', endpoint='/predictions/history').inc()
        
        try:
            _, db_manager, _ = get_services()
            
            limit = request.args.get('limit', 10, type=int)
            history = db_manager.get_prediction_history(limit)
            return jsonify({"history": history, "count": len(history)})
            
        except Exception as e:
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
            <h1>🤖 ML Model Monitoring Dashboard</h1>
            <div class="status">
                <div class="healthy">● System Healthy</div>
                <div>Last Updated: <span id="timestamp"></span></div>
            </div>
            <div class="grid" id="metrics-grid">Loading...</div>
            
            <script>
                async function updateMetrics() {
                    try {
                        const response = await fetch('/metrics');
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
