from flask import Flask
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
# from prometheus_fastapi_instrumentator import Instrumentator
from src.monitoring import ACTIVE_CONNECTIONS


def create_app(inference_service, db_manager):
    app = Flask(__name__)
    CORS(app)
    
    # Initialize Prometheus metrics
    metrics = PrometheusMetrics(app)
    # Instrumentator().instrument(app).expose(app)
    
    # Store services in app config for access in routes
    app.config['inference_service'] = inference_service
    app.config['db_manager'] = db_manager
    
    # Register routes
    from api.routes import register_routes
    register_routes(app)
    
    @app.before_request
    def before_request():
        ACTIVE_CONNECTIONS.inc()
        
    @app.after_request
    def after_request(response):
        ACTIVE_CONNECTIONS.dec()
        return response
        
    return app


