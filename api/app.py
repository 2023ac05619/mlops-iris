from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
from typing import Optional, Dict, Any, List
import httpx
import logging

from db.schemas import (
    PredictionRequest, PredictionResponse, BulkPredictionResponse,
    RetrainingRequest, NewDataSample, HealthResponse, 
    MetricsResponse, PredictionHistoryResponse
)
from src.monitoring import MonitoringService
from src.github_trigger import GitHubActionsTrigger

logger = logging.getLogger(__name__)


def create_app(inference_service, db_manager, lifespan=None):
    """Create FastAPI application with dependency injection."""
    
    app = FastAPI(
        title="MLOps Iris Classifier API",
        description="Production-ready ML API with dual model support, monitoring, and automated retraining",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup Prometheus monitoring
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app)
    
    # Store services in app state
    app.state.inference_service = inference_service
    app.state.db_manager = db_manager
    app.state.monitoring = MonitoringService(db_manager, inference_service)
    app.state.github_trigger = GitHubActionsTrigger()
    
    # Dependency injection helpers
    def get_inference_service():
        return app.state.inference_service
    
    def get_db_manager():
        return app.state.db_manager
    
    def get_monitoring():
        return app.state.monitoring
    
    def get_github_trigger():
        return app.state.github_trigger
    
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """API documentation endpoint."""
        return {
            "message": "MLOps Iris Classification API with Dual Model Support",
            "version": "3.0.0",
            "features": [
                "Dual model inference (LogisticRegression & RandomForest)",
                "Automated retraining with GitHub Actions",
                "Prometheus metrics integration",
                "DVC data versioning",
                "MLflow experiment tracking",
                "Input validation with Pydantic"
            ],
            "endpoints": {
                "/predict": "POST: Make predictions (single or both models)",
                "/predict/bulk": "POST: Bulk predictions",
                "/add_training_data": "POST: Add new labeled data (triggers retraining)",
                "/trigger_retrain": "POST: Manually trigger retraining",
                "/system_metrics": "GET: System and model metrics",
                "/predictions/history": "GET: Recent predictions history",
                "/models/info": "GET: Information about loaded models",
                "/health": "GET: Health check",
                "/metrics": "GET: Prometheus metrics"
            },
            "models_available": ["logisticregression", "randomforest"],
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/predict", response_model=BulkPredictionResponse)
    async def predict(
        request: PredictionRequest,
        inference_service=Depends(get_inference_service),
        db_manager=Depends(get_db_manager)
    ):
        """Make predictions using one or both models."""
        try:
            predictions = {}
            
            if request.model_name:
                # Single model prediction
                if request.model_name.lower() not in ["logisticregression", "randomforest"]:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid model name. Available: logisticregression, randomforest"
                    )
                
                result = inference_service.predict(request.features, request.model_name.lower())
                predictions[request.model_name.lower()] = result
                
            else:
                # Both models prediction
                for model_name in ["logisticregression", "randomforest"]:
                    try:
                        result = inference_service.predict(request.features, model_name)
                        predictions[model_name] = result
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {e}")
                        continue
            
            if not predictions:
                raise HTTPException(status_code=500, detail="No predictions could be made")
            
            # Log predictions to database
            timestamp = datetime.now().isoformat()
            for model_name, result in predictions.items():
                db_manager.log_prediction(
                    timestamp, request.features, model_name,
                    result.prediction_name, result.confidence, result.latency
                )
            
            return BulkPredictionResponse(
                predictions=predictions,
                timestamp=timestamp,
                features_used=request.features
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/bulk", response_model=List[BulkPredictionResponse])
    async def bulk_predict(
        requests: List[PredictionRequest],
        inference_service=Depends(get_inference_service),
        db_manager=Depends(get_db_manager)
    ):
        """Bulk predictions endpoint."""
        if len(requests) > 100:  # Limit bulk requests
            raise HTTPException(status_code=400, detail="Maximum 100 requests per bulk call")
        
        results = []
        for req in requests:
            try:
                # Reuse the single predict logic
                pred_result = await predict(req, inference_service, db_manager)
                results.append(pred_result)
            except Exception as e:
                logger.error(f"Error in bulk prediction: {e}")
                # Continue with other predictions
                continue
        
        return results
    
    @app.post("/add_training_data")
    async def add_training_data(
        sample: NewDataSample,
        background_tasks: BackgroundTasks,
        monitoring=Depends(get_monitoring),
        github_trigger=Depends(get_github_trigger)
    ):
        """Add new training data and optionally trigger retraining via GitHub Actions."""
        try:
            # Store training data
            success, message = monitoring.log_and_store_data(sample.features, sample.target)
            
            if not success:
                raise HTTPException(status_code=500, detail=message)
            
            # Check if retraining should be triggered
            should_retrain = monitoring.check_retrain_condition()
            
            response_data = {
                "message": message,
                "samples_pending": monitoring.db_manager.count_new_samples(),
                "retrain_threshold": monitoring.retrain_threshold,
                "retraining_triggered": False
            }
            
            if should_retrain:
                # Trigger GitHub Actions workflow
                background_tasks.add_task(
                    github_trigger.trigger_retraining_workflow,
                    reason="automatic_threshold_reached"
                )
                response_data["retraining_triggered"] = True
                response_data["message"] += " Automated retraining triggered via GitHub Actions."
            
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/trigger_retrain")
    async def trigger_retrain(
        request: RetrainingRequest,
        background_tasks: BackgroundTasks,
        monitoring=Depends(get_monitoring),
        github_trigger=Depends(get_github_trigger)
    ):
        """Manually trigger retraining via GitHub Actions."""
        try:
            should_retrain = (
                request.force_retrain or 
                monitoring.check_retrain_condition(request.trigger_threshold)
            )
            
            if should_retrain:
                # Trigger GitHub Actions workflow
                background_tasks.add_task(
                    github_trigger.trigger_retraining_workflow,
                    reason="manual_trigger" if request.force_retrain else "threshold_reached"
                )
                
                return {
                    "message": "Retraining triggered successfully via GitHub Actions",
                    "forced": request.force_retrain,
                    "samples_pending": monitoring.db_manager.count_new_samples(),
                    "workflow_triggered": True
                }
            else:
                return {
                    "message": "Not enough new samples to trigger retraining",
                    "forced": False,
                    "samples_pending": monitoring.db_manager.count_new_samples(),
                    "required_samples": request.trigger_threshold,
                    "workflow_triggered": False
                }
                
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system_metrics", response_model=MetricsResponse)
    async def system_metrics(monitoring=Depends(get_monitoring)):
        """Get comprehensive system metrics."""
        try:
            metrics_data = monitoring.get_system_metrics()
            return MetricsResponse(**metrics_data)
        except Exception as e:
            logger.error(f"Error retrieving system metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/predictions/history", response_model=PredictionHistoryResponse)
    async def predictions_history(
        limit: int = 10,
        model_name: Optional[str] = None,
        db_manager=Depends(get_db_manager)
    ):
        """Get prediction history with optional model filtering."""
        try:
            if limit > 100:
                raise HTTPException(status_code=400, detail="Maximum limit is 100")
            
            history = db_manager.get_prediction_history(limit, model_name)
            return PredictionHistoryResponse(
                history=history,
                count=len(history),
                limit=limit,
                model_filter=model_name
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/info")
    async def models_info(inference_service=Depends(get_inference_service)):
        """Get information about loaded models."""
        try:
            return {
                "models": inference_service.get_models_info(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/{model_name}/info")
    async def model_info(
        model_name: str,
        inference_service=Depends(get_inference_service)
    ):
        """Get information about a specific model."""
        try:
            if model_name.lower() not in ["logisticregression", "randomforest"]:
                raise HTTPException(status_code=404, detail="Model not found")
            
            models_info = inference_service.get_models_info()
            if model_name.lower() not in models_info:
                raise HTTPException(status_code=404, detail="Model not loaded")
            
            return {
                "model_name": model_name.lower(),
                "info": models_info[model_name.lower()],
                "timestamp": datetime.now().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
    
    return app