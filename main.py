import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routes import router as api_router
from src.inference_service import get_inference_service
from src.monitoring import setup_system_monitoring, start_metrics_exporter
from db.database import initialize_db
from config import API_HOST, API_PORT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    initialize_db()
    get_inference_service()
    start_metrics_exporter()
    print(f"API running on http://{API_HOST}:{API_PORT}")
    yield
    

app = FastAPI(
    title="MLOps Iris Classifier API",
    description="API for Iris classification with DB logging and monitoring.",
    version="1.1.0",
    lifespan=lifespan
)

# Setup system monitoring (request latency, etc.)
setup_system_monitoring(app)

# Include the API router
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
