"""
# MLOps Iris Classifier

This project demonstrates a complete MLOps pipeline for training, deploying, and monitoring an Iris flower classifier.

## Features

- **CI/CD Automation**: GitHub Actions for automated testing, building, and deployment.
- **Data Versioning**: DVC for tracking and versioning the dataset.
- **Experiment Tracking**: MLflow for logging experiments, parameters, and models.
- **FastAPI Application**: A high-performance API for serving model predictions.
- **Containerization**: Docker for packaging the application and its dependencies.
- **Monitoring**: Prometheus for collecting metrics and Grafana for visualization.
- **Automated Retraining**: A webhook to trigger retraining when new data is added.


## How to Run

1.  **Set up DVC remote storage.**
2.  **Configure GitHub secrets** for Docker Hub and SSH deployment.
3.  **Push to GitHub** to trigger the CI/CD pipeline.
"""
