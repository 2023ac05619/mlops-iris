#!/bin/bash

echo "Initializing DVC for MLOps project..."

# Initialize DVC
dvc init --no-scm

# Add data remote (using GitHub as storage)
dvc remote add -d origin https://github.com/yourusername/mlops-iris-classifier.git
dvc remote modify origin auth github

# Create directories
mkdir -p data models metrics plots

# Add initial data pipeline
dvc run -n data_loading \
    -d src/data_pipeline.py \
    -d config.py \
    -o data/iris_full.csv \
    python -c "from src.data_pipeline import DataPipeline; DataPipeline().load_and_preprocess()"

echo "DVC initialization completed!"
echo "Next steps:"
echo "1. Configure GitHub token: export GITHUB_TOKEN=your_token"
echo "2. Run: dvc push"
echo "3. Commit changes: git add . && git commit -m 'Initialize DVC'"
