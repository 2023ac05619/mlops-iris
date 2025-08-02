#!/bin/bash

echo "--- Starting Local CI Pipeline ---"
set -e

echo "[1/2] Running linter (flake8)..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
echo "✅ Linting complete."

echo "[2/2] Running unit tests (pytest)..."
python -m pytest tests/ -v
echo "✅ Testing complete."

echo "--- Local CI Pipeline Succeeded ---"