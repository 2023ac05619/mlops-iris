FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    jupyter \
    ipython \
    debugpy

# Copy source code
COPY . .

# Change ownership to mlops user
RUN chown -R mlops:mlops /app

USER mlops

# Expose ports
EXPOSE 8000 5678

# Command for development (with debugger support)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY main.py config.py ./
COPY src/ ./src/
COPY api/ ./api/
COPY db/ ./db/

# Create necessary directories
RUN mkdir -p data models logs mlruns backups

# Copy any pre-trained models or initial data if available
COPY models/ ./models/ 2>/dev/null || true
COPY data/ ./data/ 2>/dev/null || true

# Change ownership to mlops user
RUN chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Multi-arch build stage
FROM production as multi-arch

# This stage can be used for building multi-architecture images
# Useful for deployment across different platforms (AMD64, ARM64)

# Distroless stage for maximum security
FROM gcr.io/distroless/python3:latest as distroless

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application
COPY --from=production --chown=nonroot:nonroot /app /app

WORKDIR /app

# Use distroless non-root user
USER nonroot

EXPOSE 8000

# Distroless images don't have shell, so use exec form
ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# GPU-enabled stage (for future ML acceleration)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip install --no-cache-dir \
    torch \
    tensorflow-gpu \
    cupy-cuda118

# Copy application
COPY . .

# Create mlops user
RUN groupadd -r mlops && useradd -r -g mlops mlops
RUN chown -R mlops:mlops /app

USER mlops

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# Default target is production
FROM production