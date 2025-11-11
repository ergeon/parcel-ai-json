# Unified Dockerfile for parcel-ai-json property detection service
# Includes: YOLOv8-OBB, detectree, FastAPI service
# Build: docker build -t parcel-ai-json:latest .
# Run: docker run -p 8000:8000 parcel-ai-json:latest

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, GDAL, PyTorch, and SAM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for tree detection and API
RUN pip install --no-cache-dir \
    detectree \
    deepforest>=2.0.0 \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy application code
COPY parcel_ai_json/ ./parcel_ai_json/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .

# Install package in development mode
RUN pip install -e .

# Download YOLO models to bake into image (avoid first-run download)
# This makes the image larger but ensures consistent startup time
RUN python -c "\
from ultralytics import YOLO; \
import os; \
os.makedirs('/root/.ultralytics', exist_ok=True); \
print('Downloading YOLOv8m-OBB model...'); \
model = YOLO('yolov8m-obb.pt'); \
print('Model downloaded and cached'); \
"

# Pre-download detectree model data
RUN python -c "\
import detectree as dtr; \
print('Initializing detectree classifier...'); \
clf = dtr.Classifier(); \
print('Detectree ready'); \
"

# Pre-download DeepForest model (downloads on first use in DeepForestService)
# Model will be cached at /root/.cache/torch when first loaded
RUN python -c "\
print('DeepForest will download model on first use'); \
print('Model cache directory: /root/.cache/torch'); \
"

# Pre-download SAM model (ViT-B variant for speed/quality balance)
RUN mkdir -p /app/models && \
    cd /app/models && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && \
    echo "SAM ViT-B model downloaded to /app/models/"

# Create directory for temporary file uploads
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run FastAPI service
CMD ["uvicorn", "parcel_ai_json.api:app", "--host", "0.0.0.0", "--port", "8000"]
