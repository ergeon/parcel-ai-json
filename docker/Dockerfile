# Unified Dockerfile for parcel-ai-json property detection service
# Includes: YOLOv8-OBB, detectree, SAM, SAM3, FastAPI service
# Build: docker build -t parcel-ai-json:latest --build-arg HF_TOKEN=your_token .
# Run: docker run -p 8000:8000 -e HF_TOKEN=your_token parcel-ai-json:latest

FROM python:3.12-slim

# Build argument for HuggingFace token (for SAM3)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ENABLE_MPS_FALLBACK=1

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

# Install Python dependencies (excluding GroundingDINO which needs special handling)
RUN grep -v "GroundingDINO" requirements.txt > requirements_base.txt && \
    pip install --no-cache-dir -r requirements_base.txt

# Install GroundingDINO separately with --no-build-isolation
# (avoids PyTorch installation in isolated build environment)
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/IDEA-Research/GroundingDINO.git

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

# Copy pre-downloaded models from local machine (avoids network downloads)
# This saves build time and bandwidth
COPY models/ /app/models/

# Install SAM3 package from models/sam3 directory
RUN if [ -d "/app/models/sam3" ]; then \
        echo "Installing SAM3 package..." && \
        cd /app/models/sam3 && \
        pip install -e . && \
        echo "SAM3 package installed successfully"; \
    else \
        echo "SAM3 directory not found - skipping SAM3 installation"; \
    fi

# Copy YOLO models to ultralytics cache directory
RUN mkdir -p /root/.ultralytics && \
    cp /app/models/yolov8*.pt /root/.ultralytics/ && \
    echo "YOLO models copied to /root/.ultralytics/" && \
    ls -lh /root/.ultralytics/

# Verify SAM models are available
RUN echo "SAM models available at /app/models/:" && \
    ls -lh /app/models/sam_vit_*.pth

# Verify HED fence detection model is available
RUN echo "HED fence model (mixed finetune) available at /app/models/:" && \
    ls -lh /app/models/hed_fence_mixed_finetune.pth

# Verify GroundingDINO model is available
RUN echo "GroundingDINO model available at /app/models/:" && \
    ls -lh /app/models/groundingdino_swinb_cogcoor.pth

# Verify SAM3 assets are available
RUN if [ -f "/app/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz" ]; then \
        echo "SAM3 BPE vocab available at /app/models/sam3/assets/" && \
        ls -lh /app/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz; \
    else \
        echo "SAM3 assets not found - SAM3 may not work properly"; \
    fi

# Copy pre-downloaded SAM3 model to HuggingFace cache directory
# This avoids downloading ~6.4GB on first run (just like SAM, YOLO, etc.)
COPY models/huggingface_cache/ /root/.cache/huggingface/

# Verify SAM3 model is available
RUN if [ -d "/root/.cache/huggingface/hub/models--facebook--sam3" ]; then \
        echo "SAM3 model available in HuggingFace cache:" && \
        du -sh /root/.cache/huggingface/hub/models--facebook--sam3; \
    else \
        echo "SAM3 model not found - will auto-download (~6.4GB) on first use if HF_TOKEN is provided"; \
    fi

# Pre-download detectree model data
RUN python -c "\
import detectree as dtr; \
print('Initializing detectree classifier...'); \
clf = dtr.Classifier(); \
print('Detectree ready'); \
"

# Note: DeepForest ResNet50 backbone auto-downloads on first use
# (~97MB from PyTorch Hub). Pre-downloading was removed to simplify build.

# Create directory for temporary file uploads
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run FastAPI service
CMD ["uvicorn", "parcel_ai_json.api:app", "--host", "0.0.0.0", "--port", "8000"]
