#!/bin/bash
# Download SAM3 model for Docker builds
# Usage: ./scripts/download_sam3_model.sh

set -e

echo "========================================"
echo "SAM3 Model Download Script"
echo "========================================"

# Load HF_TOKEN from .env if available
if [ -f .env ]; then
    export $(cat .env | grep HF_TOKEN | xargs)
fi

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set!"
    echo "Set it in .env file or export HF_TOKEN=your_token_here"
    exit 1
fi

echo "✓ HF_TOKEN is set"

# Check if model already exists
if [ -d "models/huggingface_cache/hub/models--facebook--sam3" ]; then
    echo "✓ SAM3 model already exists in models/huggingface_cache/"
    du -sh models/huggingface_cache/hub/models--facebook--sam3
    echo ""
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download"
        exit 0
    fi
fi

# Activate virtualenv if it exists
if [ -d "venv" ]; then
    echo "Activating virtualenv..."
    source venv/bin/activate
fi

# Download SAM3 model
echo ""
echo "Downloading SAM3 model (~6.4GB)..."
echo "This may take 5-15 minutes depending on your internet connection"
echo ""

python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('Loading SAM3 model builder...')
from sam3.model_builder import build_sam3_image_model

print('Downloading SAM3 model from HuggingFace...')
model = build_sam3_image_model()
print('✓ SAM3 model downloaded successfully!')
"

# Check if model was downloaded to cache
if [ ! -d "$HOME/.cache/huggingface/hub/models--facebook--sam3" ]; then
    echo "Error: Model not found in HuggingFace cache"
    exit 1
fi

echo ""
echo "Copying model to models/huggingface_cache/..."
mkdir -p models/huggingface_cache/hub
cp -r ~/.cache/huggingface/hub/models--facebook--sam3 models/huggingface_cache/hub/

echo ""
echo "========================================"
echo "✨ SAM3 model download complete!"
echo "========================================"
du -sh models/huggingface_cache/hub/models--facebook--sam3
echo ""
echo "The model is now ready to be included in Docker builds."
echo "Build Docker image with: docker build -f docker/Dockerfile ."
