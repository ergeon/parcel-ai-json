# SAM3 Docker Setup - Complete Guide

## ✅ SAM3 Model Pre-Bundled in Docker (Just Like SAM, YOLO, GroundingDINO)

The SAM3 model is now **pre-downloaded and included** in the Docker image, consistent with all other models.

---

## 📦 Model Bundling Strategy

All models are pre-bundled in the Docker image to avoid downloads at runtime:

| Model | Size | Location in Container | Pre-bundled |
|-------|------|----------------------|-------------|
| **YOLO (vehicles, pools, amenities)** | ~100MB | `/root/.ultralytics/` | ✅ Yes |
| **SAM (original)** | ~2.4GB | `/app/models/sam_vit_h_4b8939.pth` | ✅ Yes |
| **GroundingDINO** | ~660MB | `/app/models/groundingdino_swinb_cogcoor.pth` | ✅ Yes |
| **SAM3** | **~9.6GB** | `/root/.cache/huggingface/hub/models--facebook--sam3/` | ✅ **Yes** |
| **DeepForest** | ~97MB | Auto-downloads | ❌ No (lightweight) |
| **detectree** | ~50MB | Auto-downloads | ❌ No (lightweight) |

---

## 🏗️ Docker Build Process

### 1. **Prepare SAM3 Model Locally**

If you don't have the SAM3 model yet:

```bash
# Option A: Use the download script
./scripts/download_sam3_model.sh

# Option B: Manual download
export HF_TOKEN=your_token_here
python -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"
mkdir -p models/huggingface_cache/hub
cp -r ~/.cache/huggingface/hub/models--facebook--sam3 models/huggingface_cache/hub/
```

Verify the model is present:
```bash
du -sh models/huggingface_cache/hub/models--facebook--sam3
# Expected: ~9.6G
```

### 2. **Build Docker Image**

The Dockerfile now copies the SAM3 model into the image:

```bash
# Build with SAM3 model included
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t parcel-ai-json:latest \
  -f docker/Dockerfile .
```

### 3. **Verify SAM3 in Container**

Check that SAM3 model is present:

```bash
docker run --rm parcel-ai-json:latest \
  du -sh /root/.cache/huggingface/hub/models--facebook--sam3

# Expected output: ~9.6G
```

---

## 📁 File Structure

```
parcel-ai-json/
├── models/
│   ├── sam_vit_h_4b8939.pth              # SAM original (~2.4GB)
│   ├── groundingdino_swinb_cogcoor.pth   # GroundingDINO (~660MB)
│   ├── yolov8*.pt                         # YOLO models (~100MB)
│   ├── sam3/                              # SAM3 package
│   │   ├── sam3/                          # Python package
│   │   ├── assets/                        # BPE tokenizer
│   │   └── pyproject.toml
│   └── huggingface_cache/                 # SAM3 model cache (9.6GB)
│       └── hub/
│           └── models--facebook--sam3/    # SAM3 model files
│               ├── blobs/
│               ├── refs/
│               └── snapshots/
├── docker/
│   ├── Dockerfile                         # Updated with SAM3
│   └── docker-compose.yml                 # Updated with HF_TOKEN
└── scripts/
    └── download_sam3_model.sh             # Helper script
```

---

## 🐳 Dockerfile Changes

### Key Additions:

```dockerfile
# 1. Environment variables for HuggingFace authentication
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. Install SAM3 package
RUN if [ -d "/app/models/sam3" ]; then \
        echo "Installing SAM3 package..." && \
        cd /app/models/sam3 && \
        pip install -e . && \
        echo "SAM3 package installed successfully"; \
    fi

# 3. Copy pre-downloaded SAM3 model (KEY CHANGE!)
COPY models/huggingface_cache/ /root/.cache/huggingface/

# 4. Verify SAM3 model is present
RUN if [ -d "/root/.cache/huggingface/hub/models--facebook--sam3" ]; then \
        echo "SAM3 model available in HuggingFace cache:" && \
        du -sh /root/.cache/huggingface/hub/models--facebook--sam3; \
    fi
```

---

## 🚀 Usage

### Docker Run

```bash
# Run with SAM3 pre-loaded
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  parcel-ai-json:latest
```

### Docker Compose

```bash
# Ensure HF_TOKEN is in .env file
echo "HF_TOKEN=your_token_here" >> .env

# Start services (SAM3 model already included)
docker-compose up -d

# Test SAM3 endpoint (no download, instant startup!)
curl -X POST "http://localhost:8000/segment/sam3" \
     -F "image=@test.jpg" \
     -F "center_lat=37.7749" \
     -F "center_lon=-122.4194" \
     -F "prompts=houses,cars,trees"
```

---

## 💾 Git LFS Considerations

The SAM3 model is **9.6GB**, which is too large for standard git.

### Option A: Use Git LFS (Recommended for teams)

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: apt install git-lfs  # Ubuntu

# Track large files
git lfs track "models/huggingface_cache/**"
git add .gitattributes
git commit -m "Track SAM3 model with Git LFS"

# Add and commit SAM3 model
git add models/huggingface_cache/
git commit -m "Add SAM3 model to Docker builds"
git push
```

### Option B: Exclude from Git (Recommended for solo dev)

The model is already added to `.gitignore`:

```bash
# .gitignore entry (already added):
models/huggingface_cache/
```

Download separately for Docker builds:
```bash
./scripts/download_sam3_model.sh
```

---

## 📊 Docker Image Size Impact

| Build Configuration | Image Size |
|---------------------|-----------|
| **Without SAM3** | ~4.5GB |
| **With SAM3** | **~14GB** |
| **Increase** | +9.6GB |

**Trade-off:**
- ✅ **Benefit**: No download at runtime, instant startup
- ❌ **Cost**: Larger Docker image (+9.6GB)

**Recommendation**: Pre-bundle SAM3 for consistency with other models

---

## 🔄 CI/CD Pipeline

For automated builds without Git LFS:

```yaml
# .github/workflows/docker-build.yml
- name: Download SAM3 Model
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    ./scripts/download_sam3_model.sh

- name: Build Docker Image
  run: |
    docker build \
      --build-arg HF_TOKEN=$HF_TOKEN \
      -t parcel-ai-json:latest \
      -f docker/Dockerfile .
```

---

## ✅ Benefits of Pre-Bundling SAM3

1. **Consistent with other models** - SAM, YOLO, GroundingDINO all pre-bundled
2. **Faster container startup** - No 5-15 minute download on first run
3. **No runtime network dependency** - Works in air-gapped environments
4. **Predictable performance** - No surprise downloads in production
5. **Better developer experience** - Clone, build, run immediately

---

## 🎯 Summary

**Before:**
- SAM3 model downloaded on first use (~6.4GB, 5-15 min)
- Inconsistent with other models
- Slower first startup

**After:**
- SAM3 model pre-bundled in Docker image (9.6GB)
- Consistent with SAM, YOLO, GroundingDINO
- Instant startup, no runtime downloads

**Docker build includes all models out of the box! 🎉**
