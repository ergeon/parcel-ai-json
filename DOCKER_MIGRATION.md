# Docker Migration Plan - Run All Tree Detection in Docker

## Current Status

### What's Working Now
- ✅ **DeepForest**: Running natively in Python (uses PyTorch/transformers)
- ✅ **detectree**: Configured to run in Docker (default: `detectree_use_docker=True`)
- ✅ **YOLO (vehicles/pools/amenities)**: Running natively in Python (ultralytics)
- ✅ Parallel tree detection with both DeepForest and detectree
- ✅ Simplified GeoJSON polygons from detectree (no pixel masks)
- ✅ Topology-preserving polygon simplification (0.5m tolerance)

### Current Architecture
```
PropertyDetectionService
├── VehicleDetectionService (YOLO-OBB, native Python)
├── SwimmingPoolDetectionService (YOLO-OBB, native Python)
├── AmenityDetectionService (YOLO-OBB, native Python)
└── CombinedTreeDetectionService
    ├── DeepForestService (native Python, PyTorch)
    └── DetectreeService (Docker container)
```

### Recent Changes (Last 3 Commits)
1. `eaefe24` - Use simplified GeoJSON polygons instead of pixel masks
2. `e17390a` - Add parallel tree detection with DeepForest and detectree
3. `5024b6e` - Replace detectree with DeepForest for tree detection

## Problem Statement

**Goal**: Run ALL AI/ML models (YOLO, DeepForest, detectree) inside Docker containers for:
- Consistent environment across machines
- Easier deployment
- Better isolation
- Simplified dependency management
- Production-ready containerization

## Current Docker Setup

### Existing Docker Infrastructure
- `Dockerfile` - Main container with FastAPI service
- `Makefile` - Build and run commands
- `docker-tree-detector/` - Separate detectree container (legacy)

### What Needs Migration
1. **DeepForest** - Currently native Python, needs Dockerization
2. **YOLO models** - Currently native Python, needs Dockerization
3. **Unified container** - All models in single container vs separate containers

## Migration Plan

### Option 1: Single Unified Container (Recommended)
**Pros**:
- Simpler deployment
- Faster inter-service communication
- Smaller total image size
- Easier to manage

**Cons**:
- Larger single image
- All-or-nothing updates

**Architecture**:
```
docker-compose.yml
└── parcel-ai-json (single service)
    ├── FastAPI REST API
    ├── YOLO models (ultralytics)
    ├── DeepForest (transformers/PyTorch)
    └── detectree (native Python, no Docker subprocess)
```

### Option 2: Microservices Architecture
**Pros**:
- Independent scaling
- Isolated updates
- Better resource allocation

**Cons**:
- More complex orchestration
- Network overhead
- More containers to manage

**Architecture**:
```
docker-compose.yml
├── api-service (FastAPI)
├── yolo-service (vehicles/pools/amenities)
├── deepforest-service (tree crowns)
└── detectree-service (tree coverage)
```

## Recommended Approach: Option 1 (Unified Container)

### Step 1: Update Dockerfile
**File**: `Dockerfile`

**Changes needed**:
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libspatialindex-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time (optional, for faster startup)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-obb.pt')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('weecology/deepforest-tree')"

# Copy application code
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "parcel_ai_json.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Update tree_detector.py
**File**: `parcel_ai_json/tree_detector.py`

**Change to run natively INSIDE the Docker container** (no Docker-in-Docker):
```python
class CombinedTreeDetectionService:
    def __init__(
        self,
        # DeepForest parameters
        deepforest_model_name: str = "weecology/deepforest-tree",
        deepforest_confidence_threshold: float = 0.1,
        # detectree parameters
        detectree_use_docker: bool = False,  # Run natively inside container
        detectree_docker_image: str = "parcel-tree-detector",
        detectree_save_mask: bool = False,
        detectree_extract_polygons: bool = True,
        # ...
    ):
```

**Explanation**: Since ALL code will run inside a Docker container, we set `use_docker=False` to run detectree natively WITHIN that container, avoiding Docker-in-Docker complexity.

### Step 3: Update requirements.txt
**File**: `requirements.txt`

**Ensure all dependencies are listed**:
```txt
# Core dependencies
ultralytics>=8.0.0      # YOLOv8
pillow>=9.0.0
numpy>=1.20.0
torch>=2.0.0
torchvision>=0.15.0
pyproj>=3.0.0

# Tree detection
deepforest>=2.0.0       # Individual tree crowns
detectree==0.8.0        # Tree coverage polygons

# Transformers for DeepForest
transformers>=4.0.0
timm>=0.9.0

# API dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
```

### Step 4: Create docker-compose.yml
**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  parcel-ai-json:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./output:/app/output
      - ./data:/app/data
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/torch:/root/.cache/torch
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
    restart: unless-stopped
```

### Step 5: Update Makefile
**File**: `Makefile`

```makefile
.PHONY: docker-build docker-run docker-stop docker-clean

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f

docker-logs:
	docker-compose logs -f
```

### Step 6: Test the Migration
1. Build the container: `make docker-build`
2. Run the container: `make docker-run`
3. Test API endpoint: `curl http://localhost:8000/detect`
4. Generate examples: Inside container or via API
5. Verify outputs in `./output/` directory

## Implementation Checklist

### Phase 1: Preparation
- [ ] Document current native setup performance benchmarks
- [ ] Backup current working configuration
- [ ] Test detectree in native mode (already installed)
- [ ] Verify all dependencies in requirements.txt

### Phase 2: Docker Configuration
- [ ] Update Dockerfile with all dependencies
- [ ] Add model download steps (optional optimization)
- [ ] Create docker-compose.yml
- [ ] Update Makefile with new commands
- [ ] Configure volume mounts for outputs

### Phase 3: Code Changes
- [ ] Change `detectree_use_docker=False` in tree_detector.py
- [ ] Update API to handle file paths correctly in container
- [ ] Update example generation script for Docker paths
- [ ] Add health check endpoint to API

### Phase 4: Testing
- [ ] Build container and verify no errors
- [ ] Test YOLO detection in container
- [ ] Test DeepForest detection in container
- [ ] Test detectree detection in container
- [ ] Test combined tree detection
- [ ] Generate sample outputs and verify quality
- [ ] Performance comparison: native vs Docker

### Phase 5: Documentation
- [ ] Update README.md with Docker instructions
- [ ] Add Docker troubleshooting guide
- [ ] Document environment variables
- [ ] Add deployment guide

## Performance Considerations

### Model Loading Time
- **First run**: Models download from HuggingFace/Ultralytics (slow)
- **Subsequent runs**: Models cached in volumes (fast)
- **Optimization**: Pre-download models during build

### Memory Requirements
- YOLO models: ~100MB
- DeepForest: ~200MB
- detectree: ~50MB
- **Total**: ~2GB RAM recommended

### Storage Requirements
- Docker image: ~5GB
- Model cache: ~1GB
- Output files: Varies by usage

## Migration Timeline

### Immediate Next Steps (This Session)
1. Wait for current sample generation to complete
2. Commit DOCKER_MIGRATION.md plan document
3. Review current Dockerfile structure

### Next Session - Full Docker Migration
1. Update Dockerfile with ALL dependencies (YOLO, DeepForest, detectree)
2. Set `detectree_use_docker=False` to run inside container (not Docker-in-Docker)
3. Create docker-compose.yml for orchestration
4. Add model caching via volumes
5. Test Docker build
6. Verify all models work in container
7. Generate samples entirely in Docker
8. Performance comparison

### Future Work
1. Add GPU support (CUDA)
2. Optimize image size
3. Add model caching strategies
4. Create production deployment guide

## Current File Changes Needed

### 1. parcel_ai_json/tree_detector.py
**Line 676**: Change `detectree_use_docker: bool = True` to `detectree_use_docker: bool = False`
**Why**: When running inside Docker container, detectree should run natively within that container, not spawn another Docker container (no Docker-in-Docker)

### 2. Dockerfile (major rewrite needed)
Add all ML dependencies and model downloads

### 3. docker-compose.yml (new file)
Create orchestration configuration

### 4. .dockerignore (new file)
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.git/
.pytest_cache/
output/
*.log
```

## Questions to Answer

1. **Model Persistence**: Should models be baked into image or downloaded at runtime?
   - **Recommendation**: Bake into image for faster startup

2. **API vs CLI**: How will users interact with the container?
   - **Current**: FastAPI REST API
   - **Also support**: CLI for batch processing

3. **Output Handling**: How to get results out of container?
   - **Recommendation**: Volume mounts for output directory

4. **Scaling**: Single instance or multiple replicas?
   - **Start**: Single instance
   - **Future**: Kubernetes/Docker Swarm for scaling

## Resources

- Current Dockerfile: `Dockerfile`
- Current API: `parcel_ai_json/api.py`
- Tree detector: `parcel_ai_json/tree_detector.py`
- Docker tree detector: `docker-tree-detector/` (legacy, can remove)

## Notes

- DeepForest uses PyTorch and Hugging Face transformers
- detectree uses scikit-learn and OpenCV
- YOLO uses ultralytics library
- All models are compatible with Docker
- No GPU required (CPU inference works fine)
- macOS arm64 support via Docker's emulation
