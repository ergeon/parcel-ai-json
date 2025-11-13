# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**parcel-ai-json** is a containerized AI/ML microservice for detecting property features in satellite imagery:
- Vehicles (cars, trucks) via YOLOv8-OBB
- Swimming pools via YOLOv8-OBB
- Amenities (tennis courts, basketball courts, etc.) via YOLOv8-OBB
- Trees via DeepForest + detectree
- SAM (Segment Anything Model) for general-purpose segmentation

**Key Architecture Principles:**
1. **Standalone package** - No dependencies on parcel-geojson (one-way dependency only)
2. **Docker-first deployment** - FastAPI REST API in unified container (~2-3GB)
3. **Geodesic accuracy** - Uses pyproj for proper WGS84 transformations (never hardcode Earth radius)
4. **Auto GPU detection** - Automatically uses CUDA/MPS/CPU via `device_utils.py`

## CRITICAL: Docker-First Architecture for Computer Vision Models

**ALL COMPUTER VISION MODELS RUN INSIDE DOCKER CONTAINERS**

### Architecture Overview

This project uses a **Docker-first architecture** where:
- **ALL ML models** (YOLO, DeepForest, detectree, SAM) run inside Docker containers
- **Scripts and tools** communicate with Docker containers via **REST API only**
- **Never use direct file system mounts or local model execution** in production scripts

### Why This Matters

1. **detectree** requires Docker because:
   - Uses PyTorch Lightning with specific version requirements
   - Has complex dependency chains (torch, detectron2, etc.)
   - Runs segmentation models that need isolated environment

2. **All other models** should follow the same pattern for consistency:
   - YOLO models (vehicles, pools, amenities)
   - DeepForest (tree detection)
   - SAM (Segment Anything Model)

### Correct Usage Pattern

#### ❌ WRONG - Direct File System Access
```python
# DON'T DO THIS - causes Docker volume mount errors
detector = PropertyDetectionService()
detections = detector.detect_all({
    "path": "output/examples/images/image.jpg",  # Relative path fails in Docker
    "center_lat": 37.7749,
    "center_lon": -122.4194,
})
```

#### ✅ CORRECT - Use REST API
```python
# DO THIS - Use FastAPI REST endpoint
import requests

# 1. Ensure Docker container is running
# docker-compose up -d

# 2. Call REST API endpoint
with open("output/examples/images/image.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "center_lat": 37.7749,
        "center_lon": -122.4194,
        "zoom_level": 20,
        "include_trees": True,
        "extract_tree_polygons": True
    }
    response = requests.post(
        "http://localhost:8000/api/v1/detect",
        files=files,
        data=data
    )
    detections = response.json()
```

### ⛔ STRICT PROHIBITION ⛔

**NEVER DIRECTLY INSTANTIATE MODEL SERVICES IN SCRIPTS - THIS IS STRICTLY PROHIBITED**

The following patterns are **ABSOLUTELY FORBIDDEN** in any script or tool:

```python
# ❌ FORBIDDEN - DO NOT DO THIS
from parcel_ai_json.property_detector import PropertyDetectionService
from parcel_ai_json.sam_segmentation import SAMSegmentationService

detector = PropertyDetectionService()  # FORBIDDEN!
sam_service = SAMSegmentationService()  # FORBIDDEN!
```

**Why this is strictly prohibited:**
1. **Breaks Docker-first architecture** - All models must run inside containers
2. **Causes volume mount errors** - Scripts cannot access Docker filesystem
3. **Bypasses isolation** - Models have complex dependencies requiring Docker
4. **Inconsistent behavior** - Direct instantiation works locally but fails in production

**When direct instantiation IS allowed:**
- Unit tests (with proper mocking)
- Model development/training
- Jupyter notebooks for exploration

**When direct instantiation is FORBIDDEN:**
- Production scripts (e.g., `generate_examples.py`, `create_sam_folium_map.py`)
- Data processing pipelines
- Any script intended for external use
- Command-line tools

If you find yourself tempted to instantiate a service directly, **STOP** and use the REST API instead.

### Development Workflow

1. **Start Docker container first**:
   ```bash
   docker-compose up -d
   # or
   make docker-run
   ```

2. **Test API is running**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Use REST API in all scripts**:
   - Never instantiate model services directly in scripts
   - Always use HTTP requests to Docker container API
   - Pass images as multipart/form-data uploads

### When Direct Model Usage is Acceptable

**ONLY** in these specific cases:
- **Unit tests** - Testing individual model components in isolation
- **Model development** - Training, fine-tuning, or debugging models
- **Jupyter notebooks** - Interactive exploration and prototyping

**NEVER** in:
- Production scripts
- Example generation scripts
- Data processing pipelines
- Any script intended for external use

### File Path Rules for Docker

When working with Docker containers:
- **Always use absolute paths** for volume mounts
- **Never use relative paths** like `output/examples/images`
- **Prefer REST API** over volume mounts entirely

Example:
```python
# BAD
docker_cmd = f"docker run -v output/images:/data ..."  # Fails!

# GOOD
import os
abs_path = os.path.abspath("output/images")
docker_cmd = f"docker run -v {abs_path}:/data ..."

# BEST
# Don't use volume mounts - use REST API instead!
```

### Adding This to New Scripts

When creating new scripts, **always**:
1. Check if Docker container is running (`curl http://localhost:8000/health`)
2. Use REST API endpoints (see `parcel_ai_json/api.py` for available endpoints)
3. Never instantiate `PropertyDetectionService`, `SAMSegmentationService`, etc. directly
4. Document Docker dependency clearly at top of script

### Summary

**Remember**:
- 🐳 **All CV models run in Docker**
- 🌐 **All scripts use REST API**
- 🚫 **No direct file system mounts with relative paths**
- ✅ **Test `curl http://localhost:8000/health` before running any script**

## Essential Commands

### Development Environment Setup
```bash
# Create virtualenv and install all dependencies
make install

# Activate virtualenv
source venv/bin/activate
```

### Testing
```bash
# Run all tests with coverage (must pass before any commit)
make test

# Run with verbose output
make test-verbose

# Generate HTML coverage report (target: 80%+ coverage)
make coverage-html

# Run single test file
pytest tests/test_vehicle_detector.py -v

# Run specific test
pytest tests/test_vehicle_detector.py::TestVehicleDetector::test_detect_from_image -v
```

### Code Quality (ZERO TOLERANCE POLICY)
```bash
# Format code (required before commit)
make format

# Check formatting without changes
make format-check

# Run linter (must show 0 errors)
make lint

# Run all checks (format, lint, test) - MUST PASS
make check
```

### Docker (Recommended Deployment)
```bash
# Build Docker image
make docker-build

# Run container locally
make docker-run

# View logs
make docker-logs

# Open shell in container
make docker-shell

# Stop container
make docker-stop

# Rebuild and restart
make docker-rebuild

# Docker Compose (with live code editing)
make docker-up
make docker-down
```

### Generate Examples
```bash
# Generate detection examples with all features
make generate-examples
```

## Code Architecture

### Core Module Structure
```
parcel_ai_json/
├── __init__.py              # Public API exports
├── property_detector.py     # Unified detector (orchestrates all)
├── vehicle_detector.py      # YOLOv8-OBB for vehicles
├── swimming_pool_detector.py # YOLOv8-OBB for pools
├── amenity_detector.py      # YOLOv8-OBB for amenities
├── tree_detector.py         # DeepForest + detectree
├── sam_segmentation.py      # SAM for general segmentation
├── coordinate_converter.py  # Pixel ↔ WGS84 (uses pyproj)
├── device_utils.py          # Auto GPU detection
└── api.py                   # FastAPI REST endpoints
```

### Detection Pipeline Flow
```
1. PropertyDetectionService (property_detector.py)
   ├── Initializes all individual detectors
   ├── Coordinates parallel detection calls
   └── Aggregates results into PropertyDetections

2. Individual Detectors (vehicle_detector.py, etc.)
   ├── Load YOLOv8-OBB model (auto-downloaded to ~/.ultralytics/)
   ├── Run inference on satellite image
   ├── Filter by confidence threshold
   └── Convert pixel coords → WGS84 using ImageCoordinateConverter

3. ImageCoordinateConverter (coordinate_converter.py)
   ├── Takes image metadata (center_lat, center_lon, zoom_level)
   ├── Uses pyproj.Geod for geodesic calculations
   └── Never uses hardcoded Earth radius

4. Output
   ├── List[Detection] objects (pixel + geo coordinates)
   └── GeoJSON FeatureCollection
```

### Coordinate Transformation (CRITICAL)

**ALWAYS use pyproj for geographic calculations:**
```python
from pyproj import Geod

# CORRECT
geod = Geod(ellps='WGS84')
az12, az21, distance = geod.inv(lon1, lat1, lon2, lat2)

# WRONG - Never hardcode Earth radius
R = 6371  # km - DON'T DO THIS
```

See `coordinate_converter.py` for the canonical implementation.

### GPU Support

All PyTorch models auto-detect best device:
- **CUDA** - ECS GPU instances (g4dn.xlarge)
- **MPS** - Apple Silicon (M1/M2/M3)
- **CPU** - Fallback

```python
from parcel_ai_json.device_utils import get_best_device
device = get_best_device()  # Returns "cuda", "mps", or "cpu"
```

Models automatically use detected device. See `GPU_SUPPORT.md` for details.

### FastAPI Service

The `api.py` module provides 6 REST endpoints:
- `GET /` - Health check
- `GET /health` - Detailed status
- `POST /detect` - Unified detection (all features)
- `POST /detect/vehicles` - Vehicles only
- `POST /detect/pools` - Pools only
- `POST /detect/amenities` - Amenities only
- `POST /detect/trees` - Trees only

Service runs on port 8000. API docs at `/docs`.

## Testing Standards (ZERO TOLERANCE)

### Critical Rules
1. **ALL tests must pass** - No exceptions for "pre-existing failures"
2. **Zero linter errors/warnings** - Fix immediately, no exceptions
3. **Coverage target: 80% minimum** - Check with `pytest --cov`
4. **Run checks before every commit**: `make check`

### Test Structure
```
tests/
├── test_vehicle_detector.py       # Vehicle detection
├── test_swimming_pool_detector.py # Pool detection
├── test_amenity_detector.py       # Amenity detection
├── test_tree_detector.py          # Tree detection (DeepForest + detectree)
├── test_sam_segmentation.py       # SAM segmentation
├── test_property_detector.py      # Unified detector
└── test_api.py                    # FastAPI endpoints
```

### Mocking Strategy
- Mock YOLO models for fast unit tests (no actual inference)
- Use small test images (100x100px) for integration tests
- Mock PIL.Image for I/O tests
- Never make actual model inference calls in unit tests

Example:
```python
@patch('parcel_ai_json.vehicle_detector.YOLO')
def test_detect_vehicles(mock_yolo):
    mock_results = [Mock(boxes=Mock(xyxy=[[10, 20, 50, 60]], conf=[0.9]))]
    mock_yolo.return_value.return_value = mock_results
    # ... test logic
```

## Model Management

### Auto-Downloaded Models
- **YOLOv8-OBB**: `yolov8m-obb.pt` (51MB) → `~/.ultralytics/`
- **DeepForest**: `weecology/deepforest-tree` → `~/.cache/torch/`

### Pre-Bundled in Docker
- **SAM**: `sam_vit_h_4b8939.pth` (2.4GB) → `/app/models/`
- **YOLO**: Pre-downloaded to `/root/.ultralytics/`

### Model Locations
```bash
# Local development
~/.ultralytics/yolov8m-obb.pt
~/.cache/torch/hub/weecology_deepforest-tree/

# Docker container
/root/.ultralytics/yolov8m-obb.pt
/app/models/sam_vit_h_4b8939.pth
```

Models auto-download on first use if not present.

## Common Development Workflows

### Adding a New Detection Feature
1. Create detector class in `parcel_ai_json/new_detector.py`
2. Implement detection logic with YOLOv8-OBB or other model
3. Use `ImageCoordinateConverter` for pixel → WGS84
4. Add tests in `tests/test_new_detector.py`
5. Update `PropertyDetectionService` in `property_detector.py`
6. Add API endpoint in `api.py`
7. Run `make check` - must pass with 0 errors
8. Update README.md with new feature

### Debugging Detection Issues
```bash
# Test with single image
python scripts/generate_examples.py

# Check device detection
python -c "from parcel_ai_json.device_utils import get_best_device; print(get_best_device())"

# Run API locally
make docker-run
curl -X POST http://localhost:8000/detect -F "image=@test.jpg" -F "center_lat=37.7749" -F "center_lon=-122.4194"

# Check logs
make docker-logs
```

### Performance Profiling
- Vehicle detection: ~7.5s per image (CPU), ~0.5-1s (GPU)
- SAM segmentation: ~25-35s (CPU), ~2-4s (GPU)
- Tree detection (DeepForest): ~3-5s (CPU), ~0.5-1s (GPU)

See `GPU_SUPPORT.md` for optimization strategies.

## Critical Implementation Rules

### 1. Coordinate Transformations
**NEVER hardcode geographic constants:**
- Earth radius (6371 km, 6378137 m, etc.)
- Coordinate system parameters

**ALWAYS use pyproj:**
```python
from pyproj import Geod
geod = Geod(ellps='WGS84')
az12, az21, distance = geod.inv(lon1, lat1, lon2, lat2)
```

### 2. Testing Philosophy
- Fix failing tests immediately (never commit with failures)
- No tolerance for linter errors
- Coverage must stay above 80%
- Mock external dependencies (models, APIs)

### 3. Docker-First Deployment
- Primary deployment: Docker container (~2-3GB)
- Legacy PyPI package still supported but not recommended
- All models pre-bundled in Docker image

### 4. Standalone Package
- parcel-ai-json is independent (no imports from parcel-geojson)
- parcel-geojson may optionally import parcel-ai-json
- One-way dependency: parcel-geojson → parcel-ai-json

## Project-Specific Context

### Integration with parcel-geojson
This package is standalone. The parcel-geojson package may optionally import it:
```python
# In parcel-geojson (NOT in this repo):
try:
    from parcel_ai_json import VehicleDetectionService
except ImportError:
    print("Vehicle detection skipped - install parcel-ai-json")
```

### Deployment Targets
- **AWS ECS/Fargate** - Recommended (supports GPU instances)
- **Kubernetes** - Supported
- **Docker Compose** - Development
- **Lambda** - Not supported (package too large, use parcel-geojson instead)

### Version Management
Version defined in `setup.py`. Update for releases.

## Project File Structure

### Directory Organization
```
parcel-ai-json/
├── parcel_ai_json/          # Main package source code
│   ├── __init__.py          # Public API exports
│   ├── api.py               # FastAPI REST endpoints
│   ├── property_detector.py # Unified detector (orchestrates all)
│   ├── vehicle_detector.py  # YOLOv8-OBB for vehicles
│   ├── swimming_pool_detector.py  # YOLOv8-OBB for pools
│   ├── amenity_detector.py  # YOLOv8-OBB for amenities
│   ├── tree_detector.py     # DeepForest + detectree
│   ├── sam_segmentation.py  # SAM for general segmentation
│   ├── coordinate_converter.py  # Pixel ↔ WGS84 conversions
│   └── device_utils.py      # Auto GPU detection
│
├── tests/                   # Unit and integration tests
│   ├── test_api.py
│   ├── test_vehicle_detector.py
│   ├── test_swimming_pool_detector.py
│   ├── test_amenity_detector.py
│   ├── test_tree_detector.py
│   ├── test_sam_segmentation.py
│   └── test_property_detector.py
│
├── scripts/                 # Development and testing scripts
│   ├── generate_examples.py # Generate detection examples
│   ├── test_sam.py          # Test SAM on sample images
│   ├── batch_sam_all_models.py  # Compare SAM models
│   ├── batch_sam_comparison.py  # Batch SAM comparisons
│   ├── compare_sam_models.py    # SAM model analysis
│   ├── create_sam_folium_map.py # SAM visualization
│   └── generate_via_api.py  # API testing script
│
├── docs/                    # **ALL documentation .md files go here**
│   ├── ARCHITECTURE.md      # System architecture and design
│   ├── DOCKER_MIGRATION.md  # Docker deployment guide
│   └── SAM_INTEGRATION_PLAN.md  # SAM integration roadmap
│
├── docker/                  # Docker configuration files
│   ├── Dockerfile           # Main production Dockerfile
│   ├── Dockerfile.tree      # Tree detection specialized image
│   └── docker-compose.yml   # Development compose config
│
├── models/                  # ML model checkpoints
│   ├── README.md            # Model download instructions
│   ├── sam_vit_*.pth        # SAM model checkpoints
│   └── yolov8*.pt           # YOLO model weights
│
├── output/                  # Generated outputs (not in git)
│   └── examples/            # Detection results, visualizations
│
├── examples/                # Usage example scripts
│   └── detect_vehicles_example.py
│
├── build/                   # Build artifacts (not in git)
├── dist/                    # Distribution packages (not in git)
└── venv/                    # Python virtualenv (not in git)
```

### Documentation Guidelines

**CRITICAL**: All documentation `.md` files MUST be stored in the `/docs` directory.

**Exceptions** (root-level documentation only):
- `README.md` - Main project README (user-facing)
- `CLAUDE.md` - Claude Code instructions (this file)
- `GPU_SUPPORT.md` - GPU detection documentation
- `models/README.md` - Model-specific instructions

**New documentation should go in `/docs`**:
- Architecture documents → `/docs/ARCHITECTURE.md`
- Deployment guides → `/docs/DOCKER_MIGRATION.md`
- Feature plans → `/docs/SAM_INTEGRATION_PLAN.md`
- Design decisions → `/docs/DESIGN_DECISIONS.md`
- API specifications → `/docs/API_SPEC.md`

### File Naming Conventions
- **Source files**: `snake_case.py` (e.g., `vehicle_detector.py`)
- **Test files**: `test_*.py` (e.g., `test_vehicle_detector.py`)
- **Scripts**: `snake_case.py` (e.g., `generate_examples.py`)
- **Documentation**: `SCREAMING_SNAKE_CASE.md` (e.g., `ARCHITECTURE.md`)
- **Config files**: Lowercase or project conventions (e.g., `Dockerfile`, `Makefile`)

## Related Documentation
- `README.md` - User-facing documentation and API guide
- `docs/ARCHITECTURE.md` - Detailed architecture and design decisions
- `docs/DOCKER_MIGRATION.md` - Docker deployment guide
- `GPU_SUPPORT.md` - GPU detection and optimization
- `models/README.md` - Model management and downloads
