# Parcel AI JSON

![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)
![Tests](https://img.shields.io/badge/tests-231%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

Unified property detection for satellite imagery with GeoJSON output.

## Quick Start

```bash
# 1. Build Docker image
make docker-build

# 2. Run service
make docker-run

# 3. Test API
curl http://localhost:8000/health

# 4. View interactive docs
open http://localhost:8000/docs
```

## Prerequisites

- **Python 3.10+** (for local development)
- **Docker** (for containerized deployment - recommended)
- **Make** (for build automation)
- **Git** (for version control)

**Optional:**
- **CUDA-capable GPU** (5-10x faster inference)
- **Docker Compose** (for multi-container setups)

## Features

- **Unified Detection**: Detect vehicles, pools, amenities, trees, fences, and SAM segments in one call
- **Vehicle Detection**: Cars, trucks, and other vehicles using YOLOv8-OBB (DOTA aerial dataset)
- **Swimming Pool Detection**: Detect swimming pools (DOTA class 14)
- **Amenity Detection**: Tennis courts, basketball courts, baseball diamonds, soccer fields, and track fields
- **Tree Coverage Detection**: Estimate tree coverage percentage using detectree (Docker-based)
- **Fence Detection**: Detect fence lines and boundaries using HED (Holistically-Nested Edge Detection) with VGG16 backbone
- **SAM Segmentation**: General-purpose image segmentation with semantic labeling (vehicles, driveways, buildings, etc.)
- **Grounded-SAM Detection**: Text-prompted object detection using natural language (e.g., "driveway", "patio", "shed", "gazebo") - combines GroundingDINO with SAM for open-vocabulary detection and segmentation
- **OSM Integration**: Fetch OpenStreetMap data for parcel context and geographic features
- **GeoJSON Output**: Returns detections as GeoJSON FeatureCollection with geographic coordinates
- **Coordinate Conversion**: Geodesic pixel → WGS84 transformation using pyproj
- **Standalone**: Works independently - no dependency on parcel-geojson
- **Interactive Maps**: Generate Folium visualizations with satellite overlay

## Deployment

### Docker Deployment (Recommended)

The recommended deployment method is using Docker, which bundles all dependencies including PyTorch, YOLOv8, detectree, and the FastAPI service into a single container.

**Quick Start:**

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t parcel-ai-json:latest .

# Run the service
docker run -p 8000:8000 parcel-ai-json:latest

# Or use Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Interactive docs: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json
- Health check: http://localhost:8000/health

### Using the Docker API

**Detect all features:**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "zoom_level=20" \
  -F "format=geojson"
```

**Get summary statistics only:**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "format=summary"
```

**Response (summary format):**
```json
{
  "vehicles": 5,
  "swimming_pools": 1,
  "amenities": {"tennis court": 2},
  "total_amenities": 2,
  "tree_coverage_percent": 12.3
}
```

**Detect specific features:**

```bash
# Vehicles only
curl -X POST http://localhost:8000/detect/vehicles \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194"

# Pools only
curl -X POST http://localhost:8000/detect/pools \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194"

# Amenities only
curl -X POST http://localhost:8000/detect/amenities \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194"

# Tree coverage only
curl -X POST http://localhost:8000/detect/trees \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194"

# Fences only
curl -X POST http://localhost:8000/detect/fences \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "threshold=0.1"

# Fences with Regrid probability mask (recommended for better accuracy)
curl -X POST http://localhost:8000/detect/fences \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "fence_mask=@fence_probability.png" \
  -F "threshold=0.1"

# Grounded-SAM: Text-prompted detection (custom objects)
curl -X POST http://localhost:8000/detect/grounded-sam \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "prompts=driveway, patio, deck, shed, gazebo, pergola, hot tub"
```

**Include fence detection in unified detection:**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@satellite.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "detect_fences=true" \
  -F "fence_mask=@fence_probability.png"
```

### Production Deployment

**AWS ECS/Fargate:**

```bash
# Tag and push to ECR
docker tag parcel-ai-json:latest <account>.dkr.ecr.us-west-2.amazonaws.com/parcel-ai-json:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/parcel-ai-json:latest

# Deploy via ECS task definition
```

**Kubernetes:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: parcel-ai-json
spec:
  replicas: 3
  selector:
    matchLabels:
      app: parcel-ai-json
  template:
    metadata:
      labels:
        app: parcel-ai-json
    spec:
      containers:
      - name: api
        image: parcel-ai-json:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: parcel-ai-json
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: parcel-ai-json
```

### Python Package Installation (Legacy)

If you need to use the package as a Python library (not recommended for production):

```bash
# From internal PyPI server
pip install parcel-ai-json --extra-index-url=https://erg-bot:q8zgdmot3@pypi.ergeon.in/simple/

# From GitHub
pip install git+https://github.com/ergeon/parcel-ai-json.git

# Local development
cd parcel-ai-json
pip install -e ".[dev]"
```

**Note:** When using as a Python package, you must also build and run the tree detection Docker image separately (see Dockerfile.tree).

## Usage

### Unified Property Detection (Recommended)

```python
from parcel_ai_json import PropertyDetectionService

# Initialize detector (detects vehicles, pools, and amenities)
detector = PropertyDetectionService(
    vehicle_confidence=0.25,
    pool_confidence=0.3,
    amenity_confidence=0.3,
)

# Prepare satellite image metadata
satellite_image = {
    "path": "satellite.jpg",
    "center_lat": 37.7749,  # Image center latitude (WGS84)
    "center_lon": -122.4194,  # Image center longitude (WGS84)
    "zoom_level": 20,  # Optional, default 20
}

# Option 1: Get all detections in one call
detections = detector.detect_all(satellite_image)

# Access individual detection types
print(f"Found {len(detections.vehicles)} vehicles")
print(f"Found {len(detections.swimming_pools)} pools")
print(f"Found {len(detections.amenities)} amenities")

# Get summary statistics
summary = detections.summary()
print(f"Summary: {summary}")
# Output: {'vehicles': 5, 'swimming_pools': 1, 'amenities': {'tennis court': 2}, 'total_amenities': 2}

# Option 2: Get GeoJSON directly with all features
geojson = detector.detect_all_geojson(satellite_image)

# Tree coverage is automatically included in summary
summary = detections.summary()
print(f"Summary: {summary}")
# Output: {'vehicles': 5, 'swimming_pools': 1, 'amenities': {'tennis court': 2},
#          'total_amenities': 2, 'tree_coverage_percent': 2.6}

# Save to file
import json
with open("property_detections.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
```

### Individual Detectors (Advanced Usage)

For more control, you can use individual detectors:

```python
from parcel_ai_json import VehicleDetectionService

detector = VehicleDetectionService(confidence_threshold=0.25)
vehicles = detector.detect_vehicles(satellite_image)
```

### Detectable Features

**Vehicles (YOLOv8-OBB on DOTA dataset):**
- Small vehicles (cars, motorcycles)
- Large vehicles (trucks, buses, RVs)

**Swimming Pools (YOLOv8-OBB on DOTA dataset):**
- Residential and commercial swimming pools

**Amenities (YOLOv8-OBB on DOTA dataset):**
- Tennis courts
- Basketball courts
- Baseball diamonds
- Soccer ball fields
- Ground track fields

**Trees (detectree via Docker):**
- Tree coverage percentage based on pixel classification

**Fences (HED Model):**
- Fence lines and boundaries detected using HED (Holistically-Nested Edge Detection) with VGG16 backbone
- Supports 4-channel input (RGB + Regrid fence probability mask)
- Returns fence polygons in geographic coordinates

## GeoJSON Output Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-122.419, 37.775],
          [-122.418, 37.775],
          [-122.418, 37.774],
          [-122.419, 37.774],
          [-122.419, 37.775]
        ]]
      },
      "properties": {
        "feature_type": "vehicle",
        "vehicle_class": "small vehicle",
        "confidence": 0.87,
        "pixel_bbox": [245.2, 389.1, 298.6, 445.3]
      }
    }
  ]
}
```

## Architecture

This is a **containerized microservice** with a FastAPI REST API.

**Key Components:**
- FastAPI REST API (7 endpoints)
- YOLOv8-OBB detection (vehicles, pools, amenities)
- HED fence detection (Holistically-Nested Edge Detection with VGG16)
- SAM (Segment Anything) general-purpose segmentation with semantic labeling
- detectree tree coverage analysis
- OpenStreetMap data integration
- Geodesic coordinate conversion (pyproj)
- GeoJSON generation

**Deployment:**
- Single unified Docker container (~2-3GB)
- Pre-bundled models (no first-run downloads)
- Ready for Kubernetes, ECS, or Docker Compose

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical documentation.

## Performance

- Vehicle detection: ~7.5s per address (98% of total time)
- Core GeoJSON generation: ~0.16s per address

## Model Management

The package uses `yolov8m-obb.pt` (51MB) - YOLOv8 medium model with Oriented Bounding Boxes, trained on DOTA aerial imagery dataset.

**Model Auto-Download:**
Models are automatically downloaded on first use by the ultralytics library. The detection services check for models in this order:
1. Local `models/` directory (for manual model placement)
2. `~/.ultralytics/` directory (auto-downloaded)

**To manually place models:**
```bash
# Download models to the models/ directory
mkdir -p models
cd models
# Place your .pt model files here
```

See `models/README.md` for more details on model management.

## Development

### Python Development (Local)

```bash
# Set up development environment
make install

# Run tests
make test

# Run tests with coverage
make coverage-html

# Format code
make format

# Run all checks (format, lint, test)
make check

# Generate examples
make generate-examples
```

### Docker Development

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

### Available Make Commands

Run `make help` to see all available commands.

**Development:**
- `make install` - Install package and dependencies in virtualenv
- `make dev-setup` - Quick development environment setup
- `make test` - Run tests with coverage
- `make test-verbose` - Run tests with verbose output
- `make test-watch` - Run tests in watch mode (requires pytest-watch)
- `make coverage` - Show coverage report in terminal
- `make coverage-html` - Generate HTML coverage report (opens in browser)
- `make format` - Format code with black
- `make format-check` - Check formatting without making changes
- `make lint` - Run flake8 linter
- `make check` - Run all checks (format-check, lint, test)
- `make clean` - Clean build artifacts and cache
- `make clean-all` - Clean everything including virtualenv

**Docker (Recommended):**
- `make docker-build` - Build Docker image
- `make docker-build-clean` - Build Docker image without cache (forces full rebuild)
- `make docker-run` - Run Docker container locally
- `make docker-stop` - Stop and remove Docker container
- `make docker-restart` - Restart Docker container
- `make docker-rebuild` - Rebuild and restart Docker container
- `make docker-logs` - Show Docker container logs
- `make docker-shell` - Open shell in running Docker container
- `make docker-up` - Start services with Docker Compose
- `make docker-down` - Stop Docker Compose services
- `make docker-clean` - Remove Docker image and clean build cache
- `make docker-push` - Push Docker image to registry

**Examples:**
- `make generate-examples` - Generate 3 detection examples (default)
- `make generate-examples-10` - Generate 10 detection examples
- `make generate-examples-20` - Generate 20 detection examples
- `NUM_EXAMPLES=50 make generate-examples` - Generate custom number of examples

**Package (Legacy):**
- `make build` - Build source and wheel distributions
- `make deploy` - Build and deploy package to internal PyPI
- `make tag` - Create and push git tag for current version
- `make install-ci` - Install deployment dependencies (twine, wheel)

## Testing & Scripts

### Test Datasets

The project includes 100 complete test datasets in `output/test_datasets/`:
- **Satellite images**: 512x512 PNG images from Google Maps (zoom level 20)
- **Regrid parcels**: GeoJSON property boundaries (WGS84)
- **Metadata**: CSV index with coordinates and quote IDs
- **Coverage**: Properties across USA (CA, TX, FL, IL, PA, GA, NJ, MD, etc.)

See `output/test_datasets/README.md` for details.

### Running Detections on Test Data

**Run N random detections:**

```bash
# Run 5 random detections (without SAM for speed)
python scripts/run_random_detections.py 5 --no-sam

# Run 10 with SAM segmentation
python scripts/run_random_detections.py 10

# Run 10 with Folium maps
python scripts/run_random_detections.py 10 --map

# Reproducible results with seed
python scripts/run_random_detections.py 5 --seed 42
```

**Options:**
- `--sam` / `--no-sam` - Include/exclude SAM segmentation (default: True)
- `--map` - Generate Folium visualization maps
- `--seed N` - Random seed for reproducibility

**Output:** Results saved to `output/test_datasets/results/`

### Visualization Scripts

**Create Folium map from GeoJSON detections:**

```bash
python scripts/create_folium_from_geojson.py \
  --geojson output/test_datasets/results/ADDRESS_detections.json \
  --image output/test_datasets/satellite_images/ADDRESS.jpg \
  --output map.html \
  --lat 37.7749 \
  --lon -122.4194 \
  --zoom 20
```

Features:
- Satellite image overlay
- Color-coded detection layers (vehicles, trees, fences, pools, etc.)
- Interactive popups with confidence scores and measurements
- Regrid parcel boundary visualization
- Layer controls for toggling features

**Create SAM segmentation map:**

```bash
python scripts/create_sam_folium_map.py \
  --geojson detections.json \
  --image satellite.jpg \
  --output sam_map.html \
  --lat 37.7749 \
  --lon -122.4194
```

### Available Scripts

All scripts are in the `scripts/` directory:

- **`run_random_detections.py`** - Run detections on N random test datasets
- **`create_folium_from_geojson.py`** - Generate interactive Folium maps from detection GeoJSON
- **`create_sam_folium_map.py`** - Visualize SAM segmentation results
- **`generate_examples.py`** - Generate detection examples (used by make generate-examples)
- **`find_and_copy_100_datasets.py`** - Copy test datasets from det-state-visualizer

## Package Deployment (Legacy PyPI)

For legacy PyPI deployment (not recommended - use Docker instead):

```bash
# Build package
make build

# Deploy to internal PyPI (requires pypirc.conf)
make deploy

# Create and push git tag
make tag
```

See `pypirc.conf.example` for PyPI configuration.

## License

Proprietary - Ergeon Internal Use
