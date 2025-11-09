# Architecture Documentation

## Overview

`parcel-ai-json` is a containerized microservice for detecting property features in satellite imagery using computer vision and machine learning. The service provides a REST API for detecting vehicles, swimming pools, amenities (tennis courts, basketball courts, etc.), and tree coverage.

## System Architecture

### Deployment Architecture

```
┌─────────────────────────────────────────┐
│     Docker Container (Single Image)     │
├─────────────────────────────────────────┤
│  FastAPI REST API (Port 8000)           │
│  ├─ /detect (all features)              │
│  ├─ /detect/vehicles                    │
│  ├─ /detect/pools                       │
│  ├─ /detect/amenities                   │
│  └─ /detect/trees                       │
│                                         │
│  Detection Services Layer               │
│  ├─ PropertyDetectionService            │
│  │   ├─ VehicleDetectionService         │
│  │   ├─ SwimmingPoolDetectionService    │
│  │   ├─ AmenityDetectionService         │
│  │   └─ TreeDetectionService            │
│  │                                      │
│  ML Models Layer                        │
│  ├─ YOLOv8-OBB (51MB)                   │
│  │   └─ DOTA aerial imagery dataset     │
│  └─ detectree                           │
│      └─ Tree segmentation model         │
│                                         │
│  Core Utilities                         │
│  ├─ ImageCoordinateConverter            │
│  │   └─ pyproj (geodesic transforms)    │
│  └─ GeoJSON generation                  │
└─────────────────────────────────────────┘
```

## Component Architecture

### 1. API Layer (`parcel_ai_json/api.py`)

**Responsibilities:**
- HTTP request handling and validation
- File upload management
- Response formatting (GeoJSON or summary)
- Error handling and logging
- Service initialization (singleton pattern)

**Endpoints:**
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /detect` - Unified detection (all features)
- `POST /detect/vehicles` - Vehicle detection only
- `POST /detect/pools` - Pool detection only
- `POST /detect/amenities` - Amenity detection only
- `POST /detect/trees` - Tree coverage only

**Input Format:**
```
multipart/form-data:
  - image: File (JPEG/PNG)
  - center_lat: float (WGS84 latitude)
  - center_lon: float (WGS84 longitude)
  - zoom_level: int (default: 20)
  - format: str ("geojson" or "summary")
```

**Output Formats:**

*GeoJSON Format:*
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      },
      "properties": {
        "feature_type": "vehicle|pool|amenity",
        "confidence": 0.87,
        "pixel_bbox": [x1, y1, x2, y2]
      }
    }
  ]
}
```

*Summary Format:*
```json
{
  "vehicles": 5,
  "swimming_pools": 1,
  "amenities": {"tennis court": 2},
  "total_amenities": 2,
  "tree_coverage_percent": 12.3
}
```

### 2. Detection Services Layer

#### PropertyDetectionService (`parcel_ai_json/property_detector.py`)

**Unified detection orchestrator** that coordinates all individual detection services.

```python
PropertyDetectionService(
    model_path: Optional[str] = None,
    vehicle_confidence: float = 0.25,
    pool_confidence: float = 0.3,
    amenity_confidence: float = 0.3,
    device: str = "cpu",
    tree_use_docker: bool = True,
    tree_docker_image: str = "parcel-tree-detector"
)
```

**Methods:**
- `detect_all(satellite_image)` → PropertyDetections
- `detect_all_geojson(satellite_image)` → Dict (GeoJSON)

#### VehicleDetectionService (`parcel_ai_json/vehicle_detector.py`)

Detects vehicles using YOLOv8-OBB trained on DOTA aerial dataset.

**Detected Classes:**
- Small vehicles (cars, motorcycles)
- Large vehicles (trucks, buses, RVs)

**Model:** `yolov8m-obb.pt` (51MB)

#### SwimmingPoolDetectionService (`parcel_ai_json/swimming_pool_detector.py`)

Detects swimming pools using YOLOv8-OBB (DOTA class 14).

**Model:** Shared with vehicle detector (`yolov8m-obb.pt`)

#### AmenityDetectionService (`parcel_ai_json/amenity_detector.py`)

Detects recreational amenities using YOLOv8-OBB.

**Detected Classes:**
- Tennis courts (class 4)
- Basketball courts (class 5)
- Baseball diamonds (class 3)
- Soccer ball fields (class 13)
- Ground track fields (class 6)

**Model:** Shared with vehicle/pool detector

#### TreeDetectionService (`parcel_ai_json/tree_detector.py`)

Detects tree coverage using detectree library with pixel-level classification.

**Two Operating Modes:**

1. **Native Mode (use_docker=False):**
   - Runs detectree directly (Linux/Docker container)
   - Used in containerized API deployment
   - Faster, no subprocess overhead

2. **Docker Mode (use_docker=True):**
   - Runs detectree in separate Docker container
   - Used for macOS compatibility (detectree has C extension issues on macOS)
   - Spawns subprocess with volume mounting

**Output:**
- Tree pixel count
- Total pixels
- Coverage percentage (0-100%)
- Optional tree mask PNG (green overlay)

### 3. Coordinate Conversion Layer

#### ImageCoordinateConverter (`parcel_ai_json/coordinate_converter.py`)

Converts pixel coordinates to geographic coordinates (WGS84).

**Key Features:**
- Uses pyproj for accurate geodesic transformations
- Supports Web Mercator projection (EPSG:3857)
- Accounts for zoom level and image dimensions
- Handles bbox → polygon conversion

**Algorithm:**
1. Calculate meters per pixel based on zoom level and latitude
2. Convert pixel offset to meters (from image center)
3. Use pyproj.Geod to calculate geographic coordinates
4. Return WGS84 lon/lat coordinates

**No Hardcoded Constants:**
All geographic calculations use pyproj library (as per CLAUDE.md guidelines) to ensure accuracy across different ellipsoid models.

## Data Flow

### Detection Request Flow

```
1. Client sends POST /detect
   ├─ Image file (JPEG/PNG)
   ├─ Center coordinates (lat, lon)
   └─ Zoom level

2. API Layer (api.py)
   ├─ Validate inputs
   ├─ Save uploaded file to temp directory
   └─ Create satellite_image dict

3. PropertyDetectionService.detect_all()
   ├─ VehicleDetectionService.detect_vehicles()
   │   ├─ Load YOLOv8-OBB model (lazy)
   │   ├─ Run inference on image
   │   ├─ Filter by confidence threshold
   │   ├─ Convert pixel bbox → geo polygon
   │   └─ Return VehicleDetection[]
   │
   ├─ SwimmingPoolDetectionService.detect_swimming_pools()
   │   └─ (same flow as vehicles)
   │
   ├─ AmenityDetectionService.detect_amenities()
   │   └─ (same flow as vehicles)
   │
   └─ TreeDetectionService.detect_trees()
       ├─ Load detectree classifier (lazy)
       ├─ Run pixel-level classification
       ├─ Generate tree mask PNG
       ├─ Calculate coverage statistics
       └─ Return TreeDetection

4. Convert to GeoJSON or Summary
   ├─ Each detection → GeoJSON Feature
   └─ Combine into FeatureCollection

5. Return JSON response
   └─ Clean up temp files
```

## Model Management

### YOLOv8-OBB Model

**Location:**
- Development: Auto-downloaded to `~/.ultralytics/`
- Production (Docker): Pre-downloaded during image build

**Model Details:**
- File: `yolov8m-obb.pt`
- Size: ~51MB
- Architecture: YOLOv8 Medium with Oriented Bounding Boxes
- Training: DOTA aerial imagery dataset
- Classes: 15+ (vehicles, pools, amenities, etc.)

**Performance:**
- Inference time: ~7.5s per image (CPU)
- Detection accuracy: High confidence for clear aerial imagery
- Minimum confidence threshold: 0.25-0.3 (configurable)

### Detectree Model

**Location:**
- Auto-initialized on first use
- Cached by detectree library

**Model Details:**
- Type: Random Forest classifier for tree/non-tree segmentation
- Input: RGB satellite imagery
- Output: Binary mask (1 = tree, 0 = non-tree)
- Performance: Pixel-level classification

## Deployment Options

### 1. Docker Container (Recommended)

**Single unified image** containing all dependencies.

```bash
# Build
docker build -t parcel-ai-json:latest .

# Run
docker run -p 8000:8000 parcel-ai-json:latest
```

**Advantages:**
- Consistent environment across platforms
- All dependencies bundled (PyTorch, detectree, etc.)
- No host environment pollution
- Easy scaling (Kubernetes, ECS, etc.)
- Pre-downloaded models (no first-run delay)

**Image Size:** ~2-3GB (PyTorch + models)

### 2. Docker Compose (Development)

Multi-service orchestration with volume mounting for live code editing.

```bash
docker-compose up -d
```

### 3. Kubernetes (Production)

Horizontal scaling with load balancing.

**Resource Requirements:**
- Memory: 4-8GB per pod
- CPU: 2-4 cores per pod
- Storage: Minimal (models in image)

### 4. AWS ECS/Fargate

Serverless container deployment.

```bash
# Push to ECR
docker tag parcel-ai-json:latest <account>.dkr.ecr.us-west-2.amazonaws.com/parcel-ai-json:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/parcel-ai-json:latest
```

### 5. Python Package (Legacy - Not Recommended)

Install as Python library with separate Docker for tree detection.

**Issues:**
- Requires both `pip install` AND `docker build`
- Host environment dependency conflicts
- Tree detector needs Docker-in-Docker
- Not scalable

## Performance Characteristics

### Inference Times (per image, CPU)

| Detection Type | Time | Notes |
|---------------|------|-------|
| Vehicles | ~7.5s | 98% of total time |
| Pools | ~0s | Shared YOLO inference |
| Amenities | ~0s | Shared YOLO inference |
| Trees | ~3-5s | Separate detectree inference |
| **Total** | **~10-12s** | All features |

### Optimization Opportunities

1. **GPU Acceleration:** Change `device="cuda"` for 10-20x speedup
2. **Model Quantization:** Use INT8 quantization for smaller models
3. **Batch Processing:** Process multiple images in parallel
4. **Caching:** Cache results for identical coordinates
5. **Async Processing:** Queue-based async processing for high load

## Error Handling

### API Error Codes

| Code | Error | Description |
|------|-------|-------------|
| 400 | Bad Request | Invalid image format, coordinates out of range |
| 404 | Not Found | Endpoint doesn't exist |
| 500 | Internal Server Error | Model inference failure, file I/O error |

### Graceful Degradation

- If tree detection fails, return results without tree coverage
- If YOLO model fails to load, return clear error message
- Temporary file cleanup in `finally` blocks
- Detailed error logging for debugging

## Security Considerations

1. **Input Validation:**
   - File type checking (JPEG/PNG only)
   - Coordinate range validation (-90 to 90 lat, -180 to 180 lon)
   - Zoom level bounds (1-22)

2. **File Handling:**
   - Temporary file storage with automatic cleanup
   - No permanent storage of uploaded images
   - Unique temp directories per request

3. **Resource Limits:**
   - Request timeout (10 minutes max)
   - File size limits (handled by FastAPI)
   - Memory limits (container-level)

## Extensibility

### Adding New Detection Types

1. Create new detector service in `parcel_ai_json/`
2. Add to PropertyDetectionService
3. Add API endpoint in `api.py`
4. Update Dockerfile if new dependencies needed
5. Rebuild and redeploy container

**Example: Building Detection**
```python
# 1. Create building_detector.py
class BuildingDetectionService:
    def detect_buildings(self, satellite_image):
        # Implementation
        pass

# 2. Add to PropertyDetectionService
self.building_detector = BuildingDetectionService(...)

# 3. Add API endpoint
@app.post("/detect/buildings")
async def detect_buildings(...):
    # Implementation
    pass
```

## Monitoring and Observability

### Logging

- Structured JSON logging (production)
- Request/response logging
- Error stack traces
- Performance metrics (inference time)

### Health Checks

- `GET /health` - Service health status
- Docker HEALTHCHECK - Container-level monitoring
- Model initialization status

### Metrics to Track

- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate (errors/total requests)
- Model inference time
- Memory usage
- CPU utilization

## Future Enhancements

1. **GPU Support:** Add CUDA-enabled Docker image
2. **Batch API:** Process multiple images in one request
3. **Async Processing:** Job queue with result polling
4. **Result Caching:** Redis/Memcached for repeated requests
5. **Model Versioning:** A/B testing with multiple model versions
6. **Additional Detectors:**
   - Building/structure detection
   - Driveway detection
   - Roof condition analysis
   - Property damage assessment
7. **WebSocket Streaming:** Real-time detection updates
8. **Multi-region Deployment:** Edge deployment for lower latency
