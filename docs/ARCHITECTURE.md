# Parcel AI JSON - Architecture & Context

## Overview

Standalone AI/ML package for vehicle detection in satellite imagery using YOLOv8-OBB (Oriented Bounding Boxes) trained on the DOTA aerial dataset. Returns GeoJSON with geodesic coordinate transformations.

**Key Principles**:
1. **Standalone package**: Complete vehicle detection + coordinate transformation + GeoJSON generation
2. **No circular dependencies**: parcel-ai-json is independent, parcel-geojson optionally imports it
3. **Geodesic accuracy**: Uses pyproj for proper WGS84 transformations (not approximations)

## Why a Separate Package?

### Size Constraints
- **parcel-geojson**: ~50MB (Lambda-compatible ≤250MB limit)
- **parcel-ai-json**: ~600MB runtime (PyTorch + YOLOv8 + model)
  - Package size: ~1MB (models auto-downloaded separately)
  - PyTorch: ~500MB
  - ultralytics: ~50MB
  - yolov8m-obb.pt model: 51MB (downloaded to ~/.ultralytics/ on first use)

### Deployment Options
| Environment | Packages | Size | Features |
|------------|----------|------|----------|
| **AWS Lambda** | parcel-geojson only | ~50MB | Core GeoJSON ✅ |
| **ECS Fargate** | Both | ~600MB | Core + AI ✅ |
| **Local Dev** | Both | ~600MB | Full stack ✅ |

## Architecture

### Package Structure
```
parcel-ai-json/
├── parcel_ai_json/          # Core detection services
│   ├── __init__.py
│   ├── vehicle_detector.py
│   ├── swimming_pool_detector.py
│   ├── amenity_detector.py
│   ├── tree_detector.py     # DeepForest + detectree
│   └── property_detector.py # Unified detector
├── tests/                   # Test suite
│   ├── test_vehicle_detector.py
│   ├── test_swimming_pool_detector.py
│   ├── test_amenity_detector.py
│   └── test_property_detector.py
├── docker/                  # Docker configurations
│   ├── Dockerfile
│   ├── Dockerfile.tree
│   └── docker-compose.yml
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md
│   └── DOCKER_MIGRATION.md
├── models/                  # Model files (git-ignored)
│   └── README.md           # Model management guide
├── scripts/                 # Utility scripts
│   └── generate_examples.py
├── examples/                # Example outputs
├── setup.py
├── Makefile
└── README.md
```

### Integration with parcel-geojson

**Dependency Direction** (one-way only):
```
parcel-geojson → parcel-ai-json
   (imports)      (independent)
```

**Import Pattern** (parcel-geojson optionally imports from parcel-ai-json):
```python
# In parcel_geojson/core/geojson_builder.py
if self.enable_vehicle_detection and satellite_image:
    try:
        from parcel_ai_json import VehicleDetectionService
        detector = VehicleDetectionService(...)

        # Returns GeoJSON FeatureCollection (complete)
        vehicle_geojson = detector.detect_vehicles_geojson(satellite_image)

        # Merge into main GeoJSON
        features.extend(vehicle_geojson['features'])
    except ImportError:
        print("⚠️  Vehicle detection skipped")
        print("Install with: pip install parcel-ai-json")
```

**No Reverse Dependency**: parcel-ai-json does NOT import from parcel-geojson.

## Vehicle Detection Algorithm

### Model: YOLOv8m-OBB

**Specifications**:
- **Architecture**: YOLOv8 medium with Oriented Bounding Boxes
- **Training Dataset**: DOTA (Dataset for Object Detection in Aerial Images)
- **Model Size**: 51MB
- **Input**: Satellite/aerial imagery (JPEG/PNG)
- **Output**: Rotated bounding boxes with confidence scores

**Why OBB (Oriented Bounding Boxes)?**
- Vehicles in aerial imagery appear at various angles
- Standard axis-aligned boxes waste space and reduce accuracy
- OBB provides tighter fits around rotated objects

### Detection Pipeline

```
Input: Satellite Image + Center Coordinates
    ↓
1. Load Image (PIL)
    ↓
2. YOLOv8m-OBB Inference
   - Model: yolov8m-obb.pt (auto-downloaded to ~/.ultralytics/)
   - Confidence threshold: 0.25 (default)
   - NMS threshold: 0.45
    ↓
3. Extract Detections
   - Filter by confidence
   - Filter by vehicle class (small vehicle, large vehicle, etc.)
    ↓
4. Geodesic Coordinate Transformation
   - Pixel → WGS84 using pyproj.Geod
   - Accurate ellipsoidal calculations
    ↓
5. Generate GeoJSON
   - Create VehicleDetection objects with both pixel and geo coords
   - Build FeatureCollection
    ↓
Output: GeoJSON FeatureCollection or List[VehicleDetection]
```

**Coordinate Accuracy**: Uses pyproj geodesic forward calculations (not simple approximations).

### Code Location

**Main Class**: `parcel_ai_json/vehicle_detector.py`

**Key Methods**:
```python
class VehicleDetectionService:
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.3,
        device: str = "cpu"
    )

    def detect_vehicles(
        self,
        satellite_image: Dict  # {path, center_lat, center_lon, zoom_level}
    ) -> List[VehicleDetection]

    def detect_vehicles_geojson(
        self,
        satellite_image: Dict
    ) -> Dict  # GeoJSON FeatureCollection
```

**VehicleDetection** dataclass:
```python
@dataclass
class VehicleDetection:
    pixel_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    geo_polygon: List[Tuple[float, float]]  # [(lon, lat), ...]
    confidence: float
    class_name: str  # 'small vehicle', 'large vehicle' (DOTA classes)
```

### Model Performance

**Tested on 18 Addresses**:
- **Total vehicles detected**: 85
- **Average per address**: 4.7 vehicles
- **Inference time**: ~7.5s per address (98% of total pipeline time)

**Breakdown**:
- Model loading (first call): ~2s
- Image loading: ~100ms
- Inference (torch.conv2d): ~7.4s
- Postprocessing: ~50ms

**Optimization Options**:
1. **GPU**: 5-10x faster with CUDA
2. **Quantization**: INT8 model (faster, slight accuracy loss)
3. **Smaller model**: yolov8n-obb (7.2x faster, 0% accuracy on aerial - not recommended)

## Installation

### Development Installation

```bash
# From GitHub (public repo)
pip install git+https://github.com/ergeon/parcel-ai-json.git

# From local directory
cd parcel-ai-json
pip install -e ".[dev]"
```

### Production Installation

**Standalone** (vehicle detection only):
```bash
pip install git+https://github.com/ergeon/parcel-ai-json.git
```

**With parcel-geojson** (full GeoJSON generation):
```bash
# Install both packages
pip install git+https://github.com/ergeon/parcel-geojson.git
pip install git+https://github.com/ergeon/parcel-ai-json.git
```

### No Circular Dependencies

**Old Architecture** (circular dependency):
```
parcel-geojson ⟷ parcel-ai-json  ❌ BAD
```

**New Architecture** (one-way dependency):
```
parcel-geojson → parcel-ai-json  ✅ GOOD
     (optional)     (independent)
```

parcel-ai-json can be installed and used standalone without parcel-geojson.

## Model Details

### yolov8m-obb.pt

**Source**: Ultralytics YOLOv8 medium OBB variant
**Training Dataset**: DOTA (aerial imagery)
**Size**: 51MB
**Classes**: 15 DOTA classes (plane, ship, storage-tank, **vehicle**, etc.)
**Architecture**:
- Backbone: CSPDarknet
- Neck: PANet
- Head: Decoupled OBB detection head

**Download**: Auto-downloaded to `~/.ultralytics/` on first use by the ultralytics library.

**Alternative Models** (also auto-downloadable):
- `yolov8n-obb.pt`: 6.2MB, 7.2x faster, **0% detection on aerial** (COCO-trained)
- `yolov8l-obb.pt`: 83MB, higher accuracy, slower
- `yolov8x-obb.pt`: 131MB, highest accuracy, slowest

**Why yolov8m-obb?**
- Good balance of speed vs. accuracy
- Proven on aerial imagery (DOTA dataset)
- OBB (Oriented Bounding Boxes) better for rotated vehicles
- Auto-downloaded on first use (keeps package size small)

## Testing

### Running Tests

```bash
# All vehicle detection tests
pytest tests/test_vehicle_detector.py -v

# With coverage
pytest tests/ --cov=parcel_ai_json --cov-report=html

# Single test
pytest tests/test_vehicle_detector.py::TestVehicleDetector::test_detect_from_image -v
```

### Test Coverage

**Current**: 16 tests covering:
- Model loading and initialization
- Vehicle detection from image paths
- Metadata-based detection
- Coordinate transformation
- Error handling (missing files, invalid images)
- Edge cases (no detections, low confidence)

### Key Test Cases

**Mocking Strategy**:
- Mock YOLO model for fast tests (no actual inference)
- Use small test images (100x100px) for integration tests
- Mock satellite metadata for unit tests

**Example** (tests/test_vehicle_detector.py):
```python
def test_detect_from_metadata_with_mock(self, mock_yolo):
    """Test vehicle detection with mocked YOLO model."""
    mock_results = [Mock(boxes=Mock(xyxy=[...], conf=[0.9]))]
    mock_yolo.return_value.return_value = mock_results

    service = VehicleDetectionService()
    vehicles = service.detect_vehicles_from_metadata(metadata)

    assert len(vehicles) > 0
    assert all(isinstance(v, Vehicle) for v in vehicles)
```

## Performance Optimization

### GPU Acceleration

**Enable CUDA**:
```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# YOLOv8 auto-detects and uses GPU if available
service = VehicleDetectionService()  # Will use CUDA if available
```

**Expected Speedup**: 5-10x faster on GPU (7.5s → 0.75-1.5s)

### Model Quantization

**INT8 Quantization**:
```python
# Export to INT8 ONNX
from ultralytics import YOLO

model = YOLO("yolov8m-obb.pt")
model.export(format="onnx", int8=True)
```

**Trade-offs**:
- **Speed**: ~2x faster
- **Size**: ~4x smaller (51MB → 13MB)
- **Accuracy**: 1-2% drop in mAP

### Batch Processing

**Process Multiple Images**:
```python
# Not yet implemented - future enhancement
service = VehicleDetectionService()
results = service.detect_vehicles_batch(image_paths, batch_size=4)
```

**Expected Speedup**: 1.5-2x when processing 10+ images

## Common Issues

### 1. ImportError: No module named 'parcel_geojson'

**Cause**: parcel-geojson not installed

**Solution**:
```bash
pip install git+https://github.com/ergeon/parcel-geojson.git
```

### 2. CUDA out of memory

**Cause**: GPU memory insufficient for model

**Solution**:
- Reduce batch size (if using batching)
- Use CPU inference: `device='cpu'`
- Use smaller model: yolov8s-obb.pt

### 3. Low Detection Rate

**Possible Causes**:
- Image resolution too low (<640px)
- Wrong model (COCO instead of DOTA)
- Confidence threshold too high

**Solution**:
- Ensure satellite images are at least 640x640px
- Verify using yolov8m-obb.pt (DOTA-trained)
- Lower confidence threshold (default 0.25 → 0.15)

### 4. Model Download Issues

**Error**: Model download fails or is slow

**Cause**: Network issues or firewall blocking ultralytics download

**Solution**:
- The model is automatically downloaded to `~/.ultralytics/` on first use
- First detection will take longer (~30s for 51MB download)
- Pre-download manually: `from ultralytics import YOLO; YOLO('yolov8m-obb.pt')`
- Check `~/.ultralytics/` directory for downloaded models

## Future Enhancements

### Planned Features
1. **Building Detection**: YOLOv8 for building footprints
2. **Fence Detection**: Custom model for fence line detection
3. **Batch API**: Process multiple addresses in parallel
4. **Model Caching**: Cache downloaded models in project directory
5. **TorchScript**: Compiled models for faster loading

### Model Improvements
1. **Fine-tuning**: Train on parcel-specific dataset
2. **Ensemble**: Combine multiple models for better accuracy
3. **Tracking**: Associate vehicles across multiple images
4. **Classification**: Distinguish car/truck/RV types

### Performance Targets
- **GPU**: <1s per image
- **Batch**: <5s for 10 images
- **Cold start**: <3s (currently ~2s)

## Dependencies

### Core Dependencies
- `torch>=2.0.0`: PyTorch deep learning framework (~500MB)
- `ultralytics>=8.0.0`: YOLOv8 implementation (~50MB)
- `pillow>=9.0.0`: Image loading and processing
- `numpy>=1.20.0`: Numerical operations
- `torchvision>=0.15.0`: PyTorch vision utilities
- `pyproj>=3.0.0`: Geodesic coordinate transformations (WGS84)

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `pytest-cov>=4.0.0`: Coverage reporting
- `black>=23.0.0`: Code formatting
- `flake8>=6.0.0`: Linting

### Integration
- **parcel-geojson** (separate package): Optionally imports this package for vehicle detection in GeoJSON generation
  - **Direction**: parcel-geojson → parcel-ai-json (one-way)
  - **Relationship**: parcel-ai-json is standalone and does NOT depend on parcel-geojson

## Session Continuity

When starting a new Claude Code session:

1. **Read this file first** to understand architecture
2. **Check dependencies**: Ensure required packages installed
3. **Run tests**: `pytest tests/ -v`
4. **Model auto-download**: First run will download yolov8m-obb.pt (51MB) to ~/.ultralytics/
5. **Test detection**: Run `scripts/generate_examples.py`

## Related Packages

### parcel-geojson
**Repository**: https://github.com/ergeon/parcel-geojson
**Purpose**: Core GeoJSON generation (Lambda-compatible)
**Size**: ~50MB
**Use**: `pip install git+https://github.com/ergeon/parcel-geojson.git`

## Contact & Resources

- **GitHub**: https://github.com/ergeon/parcel-ai-json
- **Tests**: `pytest tests/` (16 tests)
- **Examples**: `/examples/detect_vehicles_example.py`
- **Model**: YOLOv8m-OBB (DOTA-trained, 51MB)
