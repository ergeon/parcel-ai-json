# Parcel AI JSON

Unified property detection for satellite imagery with GeoJSON output.

## Features

- **Unified Detection**: Detect vehicles, pools, amenities, and trees in one call
- **Vehicle Detection**: Cars, trucks, and other vehicles using YOLOv8-OBB (DOTA aerial dataset)
- **Swimming Pool Detection**: Detect swimming pools (DOTA class 14)
- **Amenity Detection**: Tennis courts, basketball courts, baseball diamonds, soccer fields, and track fields
- **Tree Coverage Detection**: Estimate tree coverage percentage using detectree (Docker-based)
- **GeoJSON Output**: Returns detections as GeoJSON FeatureCollection with geographic coordinates
- **Coordinate Conversion**: Geodesic pixel → WGS84 transformation using pyproj
- **Standalone**: Works independently - no dependency on parcel-geojson
- **Interactive Maps**: Generate Folium visualizations with satellite overlay

## Installation

### From Ergeon Internal PyPI (Production)

```bash
# Install from internal PyPI server
pip install parcel-ai-json --extra-index-url=https://erg-bot:q8zgdmot3@pypi.ergeon.in/simple/
```

### From GitHub (Development)

```bash
# Install from GitHub
pip install git+https://github.com/ergeon/parcel-ai-json.git

# Or local development installation
cd parcel-ai-json
pip install -e ".[dev]"
```

This will install PyTorch (~500MB) and Ultralytics for vehicle detection.

### Tree Detection Setup (Required)

Tree detection requires Docker to run the detectree library in a Linux container:

```bash
# Build Docker image for tree detection
docker build -f Dockerfile.tree -t parcel-tree-detector .

# Test the Docker image
docker run --rm parcel-tree-detector
```

**Why Docker?** The detectree library has C extension compatibility issues on macOS. Running it in a Linux container ensures it works reliably across all platforms.

**Note:** Tree detection is automatically included in `PropertyDetectionService`. Make sure Docker is installed and the `parcel-tree-detector` image is built before using the service.

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

**Completely standalone** - handles everything internally:
- Vehicle detection (YOLOv8-OBB)
- Coordinate conversion (pixel → WGS84)
- GeoJSON generation

No dependency on parcel-geojson or any other packages.

## Performance

- Vehicle detection: ~7.5s per address (98% of total time)
- Core GeoJSON generation: ~0.16s per address

## Model Auto-Download

The package uses `yolov8m-obb.pt` (51MB) - YOLOv8 medium model with Oriented Bounding Boxes, trained on DOTA aerial imagery dataset.

**The model is automatically downloaded on first use** to `~/.ultralytics/` by the ultralytics library. This keeps the package size small (~1MB instead of ~50MB).

## Development

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

## Deployment

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
