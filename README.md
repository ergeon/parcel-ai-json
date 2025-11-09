# Parcel AI JSON

Standalone AI/ML vehicle detection for satellite imagery with GeoJSON output.

## Features

- **Vehicle Detection**: Detect vehicles in satellite imagery using YOLOv8-OBB (trained on DOTA aerial dataset)
- **GeoJSON Output**: Returns vehicle locations as GeoJSON FeatureCollection
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

## Usage

### Basic Vehicle Detection (Oriented Bounding Boxes)

```python
from parcel_ai_json import VehicleDetectionService

# Initialize detector (uses YOLOv8-OBB model, best for aerial imagery)
detector = VehicleDetectionService(
    confidence_threshold=0.25,
)

# Prepare satellite image metadata
satellite_image = {
    "path": "satellite.jpg",
    "center_lat": 37.7749,  # Image center latitude (WGS84)
    "center_lon": -122.4194,  # Image center longitude (WGS84)
    "zoom_level": 20,  # Optional, default 20
}

# Option 1: Get detections with pixel and geo coordinates
detections = detector.detect_vehicles(satellite_image)

for detection in detections:
    print(f"Found {detection.class_name}")
    print(f"  Pixel bbox: {detection.pixel_bbox}")
    print(f"  Geo polygon: {detection.geo_polygon}")
    print(f"  Confidence: {detection.confidence:.2%}")

# Option 2: Get GeoJSON directly
geojson = detector.detect_vehicles_geojson(satellite_image)

# geojson is a FeatureCollection with vehicle features
print(f"Detected {len(geojson['features'])} vehicles")

# Save to file
import json
with open("vehicles.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
```

## Swimming Pool Detection

Swimming pool detection works out of the box using YOLOv8-OBB (DOTA dataset includes "swimming pool" as class 14).

```python
from parcel_ai_json import SwimmingPoolDetectionService

# Initialize detector
detector = SwimmingPoolDetectionService(
    confidence_threshold=0.3,
)

# Detect swimming pools
satellite_image = {
    "path": "satellite.jpg",
    "center_lat": 37.7749,
    "center_lon": -122.4194,
    "zoom_level": 20,
}

# Get detections with area estimates
pools = detector.detect_swimming_pools(satellite_image)

for pool in pools:
    print(f"Swimming pool detected!")
    print(f"  Confidence: {pool.confidence:.2%}")
    print(f"  Approximate area: {pool.area_sqm:.1f} m²")
    print(f"  Location: {pool.geo_polygon[0]}")

# Or get GeoJSON directly
geojson = detector.detect_swimming_pools_geojson(satellite_image)
```

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
