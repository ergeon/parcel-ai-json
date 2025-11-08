# Parcel AI JSON

AI/ML extensions for [parcel-geojson](https://github.com/ergeon/parcel-geojson).

## Features

- **Vehicle Detection**: Detect vehicles in satellite imagery using YOLOv8-OBB (trained on DOTA aerial dataset)
- **Future**: Building detection models
- **Future**: Fence detection models

## Installation

```bash
pip install parcel-ai-json
```

This will automatically install `parcel-geojson` as a dependency along with PyTorch and Ultralytics.

## Usage

```python
from parcel_geojson import ParcelGeoJSONGenerator

# Vehicle detection automatically enabled if parcel-ai-json is installed
generator = ParcelGeoJSONGenerator(
    enable_vehicle_detection=True,
    vehicle_model_path="yolov8m-obb.pt",  # Bundled model
    vehicle_confidence=0.25,
)

geojson = generator.generate(
    regrid_json="parcel.json",
    building_json="building.json",
    satellite_image={
        "path": "satellite.jpg",
        "center_lat": 37.7749,
        "center_lon": -122.4194,
    },
)

print(f"Detected {len([f for f in geojson['features'] if f['properties']['feature_type'] == 'vehicle'])} vehicles")
```

## Why a Separate Package?

The core `parcel-geojson` package is designed to be **Lambda-compatible** (~50MB, fast startup).

This package (`parcel-ai-json`) includes:
- PyTorch (~500MB)
- YOLOv8 models (~51MB)
- Total: ~600MB

**Deployment Options:**
- **Lambda**: Use `parcel-geojson` alone (no AI features)
- **ECS Fargate**: Install both packages for full AI capabilities
- **Local**: Install both for development

## Performance

- Vehicle detection: ~7.5s per address (98% of total time)
- Core GeoJSON generation: ~0.16s per address

## Models Included

- `yolov8m-obb.pt` (51MB): YOLOv8 medium model with Oriented Bounding Boxes, trained on DOTA aerial imagery dataset

## Development

```bash
# Install in development mode
cd parcel-ai-json
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=parcel_ai_json
```

## License

MIT
