# Integration Tests

This directory contains integration tests that verify end-to-end functionality of the parcel-ai-json system.

## Test Files

- **test_sam.py** - Tests SAM segmentation model integration
- **test_sam_labeling.py** - Tests SAM segment labeling pipeline
- **test_parcel_mask_generation.py** - Tests parcel mask generation from OSM data
- **test_coordinate_roundtrip.py** - Tests coordinate transformation accuracy

## Running Integration Tests

These tests may require:
- Docker container running (`docker-compose up -d`)
- Network access for OSM data fetching
- Larger test images (not mocked)
- Actual model inference (slower than unit tests)

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_sam.py -v

# Run with output
pytest tests/integration/test_sam_labeling.py -v -s
```

## Difference from Unit Tests

**Unit tests** (`tests/test_*.py`):
- Fast execution (mocked models)
- No external dependencies
- Isolated component testing
- Run on every commit

**Integration tests** (`tests/integration/test_*.py`):
- Slower execution (real models)
- May require Docker/network
- End-to-end workflows
- Run before releases
