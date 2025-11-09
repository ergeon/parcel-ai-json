"""
Example: Vehicle Detection in Satellite Imagery

This script demonstrates how to use parcel-ai-json standalone to detect vehicles
in satellite imagery and generate GeoJSON output.

Requirements:
    pip install parcel-ai-json
"""

import json
from pathlib import Path
from parcel_ai_json import VehicleDetectionService


def example_basic_detection():
    """Basic example: Detect vehicles and get GeoJSON."""
    print("=" * 80)
    print("Example 1: Basic Vehicle Detection with GeoJSON Output")
    print("=" * 80)

    # Initialize detector
    detector = VehicleDetectionService(
        confidence_threshold=0.25,
    )

    # Satellite image metadata
    satellite_image = {
        "path": "path/to/satellite.jpg",  # Replace with actual path
        "center_lat": 38.244194,
        "center_lon": -122.612183,
        "zoom_level": 20,
    }

    print("\nDetecting vehicles in satellite image...")
    print(f"  Image: {satellite_image['path']}")
    print(
        f"  Center: ({satellite_image['center_lat']}, {satellite_image['center_lon']})"
    )

    # Get detections with both pixel and geo coordinates
    detections = detector.detect_vehicles(satellite_image)

    print(f"\n✓ Found {len(detections)} vehicles")

    for i, detection in enumerate(detections, 1):
        print(f"\n{i}. {detection.class_name}")
        print(f"   Confidence: {detection.confidence:.2%}")
        print(f"   Pixel bbox: {detection.pixel_bbox}")
        print(f"   Geo polygon: {detection.geo_polygon[0]}")  # First point

    return detections


def example_geojson_output():
    """Example: Get GeoJSON directly and save to file."""
    print("\n" + "=" * 80)
    print("Example 2: Generate GeoJSON and Save to File")
    print("=" * 80)

    # Initialize detector
    detector = VehicleDetectionService(
        confidence_threshold=0.25,
    )

    # Satellite image metadata
    satellite_image = {
        "path": "path/to/satellite.jpg",  # Replace with actual path
        "center_lat": 38.244194,
        "center_lon": -122.612183,
        "zoom_level": 20,
    }

    print("\nGenerating GeoJSON for vehicles...")

    # Get GeoJSON directly
    geojson = detector.detect_vehicles_geojson(satellite_image)

    print(f"\n✓ Generated GeoJSON with {len(geojson['features'])} vehicle features")

    # Save to file
    output_path = Path("output") / "vehicles.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")

    # Print first feature as example
    if geojson["features"]:
        print("\nExample feature:")
        print(json.dumps(geojson["features"][0], indent=2))

    return geojson


def example_custom_configuration():
    """Example: Custom detector configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Detector Configuration")
    print("=" * 80)

    # Initialize detector with custom settings
    detector = VehicleDetectionService(
        model_path="yolov8m-obb.pt",  # Use bundled model
        confidence_threshold=0.15,  # Lower threshold for more detections
        device="cpu",  # Or "cuda" for GPU
    )

    satellite_image = {
        "path": "path/to/satellite.jpg",
        "center_lat": 38.244194,
        "center_lon": -122.612183,
        "zoom_level": 20,
        # Optional: provide image dimensions to skip reading
        # "width_px": 640,
        # "height_px": 640,
    }

    print("\nDetector configuration:")
    print(f"  Confidence threshold: {detector.confidence_threshold}")
    print(f"  Device: {detector.device}")

    detections = detector.detect_vehicles(satellite_image)

    print(f"\n✓ Found {len(detections)} vehicles with confidence >= 0.15")

    return detections


if __name__ == "__main__":
    # Run all examples
    # Note: Replace "path/to/satellite.jpg" with actual satellite image path

    # Example 1: Basic detection
    example_basic_detection()

    # Example 2: GeoJSON output
    # example_geojson_output()

    # Example 3: Custom configuration
    # example_custom_configuration()
