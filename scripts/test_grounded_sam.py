"""Test Grounded-SAM detector on sample satellite images.

This script demonstrates text-prompted object detection using Grounded-SAM.
Detects property features like: driveway, patio, shed, gazebo, deck, pergola, etc.

NOTE: This is a model development/testing script (allowed exception to API-only rule).
For production use, Grounded-SAM should be integrated into the REST API.
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.grounded_sam_detector import (
    GroundedSAMDetector,
)
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter


def test_grounded_sam_detection():
    """Test Grounded-SAM on example images."""
    print("=" * 80)
    print("Testing Grounded-SAM Detector")
    print("=" * 80)

    # Example images from output/test_datasets/satellite_images/
    test_cases = [
        {
            "image_path": "output/test_datasets/satellite_images/23847_oak_meadow_dr_ramona_ca_92065.jpg",  # noqa: E501
            "center_lat": 33.0406,
            "center_lon": -116.8669,
            "zoom_level": 20,
            "name": "23847_oak_meadow_dr_ramona_ca"
        },
        {
            "image_path": "output/test_datasets/satellite_images/6337_ellsworth_ave_dallas_tx_75214.jpg",  # noqa: E501
            "center_lat": 32.8395,
            "center_lon": -96.7511,
            "zoom_level": 20,
            "name": "6337_ellsworth_ave_dallas_tx"
        },
    ]

    # Property features to detect (customize these prompts!)
    prompts = [
        "driveway",
        "patio",
        "deck",
        "shed",
        "gazebo",
        "pergola",
        "hot tub",
        "playground equipment",
        "dog house",
        "fire pit",
        "pool house",
    ]

    print(f"\nDetecting features: {', '.join(prompts)}\n")

    # Initialize detector
    print("Initializing Grounded-SAM detector...")
    print("  - GroundingDINO: models/groundingdino_swinb_cogcoor.pth")
    print("  - SAM: models/sam_vit_h_4b8939.pth")
    print("  - Device: auto-detected (CPU/CUDA/MPS)\n")

    detector = GroundedSAMDetector(
        box_threshold=0.25,
        text_threshold=0.20,
        use_sam=True,
    )

    # Process each test case
    for test_case in test_cases:
        image_path = Path(test_case["image_path"])

        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {test_case['name']}")
        print(f"Image: {image_path}")
        print(f"Location: ({test_case['center_lat']}, {test_case['center_lon']})")
        print(f"{'=' * 80}\n")

        # Load image
        image = Image.open(image_path)
        print(f"Image size: {image.size}")

        # Create coordinate converter
        converter = ImageCoordinateConverter(
            center_lat=test_case["center_lat"],
            center_lon=test_case["center_lon"],
            image_width_px=image.width,
            image_height_px=image.height,
            zoom_level=test_case["zoom_level"],
        )

        # Run detection
        print(f"\nRunning Grounded-SAM detection...")
        print(f"Prompts: {', '.join(prompts)}")

        try:
            detections = detector.detect(
                image=image,
                prompts=prompts,
                coordinate_converter=converter,
            )

            print(f"\n✓ Detection complete! Found {len(detections)} objects\n")

            # Print results
            if detections:
                print("Detections:")
                print("-" * 80)
                for i, det in enumerate(detections, 1):
                    print(f"{i}. {det.label.upper()}")
                    print(f"   Confidence: {det.confidence:.2%}")
                    print(f"   Pixel BBox: {det.pixel_bbox}")
                    print(f"   Geo BBox: {det.geo_bbox}")
                    print(f"   Area: {det.area_pixels:,} pixels" +
                          (f" ({det.area_sqm:.1f} m²)" if det.area_sqm else ""))  # noqa: E501
                    print()
            else:
                print("No detections found.")

            # Save GeoJSON output
            output_dir = Path("output/examples/grounded_sam")
            output_dir.mkdir(parents=True, exist_ok=True)

            geojson_path = output_dir / f"{test_case['name']}_detections.json"

            geojson = {
                "type": "FeatureCollection",
                "features": [det.to_geojson_feature() for det in detections],
                "metadata": {
                    "image": str(image_path),
                    "center_lat": test_case["center_lat"],
                    "center_lon": test_case["center_lon"],
                    "zoom_level": test_case["zoom_level"],
                    "prompts": prompts,
                    "num_detections": len(detections),
                }
            }

            with open(geojson_path, "w") as f:
                json.dump(geojson, f, indent=2)

            print(f"✓ Saved GeoJSON: {geojson_path}")

        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("Testing complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_grounded_sam_detection()
