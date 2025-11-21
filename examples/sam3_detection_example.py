#!/usr/bin/env python3
"""
SAM3 Open-Vocabulary Detection Example

This script demonstrates how to use SAM3 for open-vocabulary object detection
and segmentation in satellite imagery.

Requirements:
- HF_TOKEN environment variable set (HuggingFace authentication)
- SAM3 model installed (cd models/sam3 && pip install -e .)
- Satellite image with known coordinates

Usage:
    # Set HuggingFace token
    export HF_TOKEN=your_token_here

    # Run detection
    python examples/sam3_detection_example.py \\
        --image path/to/image.jpg \\
        --lat 37.7749 \\
        --lon -122.4194 \\
        --prompts "houses,cars,trees,swimming pool"

    # With custom confidence threshold
    python examples/sam3_detection_example.py \\
        --image path/to/image.jpg \\
        --lat 37.7749 \\
        --lon -122.4194 \\
        --prompts "houses,cars,trees" \\
        --confidence 0.4 \\
        --output results.geojson
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using existing environment variables")

from parcel_ai_json import SAM3SegmentationService


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 open-vocabulary detection example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to satellite image (JPEG/PNG)"
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Image center latitude (WGS84)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Image center longitude (WGS84)"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=20,
        help="Zoom level (default: 20)"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Comma-separated detection prompts (e.g., 'houses,cars,trees')"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold (0-1, default: 0.3)"
    )
    parser.add_argument(
        "--output",
        help="Output GeoJSON file path (optional)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Check HF_TOKEN
    if not os.environ.get('HF_TOKEN') and not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        print("Error: HuggingFace token not set!")
        print("Set HF_TOKEN environment variable:")
        print("  export HF_TOKEN=your_token_here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        print("Accept SAM3 model terms at: https://huggingface.co/facebook/sam3")
        sys.exit(1)

    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Parse prompts
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        print("Error: No valid prompts provided")
        sys.exit(1)

    print("=" * 80)
    print("SAM3 Open-Vocabulary Detection")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Location: ({args.lat}, {args.lon})")
    print(f"Zoom Level: {args.zoom}")
    print(f"Prompts: {', '.join(prompts)}")
    print(f"Confidence Threshold: {args.confidence}")
    print("=" * 80)

    # Initialize SAM3 service
    print("\nInitializing SAM3...")
    print("⚠️  This may take 11-12 seconds and will download ~3.44GB on first run")

    service = SAM3SegmentationService(
        device=args.device,
        confidence_threshold=args.confidence
    )

    # Prepare satellite image metadata
    satellite_image = {
        "path": str(image_path),
        "center_lat": args.lat,
        "center_lon": args.lon,
        "zoom_level": args.zoom
    }

    # Run detection
    print("\nRunning SAM3 detection...")
    cpu_time = len(prompts) * 20
    gpu_time = len(prompts) * 6
    print(f"⏱️  Expected time: ~{cpu_time}s on CPU, ~{gpu_time}s on GPU")

    results = service.segment_image(satellite_image, prompts)

    # Print results summary
    print("\n" + "=" * 80)
    print("Detection Results")
    print("=" * 80)

    total_detections = 0
    for class_name, detections in results.items():
        count = len(detections)
        total_detections += count

        print(f"\n{class_name.upper()}: {count} detected")

        if count > 0:
            # Show top 5 detections for each class
            for i, det in enumerate(detections[:5], 1):
                print(f"  {i}. Confidence: {det.confidence:.2%}, "
                      f"Area: {det.area_sqm:.1f}m², "
                      f"BBox: {det.pixel_bbox}")

            if count > 5:
                print(f"  ... and {count - 5} more")

    print("\n" + "=" * 80)
    print(f"Total Detections: {total_detections}")
    print("=" * 80)

    # Save to GeoJSON if requested
    if args.output:
        geojson = service.segment_image_geojson(satellite_image, prompts)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")
        print(f"   - Features: {len(geojson['features'])}")
        print(f"   - Classes: {', '.join(prompts)}")

    print("\n✨ Detection complete!")


if __name__ == "__main__":
    main()
