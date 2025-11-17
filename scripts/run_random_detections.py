#!/usr/bin/env python3
"""
Run detections on N random datasets from test_datasets.

Usage:
    python scripts/run_random_detections.py 5           # Run 5 random
    python scripts/run_random_detections.py 10 --map    # Run 10 with Folium maps
    python scripts/run_random_detections.py 20 --no-sam # Run 20 without SAM
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import requests


# Paths
INDEX_FILE = Path("output/test_datasets/test_index.csv")
IMAGES_DIR = Path("output/test_datasets/satellite_images")
REGRID_DIR = Path("output/test_datasets/regrid_parcels")
RESULTS_DIR = Path("output/test_datasets/results")

# API endpoint
API_URL = "http://localhost:8000/detect"


def main():
    """Run detections on N random datasets."""
    parser = argparse.ArgumentParser(
        description="Run detections on N random test datasets"
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of random datasets to process (1-100)"
    )
    parser.add_argument(
        "--sam",
        action="store_true",
        default=True,
        help="Include SAM segmentation (default: True)"
    )
    parser.add_argument(
        "--no-sam",
        action="store_false",
        dest="sam",
        help="Skip SAM segmentation"
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Generate Folium maps for each result"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate count
    if args.count < 1 or args.count > 100:
        print("Error: Count must be between 1 and 100")
        sys.exit(1)

    # Set random seed
    if args.seed:
        random.seed(args.seed)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load index
    print(f"Loading test index from {INDEX_FILE}...")
    df = pd.read_csv(INDEX_FILE)
    print(f"✓ Loaded {len(df)} datasets\n")

    # Select random samples
    print(f"Selecting {args.count} random datasets...")
    samples = df.sample(n=args.count, random_state=args.seed)
    print(f"✓ Selected {len(samples)} datasets\n")

    # Process each dataset
    print(f"{'='*80}")
    print(f"Running Detections (SAM: {args.sam}, Maps: {args.map})")
    print(f"{'='*80}\n")

    success_count = 0
    for idx, (i, row) in enumerate(samples.iterrows(), 1):
        quote_id = row['quote_id']
        image_filename = row['image_filename']
        lat = row['latitude']
        lon = row['longitude']

        print(f"[{idx}/{len(samples)}] Processing: {image_filename}")

        # File paths
        image_path = IMAGES_DIR / image_filename
        regrid_filename = image_filename.replace('.jpg', '_usa.json')
        regrid_path = REGRID_DIR / regrid_filename
        result_filename = image_filename.replace('.jpg', '_detections.json')
        result_path = RESULTS_DIR / result_filename

        # Check files exist
        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            continue
        if not regrid_path.exists():
            print(f"  ✗ Regrid parcel not found: {regrid_path}")
            continue

        # Load regrid parcel
        with open(regrid_path) as f:
            regrid_data = json.load(f)
            parcel_polygon = regrid_data['geometry']['coordinates'][0]

        # Run detection
        try:
            with open(image_path, "rb") as f:
                # Send with PNG MIME type since files are actually PNG
                files = {"image": ("image.png", f, "image/png")}
                data = {
                    "center_lat": lat,
                    "center_lon": lon,
                    "zoom_level": 20,
                    "detect_fences": "true",
                    "include_sam": "true" if args.sam else "false",
                    "regrid_parcel_polygon": json.dumps([parcel_polygon]),
                }

                print(f"  → Calling API...")
                response = requests.post(API_URL, files=files, data=data)

            if response.status_code == 200:
                # Save result
                result = response.json()
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

                feature_count = len(result.get('features', []))
                print(f"  ✓ Detection complete: {feature_count} features")
                print(f"  ✓ Saved to: {result_path}")

                # Generate Folium map if requested
                if args.map:
                    map_filename = image_filename.replace('.jpg', '_map.html')
                    map_path = RESULTS_DIR / map_filename

                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from scripts.create_folium_from_geojson import (
                        create_folium_map_from_geojson
                    )

                    create_folium_map_from_geojson(
                        geojson_path=str(result_path),
                        image_path=str(image_path),
                        output_path=str(map_path),
                        center_lat=lat,
                        center_lon=lon,
                        zoom_level=20,
                    )
                    print(f"  ✓ Map saved to: {map_path}")

                success_count += 1
            else:
                print(f"  ✗ API error: {response.status_code}")
                print(f"     {response.text[:200]}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    # Summary
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Requested:  {args.count}")
    print(f"Processed:  {success_count}")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'='*80}\n")

    if success_count > 0:
        print("✅ Detections complete!")
    else:
        print("✗ No detections completed")


if __name__ == "__main__":
    main()
