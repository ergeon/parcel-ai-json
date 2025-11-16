#!/usr/bin/env python3
"""Debug parcel polygon mask generation."""

import sys
import json
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.fence_detector import FenceDetectionService  # noqa: E402
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402

# Load parcel polygon
parcel_json_path = (
    "/Users/Alex/Documents/GitHub/det-state-visualizer/data/raw/"
    "regrid_parcels_data/"
    "43_goldeneye_ct_american_canyon_ca_94503_usa.json"
)
with open(parcel_json_path) as f:
    parcel_data = json.load(f)

# Image metadata
center_lat = 38.180297
center_lon = -122.266276
zoom_level = 20

print("=" * 80)
print("Testing Parcel Polygon Mask Generation")
print("=" * 80)
print(f"Center: ({center_lat}, {center_lon})")
print(f"Zoom level: {zoom_level}")
print()

# Create fence detector
detector = FenceDetectionService()

# Generate fence probability mask
print("Generating fence probability mask...")
mask = detector.generate_fence_probability_mask(
    parcel_polygon=parcel_data,
    center_lat=center_lat,
    center_lon=center_lon,
    zoom_level=zoom_level,
)

print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask min: {mask.min():.4f}")
print(f"Mask max: {mask.max():.4f}")
print(f"Non-zero pixels: {np.sum(mask > 0)}")
print(f"Mask mean: {mask.mean():.4f}")
print()

# Extract parcel coordinates
coords = parcel_data["geometry"]["coordinates"][0]
print(f"Parcel polygon has {len(coords)} vertices")
print("First 3 vertices (lon, lat):")
for i, (lon, lat) in enumerate(coords[:3]):
    print(f"  {i}: ({lon:.8f}, {lat:.8f})")

print()

# Create coordinate converter
converter = ImageCoordinateConverter(
    center_lat=center_lat,
    center_lon=center_lon,
    image_width_px=640,
    image_height_px=640,
    zoom_level=zoom_level,
)

# Convert parcel coords to pixels
print("Converting parcel vertices to pixel coordinates:")
pixel_coords = []
for i, (lon, lat) in enumerate(coords):
    px, py = converter.geo_to_pixel(lon, lat)
    pixel_coords.append((px, py))
    if i < 3 or i == len(coords) - 1:
        print(f"  Vertex {i}: ({lon:.8f}, {lat:.8f}) → ({px:.2f}, {py:.2f})")

print()

# Check if coordinates are within bounds
print("Checking if pixel coordinates are within 640x640 bounds:")
in_bounds = 0
out_of_bounds = 0
for i, (px, py) in enumerate(pixel_coords):
    if 0 <= px < 640 and 0 <= py < 640:
        in_bounds += 1
    else:
        out_of_bounds += 1
        if i < 5:  # Show first few out-of-bounds
            print(f"  OUT OF BOUNDS - Vertex {i}: ({px:.2f}, {py:.2f})")

print(f"In bounds: {in_bounds}/{len(pixel_coords)}")
print(f"Out of bounds: {out_of_bounds}/{len(pixel_coords)}")

if mask.max() > 0:
    print("\n✓ SUCCESS: Mask was generated with non-zero values")

    # Save mask for visual inspection
    from PIL import Image

    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save("/tmp/test_parcel_mask.png")
    print("✓ Saved mask to /tmp/test_parcel_mask.png")
else:
    print("\n✗ FAILED: Mask is all zeros - parcel polygon was not drawn!")
    print(
        "   This suggests the pixel coordinates are likely "
        "outside the 640x640 bounds"
    )
