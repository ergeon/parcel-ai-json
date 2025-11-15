#!/usr/bin/env python3
"""Test coordinate converter roundtrip accuracy."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.coordinate_converter import (  # noqa: E402
    ImageCoordinateConverter
)

# Test coordinates for 43 Goldeneye Ct
center_lat = 38.180297
center_lon = -122.266276
zoom_level = 20

# Create converter
converter = ImageCoordinateConverter(
    center_lat=center_lat,
    center_lon=center_lon,
    image_width_px=640,
    image_height_px=640,
    zoom_level=zoom_level,
)

print("=" * 80)
print("Testing Coordinate Converter Roundtrip Accuracy")
print("=" * 80)
print(f"Center: ({center_lat}, {center_lon})")
print("Image size: 640x640 pixels")
print(f"Zoom level: {zoom_level}")
print()

# Test corner points
test_pixels = [
    (0, 0, "Top-left corner"),
    (640, 0, "Top-right corner"),
    (640, 640, "Bottom-right corner"),
    (0, 640, "Bottom-left corner"),
    (320, 320, "Center"),
]

print("Pixel → Geo → Pixel Roundtrip Test:")
print("-" * 80)
for px, py, label in test_pixels:
    lon, lat = converter.pixel_to_geo(px, py)
    px_back, py_back = converter.geo_to_pixel(lon, lat)

    error_x = abs(px - px_back)
    error_y = abs(py - py_back)

    print(
        f"{label:20s}: ({px:5.1f}, {py:5.1f}) → "
        f"({lon:.8f}, {lat:.8f}) → "
        f"({px_back:5.1f}, {py_back:5.1f})"
    )
    print(f"{'':21s}Error: ({error_x:.4f}, {error_y:.4f}) pixels")

    if error_x > 1.0 or error_y > 1.0:
        print(f"{'':21s}⚠️  WARNING: Error > 1 pixel!")
    print()

print("\n" + "=" * 80)
print("Testing HED Output Scaling (512x512 → 640x640)")
print("=" * 80)

# Test scaling from 512x512 to 640x640
scale_factor = 640 / 512
test_hed_pixels = [
    (0, 0, "HED top-left"),
    (512, 0, "HED top-right"),
    (512, 512, "HED bottom-right"),
    (0, 512, "HED bottom-left"),
    (256, 256, "HED center"),
]

print("\nHED (512x512) → Scaled (640x640) → Geo → Scaled (640x640) → HED (512x512):")
print("-" * 80)
for hed_x, hed_y, label in test_hed_pixels:
    # Scale from 512x512 to 640x640
    scaled_x = hed_x * scale_factor
    scaled_y = hed_y * scale_factor

    # Convert to geographic
    lon, lat = converter.pixel_to_geo(scaled_x, scaled_y)

    # Convert back to 640x640 pixels
    back_scaled_x, back_scaled_y = converter.geo_to_pixel(lon, lat)

    # Scale back to 512x512
    back_hed_x = back_scaled_x / scale_factor
    back_hed_y = back_scaled_y / scale_factor

    error_x = abs(hed_x - back_hed_x)
    error_y = abs(hed_y - back_hed_y)

    print(
        f"{label:20s}: HED({hed_x:5.1f}, {hed_y:5.1f}) → "
        f"Scaled({scaled_x:5.1f}, {scaled_y:5.1f})"
    )
    print(f"{'':21s}→ Geo({lon:.8f}, {lat:.8f})")
    print(
        f"{'':21s}→ Scaled({back_scaled_x:5.1f}, "
        f"{back_scaled_y:5.1f}) → "
        f"HED({back_hed_x:5.1f}, {back_hed_y:5.1f})"
    )
    print(f"{'':21s}Error: ({error_x:.4f}, {error_y:.4f}) HED pixels")

    if error_x > 1.0 or error_y > 1.0:
        print(f"{'':21s}⚠️  WARNING: Error > 1 pixel!")
    print()
