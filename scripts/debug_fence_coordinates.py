#!/usr/bin/env python3
"""Debug fence coordinate transformations by visualizing intermediate steps."""

import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402

# Test address: 43 Goldeneye Ct
center_lat = 38.180297
center_lon = -122.266276
zoom_level = 20

# Load GeoJSON to see fence coordinates
geojson_path = Path(
    "output/examples/geojson/" "43_goldeneye_ct_american_canyon_ca_94503_usa.geojson"
)
with open(geojson_path) as f:
    geojson = json.load(f)

# Load satellite image
image_path = Path(
    "output/examples/images/" "43_goldeneye_ct_american_canyon_ca_94503_usa.jpg"
)
satellite_img = Image.open(image_path)

print("=" * 80)
print("Debugging Fence Coordinate Transformations")
print("=" * 80)
print(f"Satellite image size: {satellite_img.size}")
print(f"Center: ({center_lat}, {center_lon})")
print(f"Zoom: {zoom_level}")
print()

# Create coordinate converter
converter = ImageCoordinateConverter(
    center_lat=center_lat,
    center_lon=center_lon,
    image_width_px=640,
    image_height_px=640,
    zoom_level=zoom_level,
)

# Extract fence features
fence_features = [
    f for f in geojson["features"] if f["properties"].get("type") == "fence"
]
print(f"Found {len(fence_features)} fence features")
print()

# Create visualization overlay
overlay = Image.new("RGBA", satellite_img.size, (255, 255, 255, 0))
draw = ImageDraw.Draw(overlay)

for i, feature in enumerate(fence_features):
    coords = feature["geometry"]["coordinates"][0]  # Exterior ring
    print(f"Fence {i + 1}:")
    print(f"  {len(coords)} vertices (geo coordinates)")

    # Convert to pixel coordinates
    pixel_coords = []
    for lon, lat in coords:
        px, py = converter.geo_to_pixel(lon, lat)
        pixel_coords.append((px, py))

    # Show first few vertices
    print("  First 3 vertices:")
    for j in range(min(3, len(coords))):
        lon, lat = coords[j]
        px, py = pixel_coords[j]
        print(f"    ({lon:.8f}, {lat:.8f}) → ({px:.2f}, {py:.2f})")

    # Check bounds
    xs = [px for px, py in pixel_coords]
    ys = [py for px, py in pixel_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    print(
        f"  Pixel bounds: X=[{min_x:.1f}, {max_x:.1f}], "
        f"Y=[{min_y:.1f}, {max_y:.1f}]"
    )

    # Draw on overlay
    if len(pixel_coords) >= 2:
        draw.line(pixel_coords, fill=(255, 128, 0, 200), width=3)
    print()

# Composite and save
result = Image.alpha_composite(satellite_img.convert("RGBA"), overlay)
output_path = Path("/tmp/fence_debug_overlay.png")
result.save(output_path)
print(f"✓ Saved debug overlay to: {output_path}")

print()
print("=" * 80)
print("Now let's test the REVERSE transformation (pixel → geo → pixel)")
print("=" * 80)

# Test: Pick a point in HED coordinates (512x512), scale to 640x640,
# convert to geo, convert back
hed_test_points = [
    (100, 100, "HED top-left quadrant"),
    (400, 100, "HED top-right quadrant"),
    (256, 256, "HED center"),
    (100, 400, "HED bottom-left quadrant"),
    (400, 400, "HED bottom-right quadrant"),
]

scale_factor = 640 / 512

for hed_x, hed_y, label in hed_test_points:
    # Scale from 512 to 640
    scaled_x = hed_x * scale_factor
    scaled_y = hed_y * scale_factor

    # Convert to geo
    lon, lat = converter.pixel_to_geo(scaled_x, scaled_y)

    # Convert back to pixels
    back_x, back_y = converter.geo_to_pixel(lon, lat)

    # Error
    error_x = abs(scaled_x - back_x)
    error_y = abs(scaled_y - back_y)

    print(f"{label}:")
    print(f"  HED({hed_x}, {hed_y}) → Scaled({scaled_x:.1f}, {scaled_y:.1f})")
    print(f"  → Geo({lon:.8f}, {lat:.8f})")
    print(f"  → Back({back_x:.1f}, {back_y:.1f})")
    print(f"  Error: ({error_x:.4f}, {error_y:.4f}) pixels")
    print()
