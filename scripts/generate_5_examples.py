"""
Generate 5 detection examples with satellite images, Regrid parcels, and Folium maps.
"""

import os
import json
import time
import requests
from pathlib import Path

# Example addresses (residential properties in Bay Area)
ADDRESSES = [
    "456 University Avenue, Palo Alto, CA 94301",
    "123 Oak Street, San Carlos, CA 94070",
    "789 Elm Avenue, Menlo Park, CA 94025",
    "321 Pine Street, Redwood City, CA 94063",
    "555 Maple Drive, Los Altos, CA 94022",
]

# API endpoints
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REGRID_API_KEY = os.environ.get("REGRID_API_KEY", "")
DETECT_API_URL = "http://localhost:8000/detect"

# Output directory
OUTPUT_DIR = Path("output/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_coordinates(address):
    """Get lat/lon from address using Nominatim."""
    print(f"  Getting coordinates for: {address}")
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": address,
            "format": "json",
            "limit": 1,
        },
        headers={"User-Agent": "parcel-ai-json/1.0"},
    )
    results = response.json()
    if not results:
        raise ValueError(f"Address not found: {address}")

    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    print(f"    ✓ Coordinates: {lat}, {lon}")
    return lat, lon


def get_satellite_image(lat, lon, output_path, zoom=20, width=512, height=512):
    """Download satellite image from Google Static Maps."""
    print("  Downloading satellite image...")

    # Simple download using a basic tile
    # This is a placeholder - in production, use proper tile stitching
    import math
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    url = f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"    ✓ Saved to: {output_path}")
        return True
    else:
        print("    ✗ Failed to download image")
        return False


def get_regrid_parcel(lat, lon):
    """Get parcel polygon from Regrid API."""
    print("  Getting Regrid parcel data...")

    if not REGRID_API_KEY:
        print("    ⚠ REGRID_API_KEY not set, using mock data")
        # Return a simple square polygon as fallback
        offset = 0.0001
        return [
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset],
        ]

    response = requests.get(
        "https://app.regrid.com/api/v1/parcel.json",
        params={
            "token": REGRID_API_KEY,
            "lat": lat,
            "lon": lon,
        },
    )

    if response.status_code == 200:
        data = response.json()
        if data.get("parcels"):
            coords = data["parcels"][0]["geometry"]["coordinates"][0]
            print(f"    ✓ Got parcel with {len(coords)} points")
            return coords

    print("    ⚠ Using fallback polygon")
    offset = 0.0001
    return [
        [lon - offset, lat - offset],
        [lon + offset, lat - offset],
        [lon + offset, lat + offset],
        [lon - offset, lat + offset],
        [lon - offset, lat - offset],
    ]


def run_detection(image_path, lat, lon, parcel_polygon, output_json):
    """Run detection via API."""
    print("  Running detection...")

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {
            "center_lat": lat,
            "center_lon": lon,
            "zoom_level": 20,
            "detect_fences": "true",
            "include_sam": "true",
            "regrid_parcel_polygon": json.dumps([parcel_polygon]),
        }

        response = requests.post(DETECT_API_URL, files=files, data=data)

    if response.status_code == 200:
        with open(output_json, "w") as f:
            json.dump(response.json(), f, indent=2)
        print(f"    ✓ Detection complete: {output_json}")
        return True
    else:
        print(f"    ✗ Detection failed: {response.text}")
        return False


def generate_folium_map(geojson_path, image_path, lat, lon, output_html):
    """Generate Folium map from detection GeoJSON."""
    print("  Generating Folium map...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.create_folium_from_geojson import create_folium_map_from_geojson

    create_folium_map_from_geojson(
        geojson_path=str(geojson_path),
        image_path=str(image_path),
        output_path=str(output_html),
        center_lat=lat,
        center_lon=lon,
        zoom_level=20,
    )
    print(f"    ✓ Map saved: {output_html}")


def generate_example(address, index):
    """Generate one complete example."""
    print(f"\n{'='*80}")
    print(f"Example {index + 1}: {address}")
    print(f"{'='*80}")

    # Create filename-safe name
    safe_name = address.lower()
    for char in [",", " ", "."]:
        safe_name = safe_name.replace(char, "_")
    safe_name = safe_name.strip("_")

    try:
        # Get coordinates
        lat, lon = get_coordinates(address)

        # Download satellite image
        image_path = OUTPUT_DIR / f"{index + 1}_{safe_name}.jpg"
        if not get_satellite_image(lat, lon, image_path):
            print("  ✗ Skipping this example")
            return False

        # Get parcel polygon
        parcel_polygon = get_regrid_parcel(lat, lon)

        # Run detection
        json_path = OUTPUT_DIR / f"{index + 1}_{safe_name}_detections.json"
        if not run_detection(image_path, lat, lon, parcel_polygon, json_path):
            print("  ✗ Skipping this example")
            return False

        # Generate Folium map
        html_path = OUTPUT_DIR / f"{index + 1}_{safe_name}_map.html"
        generate_folium_map(json_path, image_path, lat, lon, html_path)

        print(f"\n  ✅ Example {index + 1} complete!")
        print(f"     Image: {image_path}")
        print(f"     JSON:  {json_path}")
        print(f"     Map:   {html_path}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Generate all examples."""
    print("Generating 5 detection examples...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    # Check if Docker is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("⚠ Warning: API health check failed")
    except requests.exceptions.RequestException:
        print("✗ Error: Docker container not running!")
        print("Please start it with: make docker-run")
        return

    success_count = 0
    for i, address in enumerate(ADDRESSES):
        if generate_example(address, i):
            success_count += 1
        time.sleep(2)  # Rate limiting

    print(f"\n{'='*80}")
    print(f"✅ Generated {success_count}/{len(ADDRESSES)} examples successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
