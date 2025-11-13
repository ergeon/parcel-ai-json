"""Generate folium map for a single address using Docker REST API.

IMPORTANT: This script uses the Docker REST API (http://localhost:8000)
Make sure the Docker container is running first:
    docker-compose up -d
    # or
    make docker-run
"""

import json
import os
import base64
import sys
from pathlib import Path
import requests
from PIL import Image
import folium

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402

# Configuration
API_BASE_URL = "http://localhost:8000"
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


def fetch_google_satellite_image(lat, lon, zoom, output_path, width=640, height=640):
    """Fetch satellite image from Google Maps Static API."""
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&"
        f"zoom={zoom}&"
        f"size={width}x{height}&"
        f"maptype=satellite&"
        f"key={GOOGLE_MAPS_API_KEY}"
    )

    print("  Fetching from Google Maps API...")
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def detect_via_api(image_path, lat, lon, zoom=20):
    """Run detection via Docker REST API."""
    # Check if API is available
    health_url = f"{API_BASE_URL}/health"
    try:
        health_response = requests.get(health_url, timeout=5)
        health_response.raise_for_status()
        print(f"✓ Docker API is healthy: {health_response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Docker container not running at {API_BASE_URL}")
        print("Please start it with: docker-compose up -d")
        raise RuntimeError(f"Docker API not available: {e}")

    # Call detection endpoint
    detect_url = f"{API_BASE_URL}/detect"

    with open(image_path, "rb") as f:
        files = {"image": ("image.jpg", f, "image/jpeg")}
        data = {
            "center_lat": lat,
            "center_lon": lon,
            "zoom_level": zoom,
            "format": "geojson",
            "include_sam": "true",  # Include SAM segmentation
            "sam_points_per_side": 32,
        }

        print(f"  Calling API: {detect_url}")
        print(f"  (This may take 5-10 minutes for first request with SAM...)")
        response = requests.post(detect_url, files=files, data=data, timeout=600)
        response.raise_for_status()

    return response.json()


def create_folium_map(image_path, geojson_data, center_lat, center_lon, output_path):
    """Generate interactive Folium map."""
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Create coordinate converter
    converter = ImageCoordinateConverter(
        center_lat=center_lat,
        center_lon=center_lon,
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=20,
    )

    # Get actual image bounds
    image_bounds_dict = converter.get_image_bounds()

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=20,
        min_zoom=15,
        max_zoom=22,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Add satellite image overlay
    with open(image_path, "rb") as f:
        img_data = f.read()

    img_base64 = base64.b64encode(img_data).decode()
    img_url = f"data:image/jpeg;base64,{img_base64}"

    bounds = [
        [image_bounds_dict["south"], image_bounds_dict["west"]],
        [image_bounds_dict["north"], image_bounds_dict["east"]],
    ]

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=0.7,
        interactive=False,
        cross_origin=False,
        zindex=1,
        name="Satellite Imagery",
    ).add_to(m)

    # Create feature groups
    vehicle_group = folium.FeatureGroup(name="🚗 Vehicles", show=True)
    pool_group = folium.FeatureGroup(name="🏊 Swimming Pools", show=True)
    amenity_group = folium.FeatureGroup(name="🎾 Amenities", show=True)
    tree_group = folium.FeatureGroup(name="🌳 Trees", show=True)
    sam_group = folium.FeatureGroup(name="🔷 SAM Segments", show=True)

    # Add detections
    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [[c[1], c[0]] for c in coords]

        feature_type = feature["properties"].get("feature_type", "vehicle")
        props = feature["properties"]

        if feature_type == "vehicle":
            area = props.get("area_sqm", 0)
            popup_html = f"<b>Vehicle</b><br>Area: {area:.1f} m²"
            folium.Polygon(
                locations=coords_swapped,
                color="#C70039",
                fill=True,
                fillColor="#C70039",
                fillOpacity=0.4,
                weight=2,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(vehicle_group)

        elif feature_type == "swimming_pool":
            area = props.get("area_sqm", 0)
            popup_html = f"<b>Pool</b><br>Area: {area:.1f} m²"
            folium.Polygon(
                locations=coords_swapped,
                color="#3498DB",
                fill=True,
                fillColor="#3498DB",
                fillOpacity=0.4,
                weight=2,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(pool_group)

        elif feature_type == "amenity":
            class_name = props.get("class_name", "Amenity")
            area = props.get("area_sqm", 0)
            popup_html = f"<b>{class_name}</b><br>Area: {area:.1f} m²"
            folium.Polygon(
                locations=coords_swapped,
                color="#9B59B6",
                fill=True,
                fillColor="#9B59B6",
                fillOpacity=0.4,
                weight=2,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(amenity_group)

        elif feature_type == "tree_cluster":
            area = props.get("area_sqm", 0)
            popup_html = f"<b>Tree</b><br>Area: {area:.1f} m²"
            folium.Polygon(
                locations=coords_swapped,
                color="#27AE60",
                fill=True,
                fillColor="#27AE60",
                fillOpacity=0.3,
                weight=2,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(tree_group)

        elif feature_type == "labeled_sam_segment":
            label = props.get("primary_label", "unknown")
            segment_id = props.get("segment_id", "N/A")
            confidence = props.get("label_confidence", 0.0)
            color = props.get("color", "#4A90E2")
            area = props.get("area_sqm", 0)

            popup_html = (
                f"<b>SAM Segment #{segment_id}</b><br>"
                f"Label: {label}<br>"
                f"Confidence: {confidence:.1%}<br>"
                f"Area: {area:.1f} m²"
            )
            folium.Polygon(
                locations=coords_swapped,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.3,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(sam_group)

    # Add feature groups to map
    vehicle_group.add_to(m)
    pool_group.add_to(m)
    amenity_group.add_to(m)
    tree_group.add_to(m)
    sam_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(str(output_path))
    return m


def generate_for_address(address, lat, lon, zoom=20):
    """Generate detections and folium map for address using Docker API."""
    filename = address.lower().replace(" ", "_").replace(",", "")

    print(f"Processing: {address}")
    print(f"Coordinates: {lat}, {lon}")
    print("=" * 80)

    # Create output directories
    output_dir = Path("output/examples")
    images_dir = output_dir / "images"
    geojson_dir = output_dir / "geojson"
    folium_dir = output_dir / "folium_maps"

    for dir_path in [images_dir, geojson_dir, folium_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch satellite image
    print("\nStep 1: Fetching satellite image from Google Maps...")
    image_path = images_dir / f"{filename}.jpg"

    try:
        fetch_google_satellite_image(lat, lon, zoom, str(image_path))
        print(f"✓ Satellite image saved to: {image_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Step 2: Run detections via Docker API
    print("\nStep 2: Running detections via Docker REST API...")
    try:
        geojson = detect_via_api(image_path, lat, lon, zoom)
        vehicle_count = sum(
            1
            for f in geojson.get("features", [])
            if f["properties"].get("feature_type") == "vehicle"
        )
        pool_count = sum(
            1
            for f in geojson.get("features", [])
            if f["properties"].get("feature_type") == "swimming_pool"
        )
        amenity_count = sum(
            1
            for f in geojson.get("features", [])
            if f["properties"].get("feature_type") == "amenity"
        )

        print(f"✓ Detected {vehicle_count} vehicles")
        print(f"✓ Detected {pool_count} pools")
        print(f"✓ Detected {amenity_count} amenities")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Save GeoJSON
    print("\nStep 3: Saving GeoJSON...")
    geojson_path = geojson_dir / f"{filename}.geojson"

    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"✓ GeoJSON saved to: {geojson_path}")

    # Step 4: Create folium map
    print("\nStep 4: Creating folium map...")
    folium_path = folium_dir / f"{filename}.html"

    try:
        create_folium_map(image_path, geojson, lat, lon, folium_path)
        print(f"✓ Folium map saved to: {folium_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n{'=' * 80}")
    print("✓ Successfully generated all outputs")
    print(f"  - Image: {image_path}")
    print(f"  - GeoJSON: {geojson_path}")
    print(f"  - Folium Map: {folium_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate detections and Folium map for a single address"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="2218 San Antonio St, Grand Prairie, TX 75051",
        help="Full address string",
    )
    parser.add_argument(
        "--lat", type=float, default=32.7459, help="Latitude of address center"
    )
    parser.add_argument(
        "--lon", type=float, default=-96.9978, help="Longitude of address center"
    )
    parser.add_argument(
        "--zoom", type=int, default=20, help="Google Maps zoom level (default: 20)"
    )

    args = parser.parse_args()

    generate_for_address(
        address=args.address, lat=args.lat, lon=args.lon, zoom=args.zoom
    )
