"""Generate folium map for a single address."""

import sys
import json
import os
import base64
from pathlib import Path
import requests
from PIL import Image
import folium
from folium import plugins

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.property_detector import PropertyDetectionService  # noqa: E402
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402
from parcel_ai_json.device_utils import get_best_device  # noqa: E402


def fetch_google_satellite_image(lat, lon, zoom, output_path, width=640, height=640):
    """Fetch satellite image from Google Maps Static API."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&"
        f"zoom={zoom}&"
        f"size={width}x{height}&"
        f"maptype=satellite&"
        f"key={api_key}"
    )

    print(f"  Fetching from Google Maps API...")
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(response.content)

    return output_path


def generate_folium_map_for_address(
    satellite_image_path,
    geojson_data,
    output_path,
    center_lat,
    center_lon,
    sam_segments=None,
):
    """Generate interactive Folium map."""
    if sam_segments is None:
        sam_segments = []

    # Get image dimensions
    with Image.open(satellite_image_path) as img:
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
    with open(satellite_image_path, "rb") as f:
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

    # Add detections
    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [[c[1], c[0]] for c in coords]

        feature_type = feature["properties"].get("feature_type", "vehicle")
        props = feature["properties"]

        if feature_type == "vehicle":
            popup_html = f"<b>Vehicle</b><br>Area: {props.get('area_sqm', 0):.1f} m²"
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
            popup_html = f"<b>Pool</b><br>Area: {props.get('area_sqm', 0):.1f} m²"
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
            popup_html = f"<b>{props.get('class_name', 'Amenity')}</b><br>Area: {props.get('area_sqm', 0):.1f} m²"
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
            popup_html = f"<b>Tree</b><br>Area: {props.get('area_sqm', 0):.1f} m²"
            folium.Polygon(
                locations=coords_swapped,
                color="#27AE60",
                fill=True,
                fillColor="#27AE60",
                fillOpacity=0.3,
                weight=2,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(tree_group)

    # Add feature groups to map
    vehicle_group.add_to(m)
    pool_group.add_to(m)
    amenity_group.add_to(m)
    tree_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(str(output_path))
    return m


def generate_for_address(address, lat, lon, zoom=20):
    """Generate detections and folium map for a specific address."""

    # Normalize filename
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
        fetch_google_satellite_image(
            lat=lat,
            lon=lon,
            zoom=zoom,
            output_path=str(image_path),
            width=640,
            height=640
        )
        print(f"✓ Satellite image saved to: {image_path}")
    except Exception as e:
        print(f"ERROR fetching image: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Run detections
    print("\nStep 2: Running property detections (without tree detection)...")
    detector = PropertyDetectionService(
        detectree_extract_polygons=False,
        detectree_use_docker=False
    )

    try:
        satellite_image = {
            "path": str(image_path),
            "center_lat": lat,
            "center_lon": lon,
            "zoom_level": zoom,
        }

        detections = detector.detect_all(satellite_image)

        print(f"✓ Detected {len(detections.vehicles)} vehicles")
        print(f"✓ Detected {len(detections.pools)} pools")
        print(f"✓ Detected {len(detections.amenities)} amenities")
        tree_poly_count = len(detections.tree_polygons or [])
        print(f"✓ Detected {tree_poly_count} tree polygons")

    except Exception as e:
        print(f"ERROR running detections: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Save GeoJSON
    print("\nStep 3: Saving GeoJSON...")
    geojson_path = geojson_dir / f"{filename}.geojson"
    geojson = detections.to_geojson()

    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"✓ GeoJSON saved to: {geojson_path}")

    # Step 4: Create folium map
    print("\nStep 4: Creating folium map...")
    folium_path = folium_dir / f"{filename}.html"

    try:
        generate_folium_map_for_address(
            satellite_image_path=image_path,
            geojson_data=geojson,
            output_path=folium_path,
            center_lat=lat,
            center_lon=lon,
        )
        print(f"✓ Folium map saved to: {folium_path}")
    except Exception as e:
        print(f"ERROR creating folium map: {e}")
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
    # 2218 San Antonio St, Grand Prairie, TX 75051
    generate_for_address(
        address="2218 San Antonio St, Grand Prairie, TX 75051",
        lat=32.7459,
        lon=-96.9978,
        zoom=20
    )
