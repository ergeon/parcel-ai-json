"""
✅ ARCHITECTURE COMPLIANT - Uses REST API exclusively

Create interactive folium map with SAM segments and semantic detections.

This script generates an enhanced folium map showing:
- Satellite imagery base layer
- SAM segments (general-purpose segmentation)
- Vehicles (YOLO detections)
- Swimming pools
- Amenities (tennis courts, etc.)
- Trees (DeepForest + detectree)

All detection is performed via Docker container REST API at http://localhost:8000
"""

import sys
from pathlib import Path
import requests
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import folium  # noqa: E402


def ensure_docker_running() -> bool:
    """Ensure Docker container is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✓ Docker container is running")
            return True
    except requests.exceptions.RequestException:
        pass

    print("❌ Docker container is not running!")
    print("Start it with: docker-compose up -d")
    return False


def detect_via_api(
    image_path: Path,
    center_lat: float,
    center_lon: float,
    zoom_level: int = 20
) -> Dict[str, Any]:
    """Call REST API to detect features in satellite image.

    Args:
        image_path: Path to satellite image
        center_lat: Center latitude
        center_lon: Center longitude
        zoom_level: Zoom level (default: 20)

    Returns:
        Detection results from API (GeoJSON format)
    """
    url = "http://localhost:8000/api/v1/detect"

    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        data = {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
            "include_trees": True,
            "extract_tree_polygons": True,
            "run_sam": True,
            "label_sam_segments": True,
        }

        response = requests.post(url, files=files, data=data, timeout=300)
        response.raise_for_status()
        return response.json()


def create_enhanced_folium_map(image_path: str, output_path: str):
    """Create folium map with SAM segments and all detections.

    Args:
        image_path: Path to satellite image
        output_path: Where to save the HTML map
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract state from filename and use approximate coordinates
    # Format: address_city_state_zip.jpg
    parts = image_path.stem.split("_")
    state_code = parts[-2] if len(parts) > 2 else "ca"

    # Better default coordinates by state
    state_centers = {
        "ca": (36.7783, -119.4179),  # California center
        "nj": (40.0583, -74.4057),  # New Jersey
        "tx": (31.9686, -99.9018),  # Texas
        "nc": (35.6301, -79.8064),  # North Carolina
    }

    # Special case for known cities
    if "vacaville" in image_path.stem.lower():
        center_lat, center_lon = 38.3566, -121.9877
    elif "newton" in image_path.stem.lower():
        center_lat, center_lon = 40.8998, -74.7524
    elif "holly_springs" in image_path.stem.lower():
        center_lat, center_lon = 35.6515, -78.8336
    else:
        center_lat, center_lon = state_centers.get(state_code.lower(), (37.0, -122.0))

    print(f"Creating enhanced folium map for: {image_path.name}")
    print("=" * 80)

    # Ensure Docker is running
    if not ensure_docker_running():
        raise RuntimeError("Docker container must be running. Start with: docker-compose up -d")

    # Call REST API to get all detections
    print("1. Calling REST API for detection...")
    detections = detect_via_api(image_path, center_lat, center_lon, zoom_level=20)

    # Extract data from GeoJSON response
    features = detections.get("features", [])

    # Separate by type
    sam_segments = []
    vehicles = []
    pools = []
    amenities = []
    trees_deepforest = []
    trees_detectree = []

    for feature in features:
        props = feature.get("properties", {})
        detection_type = props.get("type", "unknown")

        if detection_type == "sam_segment":
            sam_segments.append(feature)
        elif detection_type == "vehicle":
            vehicles.append(feature)
        elif detection_type == "swimming_pool":
            pools.append(feature)
        elif detection_type == "amenity":
            amenities.append(feature)
        elif detection_type == "tree" and props.get("source") == "deepforest":
            trees_deepforest.append(feature)
        elif detection_type == "tree_polygon" and props.get("source") == "detectree":
            trees_detectree.append(feature)

    print(f"   ✓ Received {len(sam_segments)} SAM segments")
    print(f"   ✓ Received {len(vehicles)} vehicles")
    print(f"   ✓ Received {len(pools)} pools")
    print(f"   ✓ Received {len(amenities)} amenities")
    print(f"   ✓ Received {len(trees_deepforest)} trees (DeepForest)")
    print(f"   ✓ Received {len(trees_detectree)} tree polygons (detectree)")

    # Get image dimensions for bounds calculation
    from PIL import Image
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    converter = ImageCoordinateConverter(
        center_lat=center_lat,
        center_lon=center_lon,
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=20,
    )

    # Get accurate image bounds
    image_bounds_dict = converter.get_image_bounds()
    bounds = [
        [image_bounds_dict["south"], image_bounds_dict["west"]],
        [image_bounds_dict["north"], image_bounds_dict["east"]],
    ]

    # Create folium map
    print("2. Creating folium map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=20,
        zoom_control=True,
        max_zoom=22,
        min_zoom=10,
    )

    # Add satellite image overlay
    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=bounds,
        opacity=0.8,
        name="Satellite Image",
    ).add_to(m)

    # Create feature groups for layer control
    sam_group = folium.FeatureGroup(
        name=f"SAM Segments ({len(sam_segments)})", show=True
    )
    vehicles_group = folium.FeatureGroup(
        name=f"Vehicles ({len(vehicles)})", show=True
    )
    pools_group = folium.FeatureGroup(
        name=f"Swimming Pools ({len(pools)})", show=True
    )
    amenities_group = folium.FeatureGroup(
        name=f"Amenities ({len(amenities)})", show=True
    )
    deepforest_group = folium.FeatureGroup(
        name=f"Trees - DeepForest ({len(trees_deepforest)})", show=True
    )
    detectree_group = folium.FeatureGroup(
        name=f"Tree Coverage - detectree ({len(trees_detectree)})", show=True
    )

    # Add SAM segments with semantic labels
    print("3. Adding labeled SAM segments to map...")
    from parcel_ai_json.sam_labeler import LABEL_SCHEMA

    for segment in sam_segments:
        props = segment["properties"]
        coords = segment["geometry"]["coordinates"][0]  # Polygon coordinates

        label = props.get("primary_label", "unknown")
        color = LABEL_SCHEMA.get(label, LABEL_SCHEMA["unknown"])["color"]

        # Build label display
        label_display = label.replace("_", " ").title()
        if props.get("label_subtype"):
            label_display += f" ({props['label_subtype']})"

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color=color,
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.3,
            popup=folium.Popup(
                f"<b>{label_display}</b><br>"
                f"Confidence: {props.get('label_confidence', 0):.2f}<br>"
                f"Source: {props.get('label_source', 'N/A')}<br>"
                f"Area: {props.get('area_sqm', 0):.1f} m² ({props.get('area_pixels', 0)} px)<br>"
                f"Stability: {props.get('stability_score', 0):.3f}<br>"
                f"Reason: {props.get('labeling_reason', 'N/A')}",
                max_width=300,
            ),
            tooltip=label_display,
        ).add_to(sam_group)

    # Add vehicles
    print("4. Adding vehicles to map...")
    for vehicle in vehicles:
        props = vehicle["properties"]
        coords = vehicle["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#800080",  # Purple
            weight=2,
            fill=True,
            fillColor="#800080",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Vehicle</b><br>"
                f"Class: {props.get('class_name', 'vehicle')}<br>"
                f"Confidence: {props.get('confidence', 0):.2f}",
                max_width=200,
            ),
            tooltip=f"Vehicle: {props.get('class_name', 'vehicle')}",
        ).add_to(vehicles_group)

    # Add swimming pools
    print("5. Adding swimming pools to map...")
    for pool in pools:
        props = pool["properties"]
        coords = pool["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#0066cc",  # Blue
            weight=2,
            fill=True,
            fillColor="#0066cc",
            fillOpacity=0.5,
            popup=folium.Popup(
                f"<b>Swimming Pool</b><br>"
                f"Confidence: {props.get('confidence', 0):.2f}<br>"
                f"Area: {props.get('area_sqm', 0):.1f} m²",
                max_width=200,
            ),
            tooltip="Swimming Pool",
        ).add_to(pools_group)

    # Add amenities
    print("6. Adding amenities to map...")
    for amenity in amenities:
        props = amenity["properties"]
        coords = amenity["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#ff8800",  # Orange
            weight=2,
            fill=True,
            fillColor="#ff8800",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Amenity</b><br>"
                f"Type: {props.get('amenity_type', 'unknown')}<br>"
                f"Confidence: {props.get('confidence', 0):.2f}<br>"
                f"Area: {props.get('area_sqm', 0):.1f} m²",
                max_width=200,
            ),
            tooltip=f"{props.get('amenity_type', 'Amenity')}",
        ).add_to(amenities_group)

    # Add trees (DeepForest)
    print("7. Adding trees to map...")
    for i, tree in enumerate(trees_deepforest):
        props = tree["properties"]
        coords = tree["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#228B22",  # Forest green
            weight=2,
            fill=True,
            fillColor="#228B22",
            fillOpacity=0.3,
            popup=folium.Popup(
                f"<b>Individual Tree (DeepForest)</b><br>"
                f"Confidence: {props.get('confidence', 0):.2f}",
                max_width=200,
            ),
            tooltip=f"Tree #{i+1}",
        ).add_to(deepforest_group)

    # Add tree coverage (detectree)
    for i, tree_poly in enumerate(trees_detectree):
        props = tree_poly["properties"]
        coords = tree_poly["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#006400",  # Dark green
            weight=2,
            fill=True,
            fillColor="#006400",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Tree Coverage (detectree)</b><br>"
                f"Area: {props.get('area_sqm', 0):.1f} m²<br>"
                f"Pixels: {props.get('area_pixels', 0)}",
                max_width=200,
            ),
            tooltip=f"Tree Cluster #{i+1}",
        ).add_to(detectree_group)

    # Add all groups to map
    sam_group.add_to(m)
    vehicles_group.add_to(m)
    pools_group.add_to(m)
    amenities_group.add_to(m)
    deepforest_group.add_to(m)
    detectree_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-zoom to fit bounds
    m.fit_bounds(bounds)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight: bold; text-align: center;">Legend</p>
    <p style="margin:5px 0;"><span style="color:#800080;">●</span> Vehicles</p>
    <p style="margin:5px 0;"><span style="color:#0066cc;">●</span> Swimming Pools</p>
    <p style="margin:5px 0;"><span style="color:#ff8800;">●</span> Amenities</p>
    <p style="margin:5px 0;"><span style="color:#228B22;">●</span> Trees (DeepForest)</p>
    <p style="margin:5px 0;"><span style="color:#006400;">●</span> Tree Coverage (detectree)</p>
    <p style="margin:5px 0;"><span style="color:#FF0000;">●</span> SAM: Vehicle</p>
    <p style="margin:5px 0;"><span style="color:#87CEEB;">●</span> SAM: Driveway</p>
    <p style="margin:5px 0;"><span style="color:#808080;">●</span> SAM: Unknown</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = f"""
    <div style="position: fixed;
                top: 10px; left: 50px; width: auto; height: auto;
                background-color: white; z-index:9999; font-size:16px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <b>{image_path.stem}</b><br>
    <small>SAM Segments + Semantic Detections</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\n✓ Enhanced folium map saved to: {output_path}")
    print("\nSummary:")
    print(f"  - SAM segments: {len(sam_segments)}")
    print(f"  - Vehicles: {len(vehicles)}")
    print(f"  - Swimming pools: {len(pools)}")
    print(f"  - Amenities: {len(amenities)}")
    print(f"  - Trees (DeepForest): {len(trees_deepforest)}")
    print(f"  - Tree polygons (detectree): {len(trees_detectree)}")
    print("\nOpen the HTML file in your browser to explore!")


if __name__ == "__main__":
    # Use the same image we tested SAM on
    image_path = "output/examples/images/672_white_oak_ln_vacaville_ca_95687.jpg"
    output_path = "output/examples/folium_maps/672_white_oak_ln_vacaville_ca_95687_with_sam.html"

    create_enhanced_folium_map(image_path, output_path)
