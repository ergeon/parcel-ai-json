"""
Generate example detection outputs using the Docker API server.

This script calls the FastAPI server running in Docker to process satellite images
and generates interactive Folium maps with all detections including tree polygons.
"""

import json
import requests
import shutil
from pathlib import Path
import folium
from folium import plugins
import base64


def generate_folium_map_with_trees(
    satellite_image_path,
    geojson_data,
    output_path,
    image_name,
    center_lat,
    center_lon,
):
    """Generate interactive Folium map with all detection features including trees.

    Args:
        satellite_image_path: Path to satellite image
        geojson_data: GeoJSON data with detections
        output_path: Output path for HTML file
        image_name: Name of the image
        center_lat: Center latitude
        center_lon: Center longitude
    """
    # Get image dimensions
    from PIL import Image

    with Image.open(satellite_image_path) as img:
        img_width, img_height = img.size

    # Create coordinate converter to get proper image bounds
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

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
        zoom_control=True,
        scrollWheelZoom=True,
        doubleClickZoom=True,
    )

    # Add satellite image as base64 encoded overlay
    if satellite_image_path.exists():
        with open(satellite_image_path, "rb") as f:
            img_data = f.read()

        img_format = (
            "jpeg"
            if satellite_image_path.suffix.lower() in [".jpg", ".jpeg"]
            else "png"
        )
        img_base64 = base64.b64encode(img_data).decode()
        img_url = f"data:image/{img_format};base64,{img_base64}"

        # Use actual image bounds from coordinate converter
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

    # Create feature groups for layer control
    vehicle_group = folium.FeatureGroup(name="🚗 Vehicles", show=True)
    pool_group = folium.FeatureGroup(name="🏊 Swimming Pools", show=True)
    amenity_group = folium.FeatureGroup(name="🎾 Amenities", show=True)
    tree_group = folium.FeatureGroup(name="🌳 Tree Clusters", show=True)

    # Count detections
    vehicle_count = 0
    pool_count = 0
    amenity_count = 0
    tree_cluster_count = 0

    # Get tree coverage info
    tree_coverage = geojson_data.get("tree_coverage", {})
    tree_coverage_percent = tree_coverage.get("tree_coverage_percent", 0.0)

    # Add detections from features
    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [[c[1], c[0]] for c in coords]  # Swap to [lat, lon]

        feature_type = feature["properties"].get("feature_type", "vehicle")
        pixel_bbox = feature["properties"].get("pixel_bbox", [0, 0, 0, 0])

        if feature_type == "swimming_pool":
            pool_count += 1
            confidence = feature["properties"]["confidence"]
            area_sqm = feature["properties"]["area_sqm"]
            popup_html = f"""
            <b>Swimming Pool Detection</b><br>
            Confidence: {confidence:.1%}<br>
            Area: {area_sqm:.1f} m²<br>
            Pixel BBox: [{pixel_bbox[0]:.0f}, {pixel_bbox[1]:.0f}, {pixel_bbox[2]:.0f}, {pixel_bbox[3]:.0f}]
            """
            tooltip_text = f"Swimming Pool ({confidence:.1%})"
            label_text = "Pool"
            fill_color = "#0099FF"
            line_color = "#0066CC"
            feature_group = pool_group

        elif feature_type == "amenity":
            amenity_count += 1
            confidence = feature["properties"]["confidence"]
            amenity_type = feature["properties"]["amenity_type"]
            area_sqm = feature["properties"]["area_sqm"]

            amenity_icons = {
                "tennis court": "🎾",
                "basketball court": "🏀",
                "baseball diamond": "⚾",
                "soccer ball field": "⚽",
                "ground track field": "🏃",
            }
            icon = amenity_icons.get(amenity_type, "🏟️")

            popup_html = f"""
            <b>{amenity_type.title()} Detection</b><br>
            Confidence: {confidence:.1%}<br>
            Area: {area_sqm:.1f} m²<br>
            Pixel BBox: [{pixel_bbox[0]:.0f}, {pixel_bbox[1]:.0f}, {pixel_bbox[2]:.0f}, {pixel_bbox[3]:.0f}]
            """
            tooltip_text = f"{amenity_type.title()} ({confidence:.1%})"
            label_text = f"{icon} {amenity_type.split()[0].title()}"
            fill_color = "#FF6B00"
            line_color = "#CC5500"
            feature_group = amenity_group

        elif feature_type == "tree_cluster":
            tree_cluster_count += 1
            area_sqm = feature["properties"]["area_sqm"]
            area_pixels = feature["properties"]["area_pixels"]

            popup_html = f"""
            <b>Tree Cluster</b><br>
            Area: {area_sqm:.1f} m²<br>
            Pixels: {area_pixels}<br>
            """
            tooltip_text = f"Tree Cluster ({area_sqm:.1f} m²)"
            label_text = f"🌳 {area_sqm:.0f}m²"
            fill_color = "#228B22"  # Forest green
            line_color = "#006400"  # Dark green
            feature_group = tree_group

        else:  # vehicle
            vehicle_count += 1
            confidence = feature["properties"]["confidence"]
            vehicle_class = feature["properties"]["vehicle_class"]
            popup_html = f"""
            <b>Vehicle Detection</b><br>
            Class: {vehicle_class}<br>
            Confidence: {confidence:.1%}<br>
            Pixel BBox: [{pixel_bbox[0]:.0f}, {pixel_bbox[1]:.0f}, {pixel_bbox[2]:.0f}, {pixel_bbox[3]:.0f}]
            """
            tooltip_text = f"{vehicle_class} ({confidence:.1%})"
            label_text = vehicle_class.title()
            fill_color = "#00FF00"
            line_color = "#00AA00"
            feature_group = vehicle_group

        # Add polygon
        folium.Polygon(
            locations=coords_swapped,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_text,
            fillColor=fill_color,
            color=line_color,
            weight=2,
            fillOpacity=0.4,
            opacity=0.8,
        ).add_to(feature_group)

        # Add label marker at center
        center_lat_poly = sum(c[0] for c in coords_swapped) / len(coords_swapped)
        center_lon_poly = sum(c[1] for c in coords_swapped) / len(coords_swapped)

        folium.Marker(
            location=[center_lat_poly, center_lon_poly],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 11px;
                    font-weight: bold;
                    color: white;
                    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
                    white-space: nowrap;
                    text-align: center;
                ">{label_text}</div>
                """
            ),
        ).add_to(feature_group)

    # Add all feature groups to map
    vehicle_group.add_to(m)
    pool_group.add_to(m)
    amenity_group.add_to(m)
    tree_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add title with all counts
    title_html = f"""
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 850px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <h4>{image_name}</h4>
        <b>Vehicles:</b> {vehicle_count} &nbsp;&nbsp;
        <b>Pools:</b> {pool_count} &nbsp;&nbsp;
        <b>Amenities:</b> {amenity_count} &nbsp;&nbsp;
        <b>Tree Clusters:</b> {tree_cluster_count} &nbsp;&nbsp;
        <b>Tree Coverage:</b> {tree_coverage_percent:.1f}%
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Add measure control
    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)

    # Add fullscreen option
    plugins.Fullscreen().add_to(m)

    # Save
    m.save(str(output_path))


def detect_via_api(image_path, center_lat, center_lon, api_url="http://localhost:8000"):
    """Call the FastAPI detection endpoint.

    Args:
        image_path: Path to satellite image
        center_lat: Center latitude
        center_lon: Center longitude
        api_url: Base URL of the API server

    Returns:
        GeoJSON response with all detections
    """
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        data = {
            "center_lat": str(center_lat),
            "center_lon": str(center_lon),
            "zoom_level": "20",
            "format": "geojson",
        }

        response = requests.post(
            f"{api_url}/detect",
            files=files,
            data=data,
            timeout=60,
        )

    response.raise_for_status()
    return response.json()


def get_default_coordinates_by_state(state_code):
    """Get default coordinates for a state (center of state)."""
    state_centers = {
        "ca": (36.7783, -119.4179),
        "nj": (40.0583, -74.4057),
        "ny": (42.1657, -74.9481),
        "tx": (31.9686, -99.9018),
        "fl": (27.7663, -81.6868),
        "pa": (40.5908, -77.2098),
        "va": (37.4316, -78.6569),
        "md": (39.0458, -76.6413),
        "sc": (33.8569, -80.9450),
        "il": (40.6331, -89.3985),
    }
    return state_centers.get(state_code.lower(), (37.0, -122.0))


def generate_examples_via_api(num_examples=5, api_url="http://localhost:8000"):
    """Generate examples using the Docker API server."""
    print("=" * 80)
    print(f"Generating {num_examples} Examples via Docker API")
    print("=" * 80)

    # Test API connection
    print(f"\nTesting API connection to {api_url}...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        health = response.json()
        print(f"✓ API Status: {health['status']}")
        print(f"✓ Detector loaded: {health.get('detector_loaded', False)}")
    except Exception as e:
        print(f"✗ Error connecting to API: {e}")
        print("  Make sure Docker container is running: docker ps")
        return

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent / "det-state-visualizer"
    satellite_dir = base_dir / "data/raw/satellite_images"
    output_dir = Path(__file__).parent.parent / "output/examples_api"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSource directory: {satellite_dir}")
    print(f"Output directory: {output_dir}")

    # Get list of satellite images
    satellite_images = list(satellite_dir.glob("*.jpg"))
    print(f"\n✓ Found {len(satellite_images)} satellite images")

    # Process images
    processed = 0
    results = []

    print(f"\nProcessing {num_examples} images via API...")
    print("-" * 80)

    for img_path in satellite_images[:num_examples]:
        img_name = img_path.name
        print(f"\n[{processed + 1}/{num_examples}] Processing: {img_name}")

        try:
            # Extract state code from filename
            parts = img_path.stem.split("_")
            state_code = None
            for part in reversed(parts):
                if len(part) == 2 and part.isalpha():
                    state_code = part
                    break

            # Get coordinates
            if state_code:
                lat, lon = get_default_coordinates_by_state(state_code)
            else:
                lat, lon = 37.0, -122.0

            print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")
            print(f"  Calling API...")

            # Call API
            geojson_data = detect_via_api(img_path, lat, lon, api_url)

            # Extract counts
            tree_coverage = geojson_data.get("tree_coverage", {})
            features = geojson_data.get("features", [])

            vehicle_count = sum(
                1 for f in features if f["properties"].get("feature_type") == "vehicle"
            )
            pool_count = sum(
                1
                for f in features
                if f["properties"].get("feature_type") == "swimming_pool"
            )
            amenity_count = sum(
                1 for f in features if f["properties"].get("feature_type") == "amenity"
            )
            tree_cluster_count = sum(
                1
                for f in features
                if f["properties"].get("feature_type") == "tree_cluster"
            )

            tree_coverage_percent = tree_coverage.get("tree_coverage_percent", 0.0)

            print(
                f"  ✓ Detected: {vehicle_count} vehicles, {pool_count} pools, "
                f"{amenity_count} amenities, {tree_cluster_count} tree clusters, "
                f"{tree_coverage_percent:.1f}% tree coverage"
            )

            # Save GeoJSON
            geojson_dir = output_dir / "geojson"
            geojson_dir.mkdir(exist_ok=True)

            output_filename = img_path.stem + "_detections.geojson"
            output_path = geojson_dir / output_filename

            with open(output_path, "w") as f:
                json.dump(geojson_data, f, indent=2)

            print(f"  ✓ Saved GeoJSON to: geojson/{output_filename}")

            # Generate Folium map
            folium_dir = output_dir / "folium_maps"
            folium_dir.mkdir(exist_ok=True)

            folium_path = folium_dir / f"{img_path.stem}.html"
            generate_folium_map_with_trees(
                satellite_image_path=img_path,
                geojson_data=geojson_data,
                output_path=folium_path,
                image_name=img_name,
                center_lat=lat,
                center_lon=lon,
            )

            print(f"  ✓ Generated Folium map: {folium_path.name}")

            results.append(
                {
                    "image": img_name,
                    "vehicles": vehicle_count,
                    "pools": pool_count,
                    "amenities": amenity_count,
                    "tree_clusters": tree_cluster_count,
                    "tree_coverage_percent": tree_coverage_percent,
                    "coordinates": {"lat": lat, "lon": lon},
                }
            )

            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Total vehicles: {sum(r['vehicles'] for r in results)}")
    print(f"Total pools: {sum(r['pools'] for r in results)}")
    print(f"Total amenities: {sum(r['amenities'] for r in results)}")
    print(f"Total tree clusters: {sum(r['tree_clusters'] for r in results)}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"total_processed": processed, "results": results},
            f,
            indent=2,
        )

    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"✓ All outputs saved to: {output_dir}")
    print(f"\n📂 Open files in {output_dir / 'folium_maps'}/ to view maps")


if __name__ == "__main__":
    generate_examples_via_api(num_examples=5)
