"""
Generate example detection outputs from satellite images using REST API.

This script processes satellite images from the det-state-visualizer project
and generates GeoJSON files with detections using the Docker REST API.

✅ ARCHITECTURE COMPLIANT - Uses REST API at http://localhost:8000/detect
"""

import argparse
import json
import csv
import shutil
import base64
import requests
from pathlib import Path
import folium
from folium import plugins


def ensure_docker_running():
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
    print("Or: make docker-run")
    return False


def load_quote_coordinates(csv_path):
    """Load quote coordinates from CSV."""
    coords_by_address = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("formatted_address", "")
            if address:
                coords_by_address[address.lower()] = {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "quote_id": row.get("quote_id", ""),
                }
    return coords_by_address


def parse_address_from_filename(filename):
    """Parse address components from satellite image filename."""
    name = Path(filename).stem
    parts = name.split("_")

    if len(parts) < 3:
        return None

    return {
        "filename": filename,
        "parts": parts,
        "raw": name,
    }


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


def detect_via_api(image_path, center_lat, center_lon, zoom_level=20):
    """Call REST API to detect features in satellite image.

    Returns:
        dict: Detection results from API
    """
    url = "http://localhost:8000/detect"

    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        data = {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
            "include_sam": True,
            "label_sam_segments": True,
        }

        response = requests.post(url, files=files, data=data, timeout=300)
        response.raise_for_status()
        return response.json()


def generate_folium_map(
    satellite_image_path,
    geojson_data,
    output_path,
    image_name,
    center_lat,
    center_lon,
    tree_count=0.0,
):
    """Generate interactive Folium map with satellite imagery and detections."""
    from PIL import Image
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

    with Image.open(satellite_image_path) as img:
        img_width, img_height = img.size

    converter = ImageCoordinateConverter(
        center_lat=center_lat,
        center_lon=center_lon,
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=20,
    )

    image_bounds_dict = converter.get_image_bounds()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=20,
        min_zoom=15,
        max_zoom=22,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Add satellite image overlay
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
    sam_group = folium.FeatureGroup(name="🔍 SAM Segments", show=False)

    # Count features
    vehicle_count = 0
    pool_count = 0
    amenity_count = 0
    tree_count_features = 0
    sam_count = 0

    # Add features from GeoJSON
    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [[c[1], c[0]] for c in coords]

        feature_type = feature["properties"].get("feature_type", "vehicle")

        if feature_type == "labeled_sam_segment":
            sam_count += 1
            segment_id = feature["properties"]["segment_id"]
            primary_label = feature["properties"]["primary_label"]
            label_confidence = feature["properties"].get("label_confidence", 0.0)
            area_sqm = feature["properties"]["area_sqm"]

            label_colors = {
                "vehicle": ("#9933FF", "#6600CC", "🚗"),
                "driveway": ("#808080", "#606060", "🛣️"),
                "building": ("#8B4513", "#654321", "🏠"),
                "pool": ("#0099FF", "#0066CC", "🏊"),
                "tree": ("#228B22", "#1B6B1B", "🌳"),
                "grass": ("#7CFC00", "#5CB300", "🌿"),
                "pavement": ("#696969", "#4A4A4A", "⬛"),
                "unknown": ("#CCCCCC", "#999999", "❓"),
            }

            fill_color, line_color, icon = label_colors.get(
                primary_label, label_colors["unknown"]
            )

            popup_html = f"""
            <b>SAM Segment #{segment_id}</b><br>
            Label: {primary_label.title()}<br>
            Confidence: {label_confidence:.1%}<br>
            Area: {area_sqm:.1f} m²
            """

            tooltip_text = (
                f"{icon} {primary_label.title()}"
                if primary_label != "unknown"
                else f"SAM #{segment_id}"
            )

            folium.Polygon(
                locations=coords_swapped,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=tooltip_text,
                fillColor=fill_color,
                color=line_color,
                weight=2,
                fillOpacity=0.4,
                opacity=0.8,
            ).add_to(sam_group)

        elif feature_type == "vehicle":
            vehicle_count += 1
            confidence = feature["properties"]["confidence"]
            vehicle_class = feature["properties"]["vehicle_class"]

            folium.Polygon(
                locations=coords_swapped,
                popup=f"<b>Vehicle</b><br>Class: {vehicle_class}<br>Confidence: {confidence:.1%}",
                tooltip=f"{vehicle_class} ({confidence:.1%})",
                fillColor="#9933FF",
                color="#6600CC",
                weight=2,
                fillOpacity=0.4,
                opacity=0.8,
            ).add_to(vehicle_group)

        elif feature_type == "swimming_pool":
            pool_count += 1
            confidence = feature["properties"]["confidence"]
            area_sqm = feature["properties"]["area_sqm"]

            folium.Polygon(
                locations=coords_swapped,
                popup=f"<b>Pool</b><br>Area: {area_sqm:.1f} m²<br>Confidence: {confidence:.1%}",
                tooltip=f"Pool ({confidence:.1%})",
                fillColor="#0099FF",
                color="#0066CC",
                weight=2,
                fillOpacity=0.4,
                opacity=0.8,
            ).add_to(pool_group)

        elif feature_type == "amenity":
            amenity_count += 1
            amenity_type = feature["properties"]["amenity_type"]
            confidence = feature["properties"]["confidence"]

            folium.Polygon(
                locations=coords_swapped,
                popup=f"<b>{amenity_type.title()}</b><br>Confidence: {confidence:.1%}",
                tooltip=f"{amenity_type.title()}",
                fillColor="#FF6B00",
                color="#CC5500",
                weight=2,
                fillOpacity=0.4,
                opacity=0.8,
            ).add_to(amenity_group)

        elif feature_type in ["tree", "tree_cluster"]:
            tree_count_features += 1

            folium.Polygon(
                locations=coords_swapped,
                popup="<b>Tree</b>",
                tooltip="Tree",
                fillColor="#228B22",
                color="#1B6B1B",
                weight=2,
                fillOpacity=0.4,
                opacity=0.8,
            ).add_to(tree_group)

    # Add groups to map
    vehicle_group.add_to(m)
    pool_group.add_to(m)
    amenity_group.add_to(m)
    tree_group.add_to(m)
    sam_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Add title
    title_html = f"""
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 900px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <h4>{image_name}</h4>
        <b>Vehicles:</b> {vehicle_count} &nbsp;&nbsp; <b>Pools:</b> {pool_count}
        &nbsp;&nbsp; <b>Amenities:</b> {amenity_count} &nbsp;&nbsp;
        <b>Trees:</b> {tree_count_features} &nbsp;&nbsp;
        <b>SAM Segments:</b> {sam_count}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    plugins.Fullscreen().add_to(m)

    m.save(str(output_path))


def generate_examples(num_examples=3):
    """Generate detection examples using REST API."""
    print("=" * 80)
    print(f"Generating {num_examples} Detection Examples (via REST API)")
    print("=" * 80)

    # Check Docker is running
    if not ensure_docker_running():
        return

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent / "det-state-visualizer"
    satellite_dir = base_dir / "data/raw/satellite_images"
    quotes_csv = base_dir / "data/raw/quote_addresses.csv"
    output_dir = Path(__file__).parent.parent / "output/examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSource directory: {satellite_dir}")
    print(f"Quotes CSV: {quotes_csv}")
    print(f"Output directory: {output_dir}")

    # Load quote coordinates
    coords_map = {}
    if quotes_csv.exists():
        print("\nLoading quote coordinates...")
        coords_map = load_quote_coordinates(quotes_csv)
        print(f"✓ Loaded {len(coords_map)} quote coordinates")

    # Get list of satellite images
    satellite_images = list(satellite_dir.glob("*.jpg"))
    print(f"\n✓ Found {len(satellite_images)} satellite images")

    # Process images
    processed = 0
    skipped = 0
    results = []

    print(f"\nProcessing {num_examples} images via REST API...")
    print("-" * 80)

    for img_path in satellite_images:
        if processed >= num_examples:
            break

        img_name = img_path.name
        print(f"\n[{processed + 1}/{num_examples}] Processing: {img_name}")

        try:
            # Parse address
            address_info = parse_address_from_filename(img_name)
            if address_info is None:
                print("  ⚠ Skipping (unable to parse filename)")
                skipped += 1
                continue

            # Get coordinates
            parts = address_info["parts"]
            state_code = None
            for part in reversed(parts):
                if len(part) == 2 and part.isalpha():
                    state_code = part
                    break

            if state_code:
                lat, lon = get_default_coordinates_by_state(state_code)
            else:
                lat, lon = 37.0, -122.0

            print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

            # Call REST API
            print("  Calling REST API at http://localhost:8000/api/v1/detect...")
            api_result = detect_via_api(img_path, lat, lon, zoom_level=20)

            # Extract GeoJSON
            geojson_data = api_result.get(
                "geojson", {"type": "FeatureCollection", "features": []}
            )

            # Count detections
            vehicles = sum(
                1
                for f in geojson_data["features"]
                if f["properties"].get("feature_type") == "vehicle"
            )
            pools = sum(
                1
                for f in geojson_data["features"]
                if f["properties"].get("feature_type") == "swimming_pool"
            )
            amenities = sum(
                1
                for f in geojson_data["features"]
                if f["properties"].get("feature_type") == "amenity"
            )
            sam_segments = sum(
                1
                for f in geojson_data["features"]
                if f["properties"].get("feature_type") == "labeled_sam_segment"
            )

            print(
                f"  ✓ Detected: {vehicles} vehicles, {pools} pools, {amenities} amenities, {sam_segments} SAM segments"
            )

            # Save GeoJSON
            geojson_dir = output_dir / "geojson"
            geojson_dir.mkdir(exist_ok=True)

            output_filename = img_path.stem + "_detections.geojson"
            output_path = geojson_dir / output_filename

            with open(output_path, "w") as f:
                json.dump(geojson_data, f, indent=2)

            print(f"  ✓ Saved to: geojson/{output_filename}")

            # Generate folium map
            folium_dir = output_dir / "folium_maps"
            folium_dir.mkdir(exist_ok=True)
            folium_path = folium_dir / f"{img_path.stem}.html"

            print("  Generating folium map...")
            generate_folium_map(
                satellite_image_path=img_path,
                geojson_data=geojson_data,
                output_path=folium_path,
                image_name=img_name,
                center_lat=lat,
                center_lon=lon,
                tree_count=0.0,
            )
            print(f"  ✓ Folium map saved to: folium_maps/{img_path.stem}.html")

            results.append(
                {
                    "image": img_name,
                    "vehicles_detected": vehicles,
                    "pools_detected": pools,
                    "amenities_detected": amenities,
                    "sam_segments": sam_segments,
                    "output_file": output_filename,
                    "coordinates": {"lat": lat, "lon": lon},
                }
            )

            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            skipped += 1
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    total_vehicles = sum(r["vehicles_detected"] for r in results)
    total_pools = sum(r["pools_detected"] for r in results)
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Total swimming pools detected: {total_pools}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_processed": processed,
                "total_skipped": skipped,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Summary saved to: {summary_path}")

    # Copy satellite images
    print("\nCopying satellite images...")
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for result in results:
        src_img = satellite_dir / result["image"]
        dst_img = images_dir / result["image"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

    print(f"✓ Copied {len(results)} images to: {images_dir}")
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"\n📂 Open {output_dir}/folium_maps/ to view individual interactive maps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate detection examples from satellite images using REST API"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to generate (default: 3)",
    )
    args = parser.parse_args()
    generate_examples(num_examples=args.num_examples)
