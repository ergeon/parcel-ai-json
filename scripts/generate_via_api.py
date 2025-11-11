#!/usr/bin/env python3
"""
Generate example detections by calling Docker REST API.

Data stays outside Docker - we send images to http://localhost:8000/detect
"""

import json
import csv
import requests
import shutil
import folium
from pathlib import Path

# Config
API_URL = "http://localhost:8000/detect"
DATA_DIR = Path("/Users/Alex/Documents/GitHub/det-state-visualizer/data/raw")
SATELLITE_DIR = DATA_DIR / "satellite_images"
QUOTES_CSV = DATA_DIR / "quote_addresses.csv"
OUTPUT_DIR = Path("output/examples")
NUM_EXAMPLES = 5

# Colors
COLORS = {
    "vehicle": ("#9933FF", "#6600CC"),  # Purple
    "swimming_pool": ("#00BFFF", "#0080FF"),
    "tree": ("#00FF00", "#00AA00"),  # DeepForest
    "tree_cluster": ("#228B22", "#006400"),  # detectree
}


def load_coords():
    """Load coordinates from CSV."""
    coords = {}
    with open(QUOTES_CSV) as f:
        for row in csv.DictReader(f):
            coords[row["filename"]] = (float(row["latitude"]), float(row["longitude"]))
    return coords


def process_image(img_path, lat, lon):
    """Send image to Docker API."""
    with open(img_path, "rb") as f:
        response = requests.post(
            API_URL,
            files={"image": f},
            data={"center_lat": lat, "center_lon": lon, "zoom_level": 20},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def create_folium_map(geojson, lat, lon, output_path):
    """Create folium map."""
    m = folium.Map(location=[lat, lon], zoom_start=20)

    groups = {
        "vehicle": folium.FeatureGroup("Vehicles 🟣"),
        "swimming_pool": folium.FeatureGroup("Pools"),
        "tree": folium.FeatureGroup("Trees (DeepForest)"),
        "tree_cluster": folium.FeatureGroup("Tree Coverage (detectree)"),
    }

    for feature in geojson.get("features", []):
        ft = feature["properties"]["feature_type"]
        if ft not in COLORS:
            continue

        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [(lat, lon) for lon, lat in coords]
        fill, line = COLORS[ft]

        # Create popup
        props = feature["properties"]
        if ft == "vehicle":
            label = props.get("vehicle_class", "vehicle").title()
            popup = (
                f"<b>Vehicle</b><br>{label}<br>Conf: {props.get('confidence', 0):.1%}"
            )
        elif ft == "swimming_pool":
            popup = f"<b>Pool</b><br>{props.get('area_sqm', 0):.1f}m²"
            label = "Pool"
        elif ft == "tree":
            popup = f"<b>Tree</b><br>Conf: {props.get('confidence', 0):.1%}"
            label = "Tree"
        elif ft == "tree_cluster":
            area = props.get("area_sqm", 0)
            popup = f"<b>Tree Cluster</b><br>{area:.1f}m²"
            label = f"{area:.0f}m²"

        folium.Polygon(
            coords_swapped,
            popup=popup,
            tooltip=label,
            fillColor=fill,
            color=line,
            weight=2,
            fillOpacity=0.4,
            opacity=0.8,
        ).add_to(groups[ft])

    for group in groups.values():
        group.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(output_path))


def main():
    print("=" * 80)
    print("Generating Examples via Docker API")
    print("=" * 80)
    print(f"API: {API_URL}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Setup
    (OUTPUT_DIR / "geojson").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "folium_maps").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    coords = load_coords()
    images = list(SATELLITE_DIR.glob("*.jpg"))
    print(f"✓ {len(coords)} coordinates, {len(images)} images\n")

    results = []
    for i, img_path in enumerate(images):
        if len(results) >= NUM_EXAMPLES:
            break
        if img_path.name not in coords:
            continue

        lat, lon = coords[img_path.name]
        print(f"[{i+1}/{NUM_EXAMPLES}] {img_path.name}")
        print(f"  Coords: ({lat}, {lon})")

        try:
            # Call Docker API
            geojson = process_image(img_path, lat, lon)

            # Count features
            counts = {}
            for f in geojson["features"]:
                ft = f["properties"]["feature_type"]
                counts[ft] = counts.get(ft, 0) + 1

            print(
                f"  ✓ {counts.get('vehicle', 0)} vehicles, "
                f"{counts.get('tree', 0)} trees, "
                f"{counts.get('tree_cluster', 0)} clusters "
                f"({geojson.get('trees', {}).get('tree_coverage_percent', 0):.1f}%)"
            )

            # Save GeoJSON
            out_name = img_path.stem
            geojson_path = OUTPUT_DIR / "geojson" / f"{out_name}_detections.geojson"
            geojson_path.write_text(json.dumps(geojson, indent=2))

            # Create folium map
            folium_path = OUTPUT_DIR / "folium_maps" / f"{out_name}.html"
            create_folium_map(geojson, lat, lon, folium_path)
            print(f"  ✓ Saved: {folium_path.name}")

            # Copy image
            shutil.copy(img_path, OUTPUT_DIR / "images" / img_path.name)

            results.append(
                {
                    "image": img_path.name,
                    "vehicles": counts.get("vehicle", 0),
                    "trees": counts.get("tree", 0),
                    "tree_clusters": counts.get("tree_cluster", 0),
                }
            )

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save summary
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"processed": len(results), "results": results}, indent=2)
    )

    print("\n" + "=" * 80)
    print(f"✓ Processed {len(results)} images")
    print(f"✓ Total vehicles: {sum(r['vehicles'] for r in results)}")
    print(f"✓ Total trees: {sum(r['trees'] for r in results)}")
    print(f"✓ Total tree clusters: {sum(r['tree_clusters'] for r in results)}")
    print(f"\n📂 Open: {OUTPUT_DIR}/folium_maps/")


if __name__ == "__main__":
    main()
