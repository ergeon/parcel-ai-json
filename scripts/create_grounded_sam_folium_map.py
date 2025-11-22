"""Create interactive Folium map from Grounded-SAM detection GeoJSON.

This script generates an enhanced Folium map showing Grounded-SAM detections
with color-coded labels and confidence scores.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import folium  # noqa: E402
from PIL import Image  # noqa: E402
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402


# Color scheme for different property features
LABEL_COLORS = {
    "driveway": "#808080",  # Gray
    "patio": "#D2B48C",  # Tan
    "deck": "#8B4513",  # Brown
    "shed": "#A0522D",  # Sienna
    "gazebo": "#CD853F",  # Peru
    "pergola": "#DEB887",  # BurlyWood
    "hot tub": "#4682B4",  # SteelBlue
    "playground equipment": "#FF6347",  # Tomato
    "dog house": "#D2691E",  # Chocolate
    "fire pit": "#FF4500",  # OrangeRed
    "pool house": "#4169E1",  # RoyalBlue
    "house": "#FF8C00",  # DarkOrange
    "pool": "#1E90FF",  # DodgerBlue
    "unknown": "#696969",  # DimGray
}


def get_label_color(label: str) -> str:
    """Get color for a label (fuzzy matching)."""
    label_lower = label.lower()

    # Exact match
    if label_lower in LABEL_COLORS:
        return LABEL_COLORS[label_lower]

    # Fuzzy match - check if any key is in the label
    for key, color in LABEL_COLORS.items():
        if key in label_lower:
            return color

    return LABEL_COLORS["unknown"]


def create_grounded_sam_folium_map(
    geojson_path: str,
    image_path: str,
    output_path: str,
):
    """Create enhanced Folium map from Grounded-SAM GeoJSON.

    Args:
        geojson_path: Path to Grounded-SAM detection GeoJSON file
        image_path: Path to satellite image
        output_path: Where to save the HTML map
    """
    geojson_path = Path(geojson_path)
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Creating Grounded-SAM Folium map")
    print("=" * 80)
    print(f"GeoJSON: {geojson_path.name}")
    print(f"Image: {image_path.name}")

    # Load GeoJSON
    print("\n1. Loading Grounded-SAM GeoJSON...")
    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    metadata = data.get("metadata", {})

    center_lat = metadata.get("center_lat")
    center_lon = metadata.get("center_lon")
    zoom_level = metadata.get("zoom_level", 20)
    prompts = metadata.get("prompts", [])

    print(f"   ✓ Loaded {len(features)} grounded detections")
    print(f"   ✓ Center: ({center_lat}, {center_lon})")
    print(f"   ✓ Prompts: {', '.join(prompts)}")

    # Get image dimensions and calculate bounds
    print("\n2. Calculating image bounds...")
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    converter = ImageCoordinateConverter(
        center_lat=center_lat,
        center_lon=center_lon,
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=zoom_level,
    )

    image_bounds_dict = converter.get_image_bounds()
    bounds = [
        [image_bounds_dict["south"], image_bounds_dict["west"]],
        [image_bounds_dict["north"], image_bounds_dict["east"]],
    ]
    print(f"   ✓ Bounds: {bounds}")

    # Create Folium map
    print("\n3. Creating Folium map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
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

    # Group features by label for layer control
    label_groups = {}

    print("\n4. Adding Grounded-SAM detections...")
    for feature in features:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"][0]

        label = props.get("label", "unknown")
        confidence = props.get("confidence", 0)
        area_pixels = props.get("area_pixels", 0)
        area_sqm = props.get("area_sqm")

        # Get color for this label
        color = get_label_color(label)

        # Create feature group for this label if not exists
        if label not in label_groups:
            count = sum(
                1 for f in features if f['properties'].get('label') == label
            )
            label_groups[label] = folium.FeatureGroup(
                name=f"{label.title()} ({count})",
                show=True
            )

        # Build popup HTML
        popup_html = f"<b>{label.upper()}</b><br>"
        popup_html += f"Confidence: {confidence:.2%}<br>"
        popup_html += f"Area: {area_pixels:,} pixels"
        if area_sqm:
            popup_html += f" ({area_sqm:.1f} m²)"

        # Add polygon
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color=color,
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.4,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{label.title()}: {confidence:.1%}",
        ).add_to(label_groups[label])

    # Add all feature groups to map
    for group in label_groups.values():
        group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-zoom to fit bounds
    m.fit_bounds(bounds)

    # Add legend with unique labels
    legend_items = []
    for label in sorted(label_groups.keys()):
        color = get_label_color(label)
        legend_items.append(
            f'<p style="margin:5px 0;">'
            f'<span style="color:{color};">●</span> {label.title()}</p>'
        )

    legend_html = f"""
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight: bold; text-align: center;">
        Grounded-SAM Detections
    </p>
    {''.join(legend_items)}
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
    <small>Grounded-SAM: {len(features)} detections</small><br>
    <small>Prompts: {', '.join(prompts[:5])}{"..." if len(prompts) > 5 else ""}</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\n✓ Grounded-SAM Folium map saved to: {output_path}")
    print("\nDetection Summary:")
    for label in sorted(label_groups.keys()):
        count = sum(1 for f in features if f['properties'].get('label') == label)
        print(f"  - {label.title()}: {count}")
    print(f"\nTotal: {len(features)} detections")
    print("\nOpen the HTML file in your browser to explore!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Folium map from Grounded-SAM detection GeoJSON"
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to Grounded-SAM detection GeoJSON file",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to satellite image",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output HTML file",
    )

    args = parser.parse_args()

    create_grounded_sam_folium_map(
        geojson_path=args.geojson,
        image_path=args.image,
        output_path=args.output,
    )
