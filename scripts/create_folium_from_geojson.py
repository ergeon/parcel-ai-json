"""
Create interactive Folium map from existing detection GeoJSON.

This script takes a detection GeoJSON file and satellite image,
and generates an enhanced Folium map with layer controls.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import folium  # noqa: E402
from PIL import Image  # noqa: E402
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402
from parcel_ai_json.sam_labeler import LABEL_SCHEMA  # noqa: E402


def create_folium_map_from_geojson(
    geojson_path: str,
    image_path: str,
    output_path: str,
    center_lat: float,
    center_lon: float,
    zoom_level: int = 20,
):
    """Create enhanced Folium map from detection GeoJSON.

    Args:
        geojson_path: Path to detection GeoJSON file
        image_path: Path to satellite image
        output_path: Where to save the HTML map
        center_lat: Image center latitude
        center_lon: Image center longitude
        zoom_level: Zoom level (default: 20)
    """
    geojson_path = Path(geojson_path)
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Creating enhanced Folium map")
    print("=" * 80)
    print(f"GeoJSON: {geojson_path.name}")
    print(f"Image: {image_path.name}")
    print(f"Center: ({center_lat}, {center_lon})")
    print()

    # Load GeoJSON
    print("1. Loading detection GeoJSON...")
    with open(geojson_path) as f:
        detections = json.load(f)

    features = detections.get("features", [])
    print(f"   ✓ Loaded {len(features)} total features")

    # Separate features by type
    labeled_sam_segments = []
    vehicles = []
    pools = []
    amenities = []
    trees = []
    tree_clusters = []
    fences = []
    osm_buildings = []
    regrid_parcel = None

    for feature in features:
        props = feature.get("properties", {})
        feature_type = props.get("feature_type", "")

        if feature_type == "labeled_sam_segment":
            labeled_sam_segments.append(feature)
        elif feature_type == "vehicle":
            vehicles.append(feature)
        elif feature_type == "swimming_pool":
            pools.append(feature)
        elif feature_type == "amenity":
            amenities.append(feature)
        elif feature_type == "tree":
            trees.append(feature)
        elif feature_type == "tree_cluster":
            tree_clusters.append(feature)
        elif feature_type == "fence":
            fences.append(feature)
        elif feature_type == "osm_building":
            osm_buildings.append(feature)
        elif feature_type == "regrid_parcel":
            regrid_parcel = feature

    print(f"   ✓ {len(labeled_sam_segments)} labeled SAM segments")
    print(f"   ✓ {len(vehicles)} vehicles")
    print(f"   ✓ {len(pools)} swimming pools")
    print(f"   ✓ {len(amenities)} amenities")
    print(f"   ✓ {len(trees)} trees (DeepForest)")
    print(f"   ✓ {len(tree_clusters)} tree clusters (detectree)")
    print(f"   ✓ {len(fences)} fence segments")
    print(f"   ✓ {len(osm_buildings)} OSM buildings")
    print(f"   ✓ {'1' if regrid_parcel else '0'} Regrid parcel")

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

    # Create feature groups for layer control
    sam_group = folium.FeatureGroup(
        name=f"SAM Segments ({len(labeled_sam_segments)})", show=True
    )
    vehicles_group = folium.FeatureGroup(name=f"Vehicles ({len(vehicles)})", show=True)
    pools_group = folium.FeatureGroup(
        name=f"Swimming Pools ({len(pools)})", show=True
    )
    amenities_group = folium.FeatureGroup(
        name=f"Amenities ({len(amenities)})", show=True
    )
    trees_group = folium.FeatureGroup(name=f"Trees ({len(trees)})", show=True)
    tree_clusters_group = folium.FeatureGroup(
        name=f"Tree Clusters ({len(tree_clusters)})", show=True
    )
    fences_group = folium.FeatureGroup(name=f"Fences ({len(fences)})", show=True)
    osm_buildings_group = folium.FeatureGroup(
        name=f"OSM Buildings ({len(osm_buildings)})", show=True
    )
    regrid_parcel_group = folium.FeatureGroup(
        name="Regrid Parcel Boundary", show=True
    )

    # Add SAM segments with semantic labels
    print("4. Adding labeled SAM segments...")
    for segment in labeled_sam_segments:
        props = segment["properties"]
        coords = segment["geometry"]["coordinates"][0]

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
                f"Area: {props.get('area_sqm', 0):.1f} m²<br>"
                f"Stability: {props.get('stability_score', 0):.3f}<br>"
                f"Reason: {props.get('labeling_reason', 'N/A')}",
                max_width=300,
            ),
            tooltip=label_display,
        ).add_to(sam_group)

    # Add vehicles
    print("5. Adding vehicles...")
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
                f"Class: {props.get('vehicle_class', 'vehicle')}<br>"
                f"Confidence: {props.get('confidence', 0):.2f}",
                max_width=200,
            ),
            tooltip=f"Vehicle: {props.get('vehicle_class', 'vehicle')}",
        ).add_to(vehicles_group)

    # Add swimming pools
    print("6. Adding swimming pools...")
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
    print("7. Adding amenities...")
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
    print("8. Adding trees...")
    for i, tree in enumerate(trees):
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
                f"<b>Tree (DeepForest)</b><br>"
                f"Confidence: {props.get('confidence', 0):.2f}",
                max_width=200,
            ),
            tooltip=f"Tree #{i+1}",
        ).add_to(trees_group)

    # Add tree clusters (detectree)
    print("9. Adding tree clusters...")
    for i, tree_cluster in enumerate(tree_clusters):
        props = tree_cluster["properties"]
        coords = tree_cluster["geometry"]["coordinates"][0]

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
        ).add_to(tree_clusters_group)

    # Add fences
    print("10. Adding fences...")
    for i, fence in enumerate(fences):
        props = fence["properties"]
        coords = fence["geometry"]["coordinates"][0]

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#8B4513",  # Brown
            weight=2,
            fill=True,
            fillColor="#8B4513",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Fence Segment</b><br>"
                f"Segment ID: {props.get('segment_id', 0)}<br>"
                f"Max Probability: {props.get('max_probability', 0):.2f}<br>"
                f"Threshold: {props.get('threshold', 0.1)}",
                max_width=200,
            ),
            tooltip=f"Fence #{i+1}",
        ).add_to(fences_group)

    # Add OSM buildings
    print("11. Adding OSM buildings...")
    for i, osm_building in enumerate(osm_buildings):
        props = osm_building["properties"]
        coords = osm_building["geometry"]["coordinates"][0]

        building_type = props.get("building_type", "yes")
        area_sqm = props.get("area_sqm", 0)
        osm_id = props.get("osm_id", "unknown")

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#FFA500",  # Orange
            weight=2,
            fill=True,
            fillColor="#FFA500",
            fillOpacity=0.3,
            popup=folium.Popup(
                f"<b>OSM Building</b><br>"
                f"Type: {building_type}<br>"
                f"Area: {area_sqm:.1f} m²<br>"
                f"OSM ID: {osm_id}",
                max_width=200,
            ),
            tooltip=f"OSM: {building_type}",
        ).add_to(osm_buildings_group)

    # Add Regrid parcel boundary
    print("12. Adding Regrid parcel boundary...")
    if regrid_parcel:
        props = regrid_parcel["properties"]
        # Handle extra nesting level in coordinates
        coords = regrid_parcel["geometry"]["coordinates"]
        # Flatten if nested too deeply
        while (
            isinstance(coords[0][0], list)
            and len(coords[0][0]) > 0
            and isinstance(coords[0][0][0], list)
        ):
            coords = coords[0]
        coords = coords[0] if isinstance(coords[0], list) else coords

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="#FF00FF",  # Magenta
            weight=3,
            fill=False,
            popup=folium.Popup(
                "<b>Regrid Parcel Boundary</b><br>"
                "Source: Regrid API",
                max_width=200,
            ),
            tooltip="Parcel Boundary",
        ).add_to(regrid_parcel_group)

    # Add all groups to map
    sam_group.add_to(m)
    vehicles_group.add_to(m)
    pools_group.add_to(m)
    amenities_group.add_to(m)
    trees_group.add_to(m)
    tree_clusters_group.add_to(m)
    fences_group.add_to(m)
    osm_buildings_group.add_to(m)
    regrid_parcel_group.add_to(m)

    # Add layer control (not collapsed - shows all layers by default)
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
    <p style="margin:5px 0;">
        <span style="color:#228B22;">●</span> Trees (DeepForest)
    </p>
    <p style="margin:5px 0;">
        <span style="color:#006400;">●</span> Tree Clusters (detectree)
    </p>
    <p style="margin:5px 0;"><span style="color:#8B4513;">●</span> Fences</p>
    <p style="margin:5px 0;"><span style="color:#FFA500;">●</span> OSM Buildings</p>
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
    <small>SAM Segments + All Detections ({len(features)} features)</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\n✓ Enhanced Folium map saved to: {output_path}")
    print("\nFeature Summary:")
    print(f"  - SAM segments: {len(labeled_sam_segments)}")
    print(f"  - Vehicles: {len(vehicles)}")
    print(f"  - Swimming pools: {len(pools)}")
    print(f"  - Amenities: {len(amenities)}")
    print(f"  - Trees: {len(trees)}")
    print(f"  - Tree clusters: {len(tree_clusters)}")
    print(f"  - Fence segments: {len(fences)}")
    print(f"  - OSM buildings: {len(osm_buildings)}")
    print(f"  - Regrid parcel: {'1' if regrid_parcel else '0'}")
    print(f"\nTotal: {len(features)} features")
    print("\nOpen the HTML file in your browser to explore!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create enhanced Folium map from detection GeoJSON"
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to detection GeoJSON file",
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
    parser.add_argument(
        "--center-lat",
        type=float,
        required=True,
        help="Image center latitude",
    )
    parser.add_argument(
        "--center-lon",
        type=float,
        required=True,
        help="Image center longitude",
    )
    parser.add_argument(
        "--zoom-level",
        type=int,
        default=20,
        help="Zoom level (default: 20)",
    )

    args = parser.parse_args()

    create_folium_map_from_geojson(
        geojson_path=args.geojson,
        image_path=args.image,
        output_path=args.output,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        zoom_level=args.zoom_level,
    )
