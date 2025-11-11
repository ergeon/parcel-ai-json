"""Create interactive folium map with SAM segments and semantic detections.

This script generates an enhanced folium map showing:
- Satellite imagery base layer
- SAM segments (general-purpose segmentation)
- Vehicles (YOLO detections)
- Swimming pools
- Amenities (tennis courts, etc.)
- Trees (DeepForest + detectree)
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.sam_segmentation import SAMSegmentationService
from parcel_ai_json.property_detector import PropertyDetectionService
import folium
from folium import plugins


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
    }

    # Special case for Vacaville, CA (detected from city name)
    if "vacaville" in image_path.stem.lower():
        center_lat, center_lon = 38.3566, -121.9877
    else:
        center_lat, center_lon = state_centers.get(state_code.lower(), (37.0, -122.0))

    satellite_image = {
        "path": str(image_path),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom_level": 20,
    }

    print(f"Creating enhanced folium map for: {image_path.name}")
    print("=" * 80)

    # Run SAM segmentation
    print("1. Running SAM segmentation...")
    sam_service = SAMSegmentationService(
        model_type="vit_b",
        device="cpu",
        points_per_side=16,  # Faster inference
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    sam_segments = sam_service.segment_image(satellite_image)
    print(f"   ✓ Found {len(sam_segments)} SAM segments")

    # Run semantic detections
    print("2. Running semantic detections (vehicles, pools, amenities, trees)...")
    from parcel_ai_json.vehicle_detector import VehicleDetectionService
    from parcel_ai_json.swimming_pool_detector import SwimmingPoolDetectionService
    from parcel_ai_json.amenity_detector import AmenityDetectionService
    from parcel_ai_json.tree_detector import TreeDetectionService
    from parcel_ai_json.property_detector import PropertyDetections

    # Run individual detections
    vehicle_service = VehicleDetectionService(confidence_threshold=0.25)
    pool_service = SwimmingPoolDetectionService(confidence_threshold=0.3)
    amenity_service = AmenityDetectionService(confidence_threshold=0.3)
    tree_service = TreeDetectionService()

    vehicles = vehicle_service.detect_vehicles(satellite_image)
    pools = pool_service.detect_swimming_pools(satellite_image)
    amenities = amenity_service.detect_amenities(satellite_image)

    # Try tree detection, fall back to empty list if Docker not available
    try:
        trees = tree_service.detect_trees(satellite_image)
    except RuntimeError as e:
        print(f"   ⚠ Tree detection skipped (Docker required): {e}")
        trees = []

    # Create detections object
    detections = PropertyDetections(
        vehicles=vehicles,
        swimming_pools=pools,
        amenities=amenities,
        trees=trees,
    )
    print(f"   ✓ Found {len(detections.vehicles)} vehicles")
    print(f"   ✓ Found {len(detections.swimming_pools)} pools")
    print(f"   ✓ Found {len(detections.amenities)} amenities")
    print(f"   ✓ Found {len(detections.trees)} trees")

    # Get image dimensions
    from PIL import Image

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    center_lat = satellite_image["center_lat"]
    center_lon = satellite_image["center_lon"]

    # Use ImageCoordinateConverter for accurate bounds calculation
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

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
    print("3. Creating folium map...")
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
        name=f"Vehicles ({len(detections.vehicles)})", show=True
    )
    pools_group = folium.FeatureGroup(
        name=f"Swimming Pools ({len(detections.swimming_pools)})", show=True
    )
    amenities_group = folium.FeatureGroup(
        name=f"Amenities ({len(detections.amenities)})", show=True
    )
    trees_group = folium.FeatureGroup(
        name=f"Trees ({len(detections.trees)})", show=True
    )

    # Add SAM segments with semi-transparent fill
    print("4. Adding SAM segments to map...")
    for i, segment in enumerate(sam_segments):
        # Use different colors based on area
        if segment.area_pixels < 500:
            color = "#3388ff"  # Blue for small segments
        elif segment.area_pixels < 2000:
            color = "#00ff00"  # Green for medium
        else:
            color = "#ff0000"  # Red for large

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in segment.geo_polygon],
            color=color,
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=folium.Popup(
                f"<b>SAM Segment #{segment.segment_id}</b><br>"
                f"Area: {segment.area_pixels} pixels<br>"
                f"Stability: {segment.stability_score:.3f}<br>"
                f"IoU: {segment.predicted_iou:.3f}",
                max_width=250,
            ),
            tooltip=f"SAM #{segment.segment_id}",
        ).add_to(sam_group)

    # Add vehicles with purple markers
    print("5. Adding vehicles to map...")
    for i, vehicle in enumerate(detections.vehicles):
        # Get center of polygon
        lons, lats = zip(*vehicle.geo_polygon[:-1])  # Exclude closing point
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in vehicle.geo_polygon],
            color="#800080",  # Purple
            weight=2,
            fill=True,
            fillColor="#800080",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Vehicle</b><br>"
                f"Class: {vehicle.class_name}<br>"
                f"Confidence: {vehicle.confidence:.2f}",
                max_width=200,
            ),
            tooltip=f"Vehicle: {vehicle.class_name}",
        ).add_to(vehicles_group)

    # Add swimming pools with blue markers
    print("6. Adding swimming pools to map...")
    for pool in detections.swimming_pools:
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in pool.geo_polygon],
            color="#0066cc",  # Blue
            weight=2,
            fill=True,
            fillColor="#0066cc",
            fillOpacity=0.5,
            popup=folium.Popup(
                f"<b>Swimming Pool</b><br>"
                f"Confidence: {pool.confidence:.2f}<br>"
                f"Area: {pool.area_sqm:.1f} m²",
                max_width=200,
            ),
            tooltip="Swimming Pool",
        ).add_to(pools_group)

    # Add amenities with orange markers
    print("7. Adding amenities to map...")
    for amenity in detections.amenities:
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in amenity.geo_polygon],
            color="#ff8800",  # Orange
            weight=2,
            fill=True,
            fillColor="#ff8800",
            fillOpacity=0.4,
            popup=folium.Popup(
                f"<b>Amenity</b><br>"
                f"Type: {amenity.amenity_type}<br>"
                f"Confidence: {amenity.confidence:.2f}<br>"
                f"Area: {amenity.area_sqm:.1f} m²",
                max_width=200,
            ),
            tooltip=f"{amenity.amenity_type}",
        ).add_to(amenities_group)

    # Add trees with green markers
    print("8. Adding trees to map...")
    for tree in detections.trees:
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in tree.geo_polygon],
            color="#228B22",  # Forest green
            weight=2,
            fill=True,
            fillColor="#228B22",
            fillOpacity=0.5,
            popup=folium.Popup(
                f"<b>Tree</b><br>"
                f"Detection: {tree.detection_method}<br>"
                f"Confidence: {tree.confidence:.2f}<br>"
                f"Area: {tree.area_sqm:.1f} m²",
                max_width=200,
            ),
            tooltip="Tree",
        ).add_to(trees_group)

    # Add all groups to map
    sam_group.add_to(m)
    vehicles_group.add_to(m)
    pools_group.add_to(m)
    amenities_group.add_to(m)
    trees_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-zoom to fit bounds
    m.fit_bounds(bounds)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight: bold; text-align: center;">Legend</p>
    <p style="margin:5px 0;"><span style="color:#3388ff;">●</span> SAM Small (&lt;500px)</p>
    <p style="margin:5px 0;"><span style="color:#00ff00;">●</span> SAM Medium (500-2000px)</p>
    <p style="margin:5px 0;"><span style="color:#ff0000;">●</span> SAM Large (&gt;2000px)</p>
    <p style="margin:5px 0;"><span style="color:#800080;">●</span> Vehicles</p>
    <p style="margin:5px 0;"><span style="color:#0066cc;">●</span> Swimming Pools</p>
    <p style="margin:5px 0;"><span style="color:#ff8800;">●</span> Amenities</p>
    <p style="margin:5px 0;"><span style="color:#228B22;">●</span> Trees</p>
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
    print(f"\nSummary:")
    print(f"  - SAM segments: {len(sam_segments)}")
    print(f"  - Vehicles: {len(detections.vehicles)}")
    print(f"  - Swimming pools: {len(detections.swimming_pools)}")
    print(f"  - Amenities: {len(detections.amenities)}")
    print(f"  - Trees: {len(detections.trees)}")
    print(f"\nOpen the HTML file in your browser to explore!")


if __name__ == "__main__":
    # Use the same image we tested SAM on
    image_path = "output/examples/images/672_white_oak_ln_vacaville_ca_95687.jpg"
    output_path = (
        "output/examples/folium_maps/672_white_oak_ln_vacaville_ca_95687_with_sam.html"
    )

    create_enhanced_folium_map(image_path, output_path)
