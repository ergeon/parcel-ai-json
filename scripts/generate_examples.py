"""
Generate 20 example vehicle detection outputs from satellite images.

This script processes satellite images from the det-state-visualizer project
and generates GeoJSON files with vehicle detections.
"""

import json
import csv
import shutil
import base64
from pathlib import Path
import folium
from folium import plugins

from parcel_ai_json.device_utils import get_best_device


def load_quote_coordinates(csv_path):
    """Load quote coordinates from CSV."""
    coords_by_address = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize address for matching
            address = row.get("formatted_address", "")
            if address:
                coords_by_address[address.lower()] = {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "quote_id": row.get("quote_id", ""),
                }
    return coords_by_address


def parse_address_from_filename(filename):
    """Parse address components from satellite image filename.

    Format: "1_austin_ave_norwood_nj_07648.jpg"
    """
    # Remove extension
    name = Path(filename).stem

    # Split by underscore
    parts = name.split("_")

    if len(parts) < 3:
        return None

    # Try to extract components
    # Format is typically: number_street_city_state_zip
    return {
        "filename": filename,
        "parts": parts,
        "raw": name,
    }


def get_default_coordinates_by_state(state_code):
    """Get default coordinates for a state (center of state)."""
    state_centers = {
        "ca": (36.7783, -119.4179),  # California
        "nj": (40.0583, -74.4057),  # New Jersey
        "ny": (42.1657, -74.9481),  # New York
        "tx": (31.9686, -99.9018),  # Texas
        "fl": (27.7663, -81.6868),  # Florida
        "pa": (40.5908, -77.2098),  # Pennsylvania
        "va": (37.4316, -78.6569),  # Virginia
        "md": (39.0458, -76.6413),  # Maryland
        "sc": (33.8569, -80.9450),  # South Carolina
        "il": (40.6331, -89.3985),  # Illinois
    }
    return state_centers.get(state_code.lower(), (37.0, -122.0))  # Default to CA


def generate_folium_map(
    satellite_image_path,
    geojson_data,
    output_path,
    image_name,
    center_lat,
    center_lon,
    tree_count=0.0,
    sam_segments=None,
):
    """Generate interactive Folium map with satellite imagery and vehicle detections.

    Args:
        satellite_image_path: Path to satellite image
        geojson_data: GeoJSON data with detections
        output_path: Output path for HTML file
        image_name: Name of the image
        center_lat: Center latitude
        center_lon: Center longitude
        tree_count: Tree coverage percentage
        sam_segments: List of SAM segment objects (optional)
    """
    if sam_segments is None:
        sam_segments = []
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

        img_format = "jpeg" if satellite_image_path.suffix.lower() in [".jpg", ".jpeg"] else "png"
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
            opacity=0.7,  # Reduced to 70% so base map shows through
            interactive=False,
            cross_origin=False,
            zindex=1,
            name="Satellite Imagery",
        ).add_to(m)

    # Add tree mask overlay if it exists
    tree_mask_path = satellite_image_path.parent / f"{satellite_image_path.stem}_tree_mask.png"
    if tree_mask_path.exists():
        with open(tree_mask_path, "rb") as f:
            tree_mask_data = f.read()
        tree_mask_base64 = base64.b64encode(tree_mask_data).decode()
        tree_mask_url = f"data:image/png;base64,{tree_mask_base64}"

        folium.raster_layers.ImageOverlay(
            image=tree_mask_url,
            bounds=bounds,
            opacity=0.7,  # Semi-transparent green overlay
            interactive=False,
            cross_origin=False,
            zindex=2,  # Above satellite image
            name="🌳 Tree Coverage",
        ).add_to(m)

    # Create feature groups for layer control
    vehicle_group = folium.FeatureGroup(name="🚗 Vehicles", show=True)
    pool_group = folium.FeatureGroup(name="🏊 Swimming Pools", show=True)
    amenity_group = folium.FeatureGroup(name="🎾 Amenities", show=True)
    tree_group = folium.FeatureGroup(name="🌳 Trees (DeepForest)", show=True)
    sam_group = folium.FeatureGroup(name=f"🔍 SAM Segments ({len(sam_segments)})", show=False)

    # Add vehicle, swimming pool, amenity, and tree detections
    vehicle_count = 0
    pool_count = 0
    amenity_count = 0
    tree_cluster_count = 0

    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"][0]
        coords_swapped = [[c[1], c[0]] for c in coords]  # Swap to [lat, lon]

        feature_type = feature["properties"].get("feature_type", "vehicle")

        if feature_type == "tree_cluster":
            tree_cluster_count += 1
            area_sqm = feature["properties"]["area_sqm"]
            area_pixels = feature["properties"]["area_pixels"]
            popup_html = f"""
            <b>Tree Cluster</b><br>
            Area: {area_sqm:.1f} m²<br>
            Pixels: {area_pixels}
            """
            tooltip_text = f"Tree Cluster ({area_sqm:.1f} m²)"
            label_text = f"{area_sqm:.0f}m²"
            fill_color = "#228B22"  # Forest green for trees
            line_color = "#1B6B1B"
            feature_group = tree_group
        elif feature_type == "tree":
            # DeepForest tree bounding box
            tree_cluster_count += 1
            confidence = feature["properties"]["confidence"]
            popup_html = f"""
            <b>Tree Detection (DeepForest)</b><br>
            Confidence: {confidence:.1%}
            """
            tooltip_text = f"Tree ({confidence:.1%})"
            label_text = "🌳"
            fill_color = "#228B22"  # Forest green for trees
            line_color = "#1B6B1B"
            feature_group = tree_group
        elif feature_type == "swimming_pool":
            confidence = feature["properties"]["confidence"]
            pixel_bbox = feature["properties"]["pixel_bbox"]
            pool_count += 1
            area_sqm = feature["properties"]["area_sqm"]
            popup_html = f"""
            <b>Swimming Pool Detection</b><br>
            Confidence: {confidence:.1%}<br>
            Area: {area_sqm:.1f} m²<br>
            Pixel BBox: [{pixel_bbox[0]:.0f}, {pixel_bbox[1]:.0f}, {pixel_bbox[2]:.0f}, {pixel_bbox[3]:.0f}]
            """
            tooltip_text = f"Swimming Pool ({confidence:.1%})"
            label_text = "Pool"
            fill_color = "#0099FF"  # Blue for pools
            line_color = "#0066CC"
            feature_group = pool_group
        elif feature_type == "amenity":
            amenity_count += 1
            amenity_type = feature["properties"]["amenity_type"]
            area_sqm = feature["properties"]["area_sqm"]

            # Icons for different amenity types
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
            fill_color = "#FF6B00"  # Orange for amenities
            line_color = "#CC5500"
            feature_group = amenity_group
        else:
            confidence = feature["properties"]["confidence"]
            pixel_bbox = feature["properties"]["pixel_bbox"]
            vehicle_count += 1
            vehicle_class = feature["properties"]["vehicle_class"]
            popup_html = f"""
            <b>Vehicle Detection</b><br>
            Class: {vehicle_class}<br>
            Confidence: {confidence:.1%}<br>
            Pixel BBox: [{pixel_bbox[0]:.0f}, {pixel_bbox[1]:.0f}, {pixel_bbox[2]:.0f}, {pixel_bbox[3]:.0f}]
            """
            tooltip_text = f"{vehicle_class} ({confidence:.1%})"
            # Capitalize first letter of each word for label
            label_text = vehicle_class.title()
            fill_color = "#9933FF"  # Purple for vehicles
            line_color = "#6600CC"
            feature_group = vehicle_group

        # Add polygon with label
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

        # Add label marker at center of bounding box
        center_lat = sum(c[0] for c in coords_swapped) / len(coords_swapped)
        center_lon = sum(c[1] for c in coords_swapped) / len(coords_swapped)

        folium.Marker(
            location=[center_lat, center_lon],
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

    # Add SAM segments with semi-transparent fill
    for segment in sam_segments:
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

    # Add all feature groups to map
    vehicle_group.add_to(m)
    pool_group.add_to(m)
    amenity_group.add_to(m)
    tree_group.add_to(m)
    sam_group.add_to(m)

    # Add layer control with checkboxes
    folium.LayerControl(collapsed=False).add_to(m)

    # Add title
    title_html = f"""
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 900px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <h4>{image_name}</h4>
        <b>Vehicles:</b> {vehicle_count} &nbsp;&nbsp; <b>Pools:</b> {pool_count} &nbsp;&nbsp; <b>Amenities:</b> {amenity_count} &nbsp;&nbsp; <b>Tree Clusters:</b> {tree_cluster_count} &nbsp;&nbsp; <b>Tree Coverage:</b> {tree_count:.1f}% &nbsp;&nbsp; <b>SAM Segments:</b> {len(sam_segments)}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Add measure control
    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)

    # Add fullscreen option
    plugins.Fullscreen().add_to(m)

    # Save
    m.save(str(output_path))


def generate_html_visualization(results, output_dir):
    """Generate HTML portfolio showing satellite images with detections."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Detection Examples</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .example {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .example:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .image-container {
            position: relative;
            width: 100%;
            aspect-ratio: 1;
            overflow: hidden;
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .image-container canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        .info {
            padding: 15px;
        }
        .filename {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .count {
            color: #666;
            font-size: 14px;
        }
        .vehicles {
            color: #4CAF50;
            font-weight: bold;
        }
        .no-vehicles {
            color: #999;
        }
    </style>
</head>
<body>
    <h1>🚗🏊 AI Detection Examples</h1>
    <div class="stats">
        <p><strong>Model:</strong> YOLOv8m-OBB (Oriented Bounding Boxes for Aerial Imagery)</p>
        <p><strong>Total Processed:</strong> {total_processed} images</p>
        <p><strong>Total Vehicles:</strong> {total_vehicles} | <strong>Total Pools:</strong> {total_pools}</p>
    </div>
    <div class="grid">
"""

    for result in results:
        vehicle_count = result["vehicles_detected"]
        pool_count = result["pools_detected"]
        has_detections = vehicle_count > 0 or pool_count > 0
        count_class = "vehicles" if has_detections else "no-vehicles"

        detection_parts = []
        if vehicle_count > 0:
            detection_parts.append(f"{vehicle_count} vehicle{'s' if vehicle_count != 1 else ''}")
        if pool_count > 0:
            detection_parts.append(f"{pool_count} pool{'s' if pool_count != 1 else ''}")

        count_text = ", ".join(detection_parts) if detection_parts else "No detections"

        html_content += f"""
        <div class="example">
            <div class="image-container">
                <img src="images/{result['image']}" alt="{result['image']}" id="img-{result['image']}">
                <canvas id="canvas-{result['image']}"></canvas>
            </div>
            <div class="info">
                <div class="filename">{result['image']}</div>
                <div class="count <{count_class}>">{count_text} detected</div>
            </div>
        </div>
"""

    html_content += """
    </div>
    <script>
        // Load and draw bounding boxes for each image
        async function loadDetections() {
"""

    for result in results:
        geojson_file = result["output_file"]
        html_content += f"""
            fetch('geojson/{geojson_file}')
                .then(r => r.json())
                .then(data => drawDetections('{result['image']}', data));
"""

    html_content += """
        }

        function drawDetections(imageName, geojson) {
            const img = document.getElementById(`img-${imageName}`);
            const canvas = document.getElementById(`canvas-${imageName}`);

            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                geojson.features.forEach(feature => {
                    const bbox = feature.properties.pixel_bbox;
                    const confidence = feature.properties.confidence;
                    const vehicleClass = feature.properties.vehicle_class;

                    // Draw bounding box
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

                    // Draw label
                    const label = `${vehicleClass} ${(confidence * 100).toFixed(0)}%`;
                    ctx.fillStyle = '#00ff00';
                    ctx.font = '12px monospace';
                    ctx.fillRect(bbox[0], bbox[1] - 16, ctx.measureText(label).width + 8, 16);
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, bbox[0] + 4, bbox[1] - 4);
                });
            };

            if (img.complete) {
                img.onload();
            }
        }

        loadDetections();
    </script>
</body>
</html>
"""

    return html_content


def generate_examples(num_examples=20):
    """Generate vehicle detection examples."""
    print("=" * 80)
    print(f"Generating {num_examples} Vehicle Detection Examples")
    print("=" * 80)

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

    # Initialize detectors with aerial imagery model (auto-downloads yolov8m-obb.pt)
    print("\nInitializing detectors...")
    print("  Model: yolov8m-obb.pt (oriented bounding boxes)")
    print("  Models will be downloaded to ~/.ultralytics/ on first use")

    from parcel_ai_json import PropertyDetectionService

    # Create property detector with combined tree detection (DeepForest + detectree)
    detector = PropertyDetectionService(
        vehicle_confidence=0.25,
        pool_confidence=0.3,
        amenity_confidence=0.3,
        tree_confidence=0.1,  # Low threshold to detect more trees with DeepForest
        tree_model_name="weecology/deepforest-tree",
        # Enable detectree polygon extraction with clustering and simplification
        detectree_extract_polygons=True,
        detectree_min_tree_area_pixels=50,  # Filter small noise
        detectree_simplify_tolerance_meters=0.5,  # Topology-preserving simplification
        detectree_use_docker=True,  # Docker mode for macOS compatibility (native crashes)
    )

    print("✓ Detector initialized (unified property detection with tree polygons)")

    # Get list of available satellite images
    satellite_images = list(satellite_dir.glob("*.jpg"))
    print(f"\n✓ Found {len(satellite_images)} satellite images")

    # Process first N images
    processed = 0
    skipped = 0
    results = []

    print(f"\nProcessing {num_examples} images...")
    print("-" * 80)

    for img_path in satellite_images:
        if processed >= num_examples:
            break

        img_name = img_path.name
        print(f"\n[{processed + 1}/{num_examples}] Processing: {img_name}")

        try:
            # Parse address from filename
            address_info = parse_address_from_filename(img_name)

            # Skip if address parsing failed
            if address_info is None:
                print("  ⚠ Skipping (unable to parse filename)")
                skipped += 1
                continue

            # Try to extract state code (usually second-to-last or last part)
            parts = address_info["parts"]
            state_code = None
            for part in reversed(parts):
                if len(part) == 2 and part.isalpha():
                    state_code = part
                    break

            # Get coordinates (default to state center)
            if state_code:
                lat, lon = get_default_coordinates_by_state(state_code)
            else:
                lat, lon = 37.0, -122.0  # Default

            print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

            satellite_image = {
                "path": str(img_path),
                "center_lat": lat,
                "center_lon": lon,
                "zoom_level": 20,
            }

            # Detect all property features
            detections = detector.detect_all(satellite_image)

            # Run SAM segmentation (ViT-H for highest accuracy)
            from parcel_ai_json.sam_segmentation import SAMSegmentationService

            # Auto-detect best device (cuda/mps/cpu)
            device = get_best_device()
            print(f"  Running SAM segmentation (device: {device})...")
            sam_service = SAMSegmentationService(
                model_type="vit_h",
                device=device,
                points_per_side=16,  # Faster inference
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )
            sam_segments = sam_service.segment_image(satellite_image)
            print(f"  ✓ SAM: {len(sam_segments)} segments detected")

            # Generate combined GeoJSON
            combined_geojson = detections.to_geojson()

            # Get summary
            summary = detections.summary()
            amenity_summary = (
                ", ".join([f"{count} {atype}" for atype, count in summary["amenities"].items()])
                if summary["amenities"]
                else "none"
            )

            print(
                f"  ✓ Detected: {summary['vehicles']} vehicles, "
                f"{summary['swimming_pools']} pools, amenities: {amenity_summary}, "
                f"tree coverage: {summary['tree_count']:.1f}%"
            )

            # Save to geojson subdirectory
            geojson_dir = output_dir / "geojson"
            geojson_dir.mkdir(exist_ok=True)

            output_filename = img_path.stem + "_detections.geojson"
            output_path = geojson_dir / output_filename

            with open(output_path, "w") as f:
                json.dump(combined_geojson, f, indent=2)

            print(f"  ✓ Saved to: geojson/{output_filename}")

            # Generate folium map immediately
            folium_dir = output_dir / "folium_maps"
            folium_dir.mkdir(exist_ok=True)
            folium_path = folium_dir / f"{img_path.stem}.html"

            print(f"  Generating folium map with {len(sam_segments)} SAM segments...")
            generate_folium_map(
                satellite_image_path=img_path,
                geojson_data=combined_geojson,
                output_path=folium_path,
                image_name=img_name,
                center_lat=lat,
                center_lon=lon,
                tree_count=summary["tree_count"],
                sam_segments=sam_segments,
            )
            print(f"  ✓ Folium map saved to: folium_maps/{img_path.stem}.html")

            results.append(
                {
                    "image": img_name,
                    "vehicles_detected": summary["vehicles"],
                    "pools_detected": summary["swimming_pools"],
                    "amenities_detected": summary["total_amenities"],
                    "tree_count": summary["tree_count"],
                    "output_file": output_filename,
                    "coordinates": {"lat": lat, "lon": lon},
                    "geojson": combined_geojson,
                    "sam_segments": sam_segments,
                }
            )

            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            skipped += 1
            continue

    # Save summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    total_vehicles = sum(r["vehicles_detected"] for r in results)
    total_pools = sum(r["pools_detected"] for r in results)
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Total swimming pools detected: {total_pools}")

    # Save results summary (exclude sam_segments from JSON as they're not serializable)
    summary_path = output_dir / "summary.json"
    results_for_json = [
        {k: v for k, v in r.items() if k != "sam_segments"} for r in results
    ]
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_processed": processed,
                "total_skipped": skipped,
                "results": results_for_json,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Summary saved to: {summary_path}")

    # Copy satellite images to output/images directory
    print("\nCopying satellite images...")
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for result in results:
        src_img = satellite_dir / result["image"]
        dst_img = images_dir / result["image"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

    print(f"✓ Copied {len(results)} images to: {images_dir}")

    # Generate HTML visualization
    print("\nGenerating HTML visualization...")
    html_content = generate_html_visualization(results, output_dir)
    html_content = (
        html_content.replace("{total_processed}", str(processed))
        .replace("{total_vehicles}", str(total_vehicles))
        .replace("{total_pools}", str(total_pools))
    )

    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"✓ HTML visualization saved to: {html_path}")

    # Folium maps already generated during processing
    folium_dir = output_dir / "folium_maps"
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"\n📂 Open {html_path} in your browser to view portfolio")
    print(f"📂 Open files in {folium_dir}/ to view individual interactive maps")


if __name__ == "__main__":
    generate_examples(num_examples=20)
