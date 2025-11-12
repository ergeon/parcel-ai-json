"""Batch comparison of SAM ViT-B vs ViT-L on multiple addresses.

Generates comparison maps and a summary CSV report.
"""

import sys
from pathlib import Path
import time
import csv
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.sam_segmentation import SAMSegmentationService  # noqa: E402
import folium  # noqa: E402
from PIL import Image  # noqa: E402


def get_coordinates_from_filename(filename: str) -> tuple:
    """Extract approximate coordinates from filename."""
    # Format: address_city_state_zip.jpg
    parts = filename.lower().replace(".jpg", "").split("_")

    # State-based coordinates (rough centers)
    state_coords = {
        "ca": (36.7783, -119.4179),
        "nj": (40.0583, -74.4057),
        "tx": (31.9686, -99.9018),
        "sc": (33.8361, -81.1637),
    }

    # Special cases
    if "vacaville" in filename:
        return (38.3566, -121.9877)
    elif "hamilton" in filename:
        return (40.2298, -74.7441)
    elif "union" in filename and "sc" in parts:
        return (34.7157, -81.6248)

    # Default by state
    state = parts[-2] if len(parts) > 2 else "ca"
    return state_coords.get(state, (37.0, -122.0))


def run_sam_on_image(image_path: Path, model_type: str) -> tuple:
    """Run SAM on a single image and return segments and time."""
    lat, lon = get_coordinates_from_filename(image_path.name)

    satellite_image = {
        "path": str(image_path),
        "center_lat": lat,
        "center_lon": lon,
        "zoom_level": 20,
    }

    sam_service = SAMSegmentationService(
        model_type=model_type,
        device="cpu",
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )

    start_time = time.time()
    segments = sam_service.segment_image(satellite_image)
    elapsed = time.time() - start_time

    return segments, elapsed, satellite_image


def batch_compare(images_dir: str, output_dir: str, max_images: int = None):
    """Run batch comparison on all images."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(images_dir.glob("*.jpg"))
    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to process")
    print("=" * 80)

    results: List[Dict] = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 80)

        # Run ViT-B
        print("  Running ViT-B...")
        segments_b, time_b, sat_img = run_sam_on_image(image_path, "vit_b")
        print(f"    ✓ {len(segments_b)} segments in {time_b:.1f}s")

        # Run ViT-L
        print("  Running ViT-L...")
        segments_l, time_l, _ = run_sam_on_image(image_path, "vit_l")
        print(f"    ✓ {len(segments_l)} segments in {time_l:.1f}s")

        # Calculate stats
        diff_segments = len(segments_l) - len(segments_b)
        diff_percent = (diff_segments / len(segments_b) * 100) if len(segments_b) > 0 else 0
        time_ratio = time_l / time_b if time_b > 0 else 0

        print(
            f"  Difference: {diff_segments:+d} segments ({diff_percent:+.1f}%), {time_ratio:.2f}x time"
        )

        # Store results
        results.append(
            {
                "image": image_path.name,
                "vit_b_segments": len(segments_b),
                "vit_b_time": time_b,
                "vit_l_segments": len(segments_l),
                "vit_l_time": time_l,
                "diff_segments": diff_segments,
                "diff_percent": diff_percent,
                "time_ratio": time_ratio,
            }
        )

        # Create comparison maps
        print("  Creating comparison maps...")
        create_maps(
            image_path,
            segments_b,
            segments_l,
            time_b,
            time_l,
            sat_img,
            output_dir,
        )

    # Save summary CSV
    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 80}")
    print("BATCH COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    # Calculate averages
    avg_b_segments = sum(r["vit_b_segments"] for r in results) / len(results)
    avg_l_segments = sum(r["vit_l_segments"] for r in results) / len(results)
    avg_b_time = sum(r["vit_b_time"] for r in results) / len(results)
    avg_l_time = sum(r["vit_l_time"] for r in results) / len(results)
    avg_diff = sum(r["diff_segments"] for r in results) / len(results)
    avg_time_ratio = sum(r["time_ratio"] for r in results) / len(results)

    print(f"Images processed: {len(results)}")
    print(f"\nAverage segments:")
    print(f"  ViT-B: {avg_b_segments:.1f}")
    print(f"  ViT-L: {avg_l_segments:.1f}")
    print(f"  Difference: {avg_diff:+.1f} ({(avg_diff/avg_b_segments*100):+.1f}%)")
    print(f"\nAverage inference time:")
    print(f"  ViT-B: {avg_b_time:.1f}s")
    print(f"  ViT-L: {avg_l_time:.1f}s")
    print(f"  Ratio: {avg_time_ratio:.2f}x")
    print(f"\n✓ Results saved to: {csv_path}")
    print(f"✓ Maps saved to: {output_dir}")


def create_maps(image_path, segments_b, segments_l, time_b, time_l, sat_img, output_dir):
    """Create comparison maps for both models."""
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    converter = ImageCoordinateConverter(
        center_lat=sat_img["center_lat"],
        center_lon=sat_img["center_lon"],
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=20,
    )

    image_bounds_dict = converter.get_image_bounds()
    bounds = [
        [image_bounds_dict["south"], image_bounds_dict["west"]],
        [image_bounds_dict["north"], image_bounds_dict["east"]],
    ]

    stem = image_path.stem

    # Create ViT-B map
    create_map(
        segments_b,
        "ViT-B",
        time_b,
        image_path,
        bounds,
        sat_img,
        output_dir / f"{stem}_vit_b.html",
    )

    # Create ViT-L map
    create_map(
        segments_l,
        "ViT-L",
        time_l,
        image_path,
        bounds,
        sat_img,
        output_dir / f"{stem}_vit_l.html",
    )


def create_map(segments, model_name, time_taken, image_path, bounds, sat_img, output_path):
    """Create a single folium map."""
    m = folium.Map(
        location=[sat_img["center_lat"], sat_img["center_lon"]],
        zoom_start=20,
        zoom_control=True,
        max_zoom=22,
        min_zoom=10,
    )

    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=bounds,
        opacity=0.8,
        name="Satellite Image",
    ).add_to(m)

    segments_group = folium.FeatureGroup(name=f"Segments ({len(segments)})", show=True)

    for seg in segments:
        if seg.area_pixels < 500:
            color = "#3388ff"
        elif seg.area_pixels < 2000:
            color = "#00ff00"
        else:
            color = "#ff0000"

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in seg.geo_polygon],
            color=color,
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=f"Segment #{seg.segment_id}<br>Area: {seg.area_pixels}px",
            tooltip=f"#{seg.segment_id}",
        ).add_to(segments_group)

    segments_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; width: auto;
                background-color: white; z-index:9999; font-size:16px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <b>SAM {model_name}</b><br>
    <small>{len(segments)} segments | {time_taken:.1f}s</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(str(output_path))


if __name__ == "__main__":
    images_dir = "output/examples/images"
    output_dir = "output/examples/folium_maps/batch_comparison"

    batch_compare(images_dir, output_dir)
