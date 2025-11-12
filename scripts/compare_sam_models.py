"""Compare SAM ViT-B vs ViT-L models on the same image.

Generates two folium maps for side-by-side comparison.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.sam_segmentation import SAMSegmentationService  # noqa: E402
import folium  # noqa: E402


def run_sam_comparison(image_path: str, output_dir: str):
    """Run both SAM models and compare results."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prepare satellite image dict
    satellite_image = {
        "path": str(image_path),
        "center_lat": 38.3566,  # Vacaville, CA
        "center_lon": -121.9877,
        "zoom_level": 20,
    }

    print(f"Comparing SAM models on: {image_path.name}")
    print("=" * 80)

    # Run ViT-B (Base)
    print("\n1. Running SAM ViT-B (Base - 358MB)...")
    start_time = time.time()
    sam_b = SAMSegmentationService(
        model_type="vit_b",
        device="cpu",
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    segments_b = sam_b.segment_image(satellite_image)
    time_b = time.time() - start_time
    print(f"   ✓ ViT-B: {len(segments_b)} segments in {time_b:.1f}s")

    # Run ViT-L (Large)
    print("\n2. Running SAM ViT-L (Large - 1.2GB)...")
    start_time = time.time()
    sam_l = SAMSegmentationService(
        model_type="vit_l",
        device="cpu",
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    segments_l = sam_l.segment_image(satellite_image)
    time_l = time.time() - start_time
    print(f"   ✓ ViT-L: {len(segments_l)} segments in {time_l:.1f}s")

    print(f"\n{'=' * 80}")
    print("COMPARISON RESULTS:")
    print(f"{'=' * 80}")
    print(f"ViT-B (Base):  {len(segments_b):4d} segments in {time_b:6.1f}s")
    print(f"ViT-L (Large): {len(segments_l):4d} segments in {time_l:6.1f}s")
    print(f"Difference:    {len(segments_l) - len(segments_b):+4d} segments")
    print(f"Time ratio:    {time_l / time_b:.2f}x slower")
    print(f"{'=' * 80}\n")

    # Create comparison maps
    from PIL import Image
    from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    converter = ImageCoordinateConverter(
        center_lat=satellite_image["center_lat"],
        center_lon=satellite_image["center_lon"],
        image_width_px=img_width,
        image_height_px=img_height,
        zoom_level=20,
    )

    image_bounds_dict = converter.get_image_bounds()
    bounds = [
        [image_bounds_dict["south"], image_bounds_dict["west"]],
        [image_bounds_dict["north"], image_bounds_dict["east"]],
    ]

    # Create ViT-B map
    print("3. Creating ViT-B comparison map...")
    create_comparison_map(
        segments_b,
        "ViT-B (Base)",
        len(segments_b),
        time_b,
        image_path,
        bounds,
        satellite_image,
        output_dir / f"{image_path.stem}_sam_vit_b.html",
    )

    # Create ViT-L map
    print("4. Creating ViT-L comparison map...")
    create_comparison_map(
        segments_l,
        "ViT-L (Large)",
        len(segments_l),
        time_l,
        image_path,
        bounds,
        satellite_image,
        output_dir / f"{image_path.stem}_sam_vit_l.html",
    )

    print(f"\n✓ Comparison maps saved to: {output_dir}")
    print("\nOpen both HTML files to compare side-by-side!")


def create_comparison_map(
    segments, model_name, count, time_taken, image_path, bounds, satellite_image, output_path
):
    """Create a folium map for one model."""
    m = folium.Map(
        location=[satellite_image["center_lat"], satellite_image["center_lon"]],
        zoom_start=20,
        zoom_control=True,
        max_zoom=22,
        min_zoom=10,
    )

    # Add satellite image
    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=bounds,
        opacity=0.8,
        name="Satellite Image",
    ).add_to(m)

    # Add segments
    segments_group = folium.FeatureGroup(name=f"SAM Segments ({count})", show=True)

    for segment in segments:
        if segment.area_pixels < 500:
            color = "#3388ff"  # Blue for small
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
                f"<b>Segment #{segment.segment_id}</b><br>"
                f"Area: {segment.area_pixels} pixels<br>"
                f"Stability: {segment.stability_score:.3f}<br>"
                f"IoU: {segment.predicted_iou:.3f}",
                max_width=250,
            ),
            tooltip=f"Segment #{segment.segment_id}",
        ).add_to(segments_group)

    segments_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)

    # Add title
    title_html = f"""
    <div style="position: fixed;
                top: 10px; left: 50px; width: auto; height: auto;
                background-color: white; z-index:9999; font-size:16px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <b>SAM {model_name}</b><br>
    <small>{count} segments | {time_taken:.1f}s inference</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight: bold; text-align: center;">Legend</p>
    <p style="margin:5px 0;"><span style="color:#3388ff;">●</span> Small (&lt;500px)</p>
    <p style="margin:5px 0;"><span style="color:#00ff00;">●</span> Medium (500-2000px)</p>
    <p style="margin:5px 0;"><span style="color:#ff0000;">●</span> Large (&gt;2000px)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))


if __name__ == "__main__":
    image_path = "output/examples/images/672_white_oak_ln_vacaville_ca_95687.jpg"
    output_dir = "output/examples/folium_maps/sam_comparison"

    run_sam_comparison(image_path, output_dir)
