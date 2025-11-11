"""Test script for SAM (Segment Anything Model) segmentation.

Tests SAM on a sample satellite image and visualizes the results.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.sam_segmentation import SAMSegmentationService
from PIL import Image
import numpy as np


def test_sam_on_satellite_image():
    """Test SAM on an existing satellite image."""

    # Find a sample satellite image from examples
    examples_dir = Path("output/examples/images")
    if not examples_dir.exists():
        print(f"ERROR: Examples directory not found: {examples_dir}")
        print("Please run generate_examples.py first to create sample images.")
        return

    # Get first .jpg file
    sample_images = list(examples_dir.glob("*.jpg"))
    if not sample_images:
        print(f"ERROR: No sample images found in {examples_dir}")
        return

    sample_image_path = sample_images[0]
    print(f"Testing SAM on: {sample_image_path}")
    print("=" * 80)

    # Load image to get center coordinates (hardcoded for now)
    # TODO: Extract from filename or metadata
    satellite_image = {
        "path": str(sample_image_path),
        "center_lat": 37.7749,  # Placeholder
        "center_lon": -122.4194,  # Placeholder
        "zoom_level": 20,
    }

    # Initialize SAM service
    print("Initializing SAM service...")
    sam_service = SAMSegmentationService(
        model_type="vit_b",  # Using ViT-B (smaller, faster)
        device="cpu",
        points_per_side=16,  # Reduced for faster inference
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,  # Minimum 100 pixels
    )

    print("Loading SAM model...")
    try:
        sam_service._load_model()
        print("✓ SAM model loaded successfully")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease download the SAM model checkpoint:")
        print("cd models && curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Run segmentation
    print("\nRunning automatic segmentation...")
    print("(This may take 30-60 seconds on CPU...)")

    try:
        segments = sam_service.segment_image(satellite_image)
    except Exception as e:
        print(f"ERROR during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Display results
    print(f"\n{'=' * 80}")
    print("SEGMENTATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Total segments found: {len(segments)}")

    if segments:
        # Show statistics
        areas_pixels = [s.area_pixels for s in segments]
        stability_scores = [s.stability_score for s in segments]
        iou_scores = [s.predicted_iou for s in segments]

        print(f"\nSegment area statistics (pixels):")
        print(f"  Min: {min(areas_pixels)}")
        print(f"  Max: {max(areas_pixels)}")
        print(f"  Mean: {sum(areas_pixels) / len(areas_pixels):.1f}")

        print(f"\nStability score statistics:")
        print(f"  Min: {min(stability_scores):.3f}")
        print(f"  Max: {max(stability_scores):.3f}")
        print(f"  Mean: {sum(stability_scores) / len(stability_scores):.3f}")

        print(f"\nPredicted IoU statistics:")
        print(f"  Min: {min(iou_scores):.3f}")
        print(f"  Max: {max(iou_scores):.3f}")
        print(f"  Mean: {sum(iou_scores) / len(iou_scores):.3f}")

        # Show first 5 segments
        print(f"\nFirst 5 segments:")
        for i, seg in enumerate(segments[:5]):
            print(f"  [{i}] ID={seg.segment_id}, area={seg.area_pixels}px, "
                  f"stability={seg.stability_score:.3f}, iou={seg.predicted_iou:.3f}")

        # Save GeoJSON
        output_dir = Path("output/examples/sam")
        output_dir.mkdir(parents=True, exist_ok=True)

        geojson_path = output_dir / f"{sample_image_path.stem}_sam_segments.geojson"
        geojson = sam_service.segment_image_geojson(satellite_image)

        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)

        print(f"\n✓ GeoJSON saved to: {geojson_path}")

        # Create visualization
        print("\nCreating visualization...")
        try:
            create_visualization(sample_image_path, segments, output_dir)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")

    else:
        print("\nNo segments found!")

    print(f"\n{'=' * 80}")


def create_visualization(image_path: Path, segments, output_dir: Path):
    """Create a visualization of SAM segments overlaid on the image."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    import random

    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Image with segments
    axes[1].imshow(img_array)

    # Overlay segments with random colors
    for seg in segments:
        mask = seg.pixel_mask
        color = [random.random(), random.random(), random.random()]

        # Create colored overlay
        colored_mask = np.zeros((*mask.shape, 3))
        colored_mask[mask] = color

        # Overlay with transparency
        axes[1].imshow(colored_mask, alpha=0.4)

        # Draw bounding box
        x1, y1, x2, y2 = seg.pixel_bbox
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        axes[1].add_patch(rect)

    axes[1].set_title(f"SAM Segmentation ({len(segments)} segments)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()

    # Save
    viz_path = output_dir / f"{image_path.stem}_sam_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Visualization saved to: {viz_path}")


if __name__ == "__main__":
    test_sam_on_satellite_image()
