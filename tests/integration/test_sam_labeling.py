"""Test script for SAM segment labeling.

Tests the overlap-based labeling system using existing detections.
"""

import sys
from pathlib import Path
import json
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parcel_ai_json.sam_segmentation import SAMSegmentationService  # noqa: E402
from parcel_ai_json.property_detector import PropertyDetectionService  # noqa: E402
from parcel_ai_json.sam_labeler import LABEL_SCHEMA  # noqa: E402


def test_sam_labeling():
    """Test SAM segment labeling on an existing satellite image."""

    # Find a sample satellite image from examples
    examples_dir = Path("output/examples/images").resolve()
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
    print(f"Testing SAM labeling on: {sample_image_path}")
    print("=" * 80)

    # Cast coordinates (we need to extract from filename in future)
    satellite_image = {
        "path": str(sample_image_path),
        "center_lat": 37.7749,  # Placeholder
        "center_lon": -122.4194,  # Placeholder
        "zoom_level": 20,
    }

    # Step 1: Run all detections
    print("\nStep 1: Running property detections...")
    property_detector = PropertyDetectionService()
    detections_result = property_detector.detect_all(satellite_image)

    print(f"✓ Detected {len(detections_result.vehicles)} vehicles")
    print(f"✓ Detected {len(detections_result.swimming_pools)} pools")
    print(f"✓ Detected {len(detections_result.amenities)} amenities")
    tree_poly_count = len(detections_result.trees.tree_polygons or [])
    print(f"✓ Detected {tree_poly_count} tree polygons")

    # Step 2: Run SAM segmentation
    print("\nStep 2: Running SAM segmentation...")
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
        sam_url = (
            "https://dl.fbaipublicfiles.com/" "segment_anything/sam_vit_b_01ec64.pth"
        )
        print(f"cd models && curl -L -o sam_vit_b_01ec64.pth {sam_url}")
        return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Step 3: Run SAM segmentation with labeling
    print("\nStep 3: Running SAM segmentation with labeling...")
    print("(This may take 30-60 seconds on CPU...)")

    try:
        labeled_segments = sam_service.segment_image_labeled(
            satellite_image,
            {
                "vehicles": detections_result.vehicles,
                "pools": detections_result.pools,
                "amenities": detections_result.amenities,
                "trees": detections_result.trees or [],
                "tree_polygons": detections_result.tree_polygons or [],
            },
            overlap_threshold=0.5,
        )
    except Exception as e:
        print(f"ERROR during segmentation: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Analyze labeling results
    print(f"\n{'=' * 80}")
    print("LABELING RESULTS")
    print(f"{'=' * 80}")
    print(f"Total segments: {len(labeled_segments)}")

    # Count labels
    label_counts = Counter(s.primary_label for s in labeled_segments)
    labeled_count = sum(
        count for label, count in label_counts.items() if label != "unknown"
    )
    unknown_count = label_counts.get("unknown", 0)

    labeled_pct = labeled_count / len(labeled_segments) * 100
    unknown_pct = unknown_count / len(labeled_segments) * 100
    print(f"\nLabeled segments: {labeled_count} ({labeled_pct:.1f}%)")
    print(f"Unknown segments: {unknown_count} ({unknown_pct:.1f}%)")

    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        pct = count / len(labeled_segments) * 100
        color = LABEL_SCHEMA.get(label, LABEL_SCHEMA["unknown"])["color"]
        print(f"  {label:20} {count:4} ({pct:5.1f}%)  {color}")

    # Show examples of each label type
    print("\nExample segments by label:")
    shown_labels = set()
    for segment in labeled_segments:
        if segment.primary_label not in shown_labels:
            shown_labels.add(segment.primary_label)
            print(
                f"  [{segment.primary_label}] "
                f"Segment #{segment.segment_id}: "
                f"area={segment.area_sqm:.1f}m², "
                f"conf={segment.label_confidence:.2f}, "
                f"source={segment.label_source}, "
                f"reason={segment.labeling_reason}"
            )

    # Step 5: Save labeled GeoJSON
    output_dir = Path("output/examples/sam")
    output_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = (
        output_dir / f"{sample_image_path.stem}_labeled_segments.geojson"
    )  # noqa: E501
    geojson = {
        "type": "FeatureCollection",
        "features": [seg.to_geojson_feature() for seg in labeled_segments],
    }

    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\n✓ Labeled GeoJSON saved to: {geojson_path}")

    # Statistics by confidence level
    print("\nLabeling confidence distribution:")
    high_conf = sum(1 for s in labeled_segments if s.label_confidence >= 0.75)
    med_conf = sum(1 for s in labeled_segments if 0.55 <= s.label_confidence < 0.75)
    low_conf = sum(1 for s in labeled_segments if 0 < s.label_confidence < 0.55)
    no_conf = sum(1 for s in labeled_segments if s.label_confidence == 0)

    print(f"  High confidence (≥0.75): {high_conf}")
    print(f"  Medium confidence (0.55-0.74): {med_conf}")
    print(f"  Low confidence (0-0.54): {low_conf}")
    print(f"  No label (0.0): {no_conf}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    test_sam_labeling()
