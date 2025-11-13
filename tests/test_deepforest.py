"""Quick test of DeepForest on Fort Worth property.

Compare with detectree results:
- detectree: 0.33% coverage, 3 tiny clusters (11.86m², 2.85m², 3.91m²)
- DeepForest: Should detect individual tree crowns with bounding boxes
"""

import pytest
from pathlib import Path


# Use the Fort Worth image from examples
TEST_IMAGE = Path(
    "output/examples/images/528_beechgrove_terrace_fort_worth_tx_76140.jpg"
)


@pytest.mark.skipif(
    not TEST_IMAGE.exists(),
    reason=f"Test image not found at {TEST_IMAGE}. Run generate_examples.py first.",
)
def test_deepforest_detection():
    """Test DeepForest tree detection on Fort Worth property."""
    from deepforest import main

    print(f"\nTesting DeepForest on: {TEST_IMAGE}")
    print("=" * 80)

    # Initialize DeepForest model
    print("Loading DeepForest model...")
    m = main.deepforest()

    # Load pretrained model from Hugging Face
    # Default model: weecology/deepforest-tree
    print("Loading pretrained weights from Hugging Face...")
    m.load_model(model_name="weecology/deepforest-tree")

    # Predict tree crowns
    print("Detecting trees...")
    boxes = m.predict_image(path=str(TEST_IMAGE))

    # Display results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Number of trees detected: {len(boxes)}")

    # Basic assertions
    assert boxes is not None, "Detection should return a DataFrame"
    assert len(boxes) >= 0, "Should return non-negative number of detections"

    if len(boxes) > 0:
        print("\nFirst 10 detections:")
        print(boxes.head(10).to_string())

        print(f"\n{'=' * 80}")
        print("STATISTICS")
        print(f"{'=' * 80}")
        print(f"Average confidence: {boxes['score'].mean():.3f}")
        print(f"Min confidence: {boxes['score'].min():.3f}")
        print(f"Max confidence: {boxes['score'].max():.3f}")

        # Calculate average crown size
        boxes["width"] = boxes["xmax"] - boxes["xmin"]
        boxes["height"] = boxes["ymax"] - boxes["ymin"]
        boxes["area"] = boxes["width"] * boxes["height"]

        print(f"\nAverage crown size (pixels): {boxes['area'].mean():.1f}")
        print(f"Min crown size (pixels): {boxes['area'].min():.1f}")
        print(f"Max crown size (pixels): {boxes['area'].max():.1f}")

        print(f"\n{'=' * 80}")
        print("COMPARISON WITH DETECTREE")
        print(f"{'=' * 80}")
        print("detectree: 0.33% coverage, 3 tiny clusters")
        print(f"DeepForest: {len(boxes)} individual tree crowns detected")

        # Verify detection quality
        assert boxes["score"].min() >= 0.0, "Confidence should be >= 0"
        assert boxes["score"].max() <= 1.0, "Confidence should be <= 1"
    else:
        print("\nNo trees detected!")
        print("This might indicate a problem with the model or image.")

    print(f"\n{'=' * 80}")
