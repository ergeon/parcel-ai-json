"""Quick test of DeepForest on Fort Worth property.

Compare with detectree results:
- detectree: 0.33% coverage, 3 tiny clusters (11.86m², 2.85m², 3.91m²)
- DeepForest: Should detect individual tree crowns with bounding boxes
"""

from deepforest import main
from pathlib import Path
import sys

# Use the Fort Worth image from examples
test_image = Path(
    "output/examples/images/528_beechgrove_terrace_fort_worth_tx_76140.jpg"
)

if not test_image.exists():
    print(f"ERROR: Could not find test image at {test_image}")
    sys.exit(1)

print(f"Testing DeepForest on: {test_image}")
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
boxes = m.predict_image(path=str(test_image))

# Display results
print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")
print(f"Number of trees detected: {len(boxes)}")

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

else:
    print("\nNo trees detected!")
    print("This might indicate a problem with the model or image.")

print(f"\n{'=' * 80}")
