#!/usr/bin/env python3
"""
Quick SAM3 test - Download and test with a sample image
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

# Enable MPS fallback to CPU for operations not yet supported on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("=" * 70)
print("SAM3 Quick Test")
print("=" * 70)

# Check if image exists
image_path = "test_images/aerial_image.jpg"

if not os.path.exists(image_path):
    print(f"\n⚠️  Image not found at: {image_path}")
    print("\nPlease save your aerial image to this location, then run:")
    print(f"  python {sys.argv[0]}")
    print("\nOr provide the image path as an argument:")
    print(f"  python {sys.argv[0]} /path/to/your/image.jpg")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"\n❌ Image not found: {image_path}")
            sys.exit(1)
    else:
        sys.exit(0)

print(f"\n📷 Using image: {image_path}")

try:
    from PIL import Image

    # Load and display image info
    image = Image.open(image_path)
    print(f"   Size: {image.size}")
    print(f"   Format: {image.format}")
    print(f"   Mode: {image.mode}")

    # Determine device
    # Note: MPS has compatibility issues with SAM3, so we use CPU for stability
    device = "cpu"
    print("\n💻 Using: CPU (MPS has compatibility issues with SAM3)")

    print("\n🚀 Loading SAM3 model...")
    print("   (This will download ~4GB on first run - please be patient)")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()

    print(f"   Moving model to {device}...")
    model = model.to(device)

    processor = Sam3Processor(model, device=device)

    print("   ✅ Model loaded!")

    print("\n🔍 Processing image...")
    inference_state = processor.set_image(image)

    # Test with a simple prompt
    print("\n📝 Testing with prompt: 'cars'")
    output = processor.set_text_prompt(
        state=inference_state,
        prompt="cars"
    )

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print("\n✅ Detection Results:")
    print(f"   Total detections: {len(masks)}")
    print(f"   Bounding boxes: {len(boxes)}")

    # Show top detections
    if len(scores) > 0:
        threshold = 0.3
        high_conf = [i for i, s in enumerate(scores) if s > threshold]
        print(f"   High confidence (>{threshold}): {len(high_conf)}")
        print(f"   Top 5 scores: {sorted(scores, reverse=True)[:5]}")

        # Show bounding boxes
        print("\n   Top 3 bounding boxes (x1, y1, x2, y2):")
        for i in range(min(3, len(boxes))):
            box = boxes[i]
            score = scores[i]
            print(f"     {i+1}. {box} (score: {score:.3f})")

    print("\n" + "=" * 70)
    print("✅ Test successful!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Try the Jupyter notebooks: jupyter notebook examples/")
    print("2. Test with different prompts: 'houses', 'trees', 'roof', etc.")
    print("3. Check the visualization examples in examples/")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMake sure you activated the virtual environment:")
    print("  source venv/bin/activate")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
