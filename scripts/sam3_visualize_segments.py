#!/usr/bin/env python3
"""
Visualize SAM3 segmentation masks for multiple classes
"""

import os
import sys
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load environment
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Determine device
device = "cpu"

# Get image path from arguments
if len(sys.argv) < 2:
    print("Usage: python visualize_segments.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Define classes to detect and their colors
classes = {
    'houses': (255, 0, 0, 120),  # Red
    'cars': (0, 255, 0, 120),  # Green
    'trees': (0, 100, 0, 100),  # Dark Green
    'roof': (255, 165, 0, 120),  # Orange
    'driveway': (128, 128, 128, 100),  # Gray
    'road': (64, 64, 64, 100),  # Dark Gray
}

# Load image
image = Image.open(image_path).convert('RGB')
print(f"Image size: {image.size}")

# Load model
print("Loading SAM3 model...")
model = build_sam3_image_model()
model = model.to(device)
processor = Sam3Processor(model, device=device, confidence_threshold=0.3)

# Create output image
output_image = image.copy()
overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

# Process each class
all_results = {}
for class_name, color in classes.items():
    print(f"\nDetecting '{class_name}'...")

    # Process image
    inference_state = processor.set_image(image)

    # Run detection
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=class_name
    )

    masks = output["masks"]
    scores = output["scores"]

    print(f"  Found {len(masks)} {class_name} (confidence > 0.3)")
    if len(scores) > 0:
        print(f"  Top scores: {[f'{s:.2f}' for s in scores[:3].tolist()]}")

    all_results[class_name] = {
        'masks': masks,
        'scores': scores,
        'count': len(masks)
    }

    # Draw masks for this class
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Convert mask to numpy
        mask_np = mask.squeeze().cpu().numpy()

        # Create colored overlay for this mask
        mask_colored = Image.new('RGBA', image.size, color)
        mask_alpha = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # Composite onto overlay
        overlay.paste(mask_colored, (0, 0), mask_alpha)

# Composite overlay onto original image
output_image = Image.alpha_composite(image.convert('RGBA'), overlay)
output_image = output_image.convert('RGB')

# Add legend
draw = ImageDraw.Draw(output_image)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    font_large = ImageFont.truetype(
        "/System/Library/Fonts/Helvetica.ttc", 16
    )
except Exception:
    font = ImageFont.load_default()
    font_large = font

# Draw legend background
legend_height = 20 + len(classes) * 22
draw.rectangle(
    [(10, 10), (200, legend_height)],
    fill=(255, 255, 255, 230),
    outline='black'
)

# Draw legend title
draw.text((15, 12), "Detected Classes:", fill='black', font=font_large)

# Draw each class in legend
y = 35
for class_name, color in classes.items():
    count = all_results[class_name]['count']
    # Draw color box
    box_color = (color[0], color[1], color[2])
    draw.rectangle([(15, y), (30, y+12)], fill=box_color, outline='black')
    # Draw label
    label = f"{class_name}: {count}"
    draw.text((35, y), label, fill='black', font=font)
    y += 22

# Save output
output_path = (
    image_path.replace('.jpg', '_segments.jpg')
    .replace('.png', '_segments.png')
)
output_image.save(output_path)
print(f"\n✅ Saved segmentation visualization to: {output_path}")

# Print summary
print("\n" + "=" * 60)
print("SEGMENTATION SUMMARY")
print("=" * 60)
for class_name, results in all_results.items():
    count = results['count']
    print(f"{class_name.upper():15s}: {count:2d} objects detected")
print("=" * 60)

# Try to open
try:
    output_image.show()
    print("\nOpening visualization...")
except Exception:
    print("\nPlease open the file manually.")
