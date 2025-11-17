# ML Model Weights

This directory contains the ML model weights used for property feature detection.

## Active Models (Currently Used)

### YOLO Models

- **`yolov8m-obb.pt`** (51 MB) - YOLOv8 Medium Oriented Bounding Box
  - Used for: Vehicles, Swimming Pools, Amenities (tennis courts, basketball courts, etc.)
  - Auto-downloaded to `~/.ultralytics/` on first use
  - Source: Ultralytics

### Fence Detection Models

- **`hed_fence_mixed_finetune.pth`** (168 MB) - HED fence detection model
  - Architecture: HED (Holistically-Nested Edge Detection) with VGG16 backbone
  - Input: 4 channels (RGB + Regrid parcel boundary mask)
  - Training: Weighted BCE Loss with 3x penalty for false negatives
  - Trained: October 31, 2025 on line-corrected dataset
  - Characteristics: Aggressive detection, high recall, prefers over-prediction
  - Source: fence_detector.py:132

### SAM Models

- **`sam_vit_h_4b8939.pth`** (2.4 GB) - Segment Anything Model (SAM) ViT-H checkpoint
  - Used for: General-purpose segmentation
  - Default model (highest accuracy)
  - Alternative models available: vit_b (375 MB), vit_l (1.2 GB) - but not included
  - Source: sam_segmentation.py:75, 128

## Tree Detection Models

Tree detection uses models that are **auto-downloaded** (not stored in this directory):

- **DeepForest** - `weecology/deepforest-tree` from HuggingFace
  - Auto-downloaded to `~/.cache/torch/hub/`
  - Used for: Individual tree crown detection
  - Source: tree_detector.py:173

- **detectree** - Runs in separate Docker container
  - Used for: Tree coverage polygons and statistics
  - Source: tree_detector.py

## Total Size

**Current models/**: 2.6 GB
- hed_fence_mixed_finetune.pth: 168 MB
- sam_vit_h_4b8939.pth: 2.4 GB
- yolov8m-obb.pt: 51 MB

## Download

YOLO models are automatically downloaded by Ultralytics on first use:

```bash
# From Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-obb.pt
```

SAM and HED models should be copied to this directory manually or will be downloaded during Docker build.

## Git Ignore

Model files (`*.pt`, `*.pth`) are git-ignored due to their large size.
YOLO models will download automatically if missing. HED and SAM models should be copied to this directory manually.

## Cleanup History

**Removed (November 2025):**
- Old fence checkpoints: hed_fence_checkpoint_best.pth, hed_fence_weighted_loss.pth
- Alternative SAM models: sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth
- Unused YOLO variants: yolov8m.pt, yolov8m-seg.pt, yolov8n.pt
- ResNet50 backbone: resnet50-0676ba61.pth
- **Total space saved: ~2.2 GB**
