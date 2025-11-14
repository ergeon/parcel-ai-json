# ML Model Weights

This directory contains the ML model weights used for object detection and fence detection.

## YOLO Models

- `yolov8n.pt` - YOLOv8 Nano (fastest, least accurate)
- `yolov8m.pt` - YOLOv8 Medium (standard detection)
- `yolov8m-seg.pt` - YOLOv8 Medium Segmentation (for pools)
- `yolov8m-obb.pt` - YOLOv8 Medium Oriented Bounding Box (for vehicles)

## Fence Detection Models

- `hed_fence_weighted_loss.pth` - HED fence detection model with weighted loss (168MB)
  - Architecture: HED (Holistically-Nested Edge Detection) with VGG16 backbone
  - Input: 4 channels (RGB + Regrid parcel boundary mask)
  - Training: Weighted BCE Loss with 3x penalty for false negatives
  - Trained: October 31, 2025 on line-corrected dataset
  - Characteristics: Aggressive detection, high recall, prefers over-prediction

## SAM Models

- `sam_vit_h_4b8939.pth` - Segment Anything Model (SAM) ViT-H checkpoint (2.4GB)

## Download

YOLO models are automatically downloaded by Ultralytics on first use:

```bash
# From Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-obb.pt
```

## Git Ignore

Model files (`*.pt`, `*.pth`) are git-ignored due to their large size (~50MB-2.4GB each).
YOLO models will download automatically if missing. HED and SAM models should be copied to this directory manually.
