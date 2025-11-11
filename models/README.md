# ML Model Weights

This directory contains the YOLO model weights used for object detection.

## Models

- `yolov8n.pt` - YOLOv8 Nano (fastest, least accurate)
- `yolov8m.pt` - YOLOv8 Medium (standard detection)
- `yolov8m-seg.pt` - YOLOv8 Medium Segmentation (for pools)
- `yolov8m-obb.pt` - YOLOv8 Medium Oriented Bounding Box (for vehicles)

## Download

These models are automatically downloaded by Ultralytics on first use. You can also manually download them:

```bash
# From Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-obb.pt
```

## Git Ignore

Model files (*.pt) are git-ignored due to their large size (~50-200MB each).
The application will download them automatically if missing.
