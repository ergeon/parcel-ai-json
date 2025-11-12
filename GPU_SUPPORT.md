# GPU Support for PyTorch Models

This document describes automatic GPU detection and usage across all environments.

## Auto-Detection

All PyTorch models (YOLO, SAM) now automatically detect and use the best available device:

1. **NVIDIA CUDA** - For ECS GPU instances (`g4dn.xlarge`, `g4dn.2xlarge`, etc.)
2. **Apple Silicon MPS** - For local Mac development (M1/M2/M3)
3. **CPU** - Fallback for Docker on Mac, non-GPU instances

## Implementation

### Device Detection Utility

`parcel_ai_json/device_utils.py` provides centralized device detection:

```python
from parcel_ai_json.device_utils import get_best_device

device = get_best_device()  # Returns "cuda", "mps", or "cpu"
```

### Models Using GPU

All PyTorch models automatically use the detected device:

- **YOLOv8-OBB** (vehicles, pools, amenities)
- **SAM** (segmentation)
- **DeepForest** (tree detection)

### API Service

The FastAPI service (`parcel_ai_json/api.py`) automatically detects and uses GPU when available:

```python
# Logs on startup:
# "Initializing PropertyDetectionService with DeepForest (device: cuda)..."
```

### Example Scripts

`scripts/generate_examples.py` auto-detects device and logs usage:

```python
# Output:
# "Running SAM segmentation (device: cuda)..."
```

## Docker Container

### Current Behavior

Docker containers **cannot access Apple Silicon MPS** (Metal is macOS-only). Containers will use CPU.

### ECS GPU Deployment

To use GPUs in ECS:

1. **Choose GPU instance type**: `g4dn.xlarge`, `g4dn.2xlarge`, etc.
2. **Update task definition**:
   ```json
   {
     "requiresCompatibilities": ["EC2"],
     "containerDefinitions": [{
       "resourceRequirements": [{
         "type": "GPU",
         "value": "1"
       }]
     }]
   }
   ```
3. **Use nvidia-docker runtime** (automatically configured on ECS GPU instances)

The container will automatically detect and use CUDA without code changes.

## Pre-Downloaded Models

All models are baked into the Docker image at build time:

### YOLO Models
- **Location**: `/root/.ultralytics/`
- **Models**: `yolov8m-obb.pt`

### SAM Models
- **Location**: `/app/models/`
- **Models**:
  - `sam_vit_b_01ec64.pth` (358MB)
  - `sam_vit_l_0b3195.pth` (~1.2GB)
  - `sam_vit_h_4b8939.pth` (~2.4GB) ← **Default**

### Tree Detection Models
- **DeepForest**: Downloads on first use to `/root/.cache/torch/`
- **detectree**: Pre-initialized classifier

## Performance Comparison

### SAM ViT-H Inference (per image)

| Environment | Device | Time |
|------------|--------|------|
| **ECS g4dn.xlarge** | CUDA (Tesla T4) | ~2-4 seconds |
| **Local Mac M2** | MPS | ~5-8 seconds |
| **Docker on Mac** | CPU | ~25-35 seconds |

### YOLO Inference (per image)

| Environment | Device | Time |
|------------|--------|------|
| **ECS g4dn.xlarge** | CUDA | ~0.1-0.2 seconds |
| **Local Mac M2** | MPS | ~0.2-0.3 seconds |
| **Docker on Mac** | CPU | ~0.5-1.0 seconds |

## Testing GPU Detection

### Local Testing

```bash
# Test device detection
python -c "from parcel_ai_json.device_utils import get_best_device; print(get_best_device())"

# Test with generation script
python scripts/generate_examples.py --num-examples 1
# Look for: "Running SAM segmentation (device: mps)..."
```

### Docker Testing

```bash
# Build and run container
docker build -f docker/Dockerfile -t parcel-ai-json:latest .
docker run -p 8000:8000 parcel-ai-json:latest

# Check logs - should show:
# "Initializing PropertyDetectionService with DeepForest (device: cpu)..."
```

### ECS GPU Testing

```bash
# SSH into ECS instance
aws ecs execute-command --cluster <cluster> --task <task-id> --command "/bin/bash" --interactive

# Inside container
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: "CUDA available: True"

# Check service logs
curl http://localhost:8000/health
# Logs should show: "device: cuda"
```

## Troubleshooting

### GPU Not Detected in ECS

1. **Verify instance type**: Must be `g4dn.*` or similar GPU instance
2. **Check task definition**: Must have `resourceRequirements` with `type: "GPU"`
3. **Verify nvidia-docker**: Run `nvidia-smi` in container
4. **Check CloudWatch logs**: Look for "device: cuda" in startup logs

### Slow Inference Despite GPU

1. **Check model location**: Models must be pre-downloaded (not downloading at runtime)
2. **Verify device**: Check logs for "device: cuda" not "device: cpu"
3. **Monitor GPU usage**: Run `nvidia-smi` during inference

### MPS Not Working on Mac

1. **Check PyTorch version**: Requires PyTorch >= 1.12
2. **Verify Apple Silicon**: MPS only on M1/M2/M3 chips
3. **Check availability**:
   ```python
   import torch
   print(torch.backends.mps.is_available())
   ```

## Future Improvements

- [ ] Add GPU memory monitoring
- [ ] Support multi-GPU inference
- [ ] Add GPU metrics to CloudWatch
- [ ] Optimize batch sizes for different GPU types
