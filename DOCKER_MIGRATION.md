# Docker Migration Plan - Run All Tree Detection in Docker

## Current Status

### What's Working Now
- ✅ **DeepForest**: Running natively in Python (uses PyTorch/transformers)
- ✅ **detectree**: Configured to run in Docker (default: `detectree_use_docker=True`)
- ✅ **YOLO (vehicles/pools/amenities)**: Running natively in Python (ultralytics)
- ✅ Parallel tree detection with both DeepForest and detectree
- ✅ Simplified GeoJSON polygons from detectree (no pixel masks)
- ✅ Topology-preserving polygon simplification (0.5m tolerance)

### Current Architecture
```
PropertyDetectionService
├── VehicleDetectionService (YOLO-OBB, native Python)
├── SwimmingPoolDetectionService (YOLO-OBB, native Python)
├── AmenityDetectionService (YOLO-OBB, native Python)
└── CombinedTreeDetectionService
    ├── DeepForestService (native Python, PyTorch)
    └── DetectreeService (Docker container)
```

### Recent Changes (Last 3 Commits)
1. `eaefe24` - Use simplified GeoJSON polygons instead of pixel masks
2. `e17390a` - Add parallel tree detection with DeepForest and detectree
3. `5024b6e` - Replace detectree with DeepForest for tree detection

## Problem Statement

**Goal**: Run ALL AI/ML models (YOLO, DeepForest, detectree) inside Docker containers for:
- Consistent environment across machines
- Easier deployment
- Better isolation
- Simplified dependency management
- Production-ready containerization

## Current Docker Setup

### Existing Docker Infrastructure
- `Dockerfile` - Main container with FastAPI service
- `Makefile` - Build and run commands
- `docker-tree-detector/` - Separate detectree container (legacy)

### What Needs Migration
1. **DeepForest** - Currently native Python, needs Dockerization
2. **YOLO models** - Currently native Python, needs Dockerization
3. **Unified container** - All models in single container vs separate containers

## Migration Plan

### Option 1: Single Unified Container (Recommended)
**Pros**:
- Simpler deployment
- Faster inter-service communication
- Smaller total image size
- Easier to manage

**Cons**:
- Larger single image
- All-or-nothing updates

**Architecture**:
```
docker-compose.yml
└── parcel-ai-json (single service)
    ├── FastAPI REST API
    ├── YOLO models (ultralytics)
    ├── DeepForest (transformers/PyTorch)
    └── detectree (native Python, no Docker subprocess)
```

### Option 2: Microservices Architecture
**Pros**:
- Independent scaling
- Isolated updates
- Better resource allocation

**Cons**:
- More complex orchestration
- Network overhead
- More containers to manage

**Architecture**:
```
docker-compose.yml
├── api-service (FastAPI)
├── yolo-service (vehicles/pools/amenities)
├── deepforest-service (tree crowns)
└── detectree-service (tree coverage)
```

## Recommended Approach: Option 1 (Unified Container)

### Step 1: Update Dockerfile
**File**: `Dockerfile`

**Changes needed**:
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libspatialindex-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time (optional, for faster startup)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-obb.pt')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('weecology/deepforest-tree')"

# Copy application code
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "parcel_ai_json.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Update tree_detector.py
**File**: `parcel_ai_json/tree_detector.py`

**Change to run natively INSIDE the Docker container** (no Docker-in-Docker):
```python
class CombinedTreeDetectionService:
    def __init__(
        self,
        # DeepForest parameters
        deepforest_model_name: str = "weecology/deepforest-tree",
        deepforest_confidence_threshold: float = 0.1,
        # detectree parameters
        detectree_use_docker: bool = False,  # Run natively inside container
        detectree_docker_image: str = "parcel-tree-detector",
        detectree_save_mask: bool = False,
        detectree_extract_polygons: bool = True,
        # ...
    ):
```

**Explanation**: Since ALL code will run inside a Docker container, we set `use_docker=False` to run detectree natively WITHIN that container, avoiding Docker-in-Docker complexity.

### Step 3: Update requirements.txt
**File**: `requirements.txt`

**Ensure all dependencies are listed**:
```txt
# Core dependencies
ultralytics>=8.0.0      # YOLOv8
pillow>=9.0.0
numpy>=1.20.0
torch>=2.0.0
torchvision>=0.15.0
pyproj>=3.0.0

# Tree detection
deepforest>=2.0.0       # Individual tree crowns
detectree==0.8.0        # Tree coverage polygons

# Transformers for DeepForest
transformers>=4.0.0
timm>=0.9.0

# API dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
```

### Step 4: Create docker-compose.yml
**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  parcel-ai-json:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./output:/app/output
      - ./data:/app/data
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/torch:/root/.cache/torch
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
    restart: unless-stopped
```

### Step 5: Update Makefile
**File**: `Makefile`

```makefile
.PHONY: docker-build docker-run docker-stop docker-clean

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f

docker-logs:
	docker-compose logs -f
```

### Step 6: Test the Migration
1. Build the container: `make docker-build`
2. Run the container: `make docker-run`
3. Test API endpoint: `curl http://localhost:8000/detect`
4. Generate examples: Inside container or via API
5. Verify outputs in `./output/` directory

## Implementation Checklist

### Phase 1: Preparation
- [ ] Document current native setup performance benchmarks
- [ ] Backup current working configuration
- [ ] Test detectree in native mode (already installed)
- [ ] Verify all dependencies in requirements.txt

### Phase 2: Docker Configuration
- [ ] Update Dockerfile with all dependencies
- [ ] Add model download steps (optional optimization)
- [ ] Create docker-compose.yml
- [ ] Update Makefile with new commands
- [ ] Configure volume mounts for outputs

### Phase 3: Code Changes
- [ ] Change `detectree_use_docker=False` in tree_detector.py
- [ ] Update API to handle file paths correctly in container
- [ ] Update example generation script for Docker paths
- [ ] Add health check endpoint to API

### Phase 4: Testing
- [ ] Build container and verify no errors
- [ ] Test YOLO detection in container
- [ ] Test DeepForest detection in container
- [ ] Test detectree detection in container
- [ ] Test combined tree detection
- [ ] Generate sample outputs and verify quality
- [ ] Performance comparison: native vs Docker

### Phase 5: Documentation
- [ ] Update README.md with Docker instructions
- [ ] Add Docker troubleshooting guide
- [ ] Document environment variables
- [ ] Add deployment guide

## Performance Considerations

### Model Loading Time
- **First run**: Models download from HuggingFace/Ultralytics (slow)
- **Subsequent runs**: Models cached in volumes (fast)
- **Optimization**: Pre-download models during build

### Memory Requirements
- YOLO models: ~100MB
- DeepForest: ~200MB
- detectree: ~50MB
- **Total**: ~2GB RAM recommended

### Storage Requirements
- Docker image: ~5GB
- Model cache: ~1GB
- Output files: Varies by usage

## Migration Timeline

### Immediate Next Steps (This Session)
1. Wait for current sample generation to complete
2. Commit DOCKER_MIGRATION.md plan document
3. Review current Dockerfile structure

### Next Session - Full Docker Migration
1. Update Dockerfile with ALL dependencies (YOLO, DeepForest, detectree)
2. Set `detectree_use_docker=False` to run inside container (not Docker-in-Docker)
3. Create docker-compose.yml for orchestration
4. Add model caching via volumes
5. Test Docker build
6. Verify all models work in container
7. Generate samples entirely in Docker
8. Performance comparison

### Future Work
1. Add GPU support (CUDA)
2. Optimize image size
3. Add model caching strategies
4. Create production deployment guide

## Current File Changes Needed

### 1. parcel_ai_json/tree_detector.py
**Line 676**: Change `detectree_use_docker: bool = True` to `detectree_use_docker: bool = False`
**Why**: When running inside Docker container, detectree should run natively within that container, not spawn another Docker container (no Docker-in-Docker)

### 2. Dockerfile (major rewrite needed)
Add all ML dependencies and model downloads

### 3. docker-compose.yml (new file)
Create orchestration configuration

### 4. .dockerignore (new file)
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.git/
.pytest_cache/
output/
*.log
```

## Questions to Answer

1. **Model Persistence**: Should models be baked into image or downloaded at runtime?
   - **Recommendation**: Bake into image for faster startup

2. **API vs CLI**: How will users interact with the container?
   - **Current**: FastAPI REST API
   - **Also support**: CLI for batch processing

3. **Output Handling**: How to get results out of container?
   - **Recommendation**: Volume mounts for output directory

4. **Scaling**: Single instance or multiple replicas?
   - **Start**: Single instance
   - **Future**: Kubernetes/Docker Swarm for scaling

## Resources

- Current Dockerfile: `Dockerfile`
- Current API: `parcel_ai_json/api.py`
- Tree detector: `parcel_ai_json/tree_detector.py`
- Docker tree detector: `docker-tree-detector/` (legacy, can remove)

## Notes

- DeepForest uses PyTorch and Hugging Face transformers
- detectree uses scikit-learn and OpenCV
- YOLO uses ultralytics library
- All models are compatible with Docker
- No GPU required (CPU inference works fine)
- macOS arm64 support via Docker's emulation

---

# AWS Deployment Options

## Overview

Once Docker container is ready, here are the AWS deployment options ranked by cost-effectiveness and use case suitability.

## Option 1: ECS Fargate Spot (RECOMMENDED)

**Best for:** Regular usage with predictable/variable traffic, event-driven processing

### Pros
- ✅ **70% cheaper** than on-demand Fargate
- ✅ **No server management** - fully managed containers
- ✅ **Auto-scaling** - scales based on load
- ✅ **Docker-native** - your container works as-is
- ✅ **No cold starts** - containers stay warm
- ✅ **Load balancing** - Application Load Balancer integration
- ✅ **Pay only when running** - cost-effective for variable workloads

### Cons
- ⚠️ Spot instances can be interrupted (rare, <5% frequency)
- ⚠️ Slightly more complex setup than Lambda

### Cost Estimate
- **Light usage** (1 hr/day): ~$5-10/month
- **Moderate** (4 hrs/day): ~$20-30/month
- **Heavy** (24/7): ~$80-100/month

### Resource Configuration
- **CPU**: 4 vCPU (recommended for parallel model inference)
- **Memory**: 16GB RAM (enough for YOLO + DeepForest + detectree)
- **Storage**: EFS for model caching (~$0.30/GB/month)

### Deployment Steps

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name parcel-ai-json

# 2. Build and push Docker image
docker build -t parcel-ai-json .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag parcel-ai-json:latest <account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest

# 3. Create ECS cluster
aws ecs create-cluster --cluster-name parcel-ai-cluster

# 4. Register task definition (see task-definition.json below)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 5. Create service with Spot capacity
aws ecs create-service \
  --cluster parcel-ai-cluster \
  --service-name parcel-ai-service \
  --task-definition parcel-ai-task \
  --launch-type FARGATE \
  --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
  --desired-count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Task Definition (task-definition.json)

```json
{
  "family": "parcel-ai-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "parcel-ai-json",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "WORKERS",
          "value": "4"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/parcel-ai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "mountPoints": [
        {
          "sourceVolume": "model-cache",
          "containerPath": "/root/.cache"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "model-cache",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxxx",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

---

## Option 2: EventBridge + ECS RunTask (Event-Driven)

**Best for:** Batch processing, scheduled jobs, event-driven workflows

### Pros
- ✅ **Zero cost when idle** - only pay when task runs
- ✅ **Event-driven** - trigger from S3, SQS, schedules, custom events
- ✅ **No persistent service** - tasks spin up on-demand
- ✅ **Perfect for batch processing** - process images in batches
- ✅ **Hybrid between Lambda and ECS** - best of both worlds

### Cons
- ⚠️ Cold start (~30-60s to spin up container)
- ⚠️ Not suitable for real-time API responses

### Use Cases
- Process images uploaded to S3
- Scheduled batch processing (e.g., nightly jobs)
- Queue-based processing (SQS → EventBridge → ECS)
- Webhook-triggered processing

### Deployment Architecture

```
S3 Upload → EventBridge Rule → ECS RunTask → Process Image → Save to S3
Queue Message → EventBridge Rule → ECS RunTask → Process Batch → Update DB
Cron Schedule → EventBridge Rule → ECS RunTask → Batch Process → Generate Reports
```

### EventBridge Rule Example

```bash
# Create EventBridge rule for S3 uploads
aws events put-rule \
  --name process-satellite-images \
  --event-pattern '{
    "source": ["aws.s3"],
    "detail-type": ["Object Created"],
    "detail": {
      "bucket": {
        "name": ["satellite-images-bucket"]
      }
    }
  }'

# Add ECS RunTask as target
aws events put-targets \
  --rule process-satellite-images \
  --targets '[{
    "Id": "1",
    "Arn": "arn:aws:ecs:us-east-1:<account>:cluster/parcel-ai-cluster",
    "RoleArn": "arn:aws:iam::<account>:role/ecsEventsRole",
    "EcsParameters": {
      "TaskDefinitionArn": "arn:aws:ecs:us-east-1:<account>:task-definition/parcel-ai-task",
      "TaskCount": 1,
      "LaunchType": "FARGATE",
      "NetworkConfiguration": {
        "awsvpcConfiguration": {
          "Subnets": ["subnet-xxx"],
          "SecurityGroups": ["sg-xxx"],
          "AssignPublicIp": "ENABLED"
        }
      },
      "PlatformVersion": "LATEST"
    }
  }]'
```

### Cost Estimate
- **Per task run**: ~$0.05-0.10 (depending on duration)
- **100 runs/month**: ~$5-10/month
- **1000 runs/month**: ~$50-100/month

---

## Option 3: AWS Lambda + Container Image

**Best for:** Sporadic usage, <10 requests/day, infrequent processing

### Pros
- ✅ **Cheapest for low usage** - pay per invocation
- ✅ **Auto-scales** - handles bursts automatically
- ✅ **Zero maintenance** - fully serverless
- ✅ **10GB container support** - can fit all models

### Cons
- ⚠️ **Cold starts** - 10-30s first invocation
- ⚠️ **15 min max timeout** - might be tight for large batches
- ⚠️ **10GB memory max** - tight for all 3 models
- ⚠️ **Model loading overhead** - downloads on every cold start

### Cost Estimate
- **Light usage** (<100 invocations/month): ~$1-5/month
- **Moderate** (100-500/month): ~$10-20/month

### Deployment

```bash
# Create Lambda function from container
aws lambda create-function \
  --function-name parcel-ai-json \
  --package-type Image \
  --code ImageUri=<account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest \
  --role arn:aws:iam::<account>:role/lambda-execution-role \
  --timeout 900 \
  --memory-size 10240 \
  --ephemeral-storage Size=10240
```

---

## Option 4: EC2 Spot Instances

**Best for:** High-volume processing, need GPU, cost-sensitive

### Pros
- ✅ **70-90% cheaper** than on-demand EC2
- ✅ **Full control** - any instance type
- ✅ **GPU support** - g4dn.xlarge for faster inference
- ✅ **Persistent** - can run 24/7

### Cons
- ⚠️ **Can be interrupted** - 2-min warning
- ⚠️ **Manual management** - need to handle instance lifecycle
- ⚠️ **Less auto-scaling** - need custom scripts

### Cost Estimate
- **CPU-only** (t3.xlarge spot): ~$20-30/month (24/7)
- **GPU** (g4dn.xlarge spot): ~$50-70/month (24/7)

### Deployment

```bash
# Launch spot instance with Docker
aws ec2 run-instances \
  --image-id ami-xxxxxx \
  --instance-type t3.xlarge \
  --instance-market-options MarketType=spot \
  --user-data '#!/bin/bash
    yum update -y
    yum install -y docker
    systemctl start docker
    $(aws ecr get-login --no-include-email)
    docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest
    docker run -p 8000:8000 <account>.dkr.ecr.us-east-1.amazonaws.com/parcel-ai-json:latest'
```

---

## Comparison Matrix

| Option | Cost (Low Usage) | Cost (High Usage) | Cold Start | Scalability | Management | Best For |
|--------|-----------------|-------------------|------------|-------------|------------|----------|
| **ECS Fargate Spot** | $5-10/mo | $80-100/mo | None | Auto | Low | Regular/variable traffic |
| **EventBridge + ECS** | $5-10/mo | $50-100/mo | 30-60s | Auto | Low | Event-driven batches |
| **Lambda Container** | $1-5/mo | $50-100/mo | 10-30s | Auto | None | Sporadic usage |
| **EC2 Spot** | $20-30/mo | $50-70/mo | None | Manual | High | 24/7 high-volume |

---

## Recommendation

**Start with ECS Fargate Spot** for these reasons:

1. ✅ Your Docker container is ready to deploy
2. ✅ No cold starts - containers stay warm
3. ✅ Auto-scaling handles traffic bursts
4. ✅ Cost-effective for regular usage
5. ✅ Easy to add EventBridge triggers later if needed

**Migration path:**
1. **Phase 1**: Deploy to ECS Fargate Spot as persistent service
2. **Phase 2**: If usage is truly event-driven, switch to EventBridge + RunTask
3. **Phase 3**: If volume grows significantly, consider EC2 Spot with GPU

---

## Next Steps for AWS Deployment

1. ✅ Complete Docker migration (current plan)
2. ⬜ Create AWS ECR repository
3. ⬜ Set up ECS cluster and task definition
4. ⬜ Configure EFS for model caching
5. ⬜ Deploy to ECS Fargate Spot
6. ⬜ Set up CloudWatch logging and monitoring
7. ⬜ Configure Application Load Balancer (if needed)
8. ⬜ Test end-to-end image processing
9. ⬜ Monitor costs and optimize
