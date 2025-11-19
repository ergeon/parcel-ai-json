# EC2 GPU Deployment Guide

This guide covers deploying the parcel-ai-json REST API to AWS EC2 GPU instances (g5.xlarge, g4dn.xlarge, etc.).

## Prerequisites

- AWS EC2 GPU instance running Ubuntu (g5.xlarge recommended)
- SSH key pair configured (`~/.ssh/ai-gpu-instance-key.pem`)
- Security group allowing:
  - SSH (port 22) from your IP
  - HTTP (port 8000) for API access
- Instance must be in "Running" state

## Quick Deployment

### Option 1: Using Make (Recommended)

Deploy to your EC2 instance with a single command:

```bash
# Deploy to default instance (44.254.121.125)
make deploy-ec2-default

# Or deploy to a specific IP
make deploy-ec2 EC2_IP=44.254.121.125
```

### Option 2: Using Deployment Script Directly

```bash
./scripts/deploy_to_ec2.sh 44.254.121.125
```

## What the Deployment Does

The deployment script automatically:

1. **Verifies connectivity** - Checks if the EC2 instance is reachable
2. **Installs Docker** - Sets up Docker CE and Docker Compose
3. **Installs NVIDIA Container Toolkit** - Enables GPU support in Docker
4. **Syncs project files** - Copies your local project to EC2 (excluding venv, .git, etc.)
5. **Builds Docker image** - Builds the production image on EC2
6. **Stops old container** - Removes any existing deployment
7. **Starts new container** - Runs with GPU support (`--gpus all`)
8. **Verifies deployment** - Checks health endpoint

## Post-Deployment Commands

### SSH into Instance

```bash
# Using make
make ec2-ssh EC2_IP=44.254.121.125

# Or directly
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125
```

### View Logs

```bash
# Using make
make ec2-logs EC2_IP=44.254.121.125

# Or directly
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125 'docker logs -f parcel-ai-json'
```

### Check Service Status

```bash
# Health endpoint
curl http://44.254.121.125:8000/health

# API documentation
open http://44.254.121.125:8000/docs
```

### Restart Container

```bash
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125 'docker restart parcel-ai-json'
```

### Stop Container

```bash
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125 'docker stop parcel-ai-json'
```

## API Endpoints

Once deployed, the API is available at:

- **Base URL**: `http://<instance-ip>:8000`
- **Health Check**: `http://<instance-ip>:8000/health`
- **API Docs**: `http://<instance-ip>:8000/docs`
- **Unified Detection**: `POST http://<instance-ip>:8000/api/v1/detect`
- **Vehicle Detection**: `POST http://<instance-ip>:8000/api/v1/detect/vehicles`
- **Pool Detection**: `POST http://<instance-ip>:8000/api/v1/detect/pools`
- **Tree Detection**: `POST http://<instance-ip>:8000/api/v1/detect/trees`

## Testing the Deployment

### Test Health Endpoint

```bash
curl http://44.254.121.125:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "device": "cuda"
}
```

### Test Detection Endpoint

```bash
curl -X POST http://44.254.121.125:8000/api/v1/detect \
  -F "file=@test_image.jpg" \
  -F "center_lat=37.7749" \
  -F "center_lon=-122.4194" \
  -F "zoom_level=20" \
  -F "include_trees=true" \
  | jq .
```

## GPU Support

The deployment automatically configures GPU support:

- **NVIDIA Container Toolkit** installed on EC2
- **Container launched with** `--gpus all` flag
- **Models automatically use CUDA** (verified via `/health` endpoint)

To verify GPU is working:

```bash
# SSH into instance
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125

# Check GPU status
nvidia-smi

# Check Docker container has GPU access
docker exec parcel-ai-json nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

## Performance Expectations

On **g5.xlarge** instance (1x NVIDIA A10G GPU):

- **Vehicle detection**: ~0.5-1s per image
- **Pool detection**: ~0.5-1s per image
- **Tree detection**: ~0.5-1s per image (DeepForest)
- **SAM segmentation**: ~2-4s per image
- **Unified detection** (all features): ~3-6s per image

Compare to **CPU-only**:
- Vehicle/Pool/Amenity: ~7.5s each
- SAM: ~25-35s
- Unified: ~40-60s

**GPU provides 10-15x speedup!**

## Troubleshooting

### Instance Not Reachable

1. Check instance is in "Running" state
2. Verify security group allows SSH (port 22) from your IP
3. Check VPC and subnet configuration
4. Try: `ssh -v -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@<instance-ip>`

### Docker Build Fails

1. Check instance has enough disk space: `df -h`
2. Check instance memory: `free -h`
3. View build logs: `docker logs parcel-ai-json`

### API Not Responding

1. Check container is running: `docker ps`
2. View logs: `docker logs parcel-ai-json`
3. Check port is open: `netstat -tlnp | grep 8000`
4. Verify health: `curl localhost:8000/health` (from within instance)

### GPU Not Detected

1. Verify GPU is available: `nvidia-smi`
2. Check NVIDIA Container Toolkit: `nvidia-container-toolkit --version`
3. Restart Docker: `sudo systemctl restart docker`
4. Check container GPU access: `docker exec parcel-ai-json nvidia-smi`

## Security Best Practices

### 1. Restrict API Access

Update security group to only allow API access (port 8000) from trusted IPs:

```bash
# Allow only from your IP
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 8000 \
  --cidr <your-ip>/32
```

### 2. Enable HTTPS

For production, use a reverse proxy (nginx) with SSL:

```bash
# Install nginx on EC2
sudo apt-get install nginx certbot python3-certbot-nginx

# Configure SSL
sudo certbot --nginx -d your-domain.com
```

### 3. Add Authentication

Add API key authentication to FastAPI endpoints (see FastAPI docs).

## Production Optimizations

### 1. Use Docker Compose

For more complex deployments, use the production compose file:

```bash
# On EC2 instance
cd ~/parcel-ai-json
docker-compose -f docker/docker-compose.prod.yml up -d
```

### 2. Enable Monitoring

Add CloudWatch monitoring:

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb
```

### 3. Set Up Log Rotation

The production docker-compose file already configures log rotation:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
```

### 4. Auto-Restart on Failure

Container is configured with `--restart unless-stopped`, so it will automatically restart on crashes or instance reboots.

## Cost Optimization

### g5.xlarge Pricing (us-west-2)

- **On-Demand**: ~$1.006/hour (~$725/month)
- **Spot**: ~$0.302/hour (~$218/month, 70% savings)

### Recommendations

1. **Use Spot Instances** for development/testing
2. **Auto-stop when idle** - Schedule shutdown during off-hours
3. **Right-size instance** - Monitor GPU utilization with `nvidia-smi`

### Auto-Stop Script

Save this on EC2 to auto-stop when idle:

```bash
#!/bin/bash
# /home/ubuntu/auto-stop-idle.sh
IDLE_THRESHOLD=3600  # Stop after 1 hour of no requests

last_request=$(docker logs parcel-ai-json 2>&1 | grep "POST /api/v1/detect" | tail -1 | cut -d' ' -f1)
if [ -z "$last_request" ]; then
    echo "No requests found, shutting down..."
    sudo shutdown -h now
fi
```

Add to crontab: `0 * * * * /home/ubuntu/auto-stop-idle.sh`

## Updating the Deployment

To deploy code changes:

```bash
# Option 1: Re-run deployment (rebuilds everything)
make deploy-ec2-default

# Option 2: SSH and rebuild manually
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125
cd ~/parcel-ai-json
git pull  # If using git
docker build -f docker/Dockerfile -t parcel-ai-json:latest .
docker stop parcel-ai-json
docker rm parcel-ai-json
docker run -d --name parcel-ai-json --gpus all -p 8000:8000 parcel-ai-json:latest
```

## Cleanup

To completely remove the deployment:

```bash
# SSH into instance
ssh -i ~/.ssh/ai-gpu-instance-key.pem ubuntu@44.254.121.125

# Stop and remove container
docker stop parcel-ai-json
docker rm parcel-ai-json

# Remove image
docker rmi parcel-ai-json:latest

# Remove project files
rm -rf ~/parcel-ai-json
```

Then terminate the EC2 instance via AWS Console.

## Support

For issues or questions:

- Check logs: `make ec2-logs EC2_IP=<instance-ip>`
- Review troubleshooting section above
- Check project documentation: `README.md`, `docs/ARCHITECTURE.md`
- Create GitHub issue: https://github.com/ergeon/parcel-ai-json/issues
