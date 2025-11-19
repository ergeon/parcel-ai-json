#!/bin/bash
# Deploy parcel-ai-json to AWS EC2 GPU instance
# Usage: ./scripts/deploy_to_ec2.sh [instance-ip]

set -e  # Exit on error

# Configuration
INSTANCE_IP="${1:-44.254.121.125}"
INSTANCE_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ai-gpu-instance-key.pem"
PROJECT_NAME="parcel-ai-json"
DOCKER_IMAGE="parcel-ai-json:latest"
API_PORT="8000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Deploying parcel-ai-json to EC2${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Instance: ${GREEN}${INSTANCE_IP}${NC}"
echo -e "User: ${GREEN}${INSTANCE_USER}${NC}"
echo -e "SSH Key: ${GREEN}${SSH_KEY}${NC}"
echo ""

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at ${SSH_KEY}${NC}"
    exit 1
fi

# Check if instance is reachable
echo -e "${YELLOW}[1/7] Checking instance connectivity...${NC}"
if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "echo 'Connected successfully'" &>/dev/null; then
    echo -e "${RED}Error: Cannot connect to instance. Make sure it's running and security groups allow SSH.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Instance is reachable${NC}"
echo ""

# Install Docker and NVIDIA Container Toolkit on EC2 instance
echo -e "${YELLOW}[2/7] Installing Docker and NVIDIA Container Toolkit...${NC}"
ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
set -e

# Update package list
echo "Updating package list..."
sudo apt-get update -qq

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y -qq ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add user to docker group
    sudo usermod -aG docker $USER
    echo "Docker installed successfully!"
else
    echo "Docker already installed"
fi

# Install NVIDIA Container Toolkit for GPU support
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    # Use generic deb repository (works for Ubuntu 24.04+)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null

    # Add generic repository for all Debian/Ubuntu distributions
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed successfully!"
else
    echo "NVIDIA Container Toolkit already installed"
fi

# Verify GPU is accessible
echo "Verifying GPU access..."
nvidia-smi || echo "Warning: nvidia-smi failed, but continuing..."
ENDSSH
echo -e "${GREEN}✓ Docker and NVIDIA toolkit installed${NC}"
echo ""

# Create project directory and copy files
echo -e "${YELLOW}[3/7] Creating project directory and copying files...${NC}"
ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "mkdir -p ~/${PROJECT_NAME}"

# Sync project files (excluding unnecessary directories)
echo "Syncing project files..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'htmlcov' \
    --exclude 'output/examples' \
    --exclude 'output/test_datasets/satellite_images' \
    --exclude 'output/test_datasets/results' \
    --exclude '.DS_Store' \
    --exclude '=0.40.0' \
    ./ "${INSTANCE_USER}@${INSTANCE_IP}:~/${PROJECT_NAME}/"

echo -e "${GREEN}✓ Project files synced${NC}"
echo ""

# Build Docker image on EC2 instance
echo -e "${YELLOW}[4/7] Building Docker image on EC2 instance...${NC}"
ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << ENDSSH
set -e
cd ~/${PROJECT_NAME}
echo "Building Docker image..."
docker build -f docker/Dockerfile -t ${DOCKER_IMAGE} .
echo "Docker image built successfully!"
ENDSSH
echo -e "${GREEN}✓ Docker image built${NC}"
echo ""

# Stop existing container if running
echo -e "${YELLOW}[5/7] Stopping existing container (if any)...${NC}"
ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << ENDSSH
docker stop ${PROJECT_NAME} 2>/dev/null || true
docker rm ${PROJECT_NAME} 2>/dev/null || true
echo "Existing container stopped"
ENDSSH
echo -e "${GREEN}✓ Existing container stopped${NC}"
echo ""

# Run Docker container with GPU support
echo -e "${YELLOW}[6/7] Starting Docker container with GPU support...${NC}"
ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << ENDSSH
set -e
echo "Starting container with GPU support..."
docker run -d \
    --name ${PROJECT_NAME} \
    --gpus all \
    --restart unless-stopped \
    -p ${API_PORT}:8000 \
    -e PYTHONUNBUFFERED=1 \
    ${DOCKER_IMAGE}

echo "Container started successfully!"
echo ""
echo "Waiting for service to be ready..."
sleep 10

# Check container status
docker ps | grep ${PROJECT_NAME}
ENDSSH
echo -e "${GREEN}✓ Container started with GPU support${NC}"
echo ""

# Verify deployment
echo -e "${YELLOW}[7/7] Verifying deployment...${NC}"
echo "Checking API health..."
if ssh -i "$SSH_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "curl -s http://localhost:${API_PORT}/health" | grep -q "status"; then
    echo -e "${GREEN}✓ API is responding${NC}"
else
    echo -e "${RED}Warning: API health check failed. Check logs with:${NC}"
    echo -e "${RED}  ssh -i $SSH_KEY ${INSTANCE_USER}@${INSTANCE_IP} 'docker logs ${PROJECT_NAME}'${NC}"
fi
echo ""

# Print success message and instructions
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "API Endpoints:"
echo -e "  Health:  ${BLUE}http://${INSTANCE_IP}:${API_PORT}/health${NC}"
echo -e "  Docs:    ${BLUE}http://${INSTANCE_IP}:${API_PORT}/docs${NC}"
echo -e "  Detect:  ${BLUE}http://${INSTANCE_IP}:${API_PORT}/api/v1/detect${NC}"
echo ""
echo -e "Useful Commands:"
echo -e "  SSH:          ${BLUE}ssh -i $SSH_KEY ${INSTANCE_USER}@${INSTANCE_IP}${NC}"
echo -e "  View logs:    ${BLUE}ssh -i $SSH_KEY ${INSTANCE_USER}@${INSTANCE_IP} 'docker logs -f ${PROJECT_NAME}'${NC}"
echo -e "  Stop:         ${BLUE}ssh -i $SSH_KEY ${INSTANCE_USER}@${INSTANCE_IP} 'docker stop ${PROJECT_NAME}'${NC}"
echo -e "  Restart:      ${BLUE}ssh -i $SSH_KEY ${INSTANCE_USER}@${INSTANCE_IP} 'docker restart ${PROJECT_NAME}'${NC}"
echo ""
echo -e "Test with:"
echo -e "  ${BLUE}curl http://${INSTANCE_IP}:${API_PORT}/health${NC}"
echo ""
