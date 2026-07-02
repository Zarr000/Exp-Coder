# Expera AI Deployment Guide

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Docker (for GPU support)
- NVIDIA GPU with CUDA 12.1+

## Quick Start

### 1. Local Development

```bash
# Build the image
docker build -f deploy/local/Dockerfile -t expera-ai:latest .

# Run with Docker Compose
cd deploy/local
docker-compose up -d
```

### 2. VPS Deployment

```bash
# Build and start
cd deploy/vps
docker-compose up -d --build

# Check status
docker-compose ps
docker-compose logs -f
```

### 3. Cloud Deployment

```bash
# Set environment variables
export GRAFANA_PASSWORD=your_secure_password

# Deploy
cd deploy/cloud
docker-compose up -d --build

# Verify
curl http://localhost:8000/health
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| MODEL_PATH | Path to model checkpoints | `/app/checkpoints/expera_coder_120m` |
| DEVICE | Compute device | `cuda` |
| CUDA_VISIBLE_DEVICES | GPU device ID | `0` |
| LOG_LEVEL | Logging level | `info` |
| METRICS_ENABLED | Enable Prometheus metrics | `true` |

## Volumes

| Volume | Description | Path |
|--------|-------------|------|
| checkpoints | Model checkpoint files | `/app/checkpoints` |
| tokenizer | Tokenizer files | `/app/tokenizer` |
| model-cache | HuggingFace cache | `/app/cache` |

## GPU Configuration

### NVIDIA Docker Setup

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

## Health Checks

- API Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- GPU Metrics: `http://localhost:8000/metrics/gpu`

## Monitoring

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

### Model Not Found

```bash
# Verify checkpoints directory
ls -la checkpoints/

# Mount correct path in docker-compose.yml
```

### Out of Memory

- Use smaller model (120M or 350M)
- Enable quantization in config
- Reduce batch size

## Production Checklist

- [ ] Set secure passwords
- [ ] Configure SSL/TLS
- [ ] Set up monitoring
- [ ] Configure backup volumes
- [ ] Set up log rotation
- [ ] Configure resource limits
- [ ] Set up health checks
- [ ] Configure auto-restart

## Scaling

### Horizontal Scaling

Edit `docker-compose.yml`:
```yaml
deploy:
  replicas: 2
```

### Vertical Scaling

Increase GPU memory in `deploy/resources/reservations/devices`.

## Security

- Run as non-root user
- Use read-only volumes where possible
- Configured with minimum required capabilities
- Use Docker secrets for sensitive data