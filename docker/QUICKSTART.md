# SochDB Docker - Quick Start

## 📦 Docker Hub

**Image:** [`sushanth53/sochdb`](https://hub.docker.com/r/sushanth53/sochdb)

## Quick Start

```bash
# Pull and run
docker pull sushanth53/sochdb:latest
docker run -d -p 50051:50051 sushanth53/sochdb:latest

# Or use docker-compose
docker compose up -d
```

## Available Profiles

- **Default**: Just SochDB gRPC server
- **dev**: Development mode with debug logging
- **web**: Adds Envoy for gRPC-Web browser support
- **monitoring**: Adds Prometheus + Grafana

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Standard Debian-based image (~50MB) |
| `Dockerfile.slim` | Minimal Alpine image (~25MB) |
| `docker-compose.yml` | Development compose |
| `docker-compose.production.yml` | Production with HA |
| `envoy.yaml` | gRPC-Web proxy config |
| `prometheus.yml` | Metrics collection |
| `Makefile` | Convenience commands |

## Endpoints

| Service | Port | URL |
|---------|------|-----|
| gRPC | 50051 | `grpc://localhost:50051` |
| gRPC-Web | 8080 | `http://localhost:8080` |
| Prometheus | 9090 | `http://localhost:9090` |
| Grafana | 3000 | `http://localhost:3000` |

See [README.md](README.md) for full documentation.
