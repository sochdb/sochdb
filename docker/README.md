# SochDB Docker Setup

Production-ready Docker configuration for SochDB gRPC server.

## � Public Docker Image

**Available on Docker Hub:** [`sushanth53/sochdb`](https://hub.docker.com/r/sushanth53/sochdb)

```bash
# Pull and run (easiest way)
docker pull sushanth53/sochdb:latest
docker run -d -p 50051:50051 sushanth53/sochdb:latest

# Or with docker-compose
docker compose up -d
```

## 🚀 Quick Start

### Using Pre-built Image (Recommended)

### Building Locally

```bash
# Build from source (from sochdb root)
cd docker
docker build -t sushanth53/sochdb
docker pull sushanth53/sochdb:latest

# Run
docker run -d \
  --name sochdb \
  -p 50051:50051 \
  -v sochdb-data:/var/lib/sochdb \
  sushanth53/sochdb:latest
```

### Building Locally

# Run the container
docker run -d \
  --name sochdb \
  -p 50051:50051 \
  -v sochdb-data:/var/lib/sochdb \
  sochdb/sochdb-grpc:latest
```

### Docker Compose (Recommended)

```bash
# Start SochDB
docker compose up -d

# View logs
docker compose logs -f sochdb

# Stop
docker compose down
```

## 📦 AvaiTags | Size | Description |
|-------|------|------|-------------|
| `sushanth53/sochdb` | `latest`, `0.4.9` | 159MB (34MB compressed) | Debian-based, production-ready |
| `sushanth53/sochdb` | `slim` | ~25MB | Alpine-based, minimal (coming soon) |

### Image Tags

- `latest` - Latest stable release
- `0.4.9` - Specific version
- `slim` - Minimal Alpine-based image (coming soon)
| `sochdb/sochdb-grpc:latest` | ~50MB | Debian-based, stable |
| `sochdb/sochdb-grpc:slim` | ~25MB | Alpine-based, minimal |

### Build Slim Image

```bash
docker build -f Dockerfile.slim -t sochdb/sochdb-grpc:slim ..
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (debug, info, warn, error) |
| `SOCHDB_DATA_DIR` | `/var/lib/sochdb` | Data directory |

### Command Line Options

```bash
docker run sochdb/sochdb-grpc:latest --help

# Options:
#   --host <HOST>    Host address to bind to [default: 0.0.0.0]
#   --port <PORT>    Port to listen on [default: 50051]
#   --debug          Enable debug logging
```

## 🏗️ Deployment Profiles

### Development

```bash
# Start with debug logging and local volume
docker compose --profile dev up -d
```

### With gRPC-Web (Browser Support)

```bash
# Start with Envoy proxy for gRPC-Web
docker compose --profile web up -d

# Access via HTTP: http://localhost:8080
```

### With Monitoring

```bash
# Start with Prometheus + Grafana
docker compose --profile monitoring up -d

# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/sochdb)
```

### Full Stack

```bash
# All profiles
docker compose --profile dev --profile web --profile monitoring up -d
```

## 🚀 Production Deployment

### Using Production Compose

```bash
# Set environment variables
export DOMAIN=sochdb.example.com
export ACME_EMAIL=admin@example.com
export GRAFANA_PASSWORD=secure-password

# Deploy
docker compose -f docker-compose.production.yml up -d
```

### Features

- **High Availability**: 3 replicas with automatic failover
- **Load Balancing**: Traefik with health checks
- **TLS**: Automatic Let's Encrypt certificates
- **Monitoring**: Prometheus + Grafana
- **Rolling Updates**: Zero-downtime deployments

## 🔌 Connecting Clients

### Python SDK

```python
from sochdb import SochDBClient

# Connect to Docker container
client = SochDBClient("localhost:50051")

# Use the client
client.put_kv("key", b"value")
value = client.get_kv("key")
```

### Go Client

```go
import "github.com/sochdb/sochdb-go"

client, err := sochdb.NewClient("localhost:50051")
if err != nil {
    log.Fatal(err)
}
defer client.Close()
```

### gRPC-Web (JavaScript)

```javascript
import { SochDBClient } from '@sochdb/client-web';

// Connect via Envoy proxy
const client = new SochDBClient('http://localhost:8080');
```

## 📊 Health Checks

### gRPC Health Check

```bash
# Install grpc_health_probe
go install github.com/grpc-ecosystem/grpc-health-probe@latest

# Check health
grpc_health_probe -addr=localhost:50051
```

### HTTP Health Check (via Envoy)

```bash
curl http://localhost:9901/ready
```

## 📁 Volume Mounts

| Mount | Purpose |
|-------|---------|
| `/var/lib/sochdb` | Persistent data storage |
| `/etc/sochdb/config.toml` | Configuration file |

### Backup Data

```bash
# Create backup
docker run --rm \
  -v sochdb-data:/data:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/sochdb-backup.tar.gz -C /data .

# Restore backup
docker run --rm \
  -v sochdb-data:/data \
  -v $(pwd):/backup:ro \
  alpine tar xzf /backup/sochdb-backup.tar.gz -C /data
```

## 🔒 Security

### Non-Root User

The container runs as non-root user `sochdb` (UID 1000).

### Network Isolation

```bash
# Create isolated network
docker network create --driver bridge sochdb-isolated

# Run with isolated network
docker run -d \
  --network sochdb-isolated \
  --name sochdb \
  sochdb/sochdb-grpc:latest
```

### TLS Configuration

For production TLS, use Traefik (see `docker-compose.production.yml`) or mount certificates:

```bash
docker run -d \
  --name sochdb \
  -p 50051:50051 \
  -v ./certs:/etc/sochdb/certs:ro \
  -e SOCHDB_TLS_CERT=/etc/sochdb/certs/server.crt \
  -e SOCHDB_TLS_KEY=/etc/sochdb/certs/server.key \
  sochdb/sochdb-grpc:latest
```

## 📈 Monitoring

### Prometheus Metrics

The server exposes metrics at `:9100/metrics`:

```bash
curl http://localhost:9100/metrics
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `sochdb_requests_total` | Total gRPC requests |
| `sochdb_request_duration_seconds` | Request latency histogram |
| `sochdb_active_connections` | Active client connections |
| `sochdb_vector_operations_total` | Vector index operations |
| `sochdb_memory_usage_bytes` | Memory usage |

### Grafana Dashboards

Pre-configured dashboards are available in `grafana/provisioning/dashboards/`.

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs sochdb

# Check if port is in use
lsof -i :50051
```

### Connection Refused

```bash
# Ensure container is running
docker ps | grep sochdb

# Test connectivity
grpc_health_probe -addr=localhost:50051
```

### Out of Memory

```bash
# Increase memory limit
docker run -d \
  --memory=4g \
  --memory-swap=4g \
  sochdb/sochdb-grpc:latest
```

### Performance Tuning

```bash
# Increase file descriptors
docker run -d \
  --ulimit nofile=65536:65536 \
  sochdb/sochdb-grpc:latest
```

## 📄 License

Apache License 2.0 - See [LICENSE](../sochdb/LICENSE)

## 🔗 Links

- [SochDB Documentation](https://sochdb.dev/docs)
- [gRPC API Reference](https://sochdb.dev/docs/api-reference)
- [Python SDK](https://github.com/sochdb/sochdb-python-sdk)
- [GitHub Repository](https://github.com/sochdb/sochdb)
