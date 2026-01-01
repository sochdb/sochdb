# How to Deploy to Production

> Production deployment checklist and best practices.

---

## Problem

You're ready to deploy ToonDB to production and need to ensure reliability, performance, and security.

---

## Solution

### Pre-Deployment Checklist

```markdown
## Infrastructure
- [ ] Adequate disk space (2x expected data size for compaction headroom)
- [ ] SSD storage (NVMe preferred for write-heavy workloads)
- [ ] Sufficient RAM (recommend: 10% of data size for caching)
- [ ] Network: Unix sockets for same-host, TCP for remote

## Configuration
- [ ] Durability settings appropriate for use case
- [ ] Connection limits configured
- [ ] Logging configured (see logging.md)
- [ ] Backup strategy in place

## Security
- [ ] File permissions restricted (700 for data directory)
- [ ] Unix socket permissions set
- [ ] No sensitive data in logs

## Monitoring
- [ ] Health check endpoint configured
- [ ] Metrics collection enabled
- [ ] Alerting rules defined
```

---

### 1. Production Configuration

Create `toondb-server-config.toml`:

```toml
[server]
# IPC settings
socket_path = "/var/run/toondb/toondb.sock"
socket_permissions = "0660"
max_connections = 1000
connection_timeout_secs = 30

# gRPC settings (if using remote access)
grpc_enabled = true
grpc_bind = "127.0.0.1:50051"  # Localhost only, use reverse proxy for external

[storage]
path = "/var/lib/toondb/data"
sync_mode = "normal"  # "full" for max durability, "off" for speed

# WAL settings
wal_size_mb = 64
checkpoint_interval_secs = 300

# Compaction
compaction_threads = 2
target_file_size_mb = 64

[index]
# HNSW settings
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50

[memory]
# Block cache size (bytes)
block_cache_size = 536870912  # 512 MB
# Write buffer size
write_buffer_size = 67108864  # 64 MB

[logging]
level = "info"
format = "json"
output = "file"
file_path = "/var/log/toondb/toondb.log"
max_size_mb = 100
max_files = 10
```

---

### 2. Systemd Service

Create `/etc/systemd/system/toondb.service`:

```ini
[Unit]
Description=ToonDB Database Server
After=network.target
Documentation=https://github.com/toondb/toondb

[Service]
Type=simple
User=toondb
Group=toondb
ExecStart=/usr/local/bin/toondb-server --config /etc/toondb/config.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/toondb /var/log/toondb /var/run/toondb

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

# Environment
Environment=RUST_LOG=toondb=info
Environment=RUST_BACKTRACE=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable toondb
sudo systemctl start toondb
```

---

### 3. Directory Structure

```bash
# Create directories
sudo mkdir -p /var/lib/toondb/data
sudo mkdir -p /var/log/toondb
sudo mkdir -p /var/run/toondb
sudo mkdir -p /etc/toondb

# Create service user
sudo useradd -r -s /bin/false toondb

# Set permissions
sudo chown -R toondb:toondb /var/lib/toondb
sudo chown -R toondb:toondb /var/log/toondb
sudo chown -R toondb:toondb /var/run/toondb
sudo chmod 700 /var/lib/toondb/data
```

---

### 4. Health Checks

#### Bash Script

```bash
#!/bin/bash
# /usr/local/bin/toondb-healthcheck

SOCKET="/var/run/toondb/toondb.sock"

if [ ! -S "$SOCKET" ]; then
    echo "CRITICAL: Socket not found"
    exit 2
fi

# Check if server responds
if timeout 5 toondb-client ping --socket "$SOCKET" > /dev/null 2>&1; then
    echo "OK: ToonDB is healthy"
    exit 0
else
    echo "CRITICAL: ToonDB not responding"
    exit 2
fi
```

#### Python Health Check

```python
from toondb import IpcClient
import sys

def health_check():
    try:
        client = IpcClient("/var/run/toondb/toondb.sock")
        stats = client.stats()
        
        # Check key metrics
        if stats.get("error_rate", 0) > 0.01:
            print(f"WARNING: High error rate: {stats['error_rate']}")
            return 1
        
        print(f"OK: connections={stats['connections_active']}")
        return 0
    except Exception as e:
        print(f"CRITICAL: {e}")
        return 2

sys.exit(health_check())
```

---

### 5. Backup Strategy

```bash
#!/bin/bash
# /usr/local/bin/toondb-backup

BACKUP_DIR="/var/backups/toondb"
DATA_DIR="/var/lib/toondb/data"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Trigger checkpoint before backup
toondb-client checkpoint --socket /var/run/toondb/toondb.sock

# Create backup (hot backup using hardlinks for SST files)
rsync -a --link-dest="$BACKUP_DIR/latest" \
    "$DATA_DIR/" "$BACKUP_DIR/backup_$DATE/"

# Update latest symlink
ln -sfn "$BACKUP_DIR/backup_$DATE" "$BACKUP_DIR/latest"

# Cleanup old backups
find "$BACKUP_DIR" -maxdepth 1 -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed: backup_$DATE"
```

Add to crontab:

```bash
# Daily backup at 3 AM
0 3 * * * /usr/local/bin/toondb-backup >> /var/log/toondb/backup.log 2>&1
```

---

### 6. Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.75-slim as builder
WORKDIR /build
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/toondb-server /usr/local/bin/
COPY --from=builder /build/target/release/toondb-client /usr/local/bin/

RUN useradd -r -s /bin/false toondb
USER toondb

VOLUME /data
EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/toondb-server"]
CMD ["--config", "/etc/toondb/config.toml"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  toondb:
    build: .
    volumes:
      - toondb_data:/data
      - ./config.toml:/etc/toondb/config.toml:ro
    ports:
      - "50051:50051"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "toondb-client", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  toondb_data:
```

---

## Discussion

### Durability vs Performance

| Setting | Durability | Write Latency | Use Case |
|---------|------------|---------------|----------|
| `sync_mode = "full"` | Maximum | ~10ms | Financial data |
| `sync_mode = "normal"` | Good | ~1ms | General use |
| `sync_mode = "off"` | Minimal | ~0.1ms | Caching, ephemeral |

### Sizing Guidelines

| Data Size | RAM | Disk | CPU |
|-----------|-----|------|-----|
| < 10 GB | 2 GB | 30 GB SSD | 2 cores |
| 10-100 GB | 8 GB | 300 GB NVMe | 4 cores |
| 100 GB - 1 TB | 32 GB | 3 TB NVMe | 8 cores |

---

## See Also

- [Logging Guide](/cookbook/logging) — Observability setup
- [Performance Guide](/concepts/performance) — Optimization tips
- [Architecture](/concepts/architecture) — System design

