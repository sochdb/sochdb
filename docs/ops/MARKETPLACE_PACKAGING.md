# SochDB Marketplace Packaging

This document describes the marketplace packaging for SochDB, ensuring compliance with cloud marketplace requirements (AWS, Azure, GCP).

## Overview

| Marketplace | Package Format | Certification |
|-------------|----------------|---------------|
| AWS | AMI + Helm | AWS Partner |
| Azure | Azure Image + Helm | Azure Certified |
| GCP | Container Image + Helm | GCP Ready |
| Kubernetes | Helm Chart + OCI | CNCF Compatible |

## SBOM (Software Bill of Materials)

### Generation

```bash
# Generate SBOM using cargo-sbom
cargo sbom --output-format spdx > sbom.spdx.json

# Or using syft for container image
syft sochdb:latest -o spdx-json > container-sbom.spdx.json
```

### Contents

- All Rust crate dependencies with versions
- License information (SPDX identifiers)
- Cryptographic hashes (SHA256)
- Vulnerability scanning results

### CycloneDX Format

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:...",
  "version": 1,
  "metadata": {
    "component": {
      "type": "application",
      "name": "sochdb",
      "version": "0.4.0"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "tokio",
      "version": "1.44.0",
      "purl": "pkg:cargo/tokio@1.44.0",
      "licenses": [{ "id": "MIT" }]
    }
  ]
}
```

## Image Signing

### Sigstore/Cosign

```bash
# Sign the container image
cosign sign --key cosign.key ghcr.io/sushanthpy/sochdb:v0.4.0

# Verify signature
cosign verify --key cosign.pub ghcr.io/sushanthpy/sochdb:v0.4.0

# Sign with OIDC (keyless)
cosign sign --oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/sushanthpy/sochdb:v0.4.0
```

### Attestation

```bash
# Generate SLSA provenance attestation
cosign attest --predicate slsa-provenance.json \
  --type slsaprovenance \
  ghcr.io/sushanthpy/sochdb:v0.4.0
```

## Reproducible Builds

### Rust Configuration

```toml
# .cargo/config.toml
[build]
rustflags = [
  "-C", "link-arg=-Wl,--build-id=none",
  "-C", "codegen-units=1",
]

[env]
SOURCE_DATE_EPOCH = "1704067200"  # Fixed timestamp
```

### Docker Build

```dockerfile
# syntax=docker/dockerfile:1
FROM rust:1.82-bookworm AS builder

# Reproducible build settings
ENV SOURCE_DATE_EPOCH=1704067200
ENV RUSTFLAGS="-C link-arg=-Wl,--build-id=none"

# Copy source with fixed timestamps
COPY --chown=rust:rust . /app
RUN find /app -exec touch -d "@${SOURCE_DATE_EPOCH}" {} +

WORKDIR /app
RUN cargo build --release --locked

# Verify reproducibility
RUN sha256sum target/release/sochdb-server
```

### Build Verification

```bash
# Build twice and compare
docker build --no-cache -t build1 .
docker build --no-cache -t build2 .

# Extract binaries
docker cp $(docker create build1):/app/target/release/sochdb-server ./binary1
docker cp $(docker create build2):/app/target/release/sochdb-server ./binary2

# Compare
sha256sum binary1 binary2
diff binary1 binary2
```

## Container Image Requirements

### Multi-Architecture

```yaml
# Build matrix
platforms:
  - linux/amd64
  - linux/arm64
```

```bash
# Multi-arch build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t ghcr.io/sushanthpy/sochdb:v0.4.0 .
```

### Image Labels (OCI)

```dockerfile
LABEL org.opencontainers.image.title="SochDB"
LABEL org.opencontainers.image.description="LLM-Optimized Embedded Database"
LABEL org.opencontainers.image.version="0.4.0"
LABEL org.opencontainers.image.vendor="Sushanth Reddy Vanagala"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"
LABEL org.opencontainers.image.source="https://github.com/sushanthpy/sochdb"
LABEL org.opencontainers.image.documentation="https://sochdb.io/docs"
```

### Security Hardening

```dockerfile
# Run as non-root
USER 1000:1000

# Read-only root filesystem
ENV SOCHDB_DATA_DIR=/data
VOLUME ["/data"]

# No new privileges
# (enforced via SecurityContext in K8s)
```

## Helm Chart Publishing

### Chart Repository

```bash
# Package chart
helm package deploy/helm/sochdb

# Push to OCI registry
helm push sochdb-0.4.0.tgz oci://ghcr.io/sushanthpy/charts

# Or to Helm repository
helm repo index . --url https://charts.sochdb.io
```

### Chart Signing

```bash
# Sign chart with GPG
helm package --sign --key 'sochdb@sochdb.io' deploy/helm/sochdb

# Verify
helm verify sochdb-0.4.0.tgz
```

### ArtifactHub Metadata

```yaml
# artifacthub-repo.yml
repositoryID: sochdb
owners:
  - name: Sushanth Reddy Vanagala
    email: sushanth@sochdb.io
```

## AWS Marketplace

### AMI Requirements

1. **No default credentials**
2. **SSH disabled or key-only**
3. **Automated security updates**
4. **CloudWatch integration**

### CloudFormation Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: SochDB Database Stack

Parameters:
  InstanceType:
    Type: String
    Default: m5.large
    AllowedValues:
      - m5.large
      - m5.xlarge
      - m5.2xlarge

Resources:
  SochDBInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: !Ref SochDBAmI
      InstanceType: !Ref InstanceType
      SecurityGroups:
        - !Ref SochDBSecurityGroup

  SochDBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: SochDB Security Group
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 50051
          ToPort: 50051
          CidrIp: 10.0.0.0/8
```

## Azure Marketplace

### Managed Application

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmSize": {
      "type": "string",
      "defaultValue": "Standard_D4s_v3"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerService/managedClusters",
      "apiVersion": "2023-07-01",
      "name": "sochdb-cluster",
      "properties": {
        "agentPoolProfiles": [
          {
            "name": "sochdbpool",
            "count": 3,
            "vmSize": "[parameters('vmSize')]"
          }
        ]
      }
    }
  ]
}
```

## GCP Marketplace

### Deployment Manager

```yaml
resources:
- name: sochdb-cluster
  type: gcp-types/container-v1:projects.zones.clusters
  properties:
    zone: us-central1-a
    cluster:
      name: sochdb
      nodePools:
      - name: sochdb-pool
        config:
          machineType: e2-standard-4
        initialNodeCount: 3
```

## Vulnerability Scanning

### Trivy Scan

```bash
# Scan container image
trivy image --severity HIGH,CRITICAL ghcr.io/sushanthpy/sochdb:v0.4.0

# Generate SARIF for GitHub Security
trivy image --format sarif --output trivy-results.sarif \
  ghcr.io/sushanthpy/sochdb:v0.4.0
```

### Grype Scan

```bash
# Scan with Grype
grype ghcr.io/sushanthpy/sochdb:v0.4.0

# Match against SBOM
grype sbom:sbom.spdx.json
```

## CI/CD Pipeline

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build
        run: cargo build --release --locked
      
      - name: Generate SBOM
        run: cargo sbom --output-format spdx > sbom.spdx.json
      
      - name: Build Container
        run: docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/sushanthpy/sochdb:${{ github.ref_name }} .
      
      - name: Scan
        run: trivy image ghcr.io/sushanthpy/sochdb:${{ github.ref_name }}
      
      - name: Sign
        run: cosign sign ghcr.io/sushanthpy/sochdb:${{ github.ref_name }}
      
      - name: Push
        run: docker push ghcr.io/sushanthpy/sochdb:${{ github.ref_name }}
```

## Compliance Checklist

- [ ] SBOM generated (SPDX/CycloneDX)
- [ ] Container image signed (Cosign)
- [ ] Vulnerability scan passed (Trivy/Grype)
- [ ] Reproducible build verified
- [ ] Multi-architecture images built
- [ ] Helm chart signed
- [ ] License compliance verified
- [ ] No default credentials
- [ ] Security hardening applied
- [ ] Documentation complete
