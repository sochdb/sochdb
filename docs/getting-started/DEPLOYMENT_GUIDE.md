# SochDB Deployment Guide

Quick start guides for deploying SochDB on every supported platform.

## Table of Contents

1. [Local Docker](#1-local-docker)
2. [Docker Compose (Production)](#2-docker-compose-production)
3. [Helm on Kubernetes](#3-helm-on-kubernetes)
4. [Azure Kubernetes Service (AKS)](#4-azure-kubernetes-service-aks)
5. [Amazon EKS](#5-amazon-eks)
6. [Google Kubernetes Engine (GKE)](#6-google-kubernetes-engine-gke)
7. [Bare Metal / VM](#7-bare-metal--vm)

---

## 1. Local Docker

```bash
# Pull the image
docker pull ghcr.io/sochdb/sochdb-grpc:latest

# Run with persistent storage
docker run -d \
  --name sochdb \
  -p 50051:50051 \
  -p 9090:9090 \
  -v sochdb-data:/var/lib/sochdb \
  ghcr.io/sochdb/sochdb-grpc:latest

# Verify health
docker exec sochdb grpc_health_probe -addr=localhost:50051

# Connect with Python SDK
pip install sochdb-client
python3 -c "
from sochdb_sdk import SochDBClient
client = SochDBClient('localhost:50051')
print(client.vectors.search('my_collection', [0.1]*768, k=5))
"
```

### Build from source

```bash
cd docker/
docker build -t sochdb/sochdb-grpc:dev -f Dockerfile ..
docker run -d -p 50051:50051 sochdb/sochdb-grpc:dev
```

---

## 2. Docker Compose (Production)

Full stack with Prometheus, Grafana, and Traefik load balancer:

```bash
cd docker/
docker compose -f docker-compose.production.yml up -d

# Access points:
#   gRPC:      localhost:50051
#   Prometheus: localhost:9090
#   Grafana:    localhost:3000 (admin/admin)
#   Traefik:    localhost:8080
```

### Environment variables

```bash
# .env file
ACME_EMAIL=admin@yourdomain.com
DOMAIN=sochdb.yourdomain.com
SOCHDB_AUTH_ENABLED=true
SOCHDB_API_KEY=your-secret-key
```

---

## 3. Helm on Kubernetes

### Prerequisites
- Kubernetes 1.24+
- Helm 3.x
- kubectl configured

### Install

```bash
# Add the SochDB Helm repository (when published)
# helm repo add sochdb https://charts.sochdb.dev

# Or install from local chart
helm install sochdb deploy/helm/sochdb/ \
  --namespace sochdb \
  --create-namespace \
  --set persistence.size=100Gi

# Verify
kubectl get pods -n sochdb
kubectl get svc -n sochdb
```

### Customize

```bash
# Production values
helm install sochdb deploy/helm/sochdb/ \
  --namespace sochdb \
  --create-namespace \
  -f deploy/helm/sochdb/values.yaml \
  --set replicaCount=3 \
  --set resources.requests.memory=4Gi \
  --set resources.limits.memory=4Gi \
  --set persistence.size=200Gi \
  --set security.tls.enabled=true \
  --set security.auth.enabled=true \
  --set ingress.enabled=true \
  --set ingress.host=sochdb.yourdomain.com
```

### Monitoring stack

```bash
# Deploy Prometheus + Grafana for SochDB
kubectl apply -f deploy/k8s-monitoring/namespace.yaml
kubectl apply -f deploy/k8s-monitoring/prometheus.yaml
kubectl apply -f deploy/k8s-monitoring/grafana.yaml

# Port-forward Grafana
kubectl port-forward -n sochdb-monitoring svc/grafana 3000:3000
# Open http://localhost:3000 (admin/admin)
```

### Upgrade

```bash
helm upgrade sochdb deploy/helm/sochdb/ \
  --set image.tag=2.1.0 \
  --wait --timeout=600s
```

---

## 4. Azure Kubernetes Service (AKS)

### Create AKS cluster

```bash
# Create resource group
az group create --name sochdb-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group sochdb-rg \
  --name sochdb-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v5 \
  --enable-managed-identity \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group sochdb-rg --name sochdb-cluster
```

### Deploy SochDB

```bash
# Install with Azure-optimized settings
helm install sochdb deploy/helm/sochdb/ \
  --namespace sochdb \
  --create-namespace \
  --set persistence.storageClass=managed-csi-premium \
  --set persistence.size=256Gi \
  --set resources.requests.memory=8Gi \
  --set resources.limits.memory=8Gi \
  --set resources.requests.cpu=2000m \
  --set ingress.enabled=true \
  --set ingress.className=azure-application-gateway \
  --set ingress.host=sochdb.yourdomain.com
```

### Azure Marketplace

```bash
# Install from Azure Marketplace (when published)
# az aks app install --resource-group sochdb-rg \
#   --cluster-name sochdb-cluster \
#   --name sochdb \
#   --marketplace-offer sochdb-aks
```

---

## 5. Amazon EKS

### Create EKS cluster

```bash
# Using eksctl
eksctl create cluster \
  --name sochdb-cluster \
  --region us-east-1 \
  --nodegroup-name sochdb-nodes \
  --node-type m5.xlarge \
  --nodes 3 \
  --managed

# Install EBS CSI driver (for persistent volumes)
eksctl create addon \
  --name aws-ebs-csi-driver \
  --cluster sochdb-cluster \
  --service-account-role-arn arn:aws:iam::role/AmazonEKS_EBS_CSI_DriverRole
```

### Deploy SochDB

```bash
# Install with AWS-optimized settings
helm install sochdb deploy/helm/sochdb/ \
  --namespace sochdb \
  --create-namespace \
  --set persistence.storageClass=gp3 \
  --set persistence.size=256Gi \
  --set resources.requests.memory=8Gi \
  --set resources.limits.memory=8Gi \
  --set ingress.enabled=true \
  --set ingress.className=alb \
  --set "ingress.annotations.alb\\.ingress\\.kubernetes\\.io/scheme=internet-facing"
```

### AWS Marketplace

```bash
# Install from AWS Marketplace (when published)
# helm install sochdb oci://<marketplace-ecr>/sochdb --version 2.0.2
```

---

## 6. Google Kubernetes Engine (GKE)

### Create GKE cluster

```bash
# Create cluster
gcloud container clusters create sochdb-cluster \
  --region us-central1 \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --enable-ip-alias

# Get credentials
gcloud container clusters get-credentials sochdb-cluster --region us-central1
```

### Deploy SochDB

```bash
# Install with GCP-optimized settings
helm install sochdb deploy/helm/sochdb/ \
  --namespace sochdb \
  --create-namespace \
  --set persistence.storageClass=premium-rwo \
  --set persistence.size=256Gi \
  --set resources.requests.memory=8Gi \
  --set resources.limits.memory=8Gi
```

### GCP Marketplace

```bash
# Install from GCP Marketplace (when published)
# Follow the GCP Console Marketplace flow
```

---

## 7. Bare Metal / VM

### Build from source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protobuf compiler
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# Build
cargo build --release -p sochdb-grpc

# Run
./target/release/sochdb-grpc-server \
  --host 0.0.0.0 \
  --port 50051 \
  --data-dir /var/lib/sochdb
```

### Systemd service

```ini
# /etc/systemd/system/sochdb.service
[Unit]
Description=SochDB gRPC Server
After=network.target

[Service]
Type=simple
User=sochdb
Group=sochdb
ExecStart=/usr/local/bin/sochdb-grpc-server --host 0.0.0.0 --port 50051 --data-dir /var/lib/sochdb
Restart=always
RestartSec=5
LimitNOFILE=65536
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable sochdb
sudo systemctl start sochdb
sudo systemctl status sochdb
```

---

## Verification

After deploying on any platform:

```bash
# Health check
grpc_health_probe -addr=<host>:50051

# Metrics
curl http://<host>:9090/metrics

# Python SDK test
pip install sochdb-client
python3 -c "
from sochdb_sdk import SochDBClient
client = SochDBClient('<host>:50051')
print('Connected!')
"
```
