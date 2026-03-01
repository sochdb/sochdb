# =============================================================================
# SochDB gRPC Server Docker Image
# =============================================================================
#
# Multi-stage build for optimal image size (~50MB final image)
#
# Build: docker build -t sochdb/sochdb-grpc:latest .
# Run:   docker run -p 50051:50051 sochdb/sochdb-grpc:latest
#
# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.91-bookworm AS builder

# Install protobuf compiler for gRPC codegen
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace Cargo files first (for caching)
COPY Cargo.toml Cargo.lock ./

# Copy all workspace members
COPY sochdb-core ./sochdb-core
COPY sochdb-kernel ./sochdb-kernel
COPY sochdb-storage ./sochdb-storage
COPY sochdb-index ./sochdb-index
COPY sochdb-query ./sochdb-query
COPY sochdb-client ./sochdb-client
COPY sochdb-grpc ./sochdb-grpc
COPY sochdb-vector ./sochdb-vector
COPY sochdb-mcp ./sochdb-mcp
COPY sochdb-wasm ./sochdb-wasm
COPY sochdb-tools ./sochdb-tools
COPY sochdb-plugin-logging ./sochdb-plugin-logging
COPY proto ./proto
COPY sochdb-fusion ./sochdb-fusion

# Build release binary
RUN cargo build --release --package sochdb-grpc --bin sochdb-grpc-server

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM debian:bookworm-slim AS runtime
COPY --from=ghcr.io/grpc-ecosystem/grpc-health-probe:v0.4.38 \
    /ko-app/grpc-health-probe /bin/grpc_health_probe
# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false sochdb

# Copy binary from builder
COPY --from=builder /build/target/release/sochdb-grpc-server /usr/local/bin/

# Create data directory
RUN mkdir -p /var/lib/sochdb && chown sochdb:sochdb /var/lib/sochdb

# Switch to non-root user
USER sochdb

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD grpc_health_probe -addr=:50051 || exit 1

# Default environment
ENV RUST_LOG=info
ENV SOCHDB_DATA_DIR=/var/lib/sochdb

# Entry point
ENTRYPOINT ["sochdb-grpc-server"]
CMD ["--host", "0.0.0.0", "--port", "50051"]
