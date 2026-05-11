#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# SochDB SDK Codegen — Generates thin client SDKs from sochdb.proto
#
# Usage:
#   ./generate.sh [python|go|node|all]
#
# Prerequisites:
#   - protoc (Protocol Buffer compiler)
#   - Language-specific plugins:
#     Python: grpcio-tools (pip install grpcio-tools)
#     Go:     protoc-gen-go, protoc-gen-go-grpc
#     Node:   grpc-tools (@grpc/grpc-tools)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(cd "$SCRIPT_DIR/../sochdb-grpc/proto" && pwd)"
PROTO_FILE="$PROTO_DIR/sochdb.proto"

if [[ ! -f "$PROTO_FILE" ]]; then
    echo "ERROR: Proto file not found at $PROTO_FILE"
    exit 1
fi

TARGET="${1:-all}"

# ============================================================================
# Python SDK
# ============================================================================
generate_python() {
    echo "==> Generating Python SDK..."
    local out_dir="$SCRIPT_DIR/python/sochdb_sdk/generated"
    mkdir -p "$out_dir"

    if command -v python3 -m grpc_tools.protoc &>/dev/null || python3 -c "import grpc_tools" 2>/dev/null; then
        python3 -m grpc_tools.protoc \
            --proto_path="$PROTO_DIR" \
            --python_out="$out_dir" \
            --grpc_python_out="$out_dir" \
            "$PROTO_FILE"
        # Create __init__.py
        touch "$out_dir/__init__.py"
        echo "    Python stubs generated in $out_dir"
    elif command -v protoc &>/dev/null; then
        protoc \
            --proto_path="$PROTO_DIR" \
            --python_out="$out_dir" \
            --grpc_python_out="$out_dir" \
            "$PROTO_FILE"
        touch "$out_dir/__init__.py"
        echo "    Python stubs generated in $out_dir"
    else
        echo "    SKIP: grpcio-tools or protoc not found"
        echo "    Install: pip install grpcio-tools"
    fi
}

# ============================================================================
# Go SDK
# ============================================================================
generate_go() {
    echo "==> Generating Go SDK..."
    local out_dir="$SCRIPT_DIR/go/sochdbv1"
    mkdir -p "$out_dir"

    if command -v protoc &>/dev/null && command -v protoc-gen-go &>/dev/null; then
        protoc \
            --proto_path="$PROTO_DIR" \
            --go_out="$out_dir" --go_opt=paths=source_relative \
            --go-grpc_out="$out_dir" --go-grpc_opt=paths=source_relative \
            "$PROTO_FILE"
        echo "    Go stubs generated in $out_dir"
    else
        echo "    SKIP: protoc-gen-go not found"
        echo "    Install: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
        echo "             go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    fi
}

# ============================================================================
# Node.js/TypeScript SDK
# ============================================================================
generate_node() {
    echo "==> Generating Node.js SDK..."
    local out_dir="$SCRIPT_DIR/node/src/generated"
    mkdir -p "$out_dir"

    if command -v protoc &>/dev/null && command -v protoc-gen-ts &>/dev/null; then
        protoc \
            --proto_path="$PROTO_DIR" \
            --ts_out="$out_dir" \
            "$PROTO_FILE"
        echo "    TypeScript stubs generated in $out_dir"
    elif command -v npx &>/dev/null; then
        # Try ts-proto via npx
        if npx --no-install ts-proto --version &>/dev/null 2>&1; then
            protoc \
                --proto_path="$PROTO_DIR" \
                --plugin="protoc-gen-ts=$(npx which ts-proto)" \
                --ts_out="$out_dir" \
                "$PROTO_FILE"
            echo "    TypeScript stubs generated in $out_dir"
        else
            echo "    SKIP: ts-proto not found"
            echo "    Install: npm install -g ts-proto"
        fi
    else
        echo "    SKIP: protoc/ts-proto not found"
    fi
}

# ============================================================================
# Main
# ============================================================================
echo "SochDB SDK Codegen"
echo "Proto: $PROTO_FILE"
echo ""

case "$TARGET" in
    python) generate_python ;;
    go)     generate_go ;;
    node)   generate_node ;;
    all)
        generate_python
        generate_go
        generate_node
        ;;
    *)
        echo "Usage: $0 [python|go|node|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
