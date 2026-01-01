#!/bin/bash
# ToonDB Feature Validation Test Runner
# Usage: ./examples/run_feature_tests.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  ToonDB Feature Validation Test Suite"
echo "=============================================="

export PYTHONPATH="${PROJECT_DIR}/toondb-python-sdk/src:${PYTHONPATH}"
export TOONDB_LIB_PATH="${PROJECT_DIR}/target/release"

case "${1:-}" in
    --quick)
        echo "Running quick smoke tests..."
        python3 "${SCRIPT_DIR}/python/toondb_feature_validation.py" 2>&1 | head -20
        ;;
    *)
        python3 "${SCRIPT_DIR}/python/toondb_feature_validation.py"
        ;;
esac

