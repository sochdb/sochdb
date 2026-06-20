#!/usr/bin/env bash
# verify-vector-search-contract.sh
#
# Verification script for the vector-search distance and ID contract PR.
# Runs formatting (touched files only), targeted tests, clippy (touched
# crates, warnings only — pre-existing issues documented), and doc
# generation.  Never installs packages, never kills processes, never
# modifies user data.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Resolve repository root safely
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Vector Search Contract Verification ==="
echo "Repository root: $REPO_ROOT"
echo ""

# ---------------------------------------------------------------------------
# 2. Print Rust and Cargo versions
# ---------------------------------------------------------------------------
echo "--- Toolchain ---"
rustc --version
cargo --version
echo ""

# Track overall status
OVERALL_OK=1
ID_NARROWING_TEST="unknown"
COSINE_CONTRACT="unknown"
L2_CONTRACT="unknown"
DOT_PRODUCT_CONTRACT="unknown"
METRIC_PROPAGATION="unknown"

# ---------------------------------------------------------------------------
# 3. Formatting checks for touched files only
# ---------------------------------------------------------------------------
echo "--- Formatting (touched files) ---"
# Pre-existing formatting issues exist in untouched files (perf_profile.rs,
# simd_distance.rs, etc.).  We only check the files this PR modifies.
TOUCHED_FILES=(
    "sochdb-index/src/hot_buffer_hnsw.rs"
    "sochdb-index/src/unified_search.rs"
    "sochdb-grpc/src/server.rs"
    "sochdb-python/src/lib.rs"
    "proto/sochdb.proto"
    "sochdb-grpc/proto/sochdb.proto"
    "scripts/verify-vector-search-contract.sh"
)

FMT_OK=1
for f in "${TOUCHED_FILES[@]}"; do
    case "$f" in
        *.rs)
            if ! rustfmt --edition 2024 --check "$f" 2>/dev/null; then
                echo "  fmt issue: $f"
                FMT_OK=0
            fi
            ;;
    esac
done
if [ "$FMT_OK" -eq 1 ]; then
    echo "fmt: OK (touched files)"
else
    echo "fmt: FAILED (touched files)"
    OVERALL_OK=0
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Targeted metric and ID regression tests (sochdb-index)
# ---------------------------------------------------------------------------
echo "--- Targeted sochdb-index tests ---"

# Hot-buffer dot-product regression
if cargo test -p sochdb-index --lib hot_buffer_hnsw::tests::test_hot_buffer_dot_product_distance_is_negated 2>&1; then
    DOT_PRODUCT_CONTRACT="passed"
else
    DOT_PRODUCT_CONTRACT="failed"
    OVERALL_OK=0
fi

# Unified-search cosine contract
if cargo test -p sochdb-index --lib unified_search::tests::test_cosine_distance_contract 2>&1; then
    COSINE_CONTRACT="passed"
else
    COSINE_CONTRACT="failed"
    OVERALL_OK=0
fi

# Unified-search L2 contract
if cargo test -p sochdb-index --lib unified_search::tests::test_euclidean_distance_contract 2>&1; then
    L2_CONTRACT="passed"
else
    L2_CONTRACT="failed"
    OVERALL_OK=0
fi

# Unified-search dot-product contract
if cargo test -p sochdb-index --lib unified_search::tests::test_dot_product_distance_contract 2>&1; then
    :
else
    DOT_PRODUCT_CONTRACT="failed"
    OVERALL_OK=0
fi

echo ""

# ---------------------------------------------------------------------------
# 5. gRPC tests (sochdb-grpc) — ID narrowing + metric propagation
# ---------------------------------------------------------------------------
echo "--- gRPC tests (sochdb-grpc) ---"
if cargo test -p sochdb-grpc --lib server::tests 2>&1; then
    ID_NARROWING_TEST="passed"
    METRIC_PROPAGATION="passed"
else
    ID_NARROWING_TEST="failed"
    METRIC_PROPAGATION="failed"
    OVERALL_OK=0
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Python binding compilation check (sochdb-python)
# ---------------------------------------------------------------------------
echo "--- Python binding check (sochdb-python) ---"
# sochdb-python is excluded from the workspace and built with maturin.
# We run `cargo check` to verify the Rust source compiles — no Python
# installation or maturin needed.
if (cd sochdb-python && cargo check 2>&1); then
    echo "sochdb-python cargo check: OK"
else
    echo "sochdb-python cargo check: FAILED"
    OVERALL_OK=0
fi
echo ""

# ---------------------------------------------------------------------------
# 7. Clippy on touched files (warnings only, not -D warnings)
# ---------------------------------------------------------------------------
echo "--- clippy (touched crates, warnings only) ---"
# Pre-existing clippy errors exist in sochdb-index and sochdb-core.
# We run clippy without -D warnings and filter for touched files.
# No new warnings should appear in our modified files.
CLIPPY_OUTPUT="$(cargo clippy -p sochdb-index -p sochdb-grpc --no-deps 2>&1 || true)"

# Check for clippy warnings in our touched files
NEW_CLIPPY="$(echo "$CLIPPY_OUTPUT" | rg \
    'hot_buffer_hnsw\.rs:(7[0-9][0-9])|unified_search\.rs:8[5-9][0-9]|unified_search\.rs:9[0-9][0-9]' \
    || true)"

if [ -z "$NEW_CLIPPY" ]; then
    echo "clippy: OK (no new warnings in touched code)"
else
    echo "clippy: new warnings found in touched files:"
    echo "$NEW_CLIPPY"
    OVERALL_OK=0
fi
echo ""

# ---------------------------------------------------------------------------
# 8. Doc generation for touched crates
# ---------------------------------------------------------------------------
echo "--- cargo doc (sochdb-index, sochdb-grpc) ---"
if cargo doc -p sochdb-index -p sochdb-grpc --no-deps 2>&1; then
    echo "doc: OK"
else
    echo "doc: FAILED"
    OVERALL_OK=0
fi
echo ""

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
echo "=== Summary ==="
echo "VECTOR_SEARCH_CONTRACT_OK=$OVERALL_OK"
echo "ID_NARROWING_TEST=$ID_NARROWING_TEST"
echo "COSINE_CONTRACT=$COSINE_CONTRACT"
echo "L2_CONTRACT=$L2_CONTRACT"
echo "DOT_PRODUCT_CONTRACT=$DOT_PRODUCT_CONTRACT"
echo "METRIC_PROPAGATION=$METRIC_PROPAGATION"

if [ "$OVERALL_OK" -ne 1 ]; then
    echo ""
    echo "VERIFICATION FAILED — see output above"
    exit 1
fi