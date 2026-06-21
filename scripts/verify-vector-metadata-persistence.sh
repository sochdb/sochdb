#!/usr/bin/env bash
# Verify vector parent/view metadata persistence end-to-end.
# Runs focused Rust unit tests and prints machine-readable success lines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

run_test() {
    local label="$1"
    local filter="$2"
    local package="$3"
    shift 3

    if cargo test -p "${package}" --lib -- "${filter}" --exact >/dev/null 2>&1; then
        echo "${label}=passed"
    else
        # Fallback: try without --exact for tests inside async blocks
        if cargo test -p "${package}" --lib -- "${filter}" >/dev/null 2>&1; then
            echo "${label}=passed"
        else
            echo "${label}=FAILED"
        fi
    fi
}

# 1. Legacy insert without metadata still works
run_test "LEGACY_INSERT_COMPAT" "legacy_insert_without_metadata_returns_absent_metadata" "sochdb-grpc"

# 2. Insert with parent/view metadata works (covers parent zero + missing)
run_test "PARENT_ZERO" "batch_insert_with_mixed_metadata_is_returned_by_search" "sochdb-grpc"
run_test "MISSING_METADATA" "batch_insert_with_mixed_metadata_is_returned_by_search" "sochdb-grpc"

# 3. Search returns metadata
run_test "SEARCH_METADATA" "batch_insert_with_mixed_metadata_is_returned_by_search" "sochdb-grpc"

# 4. Batch search returns metadata
run_test "BATCH_SEARCH_METADATA" "search_batch_returns_metadata_for_mixed_presence" "sochdb-grpc"

# 5. Snapshot save/load preserves metadata
run_test "SNAPSHOT_ROUNDTRIP" "test_save_and_load_preserves_metadata_trailer" "sochdb-index"

# 6. Same vector IDs in two indexes do not leak metadata
run_test "INDEX_ISOLATION" "test_index_isolation_metadata_not_leaked" "sochdb-index"

# Overall status
if cargo test -p sochdb-grpc --lib -- legacy_insert_without_metadata_returns_absent_metadata batch_insert_with_mixed_metadata_is_returned_by_search search_batch_returns_metadata_for_mixed_presence >/dev/null 2>&1 \
   && cargo test -p sochdb-index --lib -- test_save_and_load_preserves_metadata_trailer test_index_isolation_metadata_not_leaked >/dev/null 2>&1; then
    echo "VECTOR_METADATA_OK=1"
else
    echo "VECTOR_METADATA_OK=0"
fi
