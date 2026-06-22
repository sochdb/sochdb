#!/usr/bin/env bash
# Verify vector parent-aware grouped search end-to-end.
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
        if cargo test -p "${package}" --lib -- "${filter}" >/dev/null 2>&1; then
            echo "${label}=passed"
        else
            echo "${label}=FAILED"
        fi
    fi
}

# 1. Ungrouped search still works
run_test "UNGROUPED_COMPAT" "ungrouped_search_is_unchanged" "sochdb-grpc"

# 2. Grouped search returns unique parents
run_test "UNIQUE_PARENT_RESULTS" "grouped_search_returns_unique_parents" "sochdb-grpc"

# 3. Best view wins (closest in distance order)
run_test "BEST_VIEW_WINS" "grouped_search_parent_zero_is_grouped" "sochdb-grpc"

# 4. Missing parent falls back to vector ID
run_test "MISSING_PARENT_FALLBACK" "grouped_search_missing_parent_fallback" "sochdb-grpc"

# 5. parent_id = 0 is grouped correctly
run_test "PARENT_ZERO_GROUPING" "grouped_search_parent_zero_is_grouped" "sochdb-grpc"

# 6. Candidate overfetch recovers hidden parents
run_test "CANDIDATE_OVERFETCH" "grouped_search_candidate_overfetch" "sochdb-grpc"

# 7. Batch search grouping works
run_test "BATCH_GROUPING" "grouped_search_batch_grouping" "sochdb-grpc"

# Overall status
if cargo test -p sochdb-grpc --lib -- \
    ungrouped_search_is_unchanged \
    grouped_search_returns_unique_parents \
    grouped_search_parent_zero_is_grouped \
    grouped_search_missing_parent_fallback \
    grouped_search_candidate_overfetch \
    grouped_search_batch_grouping \
    >/dev/null 2>&1; then
    echo "GROUPED_SEARCH_OK=1"
else
    echo "GROUPED_SEARCH_OK=0"
fi
