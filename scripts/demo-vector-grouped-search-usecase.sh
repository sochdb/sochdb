#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

cargo test -p sochdb-grpc --lib grouped_search_customer_support_use_case_reduces_duplicate_parent_waste -- --nocapture

echo "GROUPED_SEARCH_USECASE_DEMO_OK=1"
