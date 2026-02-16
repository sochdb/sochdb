#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  SochDB Benchmark Runner
#  Runs the full benchmark suite across multiple configurations
#  and exports results to the results/ directory.
#
#  Usage:
#    ./run_benchmarks.sh              # all suites
#    ./run_benchmarks.sh quick        # 10K-scale smoke test
#    ./run_benchmarks.sh oltp         # OLTP only
#    ./run_benchmarks.sh analytics    # Analytics only
#    ./run_benchmarks.sh vector       # Vector only
#    ./run_benchmarks.sh criterion    # Criterion micro-benchmarks
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="results"
BIN="cargo run --release --"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo ""
}

section() {
    echo -e "${YELLOW}▶ $1${NC}"
}

done_msg() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Build once in release mode
build() {
    header "Building sochdb-bench (release)"
    cargo build --release
    done_msg "Build complete"
}

# ─── Individual Suites ────────────────────────────────────────

run_quick() {
    header "Quick Smoke Test (--all --scale 10000)"
    $BIN --all --scale 10000
    done_msg "Quick test done"
}

run_oltp() {
    header "OLTP Suite"

    section "OLTP @ 10K (warm-up)"
    $BIN --oltp --scale 10000

    section "OLTP @ 50K"
    $BIN --oltp --scale 50000

    section "OLTP @ 100K"
    $BIN --oltp --scale 100000

    done_msg "OLTP suite done"
}

run_analytics() {
    header "Analytics Suite"

    section "Analytics @ 10K"
    $BIN --analytics --scale 10000

    section "Analytics @ 50K"
    $BIN --analytics --scale 50000

    section "Analytics @ 100K"
    $BIN --analytics --scale 100000

    done_msg "Analytics suite done"
}

run_vector() {
    header "Vector Suite"

    section "Vector dim=128, k=10 @ 10K"
    $BIN --vector --dim 128 --k 10 --scale 10000

    section "Vector dim=768, k=10 @ 10K"
    $BIN --vector --dim 768 --k 10 --scale 10000

    section "Vector dim=128, k=10 @ 50K"
    $BIN --vector --dim 128 --k 10 --scale 50000

    done_msg "Vector suite done"
}

run_full_export() {
    header "Full Suite @ 50K with Export"
    mkdir -p "$RESULTS_RUN_DIR"
    $BIN --all --scale 50000 --export "$RESULTS_RUN_DIR"
    done_msg "Exported to ${RESULTS_RUN_DIR}/"
    echo "  CSV:  ${RESULTS_RUN_DIR}/benchmark_results.csv"
    echo "  JSON: ${RESULTS_RUN_DIR}/benchmark_results.json"
}

run_criterion() {
    header "Criterion Micro-Benchmarks"
    cargo bench 2>&1 | tee "${RESULTS_DIR}/criterion_${TIMESTAMP}.txt"
    done_msg "Criterion results saved to ${RESULTS_DIR}/criterion_${TIMESTAMP}.txt"
}

# ─── Main ─────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"

MODE="${1:-all}"

case "$MODE" in
    quick)
        build
        run_quick
        ;;
    oltp)
        build
        run_oltp
        ;;
    analytics)
        build
        run_analytics
        ;;
    vector)
        build
        run_vector
        ;;
    criterion)
        build
        run_criterion
        ;;
    all)
        build
        run_quick
        run_oltp
        run_analytics
        run_vector
        run_full_export
        run_criterion
        ;;
    *)
        echo "Usage: $0 {all|quick|oltp|analytics|vector|criterion}"
        exit 1
        ;;
esac

header "All Done"
echo "Results directory: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/"
