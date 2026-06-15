#!/usr/bin/env bash
# Reproducible census of `unsafe` keyword usage across the workspace.
#
# Emits a per-crate and total count of the `unsafe` keyword in Rust sources
# (src/*.rs, excluding target/ and generated output). This is the single source
# of truth for the workspace's unsafe surface (Task 0): quote this script's
# output, not a hard-coded number — the figure drifts as code changes.
#
# Note: this counts the `unsafe` *keyword* (including any occurrences in comments
# or strings), i.e. an upper bound / order-of-magnitude signal, not a precise
# count of `unsafe { }` blocks. It is intentionally simple and deterministic so
# CI can run it and humans can reproduce it.
#
# Usage:  ./unsafe_census.sh
set -euo pipefail
cd "$(dirname "$0")"

total=0
lines=""
for d in */; do
  src="${d}src"
  [ -d "$src" ] || continue
  n=$({ grep -rhoE '\bunsafe\b' "$src" --include='*.rs' 2>/dev/null || true; } | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || continue
  lines+=$(printf '%8d  %s\n' "$n" "${d%/}")$'\n'
  total=$((total + n))
done

echo "unsafe-keyword census (\\bunsafe\\b in */src/**.rs):"
printf '%s' "$lines" | sort -rn
printf '%8d  %s\n' "$total" "TOTAL"
