# PR 10: Fix `unsafe_op_in_unsafe_fn` warnings across sochdb-vector and sochdb-index

## Summary

Rust 2024 edition (and recent `rustc` nightly/stable with the lint enabled by default)
emits `unsafe_op_in_unsafe_fn` warnings whenever an `unsafe fn` body contains raw unsafe
operations (intrinsic calls, pointer dereferences, `dealloc`, etc.) without an explicit
`unsafe { }` block.  This is a soundness improvement: in Rust 2024, the body of an
`unsafe fn` is no longer implicitly an unsafe context, so every unsafe operation must be
wrapped explicitly.

This PR silences those warnings in two complementary ways:

1. **sochdb-vector** *(10 functions across 4 files)* – manual wrapping.  Each `unsafe fn`
   that calls intrinsics / uses `get_unchecked` / calls `dealloc` now wraps its body in
   `unsafe { }`, making the unsafe boundary explicit and auditable.

2. **sochdb-index** *(207 warnings across 8+ files)* – crate-level
   `#![allow(unsafe_op_in_unsafe_fn)]`.  The index crate has a very large number of
   `unsafe fn` bodies that perform raw pointer / CAS / metric operations.  Individually
   wrapping each one would be a large mechanical change that risks merge conflicts with
   ongoing work.  The `#![allow]` is a pragmatic migration shim that can be removed once
   each call site is audited and wrapped.

## Files Changed

### sochdb-vector

| File | Functions wrapped |
|------|------------------|
| `src/hot_path_layout.rs` | `free_aligned` — wrapped `dealloc` call |
| `src/simd/bps_scan.rs` | `bps_scan_avx2`, `bps_scan_avx2_u32` |
| `src/portable_simd.rs` | AVX2 `inner` (×3) + NEON `inner` (×3) |
| `src/simd_hadamard.rs` | `hadamard_avx2`, `hadamard_sse41`, `hadamard_avx512`, `hadamard_neon` |

### sochdb-index

| File | Change |
|------|--------|
| `src/lib.rs` | Added `#![allow(unsafe_op_in_unsafe_fn)]` |

## Testing

- `cargo check -p sochdb-vector` — 0 `unsafe_op_in_unsafe_fn` warnings, only pre-existing dead_code/unused warnings
- `cargo check -p sochdb-index` — 0 `unsafe_op_in_unsafe_fn` warnings
- Pre-existing `test_score_envelope` compilation error (uses non-existent `.score` field) is **not** introduced by this PR — it exists on `main` and is addressed separately in PR #49