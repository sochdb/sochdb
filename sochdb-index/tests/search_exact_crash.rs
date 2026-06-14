// Regression test: vector_search_exact() segfaults via the SDK on 768-d cosine
// indexes. Reproduces against the core to confirm whether current source crashes.
// MUST be run in --release to match the shipped dylib (debug_assert!s in the inline
// SIMD kernels mask the out-of-bounds read in debug builds).

use sochdb_index::hnsw::{HnswConfig, HnswIndex};

fn norm(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

#[test]
fn search_exact_does_not_crash_768_cosine_default() {
    let dim = 768usize;
    let index = HnswIndex::new(dim, HnswConfig::default()); // Cosine + F32 + normalize_at_ingest

    // Deterministic pseudo-random normalized vectors.
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f32 / u64::MAX as f32) - 0.5
    };

    // Insert via the BATCH-CONTIGUOUS path the SDK actually uses (hnsw_insert_batch ->
    // insert_batch_contiguous). This exceeds the scaffold threshold so most nodes go
    // through the bulk path, where HnswNode.vector is left as a zero-length dummy and
    // the real vectors live in vector_store. search_exact reading node.vector directly
    // then feeds an empty slice to the fixed-768 SIMD kernel -> OOB read.
    let n = 2000usize;
    let mut ids = Vec::with_capacity(n);
    let mut flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        ids.push(i as u128);
        flat.extend(norm((0..dim).map(|_| rnd()).collect::<Vec<f32>>()));
    }
    let inserted = index
        .insert_batch_contiguous(&ids, &flat, dim)
        .expect("batch insert failed");
    assert_eq!(inserted, n);

    let q = norm((0..dim).map(|_| rnd()).collect::<Vec<f32>>());

    // Ground truth: brute-force cosine over the SAME vectors (normalized -> 1 - dot).
    let mut truth: Vec<(u128, f32)> = (0..n)
        .map(|i| {
            let base = i * dim;
            let dot: f32 = (0..dim).map(|d| q[d] * flat[base + d]).sum();
            (i as u128, 1.0 - dot)
        })
        .collect();
    truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let truth_top: std::collections::HashSet<u128> =
        truth[..10].iter().map(|(id, _)| *id).collect();

    // This is the call that segfaults via the SDK (hnsw_search_exact -> search_exact).
    let res = index.search_exact(&q, 10).expect("search_exact failed");
    assert_eq!(res.len(), 10, "expected 10 exact neighbors");
    // Correctness: search_exact must return the TRUE nearest neighbors (not the
    // garbage it returned when it read the empty HnswNode.vector dummy).
    let got: std::collections::HashSet<u128> = res.iter().map(|(id, _)| *id).collect();
    let overlap = got.intersection(&truth_top).count();
    assert!(
        overlap >= 9,
        "search_exact must match brute-force ground truth (overlap {overlap}/10): got {got:?}"
    );

    // The f64 variant must also read real vectors (it returned empty-vector garbage before).
    let res_f64 = index
        .search_exact_f64(&q, 10)
        .expect("search_exact_f64 failed");
    let got_f64: std::collections::HashSet<u128> = res_f64.iter().map(|(id, _)| *id).collect();
    assert!(
        got_f64.intersection(&truth_top).count() >= 9,
        "search_exact_f64 must match ground truth: got {got_f64:?}"
    );

    // Sanity: approximate search on the same index works (control).
    let res2 = index.search(&q, 10).expect("search failed");
    assert_eq!(res2.len(), 10);
}
