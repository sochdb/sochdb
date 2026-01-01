//! Benchmarks for SIMD kernel performance.
//!
//! Run with: cargo bench --package toondb-vector

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use toondb_vector::simd::{bps_scan, dot_i8, visibility, dispatch};

// ============================================================================
// BPS Scan Benchmarks
// ============================================================================

fn bench_bps_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("bps_scan");
    
    for n_vec in [1000, 10000, 100000] {
        for n_blocks in [32, 48, 64] {
            // Generate test data
            let bps: Vec<u8> = (0..n_vec * n_blocks)
                .map(|i| (i % 256) as u8)
                .collect();
            let query: Vec<u8> = (0..n_blocks).map(|i| (i * 5) as u8).collect();
            let mut out = vec![0u16; n_vec];
            
            let bytes_per_iter = n_vec * n_blocks;
            group.throughput(Throughput::Bytes(bytes_per_iter as u64));
            
            group.bench_with_input(
                BenchmarkId::new("simd", format!("{}x{}", n_vec, n_blocks)),
                &(n_vec, n_blocks),
                |b, &(n, nb)| {
                    b.iter(|| {
                        bps_scan::bps_scan(
                            black_box(&bps),
                            n,
                            nb,
                            black_box(&query),
                            black_box(&mut out),
                        )
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_bps_scan_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bps_scan_u32");
    
    let n_vec = 100000;
    let n_blocks = 48;
    
    let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
    let query: Vec<u8> = (0..n_blocks).map(|i| (i * 5) as u8).collect();
    let mut out = vec![0u32; n_vec];
    
    group.throughput(Throughput::Bytes((n_vec * n_blocks) as u64));
    
    group.bench_function("simd", |b| {
        b.iter(|| {
            bps_scan::bps_scan_u32(
                black_box(&bps),
                n_vec,
                n_blocks,
                black_box(&query),
                black_box(&mut out),
            )
        })
    });
    
    group.finish();
}

// ============================================================================
// Int8 Dot Product Benchmarks
// ============================================================================

fn bench_dot_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_i8");
    
    for dim in [128, 256, 512, 768, 1024] {
        let a: Vec<i8> = (0..dim).map(|i| ((i * 3) % 127) as i8).collect();
        let b: Vec<i8> = (0..dim).map(|i| ((i * 7) % 127) as i8).collect();
        
        group.throughput(Throughput::Bytes((dim * 2) as u64)); // Both vectors
        
        group.bench_with_input(
            BenchmarkId::new("single", dim),
            &dim,
            |bench, _| {
                bench.iter(|| dot_i8::dot_i8(black_box(&a), black_box(&b)))
            },
        );
    }
    
    group.finish();
}

fn bench_dot_i8_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_i8_batch");
    
    let dim = 768;
    
    for n_vec in [10, 100, 1000] {
        let query: Vec<i8> = (0..dim).map(|i| ((i * 3) % 127) as i8).collect();
        let vectors: Vec<i8> = (0..n_vec * dim).map(|i| ((i * 7) % 127) as i8).collect();
        let scales: Vec<f32> = (0..n_vec).map(|i| 0.01 * (i + 1) as f32).collect();
        let mut results = vec![0.0f32; n_vec];
        
        group.throughput(Throughput::Elements(n_vec as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch", n_vec),
            &n_vec,
            |bench, _| {
                bench.iter(|| {
                    dot_i8::dot_i8_batch(
                        black_box(&query),
                        black_box(&vectors),
                        black_box(&scales),
                        dim,
                        black_box(&mut results),
                    )
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Visibility Check Benchmarks
// ============================================================================

fn bench_visibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("visibility");
    
    for n_rows in [100, 1000, 10000, 100000] {
        let commits: Vec<u64> = (0..n_rows)
            .map(|i| if i % 10 == 0 { 0 } else { (i * 7) as u64 })
            .collect();
        let mut mask = vec![0u8; n_rows];
        let snapshot = 5000;
        
        group.throughput(Throughput::Elements(n_rows as u64));
        
        group.bench_with_input(
            BenchmarkId::new("basic", n_rows),
            &n_rows,
            |bench, _| {
                bench.iter(|| {
                    visibility::visibility_check(
                        black_box(&commits),
                        snapshot,
                        black_box(&mut mask),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_visibility_with_txn(c: &mut Criterion) {
    let mut group = c.benchmark_group("visibility_with_txn");
    
    let n_rows = 10000;
    let commits: Vec<u64> = (0..n_rows)
        .map(|i| if i % 10 == 0 { 0 } else { (i * 7) as u64 })
        .collect();
    let txn_ids: Vec<u64> = (0..n_rows).map(|i| (i % 100) as u64).collect();
    let mut mask = vec![0u8; n_rows];
    let snapshot = 5000;
    let current_txn = 50;
    
    group.throughput(Throughput::Elements(n_rows as u64));
    
    group.bench_function("with_txn", |bench| {
        bench.iter(|| {
            visibility::visibility_check_with_txn(
                black_box(&commits),
                black_box(&txn_ids),
                snapshot,
                current_txn,
                black_box(&mut mask),
            )
        })
    });
    
    group.finish();
}

// ============================================================================
// CPU Info (printed once)
// ============================================================================

fn print_cpu_info(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_info");
    
    // Print CPU info once
    println!("\n{}", dispatch::dispatch_info());
    
    // Dummy benchmark so criterion is happy
    group.bench_function("detect", |b| {
        b.iter(|| dispatch::simd_level())
    });
    
    group.finish();
}

criterion_group!(
    benches,
    print_cpu_info,
    bench_bps_scan,
    bench_bps_scan_u32,
    bench_dot_i8,
    bench_dot_i8_batch,
    bench_visibility,
    bench_visibility_with_txn,
);

criterion_main!(benches);
