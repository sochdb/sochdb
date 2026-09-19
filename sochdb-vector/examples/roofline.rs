//! First-principles roofline probe.
//!
//! Measures (a) what the memory hierarchy on THIS machine actually delivers and
//! (b) what each kernel achieves, both in cycles/byte, so the gap between them
//! is the only thing that matters. Absolute ns/iter numbers are useless on a
//! shared, frequency-scaling box; cycles/byte against a measured ceiling is not.

use std::arch::x86_64::_rdtsc;
use std::time::Instant;

/// Measured TSC-to-wall ratio. The TSC is invariant (constant_tsc) and ticks at
/// the *base* frequency, not the turbo frequency, so a raw rdtsc delta is not a
/// core-cycle count. We need the real core-cycle rate to compare against
/// per-cycle instruction throughput limits, so calibrate the actual core clock
/// under load rather than trusting the nominal frequency.
fn calibrate_tsc_hz() -> f64 {
    let t0 = Instant::now();
    let c0 = unsafe { _rdtsc() };
    let mut sink = 0u64;
    while t0.elapsed().as_millis() < 200 {
        sink = sink.wrapping_add(1);
    }
    std::hint::black_box(sink);
    let c1 = unsafe { _rdtsc() };
    let elapsed = t0.elapsed().as_secs_f64();
    (c1 - c0) as f64 / elapsed
}

/// Minimum of N timed repetitions. On a noisy shared machine the minimum is the
/// only robust estimator: interference can only ever add time, never remove it,
/// so the minimum converges to the true cost while the mean tracks the noise.
fn min_cycles<F: FnMut()>(reps: usize, mut f: F) -> u64 {
    let mut best = u64::MAX;
    for _ in 0..reps {
        let start = unsafe { _rdtsc() };
        f();
        let end = unsafe { _rdtsc() };
        best = best.min(end - start);
    }
    best
}

/// Pure sequential read bandwidth at a given working-set size. This traces the
/// cache hierarchy: the curve's plateaus are L1/L2/L3/DRAM, and every kernel
/// that streams memory is bounded by the plateau its working set lands in.
fn measure_read_bandwidth(bytes: usize, tsc_hz: f64) -> f64 {
    let n = bytes / 8;
    let buf: Vec<u64> = (0..n as u64).collect();
    let reps = if bytes < (1 << 20) { 500 } else { 30 };

    // Eight independent accumulators: a single accumulator would serialise on
    // add latency and measure the ALU, not the memory system.
    let c = min_cycles(reps, || {
        let mut a = [0u64; 8];
        let chunks = n / 8;
        for i in 0..chunks {
            let base = i * 8;
            for (j, acc) in a.iter_mut().enumerate() {
                *acc = acc.wrapping_add(unsafe { *buf.get_unchecked(base + j) });
            }
        }
        std::hint::black_box(a);
    });

    let secs = c as f64 / tsc_hz;
    (bytes as f64) / secs / 1e9
}

/// The BPS scan is the pipeline's first stage: it streams every vector's sketch
/// and is therefore a pure bandwidth problem. Comparing its achieved GB/s to the
/// roofline at the same working-set size tells us whether it is memory-bound
/// (nothing to win without shrinking the data) or compute-bound (the kernel is
/// leaving bandwidth unused and can be fixed in place).
fn measure_bps(n_vec: usize, n_blocks: usize, tsc_hz: f64) -> (f64, usize) {
    use sochdb_vector::simd::bps_scan::bps_scan;
    let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 251) as u8).collect();
    let query: Vec<u8> = (0..n_blocks).map(|i| (i * 7 % 251) as u8).collect();
    let mut out = vec![0u16; n_vec];
    let bytes = n_vec * n_blocks;
    let reps = if bytes < (1 << 22) { 100 } else { 20 };

    let c = min_cycles(reps, || {
        bps_scan(
            std::hint::black_box(&bps),
            n_vec,
            n_blocks,
            std::hint::black_box(&query),
            std::hint::black_box(&mut out),
        );
    });
    let secs = c as f64 / tsc_hz;
    ((bytes as f64) / secs / 1e9, bytes)
}

/// Rerank throughput: dot_i8 called once per candidate, which is exactly how the
/// rerank stage uses it. Measured as GB/s over the database bytes touched so it
/// is directly comparable to the roofline and to the BPS stage.
fn measure_dot_i8(n_vec: usize, dim: usize, tsc_hz: f64) -> (f64, f64) {
    use sochdb_vector::simd::dot_i8::dot_i8;
    let db: Vec<i8> = (0..n_vec * dim).map(|i| (i % 127) as i8).collect();
    let q: Vec<i8> = (0..dim).map(|i| (i % 127) as i8).collect();
    let bytes = n_vec * dim;
    let reps = if bytes < (1 << 22) { 100 } else { 20 };

    let c = min_cycles(reps, || {
        let mut acc = 0i64;
        for i in 0..n_vec {
            acc += dot_i8(
                std::hint::black_box(&q),
                std::hint::black_box(&db[i * dim..(i + 1) * dim]),
            ) as i64;
        }
        std::hint::black_box(acc);
    });
    let secs = c as f64 / tsc_hz;
    // cycles per vector is the clearest signal for a fixed-dim kernel: it can be
    // compared directly against the instruction-issue lower bound.
    let cycles_per_vec = c as f64 / n_vec as f64;
    ((bytes as f64) / secs / 1e9, cycles_per_vec)
}

fn main() {
    let tsc_hz = calibrate_tsc_hz();
    println!("# Calibrated TSC: {:.2} GHz", tsc_hz / 1e9);
    println!();
    println!("## Memory hierarchy (sequential read) — the physical ceiling");
    println!("{:>12} {:>12} {:>14}", "working set", "GB/s", "bytes/cycle");
    let mut roof: Vec<(usize, f64)> = Vec::new();
    for kb in [16usize, 64, 256, 1024, 4096, 16384, 65536, 262144] {
        let bytes = kb * 1024;
        let gbs = measure_read_bandwidth(bytes, tsc_hz);
        roof.push((bytes, gbs));
        let bpc = gbs * 1e9 / tsc_hz;
        let label = if kb >= 1024 {
            format!("{} MB", kb / 1024)
        } else {
            format!("{} KB", kb)
        };
        println!("{:>12} {:>12.1} {:>14.2}", label, gbs, bpc);
    }

    // Nearest measured roofline point at or above a working set size.
    let ceiling = |bytes: usize| -> f64 {
        roof.iter()
            .find(|(b, _)| *b >= bytes)
            .map(|(_, g)| *g)
            .unwrap_or(roof.last().unwrap().1)
    };

    println!();
    println!("## Stage 1: bps_scan (sketch scan over all vectors)");
    println!(
        "{:>10} {:>8} {:>10} {:>10} {:>10} {:>9}",
        "n_vec", "blocks", "MB", "GB/s", "roof GB/s", "% roof"
    );
    for (n_vec, n_blocks) in [
        (100_000usize, 64usize),
        (1_000_000, 64),
        (4_000_000, 64),
        (1_000_000, 32),
    ] {
        let (gbs, bytes) = measure_bps(n_vec, n_blocks, tsc_hz);
        let r = ceiling(bytes);
        println!(
            "{:>10} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>8.0}%",
            n_vec,
            n_blocks,
            bytes as f64 / 1e6,
            gbs,
            r,
            100.0 * gbs / r
        );
    }

    println!();
    println!("## Stage 2: dot_i8 (rerank, one call per candidate)");
    println!(
        "{:>10} {:>6} {:>10} {:>10} {:>10} {:>9} {:>12}",
        "n_vec", "dim", "MB", "GB/s", "roof GB/s", "% roof", "cyc/vector"
    );
    for (n_vec, dim) in [(100_000usize, 768usize), (1_000_000, 768), (100_000, 128)] {
        let (gbs, cpv) = measure_dot_i8(n_vec, dim, tsc_hz);
        let bytes = n_vec * dim;
        let r = ceiling(bytes);
        println!(
            "{:>10} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>8.0}% {:>12.1}",
            n_vec,
            dim,
            bytes as f64 / 1e6,
            gbs,
            r,
            100.0 * gbs / r,
            cpv
        );
    }
}
