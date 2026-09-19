//! Does a scattered *prefix* read cost less than a scattered full read?
//!
//! Rerank gathers 768-byte records scattered across a 153 MB region. Scoring
//! only the first `m` dimensions touches `ceil(m/64)` cache lines per record
//! instead of 12. That is a real reduction in bytes, but it only converts into
//! time if the gather is bandwidth-bound. If it is latency-bound - one
//! dependent miss per record, regardless of how much of the record is read -
//! the prefix costs the same as the whole thing and the idea is worthless.
//!
//! This probe answers that before any pipeline work: it times the same gather
//! over the same id sequence at several prefix widths and reports the achieved
//! GB/s against the bytes actually requested.

use std::time::Instant;

fn main() {
    let n_vec: usize = 200_000;
    let dim: usize = 768;
    let n_cand: usize = 24_378;

    // 153 MB, far past the 36 MB L3, so every gather is a DRAM access.
    let data: Vec<i8> = (0..n_vec * dim).map(|i| (i % 251) as i8).collect();
    let query: Vec<i8> = (0..dim).map(|i| ((i * 7) % 127) as i8).collect();
    let _ = &query;

    // Ascending ids, matching what the bitset union now emits.
    let mut ids: Vec<u32> = Vec::with_capacity(n_cand);
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut cur = 0u32;
    let stride = (n_vec / n_cand) as u32;
    for _ in 0..n_cand {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        cur += 1 + (s >> 60) as u32 % stride.max(1);
        if cur as usize >= n_vec {
            break;
        }
        ids.push(cur);
    }
    println!(
        "{} candidates over {} vectors x {} dim ({:.1} MB region)\n",
        ids.len(),
        n_vec,
        dim,
        (n_vec * dim) as f64 / 1e6
    );
    println!(
        "{:>8} {:>10} {:>12} {:>10} {:>12}",
        "prefix", "lines/rec", "bytes req", "us", "GB/s"
    );

    let mut baseline = 0.0f64;
    for &m in &[64usize, 128, 192, 256, 384, 768] {
        let mut best = f64::MAX;
        for _ in 0..12 {
            let t = Instant::now();
            // Minimal arithmetic per byte: the point is to isolate the memory
            // system. A per-byte multiply-accumulate would scale with `m` too
            // and make a compute-bound loop look bandwidth-bound.
            let mut acc = 0u64;
            for &id in &ids {
                let off = id as usize * dim;
                let p = unsafe { data.as_ptr().add(off) as *const u64 };
                for k in 0..(m / 8) {
                    acc ^= unsafe { p.add(k).read_unaligned() };
                }
            }
            let e = t.elapsed().as_secs_f64();
            std::hint::black_box(acc);
            if e < best {
                best = e;
            }
        }
        let bytes = ids.len() * m;
        if m == 64 {
            baseline = best;
        }
        println!(
            "{:>8} {:>10} {:>12} {:>10.0} {:>12.1}",
            m,
            m.div_ceil(64),
            bytes,
            best * 1e6,
            bytes as f64 / best / 1e9
        );
    }
    println!(
        "\n   If time scales with bytes, the gather is bandwidth-bound and a\n   \
         prefix pass is worth ~6x. If time is flat, it is latency-bound\n   \
         (one miss per record) and prefix scanning buys nothing."
    );
    let _ = baseline;
}
