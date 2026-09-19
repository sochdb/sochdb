//! Is the rerank gather paying for page walks, and would hugepages fix it?
//!
//! Rerank scatters reads across a 153 MB region. At 4 KB that region is 37,500
//! pages, while this CPU's L2 TLB holds on the order of 2,000 entries, so a
//! random gather misses the TLB almost every time. Each miss is a page walk:
//! up to four dependent memory references *before* the one the code asked for.
//! If that is what the gather is actually waiting on, 2 MB pages would cover
//! the same region in 76 entries -- small enough to sit in the TLB entirely --
//! and the walks would disappear.
//!
//! That is a plausible story, and plausible stories are exactly what needs
//! measuring rather than implementing. Two things have to be true for it to pay
//! off, and they are separable:
//!
//!   1. the page walks cost meaningful time, and
//!   2. we can actually get hugepages under the data.
//!
//! This probe tests them independently. It runs the identical gather over the
//! same ids against three mappings -- ordinary anonymous memory, anonymous
//! memory with `MADV_HUGEPAGE`, and a file mmap with `MADV_HUGEPAGE`, which is
//! how `Segment` maps its data -- and reports, from `/proc/self/smaps`, how
//! much of each mapping the kernel really backed with hugepages. Timing
//! without that check would be measuring an intention rather than a mapping.
//!
//! The file-backed row is the one that decides whether any of this is
//! actionable for `Segment`, because `MADV_HUGEPAGE` on a mapped file only does
//! anything if the kernel was built with `CONFIG_READ_ONLY_THP_FOR_FS`. Where
//! it was not, the call succeeds and changes nothing, which is precisely the
//! kind of silent no-op that gets mistaken for a completed optimisation.
//!
//! ## Measured, i9-13900K, 154 MB, 24,378 scattered gathers, min of 8
//!
//! Medians of four runs on an otherwise quiet machine; the spread between runs
//! was under 5% on every row. Under load every row inflates and the ordering of
//! the bottom two can invert, so this needs a quiet box to mean anything.
//!
//! ```text
//! mapping                       ns/record  hugepage MB     vs 4KB
//! anon 4KB                           56.0            0      1.00x
//! anon MADV_HUGEPAGE                 37.2          146      1.51x
//! file mmap + MADV_HUGEPAGE          38.4            0      1.46x
//! ```
//!
//! Both halves of the hypothesis were answered, and they disagree.
//!
//! The physics is real: hugepages take the gather from 56.0 ns to 37.2 ns per
//! record, so page walks were 18.8 ns, a third of a 4 KB gather. That is a
//! large single component and exactly the effect predicted.
//!
//! It is still not worth doing, and the file-backed row is why. `Segment` does
//! not gather from anonymous 4 KB memory, so 56.0 is not its baseline -- it
//! gathers from a mapped file, which already measures 38.4 without a single
//! hugepage under it (ext4 serves the page cache in large folios, and
//! contiguous PTEs let the walker coalesce). The real headroom for `Segment` is
//! 38.4 -> 37.2, about 3% on one stage, not the 1.5x the page arithmetic
//! implies. The win the hypothesis predicted had already been collected, by the
//! filesystem, for free.
//!
//! That 3% is also not reachable by advice. `MADV_HUGEPAGE` on the file mapping
//! returned success and produced 0 MB of hugepages, matching this kernel's
//! `CONFIG_READ_ONLY_THP_FOR_FS=n`. Collecting it would mean copying every hot
//! segment into anonymous memory: 154 MB per segment of additional resident
//! memory, plus the copy, to make one pipeline stage 3% faster.
//!
//! So: do not add `MADV_HUGEPAGE` to `Segment::open`. On this kernel it does
//! nothing, and on a kernel where it works it is worth about 3% rather than the
//! 1.5x the page arithmetic implies. Re-run this probe before reopening the
//! question; it reports what the kernel actually granted, so it will say so if
//! the answer changes.

use std::io::Write;
use std::time::Instant;

const N_VEC: usize = 200_000;
const DIM: usize = 768;
const N_CAND: usize = 24_378;

/// One mapping under test, plus what the kernel actually gave us.
struct Mapping {
    label: &'static str,
    ptr: *mut u8,
    /// Bytes the kernel really backed with hugepages, from smaps.
    huge_kb: usize,
}

/// Ask smaps what this mapping is really backed by.
///
/// `madvise` returning 0 means the advice was accepted, not that it was acted
/// on, so the only honest source for "did we get hugepages" is the kernel's own
/// accounting of the range.
fn huge_kb_for(ptr: *mut u8, len: usize) -> usize {
    let start = ptr as usize;
    let end = start + len;
    let smaps = match std::fs::read_to_string("/proc/self/smaps") {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let mut in_range = false;
    let mut total = 0usize;
    for line in smaps.lines() {
        if let Some((lo, hi)) = line
            .split_once(' ')
            .and_then(|(range, _)| range.split_once('-'))
            .and_then(|(lo, hi)| {
                Some((
                    usize::from_str_radix(lo, 16).ok()?,
                    usize::from_str_radix(hi, 16).ok()?,
                ))
            })
        {
            // A header line. Ranges can be split by the kernel, so keep
            // accumulating over every vma that overlaps ours.
            in_range = lo < end && hi > start;
            continue;
        }
        if in_range
            && (line.starts_with("AnonHugePages:") || line.starts_with("FilePmdMapped:"))
            && let Some(kb) = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<usize>().ok())
        {
            total += kb;
        }
    }
    total
}

fn map_anon(label: &'static str, len: usize, advise_huge: bool) -> Mapping {
    unsafe {
        // 2 MB alignment matters: the kernel can only install a PMD-sized page
        // on a PMD-aligned boundary, so an unaligned mapping silently gets
        // fewer hugepages than it asked for. Over-allocate and trim.
        let over = len + (2 << 20);
        let raw = libc::mmap(
            std::ptr::null_mut(),
            over,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(raw, libc::MAP_FAILED, "mmap failed");

        let aligned = ((raw as usize) + (2 << 20) - 1) & !((2usize << 20) - 1);
        let ptr = aligned as *mut u8;

        if advise_huge {
            let rc = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
            assert_eq!(rc, 0, "madvise(MADV_HUGEPAGE) rejected");
        }

        // Fault every page in. Timing a gather against unfaulted memory would
        // measure minor faults, not the TLB.
        std::ptr::write_bytes(ptr, 0, len);
        for i in 0..len {
            *ptr.add(i) = (i % 251) as u8;
        }

        Mapping {
            label,
            ptr,
            huge_kb: huge_kb_for(ptr, len),
        }
    }
}

/// Map a real file the way `Segment` does, then advise hugepages on it.
fn map_file(label: &'static str, len: usize, path: &std::path::Path) -> Mapping {
    {
        let mut f = std::fs::File::create(path).expect("create backing file");
        let chunk: Vec<u8> = (0..(1 << 20)).map(|i| (i % 251) as u8).collect();
        let mut written = 0;
        while written < len {
            let n = chunk.len().min(len - written);
            f.write_all(&chunk[..n]).expect("write backing file");
            written += n;
        }
        f.sync_all().ok();
    }

    let file = std::fs::File::open(path).expect("open backing file");
    unsafe {
        use std::os::unix::io::AsRawFd;
        let raw = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        );
        assert_ne!(raw, libc::MAP_FAILED, "file mmap failed");
        let ptr = raw as *mut u8;

        // Succeeds on any kernel. Only does something where
        // CONFIG_READ_ONLY_THP_FOR_FS is enabled -- the point of the probe.
        libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
        libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);

        // Fault the whole file in so the gather hits the page cache, not the
        // device. Otherwise this row measures the SSD.
        let mut acc = 0u64;
        let mut off = 0;
        while off < len {
            acc = acc.wrapping_add(*ptr.add(off) as u64);
            off += 4096;
        }
        std::hint::black_box(acc);

        Mapping {
            label,
            ptr,
            huge_kb: huge_kb_for(ptr, len),
        }
    }
}

/// The gather under test: read one scattered `DIM`-byte record per candidate.
///
/// Deliberately the same shape as rerank -- one dependent access per record,
/// ids ascending, spread over the whole region -- because the quantity of
/// interest is the per-record stall, and that only appears when the access
/// pattern defeats the prefetcher the way the real one does.
fn gather(m: &Mapping, ids: &[u32]) -> (u64, f64) {
    let reps = 8;
    let mut best = f64::MAX;
    let mut acc = 0u64;

    for _ in 0..reps {
        let t = Instant::now();
        let mut local = 0u64;
        for &id in ids {
            let off = id as usize * DIM;
            unsafe {
                // Touch one byte per cache line, so the cost is the walk and the
                // miss rather than the bytes.
                let base = m.ptr.add(off);
                let mut d = 0;
                while d < DIM {
                    local = local.wrapping_add(*base.add(d) as u64);
                    d += 64;
                }
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / ids.len() as f64;
        acc = acc.wrapping_add(local);
        if ns < best {
            best = ns;
        }
    }
    (acc, best)
}

fn main() {
    let len = N_VEC * DIM;

    // Ascending, evenly spread ids -- what the bitset union emits.
    let stride = (N_VEC / N_CAND).max(1);
    let ids: Vec<u32> = (0..N_CAND).map(|i| ((i * stride) % N_VEC) as u32).collect();

    let tmp = std::env::var("SOCHDB_BENCH_TMP").unwrap_or_else(|_| "/tmp".into());
    let file_path = std::path::Path::new(&tmp).join("sochdb_tlb_probe.bin");

    let mappings = vec![
        map_anon("anon 4KB", len, false),
        map_anon("anon MADV_HUGEPAGE", len, true),
        map_file("file mmap + MADV_HUGEPAGE", len, &file_path),
    ];

    println!(
        "tlb_probe: {} records x {}B = {:.0} MB, {} scattered gathers, min of 8\n",
        N_VEC,
        DIM,
        len as f64 / 1e6,
        N_CAND
    );
    println!(
        "{:<28} {:>10} {:>12} {:>10}",
        "mapping", "ns/record", "hugepage MB", "vs 4KB"
    );

    let mut baseline = 0.0;
    for m in &mappings {
        let (acc, ns) = gather(m, &ids);
        std::hint::black_box(acc);
        if baseline == 0.0 {
            baseline = ns;
        }
        println!(
            "{:<28} {:>10.1} {:>12.0} {:>9.2}x",
            m.label,
            ns,
            m.huge_kb as f64 / 1024.0,
            baseline / ns
        );
    }

    let anon_huge = mappings[1].huge_kb;
    let file_huge = mappings[2].huge_kb;
    println!();
    if anon_huge == 0 {
        println!(
            "NOTE: anonymous MADV_HUGEPAGE produced no hugepages either -- check\n\
             /sys/kernel/mm/transparent_hugepage/enabled. The comparison above is\n\
             not measuring what it claims to."
        );
    } else if file_huge == 0 {
        println!(
            "NOTE: anonymous hugepages worked ({} MB) but the file mapping got none.\n\
             MADV_HUGEPAGE on a mapped file needs CONFIG_READ_ONLY_THP_FOR_FS; where\n\
             it is off, the call succeeds and does nothing. Adding it to Segment::open\n\
             would be a silent no-op, so any TLB win shown above is only reachable by\n\
             staging the hot data into anonymous memory -- which costs a copy and the\n\
             RAM, and is only worth it if the anon gap above is large.",
            anon_huge / 1024
        );
    }

    let _ = std::fs::remove_file(&file_path);
}
