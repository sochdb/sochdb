//! What a durable append actually costs on this machine.
//!
//! `durable_concurrent` measures ~190 commits/s single-threaded, which is 5.3 ms
//! per durable commit. An NVMe device programs a page in tens of microseconds,
//! so before optimising anything in the WAL it is worth knowing which part of
//! that 5.3 ms is the device and which part is the filesystem.
//!
//! The variable under test is what `fsync` has to do besides push data at the
//! device. A WAL is append-only, so every commit extends the file and changes
//! `i_size` in the inode. `fsync` is defined to persist file *metadata* as well
//! as data, so on a journalling filesystem that dirty inode forces a journal
//! transaction commit -- a second, ordered, write-and-flush that the data itself
//! never needed.
//!
//! `fdatasync` is allowed to skip the inode, but only when the metadata is not
//! required to read the data back. A file that just grew needs its new size, so
//! `fdatasync` on a growing file still commits the journal. Preallocating the
//! file removes the size change, and only then is `fdatasync` free to issue a
//! bare device flush.
//!
//! That gives four combinations, and the interesting quantity is not any one of
//! them but the spread between them, which is the filesystem tax:
//!
//!   grow + fsync       <- what TxnWal did before (txn_wal.rs:1052)
//!   grow + fdatasync
//!   prealloc + fsync
//!   prealloc + fdatasync
//!
//! Measured here, the spread is the whole story: grow+fsync and grow+fdatasync
//! are indistinguishable (~5.2 ms), and so are prealloc+fsync and grow+fsync.
//! Only prealloc *and* fdatasync together drop to ~0.97 ms. Neither half is an
//! optimisation on its own, which is why this had to be measured rather than
//! reasoned about. `fallocate` is also not enough: it leaves unwritten extents,
//! and the first write to one clears that flag, which is a metadata update and
//! so another journal commit. Visible as p99 5688 us versus 1992 us for a
//! runway that was genuinely zero-filled.
//!
//! The remaining ~0.97 ms is not filesystem overhead, and the `direct`/`dsync`
//! configs exist to prove it. The hypothesis was that the cost is the kernel's
//! write-then-FLUSH protocol -- two device round trips, where FLUSH must drain
//! the drive's entire volatile cache -- and that `O_DIRECT|O_DSYNC` would
//! collapse it into one FUA write. It does not. `O_DSYNC` measures 969 us,
//! identical to fdatasync's 967, and `O_DIRECT|O_DSYNC` is 3408 us, 3.5x
//! *worse*: bypassing the page cache costs far more than the barrier it saves.
//!
//! So ~0.96 ms is this device (a 990 PRO has no power-loss protection, so every
//! barrier must actually move DRAM to NAND), and it is a fixed cost: 968 us
//! covering 8 records, 960 us covering 64. Durable throughput is therefore
//! barrier_cost / batch_size, and batching is the only remaining lever -- which
//! is what `group_commit.rs` exists to pull.
//!
//! Run against a directory on real storage. On tmpfs every row is a memcpy.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Instant;

/// A page-aligned buffer. `O_DIRECT` bypasses the page cache, so the kernel
/// DMAs straight out of userspace and requires the buffer address, the file
/// offset and the length to all be block-aligned.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), align).expect("layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned alloc failed");
        Self { ptr, len, layout }
    }

    fn fill(&mut self, b: u8) {
        unsafe { std::ptr::write_bytes(self.ptr, b, self.len) };
    }
}

impl std::ops::Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// `fallocate` the whole file and fsync once, so the extent map and the size are
/// already durable before the timed region starts.
///
/// This is not sufficient on its own. `fallocate` produces *unwritten* extents:
/// the blocks are reserved but flagged as never-written so reads return zeros.
/// The first write to such an extent has to clear that flag, which is an extent
/// tree update, which is metadata, which is a journal commit -- the exact cost
/// preallocating was meant to avoid. `zero_fill` below actually writes the
/// blocks so the extents are already in the written state.
fn preallocate(file: &File, len: u64) -> std::io::Result<()> {
    let rc = unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len as libc::off_t) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    file.sync_all()
}

/// Write real zeros over the whole file so every extent is in the written state,
/// then rewind. After this a write is a pure in-place overwrite: no size change,
/// no extent conversion, nothing for the journal to record.
fn zero_fill(file: &mut File, len: u64) -> std::io::Result<()> {
    let chunk = vec![0u8; 1 << 20];
    let mut remaining = len;
    while remaining > 0 {
        let n = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..n])?;
        remaining -= n as u64;
    }
    file.sync_all()?;
    file.seek(SeekFrom::Start(0))?;
    Ok(())
}

#[derive(Default)]
struct Cfg {
    name: &'static str,
    /// `none`, `fallocate` or `zerofill`
    prep: &'static str,
    datasync: bool,
    /// Pad each record out to this many bytes. A WAL record smaller than a
    /// sector forces the layer below to read-modify-write the containing block.
    align: usize,
    /// Durability barriers are issued once per this many records. This is group
    /// commit expressed at the syscall level, and it is the ceiling any
    /// in-process batching scheme can reach.
    batch: usize,
    /// `O_DIRECT`: DMA straight from userspace, no page cache.
    direct: bool,
    /// `O_DSYNC`: the write itself is the durability barrier, so no separate
    /// flush is issued. The kernel can satisfy this with a single FUA write
    /// command instead of a write followed by a cache flush.
    dsync: bool,
}

fn run(dir: &Path, cfg: &Cfg, ops: usize, record: usize) {
    let path = dir.join(format!(
        "roofline_{}.wal",
        cfg.name.replace(['+', ' ', '/', '|', '='], "_")
    ));
    let _ = std::fs::remove_file(&path);

    let rec = record.max(cfg.align);
    let batch = cfg.batch.max(1);
    let span = (ops as u64 + 16) * rec as u64;

    // Prepare through an ordinary buffered handle: zero_fill writes at arbitrary
    // lengths and offsets, which an O_DIRECT handle would reject.
    {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .expect("open");
        match cfg.prep {
            "fallocate" => preallocate(&file, span).expect("fallocate"),
            "zerofill" => zero_fill(&mut file, span).expect("zerofill"),
            _ => {}
        }
    }

    let mut flags = 0;
    if cfg.direct {
        flags |= libc::O_DIRECT;
    }
    if cfg.dsync {
        flags |= libc::O_DSYNC;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(flags)
        .open(&path)
        .expect("open test handle");

    let chunk = rec * batch;
    let mut buf = AlignedBuf::new(chunk, 4096);
    buf.fill(0xAB);

    let rounds = (ops / batch).max(1);
    let mut lat = Vec::with_capacity(rounds);
    let mut off = 0u64;

    for _ in 0..rounds {
        let t0 = Instant::now();
        if cfg.dsync {
            // One write, already durable when it returns. There is no second
            // trip to the device to flush a cache.
            file.write_all_at(&buf[..chunk], off).expect("write");
        } else {
            for i in 0..batch {
                file.write_all_at(&buf[..rec], off + (i * rec) as u64)
                    .expect("write");
            }
            if cfg.datasync {
                file.sync_data().expect("sync_data");
            } else {
                file.sync_all().expect("sync_all");
            }
        }
        // Charge the whole barrier to the batch, then report per record.
        lat.push(t0.elapsed().as_nanos() as u64 / batch as u64);
        off += chunk as u64;
    }

    lat.sort_unstable();
    let mean = lat.iter().sum::<u64>() as f64 / lat.len().max(1) as f64;
    println!(
        "{:<24} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>10.0}",
        cfg.name,
        lat[0] as f64 / 1000.0,
        percentile(&lat, 0.50) as f64 / 1000.0,
        percentile(&lat, 0.99) as f64 / 1000.0,
        mean / 1000.0,
        1_000_000_000.0 / mean,
    );
    let _ = std::fs::remove_file(&path);
}

fn main() {
    let dir = std::env::var("SOCHDB_BENCH_TMP")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).expect("mkdir");

    let ops: usize = std::env::var("OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let record: usize = std::env::var("RECORD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    println!(
        "durable append roofline: {} ops x {} B in {}",
        ops,
        record,
        dir.display()
    );
    println!(
        "{:<24} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "mode", "min us", "p50 us", "p99 us", "mean us", "ops/s"
    );
    println!("{}", "-".repeat(76));

    let cfgs = [
        // What TxnWal did before the runway.
        Cfg {
            name: "grow+fsync",
            prep: "none",
            batch: 1,
            ..Default::default()
        },
        Cfg {
            name: "grow+fdatasync",
            prep: "none",
            datasync: true,
            batch: 1,
            ..Default::default()
        },
        Cfg {
            name: "fallocate+fdatasync",
            prep: "fallocate",
            datasync: true,
            batch: 1,
            ..Default::default()
        },
        // Extents already written, so nothing metadata-shaped remains.
        // This is what TxnWal does now.
        Cfg {
            name: "zerofill+fdatasync",
            prep: "zerofill",
            datasync: true,
            batch: 1,
            ..Default::default()
        },
        Cfg {
            name: "zerofill+4KB",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 1,
            ..Default::default()
        },
        // The group-commit ceiling: one barrier per N records.
        Cfg {
            name: "zerofill+4KB batch=8",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 8,
            ..Default::default()
        },
        Cfg {
            name: "zerofill+4KB batch=64",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 64,
            ..Default::default()
        },
        // Is the ~1 ms barrier the device, or the protocol used to reach it?
        // fdatasync is write-then-FLUSH: two trips, and the FLUSH must drain the
        // drive's whole volatile write cache. O_DSYNC lets the kernel issue one
        // FUA write instead, which commits only these blocks.
        Cfg {
            name: "O_DSYNC 4KB",
            prep: "zerofill",
            align: 4096,
            batch: 1,
            dsync: true,
            ..Default::default()
        },
        Cfg {
            name: "O_DIRECT|O_DSYNC 4KB",
            prep: "zerofill",
            align: 4096,
            batch: 1,
            direct: true,
            dsync: true,
            ..Default::default()
        },
        Cfg {
            name: "O_DIRECT|O_DSYNC b=8",
            prep: "zerofill",
            align: 4096,
            batch: 8,
            direct: true,
            dsync: true,
            ..Default::default()
        },
        Cfg {
            name: "O_DIRECT|O_DSYNC b=64",
            prep: "zerofill",
            align: 4096,
            batch: 64,
            direct: true,
            dsync: true,
            ..Default::default()
        },
    ];
    for c in &cfgs {
        run(dir, c, ops, record);
    }
}
