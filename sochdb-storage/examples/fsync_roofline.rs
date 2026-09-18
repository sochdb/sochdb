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
//!   grow + fsync       <- what TxnWal does today (txn_wal.rs:1052)
//!   grow + fdatasync
//!   prealloc + fsync
//!   prealloc + fdatasync
//!
//! Run against a directory on real storage. On tmpfs every row is a memcpy.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Instant;

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
}

fn run(dir: &Path, cfg: &Cfg, ops: usize, record: usize) {
    let path = dir.join(format!(
        "roofline_{}.wal",
        cfg.name.replace(['+', ' ', '/'], "_")
    ));
    let _ = std::fs::remove_file(&path);
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&path)
        .expect("open");

    let rec = record.max(cfg.align);
    let span = (ops as u64 + 16) * rec as u64;
    match cfg.prep {
        "fallocate" => preallocate(&file, span).expect("fallocate"),
        "zerofill" => zero_fill(&mut file, span).expect("zerofill"),
        _ => {}
    }

    let buf = vec![0xABu8; rec];
    let mut lat = Vec::with_capacity(ops / cfg.batch + 1);

    let mut pending = 0usize;
    let mut t0 = Instant::now();
    for _ in 0..ops {
        file.write_all(&buf).expect("write");
        pending += 1;
        if pending == cfg.batch {
            if cfg.datasync {
                file.sync_data().expect("sync_data");
            } else {
                file.sync_all().expect("sync_all");
            }
            // Charge the whole barrier to the batch, then report per record.
            lat.push(t0.elapsed().as_nanos() as u64 / cfg.batch as u64);
            pending = 0;
            t0 = Instant::now();
        }
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
        // What TxnWal does today.
        Cfg {
            name: "grow+fsync",
            prep: "none",
            datasync: false,
            align: 0,
            batch: 1,
        },
        Cfg {
            name: "grow+fdatasync",
            prep: "none",
            datasync: true,
            align: 0,
            batch: 1,
        },
        Cfg {
            name: "fallocate+fsync",
            prep: "fallocate",
            datasync: false,
            align: 0,
            batch: 1,
        },
        Cfg {
            name: "fallocate+fdatasync",
            prep: "fallocate",
            datasync: true,
            align: 0,
            batch: 1,
        },
        // Extents already written, so nothing metadata-shaped remains.
        Cfg {
            name: "zerofill+fdatasync",
            prep: "zerofill",
            datasync: true,
            align: 0,
            batch: 1,
        },
        // Same, with records padded to a sector and to a page.
        Cfg {
            name: "zerofill+512B",
            prep: "zerofill",
            datasync: true,
            align: 512,
            batch: 1,
        },
        Cfg {
            name: "zerofill+4KB",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 1,
        },
        // The group-commit ceiling: one barrier per N records.
        Cfg {
            name: "zerofill+4KB batch=8",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 8,
        },
        Cfg {
            name: "zerofill+4KB batch=64",
            prep: "zerofill",
            datasync: true,
            align: 4096,
            batch: 64,
        },
    ];
    for c in &cfgs {
        run(dir, c, ops, record);
    }
}
