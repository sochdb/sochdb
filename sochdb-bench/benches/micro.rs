//! Criterion microbenchmarks for SochDB vs SQLite point operations.
//!
//! Run with: `cargo bench --bench micro`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use sochdb_bench::adapters::sochdb_adapter::SochDbAdapter;
use sochdb_bench::adapters::sqlite_adapter::SqliteAdapter;
use sochdb_bench::{BenchDb, DataGen};
use tempfile::TempDir;

fn bench_point_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_write");
    let value_size = 256;

    for db_name in ["SochDB", "SQLite"] {
        group.bench_with_input(
            BenchmarkId::new(db_name, value_size),
            &value_size,
            |b, &sz| {
                let tmp = TempDir::new().unwrap();
                let mut db: Box<dyn BenchDb> = match db_name {
                    "SochDB" => Box::new(SochDbAdapter::new(tmp.path()).unwrap()),
                    "SQLite" => Box::new(SqliteAdapter::new(tmp.path()).unwrap()),
                    _ => unreachable!(),
                };
                db.setup_kv_table().unwrap();

                let mut gen = DataGen::new(42);
                let mut i = 0u64;

                b.iter(|| {
                    let key = gen.kv_key(i);
                    let val = gen.random_value(sz);
                    db.put(&key, &val).unwrap();
                    i += 1;
                });
            },
        );
    }
    group.finish();
}

fn bench_point_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_read");
    let n = 10_000usize;

    for db_name in ["SochDB", "SQLite"] {
        group.bench_function(BenchmarkId::new(db_name, n), |b| {
            let tmp = TempDir::new().unwrap();
            let mut db: Box<dyn BenchDb> = match db_name {
                "SochDB" => Box::new(SochDbAdapter::new(tmp.path()).unwrap()),
                "SQLite" => Box::new(SqliteAdapter::new(tmp.path()).unwrap()),
                _ => unreachable!(),
            };
            db.setup_kv_table().unwrap();

            // Pre-populate.
            let mut gen = DataGen::new(42);
            for i in 0..n {
                let key = gen.kv_key(i as u64);
                let val = gen.random_value(256);
                db.put(&key, &val).unwrap();
            }

            let data_gen = DataGen::new(42);
            let mut read_gen = DataGen::new(99);
            let indices = read_gen.shuffled_indices(n);
            let mut idx = 0;

            b.iter(|| {
                let key = data_gen.kv_key(indices[idx % n] as u64);
                let _ = db.get(&key).unwrap();
                idx += 1;
            });
        });
    }
    group.finish();
}

fn bench_batch_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_write_1000");

    for db_name in ["SochDB", "SQLite"] {
        group.bench_function(BenchmarkId::new(db_name, 1000), |b| {
            let tmp = TempDir::new().unwrap();
            let mut db: Box<dyn BenchDb> = match db_name {
                "SochDB" => Box::new(SochDbAdapter::new(tmp.path()).unwrap()),
                "SQLite" => Box::new(SqliteAdapter::new(tmp.path()).unwrap()),
                _ => unreachable!(),
            };
            db.setup_kv_table().unwrap();

            let mut gen = DataGen::new(42);
            let mut offset = 0u64;

            b.iter(|| {
                let keys: Vec<Vec<u8>> = (0..1000).map(|_| {
                    offset += 1;
                    format!("bm:{:08x}", offset).into_bytes()
                }).collect();
                let vals: Vec<Vec<u8>> = (0..1000).map(|_| gen.random_value(256)).collect();
                let pairs: Vec<(&[u8], &[u8])> = keys.iter().zip(vals.iter())
                    .map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
                db.batch_put(&pairs).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_point_write, bench_point_read, bench_batch_write);
criterion_main!(benches);
