// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Microbenchmarks for Component-Level Performance Analysis
//!
//! This module provides isolated benchmarks for individual components
//! to identify actual performance bottlenecks with ±10% accuracy.
//!
//! ## Measured Components
//!
//! - **WAL write path**: write_no_flush_refs() overhead
//! - **MVCC tracking**: record_write() with HashSet insertion
//! - **PackedRow serialization**: pack_slice() encoding
//! - **DashMap operations**: concurrent HashMap performance
//! - **Memory allocation**: Vec<u8>, String creation overhead
//! - **Key formatting**: write!() vs KeyBuffer comparison
//!
//! ## Features
//!
//! - `jemalloc` - Use jemalloc as the global allocator for better performance
//!
//! ## Usage
//!
//! ```bash
//! # Default system allocator
//! cargo bench --package benchmarks -- --bench microbench
//!
//! # With jemalloc (better performance)
//! cargo bench --package benchmarks --features jemalloc -- --bench microbench
//! ```

// Use jemalloc as global allocator when feature is enabled
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use toondb_core::ToonValue;
use toondb_storage::key_buffer::KeyBuffer;
use toondb_storage::packed_row::{PackedColumnDef, PackedColumnType, PackedRow, PackedTableSchema};

/// Setup: Create a 4-column schema for PackedRow benchmarks
fn create_4col_schema() -> PackedTableSchema {
    PackedTableSchema {
        name: "users".to_string(),
        columns: vec![
            PackedColumnDef {
                name: "id".to_string(),
                col_type: PackedColumnType::Int64,
                nullable: false,
            },
            PackedColumnDef {
                name: "name".to_string(),
                col_type: PackedColumnType::Text,
                nullable: false,
            },
            PackedColumnDef {
                name: "email".to_string(),
                col_type: PackedColumnType::Text,
                nullable: false,
            },
            PackedColumnDef {
                name: "score".to_string(),
                col_type: PackedColumnType::Int64,
                nullable: false,
            },
        ],
    }
}

/// Setup: Create test values HashMap for PackedRow
fn create_test_values_map() -> HashMap<String, ToonValue> {
    let mut values = HashMap::new();
    values.insert("id".to_string(), ToonValue::Int(12345));
    values.insert("name".to_string(), ToonValue::Text("Test User".to_string()));
    values.insert(
        "email".to_string(),
        ToonValue::Text("test@example.com".to_string()),
    );
    values.insert("score".to_string(), ToonValue::Int(42));
    values
}

/// Setup: Create test values slice for PackedRow::pack_slice
fn create_test_values_slice() -> [Option<ToonValue>; 4] {
    [
        Some(ToonValue::Int(12345)),
        Some(ToonValue::Text("Test User".to_string())),
        Some(ToonValue::Text("test@example.com".to_string())),
        Some(ToonValue::Int(42)),
    ]
}

// =============================================================================
// Benchmark: Key Formatting
// =============================================================================

/// Benchmark: format!() macro for key construction
pub fn bench_format_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_formatting");
    group.throughput(Throughput::Elements(1));

    // Current approach: format!() with heap allocation
    group.bench_function("format_macro", |b| {
        b.iter(|| {
            let key = format!("{}/{}", black_box("users"), black_box(12345_u64));
            black_box(key)
        })
    });

    // Current approach: Cursor + write!()
    group.bench_function("cursor_write", |b| {
        b.iter(|| {
            use std::io::Write;
            let mut key_buf = [0u8; 128];
            let mut cursor = std::io::Cursor::new(&mut key_buf[..]);
            write!(cursor, "{}/{}", black_box("users"), black_box(12345_u64)).ok();
            let len = cursor.position() as usize;
            black_box(len)
        })
    });

    // Optimized: KeyBuffer (stack allocated, no heap)
    group.bench_function("key_buffer", |b| {
        b.iter(|| {
            let key = KeyBuffer::format_row_key(black_box("users"), black_box(12345));
            black_box(key)
        })
    });

    // Compare column key formatting
    group.bench_function("key_buffer_column", |b| {
        b.iter(|| {
            let key = KeyBuffer::format_column_key(
                black_box("users"),
                black_box(12345),
                black_box("email"),
            );
            black_box(key)
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: DashMap Operations
// =============================================================================

/// Benchmark: DashMap insert/lookup performance
pub fn bench_dashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("dashmap_ops");

    // Single insert
    group.bench_function("insert_100b", |b| {
        let map: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("k{:08}", i).into_bytes();
            let val = vec![0u8; 100];
            map.insert(black_box(key), black_box(val));
            i = i.wrapping_add(1);
        })
    });

    // Lookup (pre-populated map)
    group.bench_function("lookup_100b", |b| {
        let map: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        for i in 0..10000 {
            let key = format!("k{:08}", i).into_bytes();
            map.insert(key, vec![0u8; 100]);
        }
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("k{:08}", i % 10000).into_bytes();
            black_box(map.get(&key));
            i = i.wrapping_add(1);
        })
    });

    // Entry API (upsert pattern)
    group.bench_function("entry_api", |b| {
        let map: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("k{:08}", i).into_bytes();
            map.entry(key).or_insert_with(|| vec![0u8; 100]);
            i = i.wrapping_add(1);
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: HashSet Operations (MVCC tracking simulation)
// =============================================================================

/// Benchmark: HashSet<Vec<u8>> vs HashSet<[u8; 32]> for MVCC write_set
pub fn bench_mvcc_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc_tracking");

    // Current: HashSet<Vec<u8>> with key.to_vec()
    group.bench_function("hashset_vec_insert", |b| {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("users/{}", i).into_bytes();
            set.insert(black_box(key));
            i = i.wrapping_add(1);
            if i.is_multiple_of(10000) {
                set.clear();
            }
        })
    });

    // Alternative: HashSet<u64> with hash of key
    group.bench_function("hashset_hash_insert", |b| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut set: HashSet<u64> = HashSet::new();
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("users/{}", i);
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            set.insert(black_box(hash));
            i = i.wrapping_add(1);
            if i.is_multiple_of(10000) {
                set.clear();
            }
        })
    });

    // Intersection check (SSI validation simulation)
    group.bench_function("hashset_is_disjoint_1000", |b| {
        let set1: HashSet<Vec<u8>> = (0..1000)
            .map(|i| format!("key_{:06}", i).into_bytes())
            .collect();
        let set2: HashSet<Vec<u8>> = (500..1500)
            .map(|i| format!("key_{:06}", i).into_bytes())
            .collect();

        b.iter(|| black_box(set1.is_disjoint(&set2)))
    });

    group.finish();
}

// =============================================================================
// Benchmark: PackedRow Serialization
// =============================================================================

/// Benchmark: PackedRow::pack and pack_slice
pub fn bench_packed_row(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_row");

    let schema = create_4col_schema();

    // pack() with HashMap lookup
    group.bench_function("pack_hashmap", |b| {
        let values = create_test_values_map();
        b.iter(|| {
            let packed = PackedRow::pack(black_box(&schema), black_box(&values));
            black_box(packed)
        })
    });

    // pack_slice() with ordered values (preferred for hot path)
    group.bench_function("pack_slice", |b| {
        let values = create_test_values_slice();
        let refs: Vec<Option<&ToonValue>> = values.iter().map(|v| v.as_ref()).collect();
        b.iter(|| {
            let packed = PackedRow::pack_slice(black_box(&schema), black_box(&refs));
            black_box(packed)
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: Memory Allocation Overhead
// =============================================================================

/// Benchmark: Various allocation patterns
pub fn bench_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocations");

    // Vec<u8> with capacity
    group.bench_function("vec_u8_cap_100", |b| {
        b.iter(|| {
            let v: Vec<u8> = Vec::with_capacity(100);
            black_box(v)
        })
    });

    // Vec<u8> with zeroed data
    group.bench_function("vec_u8_zeroed_100", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![0u8; 100];
            black_box(v)
        })
    });

    // String formatting
    group.bench_function("format_string_short", |b| {
        b.iter(|| black_box(format!("User {}", black_box(12345))))
    });

    group.bench_function("format_string_email", |b| {
        b.iter(|| black_box(format!("user{}@example.com", black_box(12345))))
    });

    // to_vec() cost
    group.bench_function("slice_to_vec_20", |b| {
        let data = b"users/12345678901234";
        b.iter(|| black_box(black_box(data).to_vec()))
    });

    // Clone String
    group.bench_function("string_clone", |b| {
        let s = "test@example.com".to_string();
        b.iter(|| black_box(black_box(&s).clone()))
    });

    group.finish();
}

// =============================================================================
// Benchmark: Lock Contention Simulation
// =============================================================================

/// Benchmark: Mutex acquisition patterns
pub fn bench_lock_patterns(c: &mut Criterion) {
    use parking_lot::Mutex;

    let mut group = c.benchmark_group("lock_patterns");

    // Uncontended mutex (best case)
    group.bench_function("mutex_uncontended", |b| {
        let mutex = Mutex::new(0u64);
        b.iter(|| {
            let mut guard = mutex.lock();
            *guard += 1;
            black_box(*guard);
        })
    });

    // Multiple writes under single lock (batched pattern)
    group.bench_function("mutex_batched_1000", |b| {
        let mutex = Mutex::new(Vec::with_capacity(10000));
        b.iter(|| {
            let mut guard = mutex.lock();
            for i in 0..1000u64 {
                guard.push(i);
            }
            guard.clear();
            black_box(&*guard);
        })
    });

    // Multiple lock acquisitions (current pattern simulation)
    group.bench_function("mutex_per_write_1000", |b| {
        let mutex = Mutex::new(Vec::with_capacity(10000));
        b.iter(|| {
            for i in 0..1000u64 {
                let mut guard = mutex.lock();
                guard.push(i);
            }
            mutex.lock().clear();
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: End-to-End Insert Path Simulation
// =============================================================================

/// Benchmark: Simulated insert path component breakdown
pub fn bench_insert_path_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_path");

    let schema = create_4col_schema();

    // Full simulated insert (without actual WAL/memtable)
    group.bench_function("simulated_full_insert", |b| {
        let map: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        let mut mvcc_set: HashSet<Vec<u8>> = HashSet::new();
        let mut i = 0u64;

        b.iter(|| {
            // 1. Key construction (current: format!)
            let key = format!("users/{}", i);
            let key_bytes = key.as_bytes();

            // 2. MVCC tracking (current: to_vec())
            mvcc_set.insert(key_bytes.to_vec());

            // 3. Value construction
            let values = create_test_values_slice();
            let refs: Vec<Option<&ToonValue>> = values.iter().map(|v| v.as_ref()).collect();
            let packed = PackedRow::pack_slice(&schema, &refs);

            // 4. DashMap insert
            map.insert(key_bytes.to_vec(), packed.as_bytes().to_vec());

            i = i.wrapping_add(1);
            if i.is_multiple_of(10000) {
                map.clear();
                mvcc_set.clear();
            }

            black_box(&map);
        })
    });

    // Optimized path simulation
    group.bench_function("simulated_optimized_insert", |b| {
        let map: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        let mut mvcc_set: HashSet<u64> = HashSet::new();
        let mut i = 0u64;

        // Pre-create values (simulating pre-allocation)
        let values = create_test_values_slice();
        let refs: Vec<Option<&ToonValue>> = values.iter().map(|v| v.as_ref()).collect();

        b.iter(|| {
            // 1. Key construction (optimized: KeyBuffer)
            let key = KeyBuffer::format_row_key("users", i);
            let key_bytes = key.as_bytes();

            // 2. MVCC tracking (optimized: hash only)
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            key_bytes.hash(&mut hasher);
            mvcc_set.insert(hasher.finish());

            // 3. Value serialization
            let packed = PackedRow::pack_slice(&schema, &refs);

            // 4. DashMap insert
            map.insert(key_bytes.to_vec(), packed.as_bytes().to_vec());

            i = i.wrapping_add(1);
            if i.is_multiple_of(10000) {
                map.clear();
                mvcc_set.clear();
            }

            black_box(&map);
        })
    });

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    name = microbench;
    config = Criterion::default()
        .sample_size(1000)
        .measurement_time(Duration::from_secs(5));
    targets =
        bench_format_key,
        bench_dashmap,
        bench_mvcc_tracking,
        bench_packed_row,
        bench_allocations,
        bench_lock_patterns,
        bench_insert_path_breakdown,
);

criterion_main!(microbench);

// =============================================================================
// Unit Tests for Benchmark Correctness
// =============================================================================

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_key_buffer_equivalence() {
        // Verify KeyBuffer produces same output as format!
        let formatted = format!("users/{}", 12345);
        let key_buf = KeyBuffer::format_row_key("users", 12345);
        assert_eq!(formatted.as_bytes(), key_buf.as_bytes());
    }

    #[test]
    fn test_packed_row_schema() {
        let schema = create_4col_schema();
        assert_eq!(schema.columns.len(), 4);
        assert_eq!(schema.columns[0].name, "id");
    }

    #[test]
    fn test_values_slice_consistency() {
        let values = create_test_values_slice();
        assert!(matches!(values[0], Some(ToonValue::Int(12345))));
        assert!(matches!(values[3], Some(ToonValue::Int(42))));
    }
}
