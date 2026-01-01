#!/usr/bin/env python3
"""Run pure Rust HNSW performance test."""

import subprocess
import time

def run_rust_test():
    print("=" * 80)
    print("RUNNING PURE RUST HNSW PERFORMANCE TEST")
    print("=" * 80)
    
    # Create a simple Rust test program
    rust_code = '''
use std::time::Instant;
use toondb_index::hnsw::{HnswConfig, HnswIndex};
use rand::Rng;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn main() {
    println!("Testing pure Rust HNSW performance...");
    
    let dimension = 768;
    let n_vectors = 1000;
    let config = HnswConfig {
        max_connections: 8,
        ef_construction: 25,
        ..Default::default()
    };
    
    println!("Configuration: {}D, {} vectors, M={}, ef={}", 
             dimension, n_vectors, config.max_connections, config.ef_construction);
    
    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| generate_random_vector(dimension))
        .collect();
    
    // Create index
    let index = HnswIndex::new(dimension, config);
    
    // Warmup
    for i in 0..10 {
        index.insert(i + 1000000, vectors[i].clone()).unwrap();
    }
    
    println!("Starting timed insertion...");
    let start = Instant::now();
    
    for (i, vector) in vectors.iter().enumerate() {
        index.insert(i as u128, vector.clone()).unwrap();
    }
    
    let elapsed = start.elapsed();
    let throughput = n_vectors as f64 / elapsed.as_secs_f64();
    
    println!("Results:");
    println!("  Time: {:.3}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} vec/s", throughput);
    
    // Test search correctness
    let query = &vectors[0];
    if let Ok(results) = index.search(query, 1) {
        if !results.is_empty() {
            let (found_id, distance) = &results[0];
            println!("  Search test: ID {}, distance {:.6}", found_id, distance);
        }
    }
    
    println!("Pure Rust HNSW performance: {:.0} vec/s", throughput);
}
'''
    
    # Write the Rust code to a temporary file
    with open('/tmp/rust_hnsw_test.rs', 'w') as f:
        f.write(rust_code)
    
    # Create a temporary Cargo project
    cargo_toml = '''[package]
name = "rust_hnsw_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "rust_hnsw_test"
path = "/tmp/rust_hnsw_test.rs"

[dependencies]
toondb-index = { path = "/Users/sushanth/toondb/toondb-index" }
rand = "0.8"
'''
    
    with open('/tmp/Cargo.toml', 'w') as f:
        f.write(cargo_toml)
    
    print("Compiling and running Rust test...")
    
    try:
        result = subprocess.run([
            'cargo', 'run', '--release', '--manifest-path', '/tmp/Cargo.toml'
        ], capture_output=True, text=True, timeout=120)
        
        print("Rust output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"Rust test failed with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("Rust test timed out")
    except Exception as e:
        print(f"Error running Rust test: {e}")

if __name__ == '__main__':
    run_rust_test()