// Insert Performance Profiler
// Run with: cargo run --release --bin insert_profile

use std::time::Instant;
use toondb_index::hnsw::{HnswConfig, HnswIndex};

fn main() {
    println!("{}", "=".repeat(60));
    println!("  ToonDB Insert Performance Profile");
    println!("{}", "=".repeat(60));
    
    let dims = [128, 768];
    let sizes = [1000, 5000, 10000];
    
    for &dim in &dims {
        for &n in &sizes {
            profile_insert(n, dim);
        }
    }
}

fn profile_insert(n: usize, dim: usize) {
    // Generate test vectors
    let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..(n * dim) {
        vectors.push((i as f32 * 0.001).sin());
    }
    let ids: Vec<u128> = (0..n as u128).collect();
    
    // Create index with faster settings (matching ChromaDB defaults more closely)
    let config = HnswConfig {
        max_connections: 16,
        max_connections_layer0: 32,
        ef_construction: 48,  // Lower ef_construction for faster inserts
        ef_search: 50,
        ..Default::default()
    };
    
    println!("\n--- {}D × {} vectors ---", dim, n);
    
    // Batch insert timing
    let index = HnswIndex::new(dim, config.clone());
    let start = Instant::now();
    let inserted = index.insert_batch_contiguous(&ids, &vectors, dim).unwrap();
    let batch_time = start.elapsed();
    let batch_rate = inserted as f64 / batch_time.as_secs_f64();
    
    println!("Batch insert: {} vectors in {:?} ({:.0} vec/sec)", 
             inserted, batch_time, batch_rate);
    
    // Verify correctness with search
    let query = &vectors[0..dim];
    let results = index.search(query, 10).unwrap();
    println!("Search sanity check: {} results, first ID = {}", 
             results.len(), 
             results.first().map(|(id, _)| *id).unwrap_or(0));
    
    // Compare with individual inserts for small sizes
    if n <= 1000 {
        let index2 = HnswIndex::new(dim, config);
        let start = Instant::now();
        for i in 0..n {
            let vec_slice = &vectors[i * dim..(i + 1) * dim];
            index2.insert(ids[i], vec_slice.to_vec()).unwrap();
        }
        let individual_time = start.elapsed();
        let individual_rate = n as f64 / individual_time.as_secs_f64();
        
        println!("Individual insert: {} vectors in {:?} ({:.0} vec/sec)", 
                 n, individual_time, individual_rate);
        println!("Speedup: {:.1}x", batch_rate / individual_rate);
    }
}
