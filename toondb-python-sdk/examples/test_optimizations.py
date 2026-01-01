#!/usr/bin/env python3
"""
Quick performance test to measure our HNSW optimizations
"""
import subprocess
import time
import tempfile
import os

def measure_insertion_performance():
    """Measure ToonDB vector insertion performance"""
    
    # Create a simple test script to measure insertion speed
    test_script = """
use std::time::Instant;
use toondb_index::hnsw::HnswIndex;
use toondb_index::vector_quantized::{QuantizedVector, Precision};
use toondb_index::distance::DistanceMetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 768;
    let num_vectors = 1000;
    
    // Create HNSW index with optimized parameters
    let mut index = HnswIndex::new(
        dimension,
        DistanceMetric::Cosine,
        16,  // max_m
        48,  // ef_construction (adaptive will optimize this)
        200  // max_m0
    )?;
    
    println!("Testing {} vectors of dimension {}", num_vectors, dimension);
    
    // Generate test vectors
    let mut vectors = Vec::new();
    for i in 0..num_vectors {
        let data: Vec<f32> = (0..dimension)
            .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
            .collect();
        vectors.push(QuantizedVector::from_f32_slice(&data, Precision::Full)?);
    }
    
    // Measure insertion time
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        index.insert(i as u128 + 1, vector.clone())?;
        if i > 0 && i % 100 == 0 {
            let elapsed = start.elapsed().as_millis();
            let rate = (i + 1) as f64 / elapsed as f64 * 1000.0;
            println!("Inserted {} vectors: {:.0} vec/s", i + 1, rate);
        }
    }
    
    let total_elapsed = start.elapsed();
    let insertion_rate = num_vectors as f64 / total_elapsed.as_secs_f64();
    
    println!("\\n=== FINAL RESULTS ===");
    println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
    println!("Insertion rate: {:.0} vectors/second", insertion_rate);
    println!("Average per vector: {:.2}ms", total_elapsed.as_secs_f64() * 1000.0 / num_vectors as f64);
    
    // Test search performance
    let query = &vectors[0];
    let search_start = Instant::now();
    let results = index.search(query, 10, 50)?;
    let search_time = search_start.elapsed();
    
    println!("Search time: {:.2}ms for top-10", search_time.as_secs_f64() * 1000.0);
    println!("Found {} results", results.len());
    
    Ok(())
}
"""
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
        f.write(test_script)
        temp_file = f.name
    
    try:
        # Write to the src/bin directory
        bin_path = "/Users/sushanth/toondb/toondb-index/src/bin/test_perf.rs"
        with open(bin_path, 'w') as f:
            f.write(test_script)
        
        print("🚀 Running ToonDB HNSW insertion performance test...")
        
        # Run the test
        result = subprocess.run([
            'cargo', 'run', '--release', '--bin', 'test_perf'
        ], capture_output=True, text=True, cwd='/Users/sushanth/toondb/toondb-index')
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("\n" + "="*60)
            print("OUTPUT:")
            print("="*60)
            print(result.stdout)
            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr)
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        # Cleanup
        if os.path.exists(bin_path):
            os.remove(bin_path)
            
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    measure_insertion_performance()