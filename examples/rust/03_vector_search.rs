//! Vector Search Example
//! 
//! This example demonstrates vector similarity search:
//! - Creating a vector index with HNSW
//! - Bulk loading embeddings
//! - Finding nearest neighbors
//! - Distance metrics

use toondb::{VectorIndex, VectorIndexConfig, DistanceMetric};
use anyhow::Result;

fn main() -> Result<()> {
    // Configuration for the vector index
    let config = VectorIndexConfig {
        dimension: 128,                    // Embedding dimension
        metric: DistanceMetric::Cosine,    // Cosine similarity
        m: 16,                             // HNSW connections per node
        ef_construction: 100,              // Construction quality factor
        ef_search: 50,                     // Search quality factor
    };

    let index = VectorIndex::new("./vector_index", config)?;
    println!("✓ Vector index created");

    // Generate sample embeddings (in practice, use a real embedding model)
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    
    for i in 0..100 {
        // Create a simple pattern-based embedding for demo
        let mut vec = vec![0.0f32; 128];
        for j in 0..128 {
            vec[j] = ((i * j) % 256) as f32 / 255.0;
        }
        vectors.push(vec);
        labels.push(format!("document_{}", i));
    }

    // Bulk build the index
    index.bulk_build(&vectors, Some(&labels))?;
    println!("✓ Indexed {} vectors", vectors.len());

    // Create a query vector
    let mut query = vec![0.0f32; 128];
    for j in 0..128 {
        query[j] = ((42 * j) % 256) as f32 / 255.0;  // Similar to document_42
    }

    // Search for nearest neighbors
    let results = index.query(&query, 5)?;  // k=5

    println!("\n✓ Top 5 nearest neighbors:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (distance: {:.4})", 
            i + 1,
            result.label.as_deref().unwrap_or("<no label>"),
            result.distance);
    }

    // Distance utility functions
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.707, 0.707, 0.0];

    let cosine_dist = toondb::vector::cosine_distance(&a, &b);
    let euclidean_dist = toondb::vector::euclidean_distance(&a, &b);

    println!("\n✓ Distance calculations:");
    println!("  Cosine distance: {:.4}", cosine_dist);
    println!("  Euclidean distance: {:.4}", euclidean_dist);

    // Normalize a vector
    let v = vec![3.0, 4.0];
    let normalized = toondb::vector::normalize(&v);
    println!("  Normalized [3, 4]: {:?}", normalized);

    Ok(())
}
