//! Vector Search Example
//!
//! This example demonstrates vector similarity search:
//! - Creating a vector collection
//! - Bulk loading embeddings
//! - Finding nearest neighbors

use sochdb::vectors::VectorCollection;
use sochdb::SochConnection;
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = SochConnection::open("./vector_db")?;
    let arc_conn = Arc::new(conn);

    // Create a vector collection with 128 dimensions
    let mut collection = VectorCollection::create(&arc_conn, "demo", 128)?;
    println!("✓ Vector collection created");

    // Generate sample embeddings (in practice, use a real embedding model)
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for i in 0..100 {
        let mut vec = vec![0.0f32; 128];
        for j in 0..128 {
            vec[j] = ((i * j) % 256) as f32 / 255.0;
        }
        vectors.push(vec);
        labels.push(format!("document_{}", i));
    }

    // Bulk add vectors
    let ids: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    collection.add(&ids, &vectors)?;
    println!("✓ Indexed {} vectors", vectors.len());

    // Create a query vector (similar to document_42)
    let mut query = vec![0.0f32; 128];
    for j in 0..128 {
        query[j] = ((42 * j) % 256) as f32 / 255.0;
    }

    // Search for nearest neighbors
    let results = collection.search(&query, 5)?;

    println!("\n✓ Top 5 nearest neighbors:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} (distance: {:.4})",
            i + 1,
            result.id,
            result.distance
        );
    }

    Ok(())
}
