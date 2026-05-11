//! Basic SochDB Operations Example
//!
//! This example demonstrates fundamental key-value operations:
//! - Opening a database
//! - Put, Get, Delete operations
//! - Path-based hierarchical keys
//! - Prefix scanning

use sochdb::Connection;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Open or create a database
    let conn = Connection::open("./example_db")?;
    println!("✓ Database opened");

    // Basic key-value operations
    conn.put(b"greeting", b"Hello, SochDB!")?;
    println!("✓ Key 'greeting' written");

    let value = conn.get(b"greeting")?;
    match value {
        Some(v) => println!("✓ Read value: {}", String::from_utf8_lossy(&v)),
        None => println!("Key not found"),
    }

    // Path-based hierarchical keys (using slash-separated strings)
    conn.put_path("users/alice/name", b"Alice Smith")?;
    conn.put_path("users/alice/email", b"alice@example.com")?;
    conn.put_path("users/bob/name", b"Bob Jones")?;
    println!("✓ Hierarchical data stored");

    // Read by path
    if let Some(name) = conn.get_path("users/alice/name")? {
        println!("✓ Alice's name: {}", String::from_utf8_lossy(&name));
    }

    // Delete a key
    conn.delete(b"greeting")?;
    println!("✓ Key 'greeting' deleted");

    // Verify deletion
    match conn.get(b"greeting")? {
        Some(_) => println!("Key still exists"),
        None => println!("✓ Key confirmed deleted"),
    }

    // Prefix scan
    let results = conn.scan_path("users/")?;
    println!("\n✓ Prefix scan results:");
    for (key, value) in results {
        println!(
            "  {} = {}",
            key,
            String::from_utf8_lossy(&value)
        );
    }

    Ok(())
}
