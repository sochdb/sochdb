//! Basic ToonDB Operations Example
//! 
//! This example demonstrates fundamental key-value operations:
//! - Opening a database
//! - Put, Get, Delete operations
//! - Path-based hierarchical keys
//! - Context manager pattern

use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    // Open or create a database
    let db = Database::open("./example_db")?;
    println!("✓ Database opened");

    // Basic key-value operations
    db.put(b"greeting", b"Hello, ToonDB!")?;
    println!("✓ Key 'greeting' written");

    let value = db.get(b"greeting")?;
    match value {
        Some(v) => println!("✓ Read value: {}", String::from_utf8_lossy(&v)),
        None => println!("Key not found"),
    }

    // Path-based hierarchical keys
    db.put_path(&["users", "alice", "name"], b"Alice Smith")?;
    db.put_path(&["users", "alice", "email"], b"alice@example.com")?;
    db.put_path(&["users", "bob", "name"], b"Bob Jones")?;
    println!("✓ Hierarchical data stored");

    // Read by path
    if let Some(name) = db.get_path(&["users", "alice", "name"])? {
        println!("✓ Alice's name: {}", String::from_utf8_lossy(&name));
    }

    // Delete a key
    db.delete(b"greeting")?;
    println!("✓ Key 'greeting' deleted");

    // Verify deletion
    match db.get(b"greeting")? {
        Some(_) => println!("Key still exists"),
        None => println!("✓ Key confirmed deleted"),
    }

    // Query prefix scan
    let results = db.query("users/")
        .limit(10)
        .execute()?;
    
    println!("\n✓ Prefix scan results:");
    for (key, value) in results {
        println!("  {} = {}", 
            String::from_utf8_lossy(&key),
            String::from_utf8_lossy(&value));
    }

    Ok(())
}
