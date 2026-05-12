// Copyright 2025 SochDB Authors
//
// Licensed under the Apache License, Version 2.0

//! SQL Query Examples
//!
//! Demonstrates SQL support in SochDB:
//! - CREATE TABLE, INSERT, UPDATE, DELETE
//! - SELECT with WHERE, ORDER BY, LIMIT
//! - Schema management
//!
//! Note: SQL execution uses SochConnection (in-memory with SQL support).
//! For durable storage, use DurableConnection with path-based APIs.

use sochdb::SochConnection;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let sep = "=".repeat(60);
    println!("{}", sep);
    println!("SochDB SQL Query Examples");
    println!("{}", sep);

    // Open in-memory database (SQL execution uses SochConnection)
    let db_path = "./demo_sql_db_rust";
    println!("\n📂 Opening database: {}", db_path);

    // Clean up existing database
    let _ = std::fs::remove_dir_all(db_path);

    let conn = SochConnection::open(db_path)?;
    println!("✓ Database opened successfully");

    // Run demonstrations
    create_tables(&conn)?;
    insert_data(&conn)?;
    select_queries(&conn)?;
    update_operations(&conn)?;
    delete_operations(&conn)?;
    complex_queries(&conn)?;

    println!("\n{}", sep);
    println!("✓ All SQL examples completed successfully!");
    println!("{}", sep);

    Ok(())
}

fn create_tables(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n📝 Creating Tables with SQL");
    println!("{}", "=".repeat(60));

    // Create users table
    conn.execute_sql(
        r#"
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            created_at TEXT
        )
        "#,
    )?;
    println!("✓ Created 'users' table");

    // Create posts table
    conn.execute_sql(
        r#"
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            likes INTEGER DEFAULT 0,
            published_at TEXT
        )
        "#,
    )?;
    println!("✓ Created 'posts' table");

    Ok(())
}

fn insert_data(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n📥 Inserting Data with SQL");
    println!("{}", "=".repeat(60));

    // Insert users
    let users = vec![
        (1, "Alice", "alice@example.com", 30, "2024-01-01"),
        (2, "Bob", "bob@example.com", 25, "2024-01-02"),
        (3, "Charlie", "charlie@example.com", 35, "2024-01-03"),
        (4, "Diana", "diana@example.com", 28, "2024-01-04"),
    ];

    for (id, name, email, age, created_at) in users {
        conn.execute_sql(&format!(
            "INSERT INTO users (id, name, email, age, created_at) VALUES ({}, '{}', '{}', {}, '{}')",
            id, name, email, age, created_at
        ))?;
        println!("  ✓ Inserted user: {}", name);
    }

    // Insert posts
    let posts = vec![
        (1, 1, "First Post", "Hello World!", 10, "2024-01-05"),
        (2, 1, "Second Post", "SochDB is awesome", 25, "2024-01-06"),
        (3, 2, "Bob's Thoughts", "SQL queries are easy", 15, "2024-01-07"),
        (4, 3, "Charlie's Guide", "Database tips", 30, "2024-01-08"),
        (5, 3, "Advanced Topics", "Performance tuning", 50, "2024-01-09"),
    ];

    for (id, user_id, title, content, likes, published_at) in posts {
        conn.execute_sql(&format!(
            "INSERT INTO posts (id, user_id, title, content, likes, published_at) VALUES ({}, {}, '{}', '{}', {}, '{}')",
            id, user_id, title, content, likes, published_at
        ))?;
        println!("  ✓ Inserted post: {}", title);
    }

    Ok(())
}

fn select_queries(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n🔍 Running SELECT Queries");
    println!("{}", "=".repeat(60));

    // Simple SELECT
    println!("\n1. Select all users:");
    let result = conn.query_sql("SELECT * FROM users")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            println!("   Found {} users", rows.len());
            for row in &rows {
                println!(
                    "   - {} ({})",
                    row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("email").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => println!("   Unexpected result type"),
    }

    // SELECT with WHERE clause
    println!("\n2. Users older than 28:");
    let result = conn.query_sql("SELECT name, age FROM users WHERE age > 28")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} years old",
                    row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("age").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => println!("   Unexpected result type"),
    }

    // SELECT with ORDER BY
    println!("\n3. Posts ordered by likes (descending):");
    let result = conn.query_sql("SELECT title, likes FROM posts ORDER BY likes DESC")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} likes",
                    row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => println!("   Unexpected result type"),
    }

    // SELECT with LIMIT
    println!("\n4. Top 3 most liked posts:");
    let result = conn.query_sql("SELECT title, likes FROM posts ORDER BY likes DESC LIMIT 3")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} likes",
                    row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => println!("   Unexpected result type"),
    }

    Ok(())
}

fn update_operations(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n✏️  UPDATE Operations");
    println!("{}", "=".repeat(60));

    // Update single row
    println!("\n1. Update Alice's age:");
    conn.execute_sql("UPDATE users SET age = 31 WHERE name = 'Alice'")?;
    let result = conn.query_sql("SELECT name, age FROM users WHERE name = 'Alice'")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            if let Some(row) = rows.first() {
                println!("   Alice's new age: {:?}", row.get("age"));
            }
        }
        _ => {}
    }

    // Update multiple rows
    println!("\n2. Increment likes on all posts by user_id = 1:");
    conn.execute_sql("UPDATE posts SET likes = likes + 5 WHERE user_id = 1")?;
    let result = conn.query_sql("SELECT title, likes FROM posts WHERE user_id = 1")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} likes",
                    row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => {}
    }

    Ok(())
}

fn delete_operations(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n🗑️  DELETE Operations");
    println!("{}", "=".repeat(60));

    // Count before delete
    let result = conn.query_sql("SELECT COUNT(*) as total FROM posts")?;
    let before_count = match &result {
        sochdb::ast_query::QueryResult::Select(rows) => rows.len(),
        _ => 0,
    };
    println!("Posts before delete: {}", before_count);

    // Delete specific post
    conn.execute_sql("DELETE FROM posts WHERE id = 5")?;
    println!("✓ Deleted post with id = 5");

    // Count after delete
    let result = conn.query_sql("SELECT COUNT(*) as total FROM posts")?;
    let after_count = match &result {
        sochdb::ast_query::QueryResult::Select(rows) => rows.len(),
        _ => 0,
    };
    println!("Posts after delete: {}", after_count);

    Ok(())
}

fn complex_queries(conn: &SochConnection) -> Result<(), Box<dyn Error>> {
    println!("\n🎯 Complex Queries");
    println!("{}", "=".repeat(60));

    // SELECT with multiple conditions
    println!("\n1. Users aged 25-30:");
    let result = conn.query_sql(
        "SELECT name, age, email FROM users WHERE age >= 25 AND age <= 30 ORDER BY age"
    )?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} years ({})",
                    row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("age").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("email").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => {}
    }

    // SELECT with pattern matching
    println!("\n2. Posts with 'Post' in title:");
    let result = conn.query_sql("SELECT title, likes FROM posts WHERE title LIKE '%Post%'")?;
    match result {
        sochdb::ast_query::QueryResult::Select(rows) => {
            for row in &rows {
                println!(
                    "   - {}: {} likes",
                    row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                    row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default()
                );
            }
        }
        _ => {}
    }

    Ok(())
}
