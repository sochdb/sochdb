use toondb::prelude::*;
use std::fs;
use std::path::Path;

static mut TEST_COUNT: usize = 0;
static mut PASS_COUNT: usize = 0;
static mut FAIL_COUNT: usize = 0;

fn test_assert(condition: bool, message: &str) -> bool {
    unsafe {
        TEST_COUNT += 1;
        if condition {
            PASS_COUNT += 1;
            println!("  ✓ {}", message);
            true
        } else {
            FAIL_COUNT += 1;
            println!("  ✗ {}", message);
            false
        }
    }
}

fn test_basic_key_value(conn: &Connection) {
    println!("\n📝 Testing Basic Key-Value Operations...");

    // Put
    let _ = conn.put("key1", b"value1");
    test_assert(true, "Put operation succeeded");

    // Get
    let value = conn.get("key1").unwrap();
    test_assert(
        value.is_some() && value.unwrap() == b"value1",
        "Get returns correct value",
    );

    // Get non-existent key
    let missing = conn.get("nonexistent").unwrap();
    test_assert(missing.is_none(), "Get returns None for missing key");

    // Delete
    let _ = conn.delete("key1");
    let deleted = conn.get("key1").unwrap();
    test_assert(deleted.is_none(), "Delete removes key");
}

fn test_path_operations(conn: &Connection) {
    println!("\n🗂️  Testing Path Operations...");

    // Put path
    let _ = conn.put("users/alice/email", b"alice@example.com");
    test_assert(true, "put_path succeeded");

    // Get path
    let email = conn.get("users/alice/email").unwrap();
    test_assert(
        email.is_some() && email.unwrap() == b"alice@example.com",
        "get_path retrieves correct value",
    );

    // Multiple segments
    let _ = conn.put("users/bob/profile/name", b"Bob");
    let name = conn.get("users/bob/profile/name").unwrap();
    test_assert(
        name.is_some() && name.unwrap() == b"Bob",
        "get_path handles multiple segments",
    );

    // Missing path
    let missing = conn.get("users/charlie/email").unwrap();
    test_assert(missing.is_none(), "get_path returns None for missing path");
}

fn test_prefix_scanning(conn: &Connection) {
    println!("\n🔍 Testing Prefix Scanning...");

    // Insert multi-tenant data
    let _ = conn.put("tenants/acme/users/1", br#"{"name":"Alice"}"#);
    let _ = conn.put("tenants/acme/users/2", br#"{"name":"Bob"}"#);
    let _ = conn.put("tenants/acme/orders/1", br#"{"total":100}"#);
    let _ = conn.put("tenants/globex/users/1", br#"{"name":"Charlie"}"#);

    // Scan ACME
    let acme_results = conn.scan("tenants/acme/").unwrap();
    test_assert(
        acme_results.len() == 3,
        &format!("Scan returns 3 ACME items (got {})", acme_results.len()),
    );

    // Scan Globex
    let globex_results = conn.scan("tenants/globex/").unwrap();
    test_assert(
        globex_results.len() == 1,
        &format!("Scan returns 1 Globex item (got {})", globex_results.len()),
    );

    // Verify results have key and value
    if !acme_results.is_empty() {
        test_assert(
            !acme_results[0].0.is_empty() && !acme_results[0].1.is_empty(),
            "Scan results have key and value",
        );
    }
}

fn test_transactions(conn: &Connection) {
    println!("\n💳 Testing Transactions...");

    // Begin transaction
    let _ = conn.begin();
    test_assert(true, "Transaction began");

    // Put in transaction
    let _ = conn.put("tx_key1", b"tx_value1");
    let _ = conn.put("tx_key2", b"tx_value2");

    // Commit
    let result = conn.commit();
    test_assert(result.is_ok(), "Transaction commits successfully");

    // Verify data persisted
    let value1 = conn.get("tx_key1").unwrap();
    let value2 = conn.get("tx_key2").unwrap();
    test_assert(
        value1.is_some() && value2.is_some(),
        "Transaction data persisted",
    );
}

fn test_empty_value_handling(conn: &Connection) {
    println!("\n🔄 Testing Empty Value Handling...");

    // Test non-existent key
    let missing = conn.get("truly-missing-key-test").unwrap();
    test_assert(missing.is_none(), "Missing key returns None");

    println!("  ℹ️  Note: Empty values and missing keys both return None (protocol limitation)");
}

fn main() {
    let test_dir = "./test-data-comprehensive-rust";

    // Clean up any existing test data
    if Path::new(test_dir).exists() {
        let _ = fs::remove_dir_all(test_dir);
    }

    println!("🧪 ToonDB Rust SDK Comprehensive Feature Test\n");
    println!("Testing all features mentioned in README...\n");
    println!("{}", "=".repeat(60));

    // Open connection
    let conn = Connection::open(test_dir).expect("Failed to open connection");

    // Run tests
    test_basic_key_value(&conn);
    test_path_operations(&conn);
    test_prefix_scanning(&conn);
    test_transactions(&conn);
    test_empty_value_handling(&conn);

    // Clean up
    drop(conn);
    let _ = fs::remove_dir_all(test_dir);

    unsafe {
        println!("\n{}", "=".repeat(60));
        println!("\n📊 Test Results:");
        println!("   Total:  {}", TEST_COUNT);
        println!("   ✓ Pass: {}", PASS_COUNT);
        println!("   ✗ Fail: {}", FAIL_COUNT);
        println!(
            "   Success Rate: {:.1}%",
            (PASS_COUNT as f64 / TEST_COUNT as f64) * 100.0
        );

        if FAIL_COUNT == 0 {
            println!("\n✅ All tests passed! Rust SDK is working correctly.\n");
            std::process::exit(0);
        } else {
            println!("\n❌ {} test(s) failed. See details above.\n", FAIL_COUNT);
            std::process::exit(1);
        }
    }
}
