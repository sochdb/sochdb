//! Transaction Example
//!
//! This example demonstrates ACID transactions:
//! - Manual transaction control with begin/commit/abort
//! - Read operations within transactions
//! - Rollback on error

use sochdb::Connection;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("./txn_example_db")?;
    println!("✓ Database opened");

    // Transaction 1: Write initial balances
    conn.begin_txn()?;
    conn.put(b"accounts/alice/balance", b"1000")?;
    conn.put(b"accounts/bob/balance", b"500")?;
    println!("✓ Transaction: wrote initial balances");
    conn.commit_txn()?;
    println!("✓ Transaction committed");

    // Transaction 2: Simulate a transfer
    conn.begin_txn()?;

    // Read current balances
    let alice_bytes = conn.get(b"accounts/alice/balance")?.unwrap_or_default();
    let alice_balance: i64 = String::from_utf8_lossy(&alice_bytes)
        .parse()
        .unwrap_or(0);

    let bob_bytes = conn.get(b"accounts/bob/balance")?.unwrap_or_default();
    let bob_balance: i64 = String::from_utf8_lossy(&bob_bytes)
        .parse()
        .unwrap_or(0);

    let transfer_amount = 250;

    // Update balances
    conn.put(
        b"accounts/alice/balance",
        (alice_balance - transfer_amount).to_string().as_bytes(),
    )?;
    conn.put(
        b"accounts/bob/balance",
        (bob_balance + transfer_amount).to_string().as_bytes(),
    )?;

    println!("✓ Transfer: Alice -> Bob: ${}", transfer_amount);
    conn.commit_txn()?;

    // Verify final balances
    let alice = conn
        .get(b"accounts/alice/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();
    let bob = conn
        .get(b"accounts/bob/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();

    println!("\n✓ Final balances:");
    println!("  Alice: ${}", alice);
    println!("  Bob: ${}", bob);

    // Transaction 3: Rollback on error
    conn.begin_txn()?;
    conn.put(b"accounts/alice/balance", b"9999")?;
    println!("✓ Transaction rolled back");
    conn.abort_txn()?;

    // Verify balance unchanged after rollback
    let alice_after = conn
        .get(b"accounts/alice/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();
    println!("✓ Alice's balance after rollback: ${}", alice_after);

    Ok(())
}
