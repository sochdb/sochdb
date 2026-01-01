//! Transaction Example
//! 
//! This example demonstrates ACID transactions:
//! - Using with_transaction for automatic commit/rollback
//! - Manual transaction control
//! - Read operations within transactions

use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    let db = Database::open("./txn_example_db")?;
    println!("✓ Database opened");

    // Automatic transaction with closure (recommended)
    db.with_transaction(|txn| {
        txn.put(b"accounts/alice/balance", b"1000")?;
        txn.put(b"accounts/bob/balance", b"500")?;
        println!("✓ Transaction: wrote initial balances");
        Ok(())
    })?;
    println!("✓ Transaction committed automatically");

    // Simulate a transfer
    db.with_transaction(|txn| {
        // Read current balances
        let alice_balance: i64 = txn.get(b"accounts/alice/balance")?
            .map(|v| String::from_utf8_lossy(&v).parse().unwrap_or(0))
            .unwrap_or(0);
        let bob_balance: i64 = txn.get(b"accounts/bob/balance")?
            .map(|v| String::from_utf8_lossy(&v).parse().unwrap_or(0))
            .unwrap_or(0);

        let transfer_amount = 250;

        // Update balances
        txn.put(b"accounts/alice/balance", 
                (alice_balance - transfer_amount).to_string().as_bytes())?;
        txn.put(b"accounts/bob/balance", 
                (bob_balance + transfer_amount).to_string().as_bytes())?;

        println!("✓ Transfer: Alice -> Bob: ${}", transfer_amount);
        Ok(())
    })?;

    // Verify final balances
    let alice = db.get(b"accounts/alice/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();
    let bob = db.get(b"accounts/bob/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();

    println!("\n✓ Final balances:");
    println!("  Alice: ${}", alice);
    println!("  Bob: ${}", bob);

    // Transaction rollback on error
    let result = db.with_transaction(|txn| {
        txn.put(b"accounts/alice/balance", b"9999")?;
        // Simulate an error
        Err(anyhow::anyhow!("Simulated failure"))
    });

    match result {
        Ok(_) => println!("Transaction committed"),
        Err(e) => println!("✓ Transaction rolled back: {}", e),
    }

    // Verify balance unchanged after rollback
    let alice_after = db.get(b"accounts/alice/balance")?
        .map(|v| String::from_utf8_lossy(&v).to_string())
        .unwrap_or_default();
    println!("✓ Alice's balance after rollback: ${}", alice_after);

    Ok(())
}
