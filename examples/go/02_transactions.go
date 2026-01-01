// Transaction Example
//
// Demonstrates ACID transactions:
// - Using WithTransaction for automatic commit/rollback
// - Read operations within transactions
// - Error handling and rollback

package main

import (
	"fmt"
	"log"
	"strconv"

	toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
	db, err := toondb.Open("./txn_example_db")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	fmt.Println("✓ Database opened")

	// Automatic transaction with callback (recommended)
	err = db.WithTransaction(func(txn *toondb.Transaction) error {
		if err := txn.Put("accounts/alice/balance", []byte("1000")); err != nil {
			return err
		}
		if err := txn.Put("accounts/bob/balance", []byte("500")); err != nil {
			return err
		}
		fmt.Println("✓ Transaction: wrote initial balances")
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Transaction committed automatically")

	// Simulate a transfer
	err = db.WithTransaction(func(txn *toondb.Transaction) error {
		// Read current balances
		aliceData, _ := txn.Get("accounts/alice/balance")
		bobData, _ := txn.Get("accounts/bob/balance")

		aliceBalance, _ := strconv.Atoi(string(aliceData))
		bobBalance, _ := strconv.Atoi(string(bobData))

		transferAmount := 250

		// Update balances atomically
		if err := txn.Put("accounts/alice/balance", 
			[]byte(strconv.Itoa(aliceBalance-transferAmount))); err != nil {
			return err
		}
		if err := txn.Put("accounts/bob/balance", 
			[]byte(strconv.Itoa(bobBalance+transferAmount))); err != nil {
			return err
		}

		fmt.Printf("✓ Transfer: Alice -> Bob: $%d\n", transferAmount)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	// Verify final balances
	alice, _ := db.Get("accounts/alice/balance")
	bob, _ := db.Get("accounts/bob/balance")

	fmt.Println("\n✓ Final balances:")
	fmt.Printf("  Alice: $%s\n", string(alice))
	fmt.Printf("  Bob: $%s\n", string(bob))

	// Transaction rollback on error
	err = db.WithTransaction(func(txn *toondb.Transaction) error {
		txn.Put("accounts/alice/balance", []byte("9999"))
		return fmt.Errorf("simulated failure")
	})
	if err != nil {
		fmt.Printf("✓ Transaction rolled back: %v\n", err)
	}

	// Verify balance unchanged after rollback
	aliceAfter, _ := db.Get("accounts/alice/balance")
	fmt.Printf("✓ Alice's balance after rollback: $%s\n", string(aliceAfter))
}
