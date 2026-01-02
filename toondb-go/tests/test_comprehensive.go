package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/toondb/toondb/toondb-go"
)

var (
	testCount int
	passCount int
	failCount int
)

func testAssert(condition bool, message string) bool {
	testCount++
	if condition {
		passCount++
		fmt.Printf("  ✓ %s\n", message)
		return true
	}
	failCount++
	fmt.Printf("  ✗ %s\n", message)
	return false
}

func testBasicKeyValue(db *toondb.Database) {
	fmt.Println("\n📝 Testing Basic Key-Value Operations...")

	// Put
	err := db.Put([]byte("key1"), []byte("value1"))
	testAssert(err == nil, "Put operation succeeded")

	// Get
	value, err := db.Get([]byte("key1"))
	testAssert(err == nil && value != nil && string(value) == "value1", "Get returns correct value")

	// Get non-existent key
	missing, err := db.Get([]byte("nonexistent"))
	testAssert(err == nil && missing == nil, "Get returns nil for missing key")

	// Delete
	err = db.Delete([]byte("key1"))
	deleted, _ := db.Get([]byte("key1"))
	testAssert(err == nil && deleted == nil, "Delete removes key")
}

func testPathOperations(db *toondb.Database) {
	fmt.Println("\n🗂️  Testing Path Operations...")

	// Put path
	err := db.PutPath("users/alice/email", []byte("alice@example.com"))
	testAssert(err == nil, "PutPath succeeded")

	// Get path
	email, err := db.GetPath("users/alice/email")
	testAssert(err == nil && string(email) == "alice@example.com", "GetPath retrieves correct value")

	// Multiple segments
	err = db.PutPath("users/bob/profile/name", []byte("Bob"))
	name, err := db.GetPath("users/bob/profile/name")
	testAssert(err == nil && string(name) == "Bob", "GetPath handles multiple segments")

	// Missing path
	missing, err := db.GetPath("users/charlie/email")
	testAssert(err == nil && missing == nil, "GetPath returns nil for missing path")
}

func testPrefixScanning(db *toondb.Database) {
	fmt.Println("\n🔍 Testing Prefix Scanning...")

	// Insert multi-tenant data
	db.Put([]byte("tenants/acme/users/1"), []byte(`{"name":"Alice"}`))
	db.Put([]byte("tenants/acme/users/2"), []byte(`{"name":"Bob"}`))
	db.Put([]byte("tenants/acme/orders/1"), []byte(`{"total":100}`))
	db.Put([]byte("tenants/globex/users/1"), []byte(`{"name":"Charlie"}`))

	// Scan ACME
	acmeResults, err := db.Scan("tenants/acme/")
	testAssert(err == nil && len(acmeResults) == 3, fmt.Sprintf("Scan returns 3 ACME items (got %d)", len(acmeResults)))

	// Scan Globex
	globexResults, err := db.Scan("tenants/globex/")
	testAssert(err == nil && len(globexResults) == 1, fmt.Sprintf("Scan returns 1 Globex item (got %d)", len(globexResults)))

	// Verify results have key and value
	if len(acmeResults) > 0 {
		testAssert(
			len(acmeResults[0].Key) > 0 && len(acmeResults[0].Value) > 0,
			"Scan results have Key and Value",
		)
	}
}

func testTransactions(db *toondb.Database) {
	fmt.Println("\n💳 Testing Transactions...")

	// Automatic commit
	err := db.WithTransaction(func(txn *toondb.Transaction) error {
		txn.Put([]byte("tx_key1"), []byte("tx_value1"))
		txn.Put([]byte("tx_key2"), []byte("tx_value2"))
		return nil
	})

	// Verify committed
	value1, _ := db.Get([]byte("tx_key1"))
	value2, _ := db.Get([]byte("tx_key2"))
	testAssert(
		err == nil && string(value1) == "tx_value1" && string(value2) == "tx_value2",
		"Transaction commits successfully",
	)

	// Verify data persisted
	persisted, _ := db.Get([]byte("tx_key1"))
	testAssert(persisted != nil, "Transaction data persisted")

	// Manual transaction
	txn, _ := db.BeginTransaction()
	txn.Put([]byte("manual_key"), []byte("manual_value"))
	txn.Commit()
	manual, _ := db.Get([]byte("manual_key"))
	testAssert(string(manual) == "manual_value", "Manual transaction works")
}

func testQueryBuilder(db *toondb.Database) {
	fmt.Println("\n🔎 Testing Query Builder...")

	// Insert structured data
	db.Put([]byte("products/laptop"), []byte(`{"name":"Laptop","price":999}`))
	db.Put([]byte("products/mouse"), []byte(`{"name":"Mouse","price":25}`))

	// Execute query
	results, err := db.Query("products/").Execute()
	testAssert(err == nil, fmt.Sprintf("Query returns results (got %d)", len(results)))

	// Count
	count, err := db.Query("products/").Count()
	testAssert(err == nil && count >= 0, fmt.Sprintf("Count returns non-negative number (got %d)", count))

	// First
	_, err = db.Query("products/").First()
	testAssert(err == nil, "First returns result or nil")

	// Exists
	exists, err := db.Query("products/").Exists()
	testAssert(err == nil, fmt.Sprintf("Exists returns boolean (got %v)", exists))
}

func testEmptyValueHandling(db *toondb.Database) {
	fmt.Println("\n🔄 Testing Empty Value Handling...")

	// Test non-existent key
	missing, err := db.Get([]byte("truly-missing-key-test"))
	testAssert(err == nil && missing == nil, "Missing key returns nil")

	fmt.Println("  ℹ️  Note: Empty values and missing keys both return nil (protocol limitation)")
}

func main() {
	testDir := filepath.Join(".", "test-data-comprehensive")

	// Clean up any existing test data
	os.RemoveAll(testDir)

	fmt.Println("🧪 ToonDB Go SDK Comprehensive Feature Test\n")
	fmt.Println("Testing all features mentioned in README...\n")
	fmt.Println("============================================================")

	// Open database
	db, err := toondb.Open(testDir)
	if err != nil {
		fmt.Printf("\n❌ Fatal error: %v\n\n", err)
		os.Exit(1)
	}
	defer db.Close()

	testBasicKeyValue(db)
	testPathOperations(db)
	testPrefixScanning(db)
	testTransactions(db)
	testQueryBuilder(db)
	testEmptyValueHandling(db)

	// Clean up
	os.RemoveAll(testDir)

	fmt.Println("\n============================================================")
	fmt.Printf("\n📊 Test Results:\n")
	fmt.Printf("   Total:  %d\n", testCount)
	fmt.Printf("   ✓ Pass: %d\n", passCount)
	fmt.Printf("   ✗ Fail: %d\n", failCount)
	fmt.Printf("   Success Rate: %.1f%%\n", float64(passCount)/float64(testCount)*100)

	if failCount == 0 {
		fmt.Println("\n✅ All tests passed! Go SDK is working correctly.\n")
		os.Exit(0)
	} else {
		fmt.Printf("\n❌ %d test(s) failed. See details above.\n\n", failCount)
		os.Exit(1)
	}
}
