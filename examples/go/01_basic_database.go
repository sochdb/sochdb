// Basic ToonDB Operations Example
//
// Demonstrates fundamental key-value operations:
// - Opening a database
// - Put, Get, Delete operations
// - Path-based hierarchical keys
// - Prefix scan queries

package main

import (
	"fmt"
	"log"

	toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
	// Open or create a database
	db, err := toondb.Open("./example_db")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	fmt.Println("✓ Database opened")

	// Basic key-value operations
	err = db.Put("greeting", []byte("Hello, ToonDB!"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Key 'greeting' written")

	value, err := db.Get("greeting")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ Read value: %s\n", string(value))

	// Path-based hierarchical keys
	err = db.PutPath([]string{"users", "alice", "name"}, []byte("Alice Smith"))
	if err != nil {
		log.Fatal(err)
	}
	err = db.PutPath([]string{"users", "alice", "email"}, []byte("alice@example.com"))
	if err != nil {
		log.Fatal(err)
	}
	err = db.PutPath([]string{"users", "bob", "name"}, []byte("Bob Jones"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Hierarchical data stored")

	// Read by path
	aliceName, err := db.GetPath([]string{"users", "alice", "name"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ Alice's name: %s\n", string(aliceName))

	// Delete a key
	err = db.Delete("greeting")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Key 'greeting' deleted")

	// Verify deletion
	_, err = db.Get("greeting")
	if err == toondb.ErrNotFound {
		fmt.Println("✓ Key confirmed deleted")
	}

	// Query prefix scan
	results, err := db.Query("users/").
		Limit(10).
		Execute()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n✓ Prefix scan results:")
	for _, kv := range results {
		fmt.Printf("  %s = %s\n", kv.Key, string(kv.Value))
	}
}
