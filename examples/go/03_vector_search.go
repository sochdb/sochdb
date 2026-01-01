// Vector Search Example
//
// Demonstrates vector similarity search:
// - Creating a vector index with HNSW
// - Bulk loading embeddings
// - Finding nearest neighbors
// - Distance utilities

package main

import (
	"fmt"
	"log"

	toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
	// Configuration for the vector index
	config := toondb.VectorIndexConfig{
		Dimension:      128,                      // Embedding dimension
		Metric:         toondb.DistanceMetricCosine, // Cosine similarity
		M:              16,                       // HNSW connections per node
		EfConstruction: 100,                      // Construction quality factor
	}

	index, err := toondb.NewVectorIndex("./vector_index", config)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Vector index created")

	// Generate sample embeddings (in practice, use a real embedding model)
	vectors := make([][]float32, 100)
	labels := make([]string, 100)

	for i := 0; i < 100; i++ {
		// Create a simple pattern-based embedding for demo
		vec := make([]float32, 128)
		for j := 0; j < 128; j++ {
			vec[j] = float32((i*j)%256) / 255.0
		}
		vectors[i] = vec
		labels[i] = fmt.Sprintf("document_%d", i)
	}

	// Bulk build the index
	err = index.BulkBuild(vectors, labels)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ Indexed %d vectors\n", len(vectors))

	// Create a query vector (similar to document_42)
	query := make([]float32, 128)
	for j := 0; j < 128; j++ {
		query[j] = float32((42*j)%256) / 255.0
	}

	// Search for nearest neighbors
	results, err := index.Query(query, 5) // k=5
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n✓ Top 5 nearest neighbors:")
	for i, result := range results {
		label := "<no label>"
		if result.Label != "" {
			label = result.Label
		}
		fmt.Printf("  %d. %s (distance: %.4f)\n", i+1, label, result.Distance)
	}

	// Distance utility functions
	a := []float32{1.0, 0.0, 0.0}
	b := []float32{0.707, 0.707, 0.0}

	cosineDist := toondb.ComputeCosineDistance(a, b)
	euclideanDist := toondb.ComputeEuclideanDistance(a, b)

	fmt.Println("\n✓ Distance calculations:")
	fmt.Printf("  Cosine distance: %.4f\n", cosineDist)
	fmt.Printf("  Euclidean distance: %.4f\n", euclideanDist)

	// Normalize a vector
	v := []float32{3.0, 4.0}
	normalized := toondb.NormalizeVector(v)
	fmt.Printf("  Normalized [3, 4]: [%.2f, %.2f]\n", normalized[0], normalized[1])
}
