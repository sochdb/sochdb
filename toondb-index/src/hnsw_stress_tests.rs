// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    #[ignore] // Stress test - run manually with `cargo test -- --ignored`
    fn test_concurrent_insert_stress() {
        let config = HnswConfig::default();
        let index = Arc::new(HnswIndex::new(128, config));
        let num_threads = 8;
        let vectors_per_thread = 1_000;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    for i in 0..vectors_per_thread {
                        let id = (thread_id * vectors_per_thread + i) as u128;
                        let mut vector = vec![0.0; 128];
                        vector[0] = (id as f32).sin();
                        vector[1] = (id as f32).cos();
                        index.insert(id, vector).expect("Insert failed");
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all vectors were inserted
        let stats = index.stats();
        assert_eq!(stats.num_nodes, (num_threads * vectors_per_thread) as u64);
    }
    
    #[test]
    #[ignore] // Stress test - run manually
    fn test_concurrent_mixed_workload_stress() {
        let config = HnswConfig::default();
        let index = Arc::new(HnswIndex::new(128, config));
        
        // Pre-populate with some vectors
        for i in 0..5_000 {
            let mut vector = vec![0.0; 128];
            vector[0] = (i as f32).sin();
            vector[1] = (i as f32).cos();
            index.insert(i, vector).unwrap();
        }
        
        let num_threads = 16;
        let operations_per_thread = 500;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        if i % 3 == 0 {
                            // Insert
                            let id = (5_000 + thread_id * operations_per_thread + i) as u128;
                            let mut vector = vec![0.0; 128];
                            vector[0] = (id as f32).sin();
                            vector[1] = (id as f32).cos();
                            index.insert(id, vector).expect("Insert failed");
                        } else {
                            // Search
                            let mut query = vec![0.0; 128];
                            query[0] = (thread_id as f32).sin();
                            query[1] = (thread_id as f32).cos();
                            index.search(&query, 10).expect("Search failed");
                        }
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify index is still functional
        let query = vec![1.0; 128];
        let results = index.search(&query, 10).unwrap();
        assert!(!results.is_empty());
    }
    
    #[test]
    #[ignore] // Stress test - run manually
    fn test_concurrent_layer_access_stress() {
        // This specifically tests that different threads can access different layers
        // of the same node concurrently without deadlocks or data races
        let config = HnswConfig::default();
        let index = Arc::new(HnswIndex::new(128, config));
        
        // Insert a bunch of vectors to create multi-layer nodes
        for i in 0..10_000 {
            let mut vector = vec![0.0; 128];
            vector[0] = (i as f32).sin();
            vector[1] = (i as f32).cos();
             vector[2] = i as f32;
            index.insert(i, vector).unwrap();
        }
        
        // Now hammer the index with concurrent searches
        // This will cause concurrent access to layers of various nodes
        let num_threads = 32;
        let searches_per_thread = 200;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    for i in 0..searches_per_thread {
                        let mut query = vec![0.0; 128];
                        query[0] = ((thread_id + i) as f32).sin();
                        query[1] = ((thread_id + i) as f32).cos();
                        query[2] = (thread_id * searches_per_thread + i) as f32;
                        index.search(&query, 20).expect("Search failed");
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
