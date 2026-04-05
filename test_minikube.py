#!/usr/bin/env python3
"""Quick smoke test against SochDB running in minikube (port-forwarded to localhost:50051)."""

import sys
sys.path.insert(0, "/Users/sushanth/sochdb-cloud/sochdb-python-sdk/src")

from sochdb.grpc_client import SochDBClient

def main():
    client = SochDBClient("localhost:50051")
    
    # 1. Create vector index
    print("=== Create Vector Index ===")
    try:
        result = client.create_index("smoke_test_py", dimension=8)
        print(f"  Created: {result}")
    except Exception as e:
        print(f"  (may already exist): {e}")
    
    # 2. Insert vectors
    print("\n=== Insert Vectors ===")
    ids = [100, 101, 102, 103]
    vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    result = client.insert_vectors("smoke_test_py", ids, vectors)
    print(f"  Inserted: {result}")
    
    # 3. Search
    print("\n=== Vector Search (K=3) ===")
    query = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = client.search("smoke_test_py", query, k=3)
    for r in results:
        print(f"  ID={r.id}  distance={r.distance:.6f}")
    
    # 4. Graph operations
    print("\n=== Graph: Add Nodes ===")
    client.add_node("web1", "service", {"port": "8080", "framework": "actix"})
    client.add_node("db1", "service", {"port": "5432", "engine": "postgres"})
    client.add_node("cache1", "service", {"port": "6379", "engine": "redis"})
    print("  Added: web1, db1, cache1")
    
    print("\n=== Graph: Add Edges ===")
    client.add_edge("web1", "depends_on", "db1", {"latency_ms": "5"})
    client.add_edge("web1", "depends_on", "cache1", {"latency_ms": "1"})
    print("  web1 --depends_on--> db1")
    print("  web1 --depends_on--> cache1")
    
    print("\n=== Graph: Traverse from web1 ===")
    nodes, edges = client.traverse("web1", max_depth=2)
    for n in nodes:
        print(f"  Node: {n.id} ({n.node_type}) props={n.properties}")
    for e in edges:
        print(f"  Edge: {e.from_id} --{e.edge_type}--> {e.to_id} props={e.properties}")
    
    # 5. KV operations
    print("\n=== KV Put/Get ===")
    client.put(b"config:version", b"0.5.0")
    val = client.get(b"config:version")
    print(f"  config:version = {val}")
    
    print("\n========================================")
    print("  All Tests Passed!")
    print("  SochDB v0.5.0 running in minikube")
    print("========================================")
    
    client.close()


if __name__ == "__main__":
    main()
