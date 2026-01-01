#!/usr/bin/env python3
"""
Example 06: JSON Documents - Store and query JSON documents
"""

import os, shutil, sys, json
from datetime import datetime
from typing import Optional, Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from toondb import Database

DB_PATH = "./example_06_db"

def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

class DocumentStore:
    def __init__(self, db: Database, collection: str):
        self.db = db
        self.collection = collection
    
    def insert(self, doc: Dict, doc_id: str = None) -> str:
        if not doc_id:
            import uuid
            doc_id = str(uuid.uuid4())[:8]
        
        meta = {"created_at": datetime.utcnow().isoformat(), "version": 1}
        self.db.put(f"doc:{self.collection}:{doc_id}".encode(), json.dumps(doc).encode())
        self.db.put(f"meta:{self.collection}:{doc_id}".encode(), json.dumps(meta).encode())
        return doc_id
    
    def get(self, doc_id: str) -> Optional[Dict]:
        data = self.db.get(f"doc:{self.collection}:{doc_id}".encode())
        return json.loads(data.decode()) if data else None
    
    def update(self, doc_id: str, updates: Dict) -> bool:
        doc = self.get(doc_id)
        if not doc:
            return False
        doc.update(updates)
        self.db.put(f"doc:{self.collection}:{doc_id}".encode(), json.dumps(doc).encode())
        return True
    
    def list_all(self, limit: int = 100) -> List[Dict]:
        prefix = f"doc:{self.collection}:".encode()
        docs = []
        for key, val in self.db.scan(prefix, f"doc:{self.collection};".encode()):
            if len(docs) >= limit:
                break
            docs.append(json.loads(val.decode()))
        return docs

def main():
    print("=" * 60)
    print("ToonDB - Example 06: JSON Documents")
    print("=" * 60)
    
    cleanup()
    
    with Database.open(DB_PATH) as db:
        docs = DocumentStore(db, "articles")
        
        # Insert
        id1 = docs.insert({"title": "Getting Started", "author": "Alice", "views": 100})
        id2 = docs.insert({"title": "Advanced Queries", "author": "Bob", "views": 50})
        print(f"✓ Inserted articles: {id1}, {id2}")
        
        # Get
        article = docs.get(id1)
        print(f"✓ Retrieved: {article['title']}")
        
        # Update
        docs.update(id1, {"views": 150})
        print(f"✓ Updated views to 150")
        
        # List
        all_docs = docs.list_all()
        print(f"✓ Total documents: {len(all_docs)}")
    
    cleanup()
    print("✓ Example completed!")

if __name__ == "__main__":
    main()
