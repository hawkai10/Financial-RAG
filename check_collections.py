#!/usr/bin/env python3
"""Quick check of per-model Chroma collection counts."""

from parent_child.chroma_child_store import ChromaChildStore
import sqlite3
import os

def main():
    print("Checking dual-encoder collections...")
    
    # Check per-model collections
    collections = [
        "children_baai_bge_small_en_v1_5",
        "children_thenlper_gte_small"
    ]
    
    for coll in collections:
        try:
            store = ChromaChildStore(collection=coll)
            count = store.count()
            print(f"  {coll}: {count} vectors (persist_dir: {store.persist_dir})")
        except Exception as e:
            print(f"  {coll}: ERROR - {e}")
    
    # Check parent count
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        parent_db = os.path.join(base, 'parent_child', 'parents.db')
        if os.path.exists(parent_db):
            conn = sqlite3.connect(parent_db)
            count = conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
            conn.close()
            print(f"  parents.db: {count} parents")
        else:
            print(f"  parents.db: NOT FOUND at {parent_db}")
    except Exception as e:
        print(f"  parents.db: ERROR - {e}")

if __name__ == "__main__":
    main()
