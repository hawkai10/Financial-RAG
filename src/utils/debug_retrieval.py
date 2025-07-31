#!/usr/bin/env python3
"""
Debug RAG retrieval issues
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from txtai import Embeddings
import sqlite3
from .paths import CHUNKS_DB

def test_retrieval():
    print("=== RAG Retrieval Debug ===")
    
    # Test 1: Check database
    print("\n1. Checking database...")
    conn = sqlite3.connect(str(CHUNKS_DB))
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM chunks')
    chunk_count = cursor.fetchone()[0]
    print(f"Total chunks in database: {chunk_count}")
    
    if chunk_count > 0:
        cursor.execute('SELECT chunk_id, document_name, SUBSTR(chunk_text, 1, 100) FROM chunks LIMIT 3')
        sample_chunks = cursor.fetchall()
        print("\nSample chunks:")
        for chunk in sample_chunks:
            print(f"  ID: {chunk[0]}, Doc: {chunk[1]}, Text: {chunk[2]}...")
    
    conn.close()
    
    # Test 2: Check embedding index
    print("\n2. Checking embedding index...")
    try:
        embeddings = Embeddings({'path': 'sentence-transformers/all-MiniLM-L6-v2', 'content': True})
        embeddings.load('business-docs-index')
        embed_count = embeddings.count()
        print(f"Embedding index documents: {embed_count}")
        
        if embed_count > 0:
            # Test 3: Simple search
            print("\n3. Testing search...")
            test_queries = [
                "financial performance",
                "revenue",
                "profit",
                "business"
            ]
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                results = embeddings.search(query, 3)
                print(f"Results: {len(results)}")
                
                for i, result in enumerate(results):
                    print(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.3f}")
                    if 'text' in result:
                        print(f"     Text: {result['text'][:100]}...")
        else:
            print("ERROR: No documents in embedding index!")
            
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()
