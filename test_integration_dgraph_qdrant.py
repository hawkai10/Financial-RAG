"""
Integration test for Dgraph and Qdrant connectivity and schema operations.
This script checks:
- Dgraph: schema mutation, query, and statistics
- Qdrant: collection creation, upsert, and search
"""

import sys
import requests
from adaptive_dgraph_manager import AdaptiveDgraphManager
from adaptive_qdrant_manager import AdaptiveQdrantManager
from dynamic_schema_manager import DynamicSchemaManager


def test_dgraph():
    print("\n--- Dgraph Test ---")
    try:
        dgraph = AdaptiveDgraphManager()
        stats = dgraph.get_statistics()
        print("Dgraph statistics:", stats)
        # Try a simple schema mutation (add a test predicate)
        schema = "test_predicate: string ."
        result = dgraph._execute_schema_mutation(schema)
        print("Schema mutation result:", result)
        # Try a simple query (should not fail)
        query = "{ all(func: has(test_predicate)) { uid test_predicate } }"
        resp = requests.post(f"{dgraph.dgraph_url}/query", data=query)
        print("Query status:", resp.status_code)
        print("Query response:", resp.text[:200])
        print("Dgraph test PASSED\n")
    except Exception as e:
        print("Dgraph test FAILED:", e)
        sys.exit(1)

def test_qdrant():
    print("\n--- Qdrant Test ---")
    try:
        qdrant = AdaptiveQdrantManager()
        # Check collection exists or create
        collection_name = getattr(qdrant, 'collection_name', 'adaptive_chunks')
        client = getattr(qdrant, 'client', None)
        if client is not None:
            collections = client.get_collections().collections
            print("Qdrant collections:", [c.name for c in collections])
            # Upsert a test point
            import numpy as np
            import uuid
            vector = np.random.rand(384).tolist()
            payload = {"test_field": "test_value"}
            point_id = str(uuid.uuid4())
            client.upsert(collection_name=collection_name, points=[{"id": point_id, "vector": vector, "payload": payload}])
            print("Upserted test point to Qdrant.")
            # Search
            search_result = client.search(collection_name=collection_name, query_vector=vector, limit=1)
            print("Qdrant search result:", search_result)
            print("Qdrant test PASSED\n")
        else:
            print("Qdrant client not found in AdaptiveQdrantManager.")
            sys.exit(1)
    except Exception as e:
        print("Qdrant test FAILED:", e)
        sys.exit(1)

def main():
    test_dgraph()
    test_qdrant()
    print("\nAll integration tests completed.")

if __name__ == "__main__":
    main()
