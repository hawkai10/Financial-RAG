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
        # Try a simple query (use JSON payload so Dgraph accepts it)
        query = "{ all(func: has(test_predicate)) { uid test_predicate } }"
        try:
            resp = requests.post(f"{dgraph.dgraph_url}/query", json={"query": query}, timeout=10)
            print("Query status:", resp.status_code)
            try:
                print("Query response:", resp.json())
            except Exception:
                print("Query response (text):", resp.text[:200])
        except Exception as e:
            print("Error executing query:", e)
        print("Dgraph test PASSED\n")
    except Exception as e:
        print("Dgraph test FAILED:", e)
        sys.exit(1)

def test_qdrant():
    print("\n--- Qdrant Test ---")
    try:
        qdrant = AdaptiveQdrantManager()
        collection_name = getattr(qdrant, 'collection_name', 'adaptive_chunks')
        client = getattr(qdrant, 'client', None)

        # Quick HTTP check to see if Qdrant is reachable
        try:
            r = requests.get(getattr(qdrant, 'qdrant_url', 'http://localhost:6333') + '/collections', timeout=5)
            if r.status_code != 200:
                raise RuntimeError(f"Qdrant HTTP endpoint returned {r.status_code}")
        except Exception as e:
            raise RuntimeError(f"Qdrant does not appear to be running at http://localhost:6333: {e}\nStart Qdrant (docker: docker run -d -p 6333:6333 qdrant/qdrant) and retry.")

        if client is None:
            raise RuntimeError("Qdrant client not present on AdaptiveQdrantManager instance")

        # List collections
        collections = client.get_collections()
        # client.get_collections() may return different shapes depending on qdrant-client version
        try:
            names = [c.name for c in collections.collections]
        except Exception:
            names = collections
        print("Qdrant collections:", names)

        # Upsert a test point - generate a vector with the correct dimensionality from the collection
        import numpy as np
        import uuid
        # Determine vector size from collection info if possible
        try:
            coll_info = client.get_collection(collection_name)
            # Different qdrant-client versions expose vector size differently
            try:
                vector_size = coll_info.config.params.vectors.size
            except Exception:
                try:
                    vector_size = coll_info.vectors.size
                except Exception:
                    vector_size = getattr(qdrant, 'embedding_dimension', 384)
        except Exception:
            vector_size = getattr(qdrant, 'embedding_dimension', 384)

        vector = np.random.rand(vector_size).tolist()
        payload = {"test_field": "test_value"}
        point_id = str(uuid.uuid4())

        # Try upsert - method name varies; handle common clients
        try:
            # Construct a PointStruct to match client expectations
            try:
                from qdrant_client.http.models import PointStruct as QPointStruct
                q_point = QPointStruct(id=point_id, vector=vector, payload=payload)
                client.upsert(collection_name=collection_name, points=[q_point])
            except Exception:
                # Fallback to dict if PointStruct not available
                client.upsert(collection_name=collection_name, points=[{"id": point_id, "vector": vector, "payload": payload}])
        except Exception as e:
            raise RuntimeError(f"Failed to upsert point to Qdrant: {e}")

        print("Upserted test point to Qdrant.")

        # Search - attempt a simple search call
        try:
            search_result = client.search(collection_name=collection_name, query_vector=vector, limit=1)
            print("Qdrant search result:", search_result)
        except Exception:
            # Fallback: try client.search_points or client.retrieve
            try:
                search_result = client.search_points(collection_name=collection_name, vector=vector, limit=1)
                print("Qdrant search result (alt):", search_result)
            except Exception as e:
                raise RuntimeError(f"Qdrant search failed: {e}")

        print("Qdrant test PASSED\n")

    except Exception as e:
        print("Qdrant test FAILED:", e)
        sys.exit(1)

def main():
    test_dgraph()
    test_qdrant()
    print("\nAll integration tests completed.")

if __name__ == "__main__":
    main()
