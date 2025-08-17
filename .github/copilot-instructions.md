# Copilot Instructions for Financial-RAG

## Project Overview
- **Purpose:** Enterprise-grade Retrieval-Augmented Generation (RAG) for financial documents, with hierarchical map-reduce, advanced chunking, and robust graph/vector search.
- **Core Components:**
  - `rag_app.py`, `api_server.py`: Main API entrypoints (Flask-based)
  - `adaptive_dgraph_manager.py`: Dgraph (graph DB) manager, handles schema, entities, relationships, and facets
  - `adaptive_qdrant_manager.py`: Qdrant (vector DB) manager, handles vector storage, filtering, and search
  - `enhanced_json_chunker.py`: Chunking and embedding pipeline, produces EnhancedChunk objects
  - `dynamic_schema_manager.py`: Dynamically evolves Dgraph schema based on content analysis
  - `full_agent.py`, `mini_agent.py`: Orchestration/agent logic for query processing
  - `amber-ai-search/`: React+TS frontend for search and QA

## Data Flow & Architecture
- **Pipeline:**
  1. Documents are chunked (`enhanced_json_chunker.py`), producing EnhancedChunk objects with precomputed embeddings and ML metadata.
  2. Chunks are stored in Dgraph (entities/relationships) and Qdrant (vectors, slim metadata only).
  3. Queries are routed through hierarchical processors, which batch, parallelize, and aggregate results.
  4. Results are reranked, deduplicated, and streamed to the user.
- **Dgraph:**
  - Schema is evolved dynamically; entity edges use `[uid] @reverse @count`, entity_type/entity_value/document_type are always indexed.
  - Facets are encoded via a centralized helper; mutations are validated for facet safety.
  - Reverse queries are robust: if entity_type is missing, all `~has_*` edges are traversed.
- **Qdrant:**
  - Only slim metadata (IDs, doc type, snippet) is stored in payload; full text is in Dgraph.
  - Filtering uses MatchAny for lists; filter-only queries use scroll, not zero vector search.

## Developer Workflows
- **Setup:** Use `setup_ml_system.py` for end-to-end setup. It polls for Dgraph/Qdrant health (no input()), suitable for CI/containers.
- **Testing:** Run `test_ml_system.py` for integration tests.
- **Schema Evolution:** All schema changes are managed via `dynamic_schema_manager.py` and are applied automatically.
- **Chunking/Embedding:** All chunking and embedding logic is in `enhanced_json_chunker.py`.

## Project Conventions
- **No keyword-based logic:** All retrieval and processing is ML/embedding-based.
- **All entity/relationship edges in Dgraph use `[uid]` with `@reverse @count`.
- **Facets:** Use the `_build_facet_mutation` helper for all edge facets; never mix node fields and edge facets in the same object.
- **Qdrant payloads:** Only store minimal metadata; do not store full content.
- **Hierarchical processing:** Use batch/parallel/aggregation patterns for large queries.

## Integration Points
- **Dgraph:** Runs at `http://localhost:8080` (default)
- **Qdrant:** Runs at `http://localhost:6333` (default)
- **Frontend:** `amber-ai-search/` (React+Vite)

## Examples
- See `test_ml_system.py` for end-to-end usage and integration patterns.
- See `dynamic_schema_manager.py` for schema evolution logic and conventions.
- See `adaptive_dgraph_manager.py` for robust graph queries and facet-safe mutations.

---
For new AI agents: Always check for dynamic schema, use the provided helpers for facets and mutations, and follow the minimal-payload pattern for Qdrant.
