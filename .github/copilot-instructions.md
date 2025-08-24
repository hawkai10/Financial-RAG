# Copilot Instructions for Financial-RAG

## Project Overview
- Purpose: Retrieval-Augmented Generation (RAG) for financial documents with hierarchical parent/child chunking, dual-encoder retrieval, reranking, and streaming UI.
- Core components (current):
  - `api_server.py`: Flask API (search, search-stream, filters, pdf, recent-documents)
  - `rag_backend.py`: RAG pipeline (dual-encoder retrieval, early filters, rerank, synthesis)
  - `document_reranker.py`: Cross-encoder reranking utilities
  - `local_embedder.py`: Local SentenceTransformer wrapper using `local_models/`
  - `parent_child/`:
    - `parent_child_chunker.py`: Chunking parents/children
    - `parent_store.py`: SQLite-backed parent contexts (`parents.db`)
    - `chroma_child_store.py`: Chroma vector store for child chunks
    - `vector_store_factory.py`: Returns Chroma child store (Chroma-only)
    - `pipeline.py`, `retriever.py`: Ingestion/retrieval helpers
  - `extraction.py`: Marker-based extraction from source docs to JSON
  - `ingest_all.py`: Batch ingestion/orchestration
  - `amber-ai-search/`: React + TypeScript frontend (search, filters, streaming)

## Data Flow & Architecture
- Storage (Chroma-only):
  - Child chunks are embedded with two local models (BAAI bge-small and GTE-small) and indexed in per-model Chroma collections: `children_baai_bge_small_en_v1_5` and `children_thenlper_gte_small`.
  - Parent contexts are stored in SQLite (`parent_child/parents.db`).
- Pipeline:
  1) Extraction (`extraction.py`): Uses Marker to convert PDFs/office docs to structured JSON.
  2) Chunking & Embedding (`parent_child/*`): Builds parent/child chunks; embeds children and writes to Chroma with minimal metadata (ids, parent_id, document_id, snippet, optional mtime, ext).
  3) Retrieval (`rag_backend.py`):
     - Normalizes the user query and generates related variants.
     - Dual dense retrieval over both collections; optional multivector aggregation.
     - Hybrid scoring with BM25 over candidate child snippets.
     - Early filtering: applies fileType and timeRange to child candidates before reranking; then filters parents and prunes children accordingly.
     - Reranks children (cross-encoder optional), selects top parents, synthesizes concise answer (Gemini).
  4) Streaming: `/search-stream` sends chunks first, then the AI answer via SSE.

Notes:
- Dgraph and Qdrant have been removed; the system is Chroma-only with minimal payloads.
- FileType and TimeRange filters are enforced early on children to prevent leakage and ranking distortion.

## Developer Workflows
- Setup: Local-only. No external DB health checks.
  - Backend: `python api_server.py`
  - Frontend: `cd amber-ai-search && npm install && npm run dev`
- Ingestion: `python ingest_all.py` to process Source_Documents via extraction → chunk/emb → index.
- Testing: Use `scripts/run_end_to_end.py` and `scripts/test_full_flow.py` for smoke/integration.
- Logs: Retrieval traces in `test_logs/`; chunk logs in `chunk_logs/`.

## Conventions & Filters
- Retrieval is ML/embedding-based; avoid keyword-only heuristics.
- Chroma vector payloads must stay minimal; full content remains on disk/parents.
- File types (UI ↔ backend): `pdf`, `word`, `excel`, `ppt`, `html`, `txt`, `email`, `compressed`, `page`.
- Time ranges: `all`, `3days`, `week`, `month`, `3months`, `year`, `5years`, `custom` (ISO dates supported). Applied to file mtimes.
- Friendly empty-state: When filters remove all candidates, backend returns a concise message instead of raw retrieval errors.

## Integration Points
- No external DB endpoints.
- Endpoints:
  - POST `/search` (JSON): returns documents + aiResponse
  - POST `/search-stream` (JSON): SSE stream (chunks → answer → complete)
  - GET `/filters`: fileTypes, timeRanges
  - GET `/pdf?path=...`: secure PDF serving from `Source_Documents`
  - GET `/recent-documents`: scans `Source_Documents` by mtime
- Frontend: `amber-ai-search/` uses staged filters + Apply; streaming is default, with automatic fallback to non-streaming on errors.

## Examples
- Ingestion + retrieval: `scripts/run_end_to_end.py`
- Retrieval internals: `parent_child/chroma_child_store.py`, `parent_child/retriever.py`
- Full RAG path: `rag_backend.py` (see `execute_single_strategy` and `rag_query_enhanced`)

---
For new AI agents: Use the Chroma child store, keep payloads minimal, honor fileType/timeRange early, and stream chunks before answers.
