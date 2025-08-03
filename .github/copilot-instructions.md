# GitHub Copilot Instructions

This document provides essential guidance for AI agents working on the Financial-RAG codebase.

## 🏛️ Architecture Overview

The system is a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed for financial document analysis. Understanding the data flow and component responsibilities is crucial.

1.  **Orchestration**: `rag_backend.py` is the central orchestrator. It manages the end-to-end process from query intake to final answer synthesis.
2.  **Query Understanding**: `unified_query_processor.py` is the first step. It analyzes the user's query to determine the `strategy` ("Standard", "Analyse", or "Aggregation"). This strategy dictates how other components behave.
3.  **Chunk Management**: The `chunk_manager.py` provides an efficient, memory-mapped interface to the `contextualized_chunks.json` data. It uses an index (`chunks.index.json`) for lazy loading, which is a critical performance optimization. **Do not read `contextualized_chunks.json` directly; use the `ChunkManager`**.
4.  **Retrieval**: `progressive_retrieval.py` implements a smart, two-stage retrieval process. It fetches a small set of chunks, assesses their quality, and decides whether to fetch more. This is more efficient than traditional fixed-`top_k` retrieval.
5.  **Reranking**: `document_reranker.py` takes the retrieved chunks and uses a `cross-encoder/ms-marco-MiniLM-L-12-v2` model to calculate a precise relevance score. This is a key step for ensuring high-quality context for the LLM.
6.  **Answer Synthesis**: `rag_backend.py` uses the reranked chunks to build a prompt for the Gemini API and generate the final answer.

## 🧑‍💻 Developer Workflows

-   **Initial Setup**: Before running any code, you **must** initialize the chunks database. This is a critical first step.
    ```bash
    python init_chunks_db.py
    ```
-   **Testing the Full Pipeline**: To test end-to-end functionality, use the `test_new_system.py` script.
    ```bash
    python test_new_system.py
    ```
-   **Cache Management**: The system uses multiple layers of caching. If you are not seeing changes reflected, clear the caches.
    ```bash
    python simple_clear_cache.py
    ```
-   **Debugging**: For targeted debugging, use `debug_retrieval.py` to inspect the retrieval and reranking process for a given query.

## 🧩 Project Conventions & Patterns

-   **Strategy-Aware Logic**: When modifying the pipeline, always consider the `strategy` variable. The system's behavior (e.g., number of chunks to retrieve, reranking logic, prompting) changes based on whether the strategy is "Standard", "Analyse", or "Aggregation". For example, reranking is skipped for the "Aggregation" strategy to preserve all initial results.
    ```python
    # In document_reranker.py
    if strategy == "Aggregation":
        logger.info("Skipping reranking for aggregation query...")
        return chunks[:top_k], rerank_info
    ```
-   **Scoring is Multi-Stage**: A chunk's relevance is determined by a `final_rerank_score`. This is a weighted average of the initial `retrieval_score` and the much more important `cross_encoder_score`. When working with chunk relevance, always refer to `final_rerank_score`.
    ```python
    # In document_reranker.py
    chunk["final_rerank_score"] = float((normalized_scores[i] * 0.8) + (retrieval_score * 0.2))
    ```
-   **Configuration**: All key parameters (file paths, model names, thresholds) are centralized in `config.py`. Do not hard-code these values elsewhere.
-   **Efficient Chunk Access**: When you need to get the content of a chunk, do not read the main JSON file. Use the `ChunkManager` or the `get_chunk_by_id_enhanced` function in `rag_backend.py`, which uses multiple layers of caching and the memory-mapped `ChunkManager`.

## 🔗 External Dependencies & Integrations

-   **Primary Data Source**: `contextualized_chunks.json`.
-   **Databases**: `chunks.db` (for chunk content) and `feedback.db` (for user feedback and performance metrics). Both are SQLite.
-   **LLM**: The system uses the Gemini API for query processing and answer generation. API calls are centralized in `call_gemini_enhanced` within `rag_backend.py`.
-   **Key Python Libraries**: `txtai` (for embeddings), `sentence-transformers` (for the cross-encoder), and `numpy` (for scoring calculations).
