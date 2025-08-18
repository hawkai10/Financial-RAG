"""
DEPRECATED: ml_rag_backend

The classic ML RAG backend is no longer used. The repository now uses the
parent–child pipeline exclusively for ingestion and retrieval.

Use parent_child.pipeline.ParentChildPipeline for ingestion and
parent_child.retriever.ParentContextRetriever for search.
"""

class MLBasedRAGBackend:  # kept for import compatibility in old tests/scripts
    def __init__(self, *_, **__):
        raise RuntimeError(
            "MLBasedRAGBackend is deprecated. Use parent_child pipeline instead."
        )
