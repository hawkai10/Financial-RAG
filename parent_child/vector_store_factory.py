import os


def get_child_vector_store(collection: str | None = None, table: str | None = None):
    """Return a child vector store instance.

    Parameters:
    - collection: name for backends that support named collections (e.g., Chroma, Qdrant)
    - table: name for pgvector backend

    Falls back to defaults if not provided.
    """
    backend = os.getenv("CHILD_VECTOR_BACKEND", "chroma").lower()
    if backend == "chroma":
        from .chroma_child_store import ChromaChildStore
        return ChromaChildStore(collection=collection)
    if backend == "qdrant":
        from .child_vector_store import ChildVectorStore
        return ChildVectorStore(collection=collection) if collection else ChildVectorStore()
    if backend == "pgvector":
        try:
            from .pgvector_child_store import PGVectorChildStore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PGVector backend requested but dependencies are missing: psycopg and pgvector") from e
        return PGVectorChildStore(table=table) if table else PGVectorChildStore()
    # default
    from .chroma_child_store import ChromaChildStore
    return ChromaChildStore(collection=collection)
