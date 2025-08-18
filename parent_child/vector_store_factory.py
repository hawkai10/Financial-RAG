import os


def get_child_vector_store():
    backend = os.getenv("CHILD_VECTOR_BACKEND", "chroma").lower()
    if backend == "chroma":
        from .chroma_child_store import ChromaChildStore

        return ChromaChildStore()
    if backend == "qdrant":
        from .child_vector_store import ChildVectorStore

        return ChildVectorStore()
    if backend == "pgvector":
        try:
            from .pgvector_child_store import PGVectorChildStore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PGVector backend requested but dependencies are missing: psycopg and pgvector") from e
        return PGVectorChildStore()
    # default
    from .chroma_child_store import ChromaChildStore

    return ChromaChildStore()
