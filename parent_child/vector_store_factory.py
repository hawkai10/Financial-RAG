def get_child_vector_store(collection: str | None = None, table: str | None = None):
    """Return a child vector store instance.

    Parameters:
    - collection: name for backends that support named collections (Chroma)
    - table: name for pgvector backend

    Falls back to defaults if not provided.
    """
    # Permanently use Chroma backend
    from .chroma_child_store import ChromaChildStore
    return ChromaChildStore(collection=collection)
