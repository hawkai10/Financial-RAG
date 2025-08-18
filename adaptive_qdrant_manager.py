"""
DEPRECATED: adaptive_qdrant_manager

This module belonged to the legacy classic pipeline. The project now uses the
parent–child pipeline exclusively. Please switch to `parent_child` modules.

Status: Stub only. Do not use.
"""

class AdaptiveQdrantManager:  # kept for import compatibility
    def __init__(self, *_, **__):
        raise RuntimeError(
            "AdaptiveQdrantManager is deprecated. Use parent_child/*.py (Chroma/pgvector via factory)."
        )

    def __getattr__(self, name):  # pragma: no cover
        raise RuntimeError(
            "Deprecated legacy manager accessed. Migrate to parent_child pipeline."
        )
