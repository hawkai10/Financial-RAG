"""
DEPRECATED: enhanced_json_chunker

Classic pipeline chunker has been retired. Use parent_child.parent_child_chunker
and parent_child.pipeline for ingestion and retrieval.

Status: Stub only. Do not use.
"""

class EnhancedJSONChunker:  # kept for import compatibility
    def __init__(self, *_, **__):
        raise RuntimeError(
            "EnhancedJSONChunker is deprecated. Use parent_child.parent_child_chunker + pipeline."
        )

class EnhancedChunk:  # minimal symbol for compatibility
    pass
