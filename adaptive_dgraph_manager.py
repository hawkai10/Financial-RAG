"""
DEPRECATED: adaptive_dgraph_manager

This module belonged to the legacy classic pipeline (Dgraph). The project now
uses the parent–child pipeline exclusively.

Status: Stub only. Do not use.
"""

class AdaptiveDgraphManager:  # kept for import compatibility
    def __init__(self, *_, **__):
        raise RuntimeError(
            "AdaptiveDgraphManager is deprecated. Use parent_child pipeline; Dgraph is not required."
        )

    def __getattr__(self, name):  # pragma: no cover
        raise RuntimeError(
            "Deprecated legacy manager accessed. Migrate to parent_child pipeline."
        )
