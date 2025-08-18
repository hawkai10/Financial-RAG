"""
DEPRECATED: dynamic_schema_manager

Legacy Dgraph schema evolution helper. Parent–child pipeline does not use Dgraph.

Status: Stub only. Do not use.
"""

class DynamicSchemaManager:  # kept for import compatibility
    def __init__(self, *_, **__):
        raise RuntimeError(
            "DynamicSchemaManager is deprecated. Parent–child pipeline does not use Dgraph."
        )
