from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING


def _get_distance_ops(distance: str) -> str:
    d = (distance or "cosine").lower()
    if d in ("cos", "cosine"):
        return "vector_cosine_ops"
    if d in ("l2", "euclidean"):
        return "vector_l2_ops"
    if d in ("ip", "inner", "inner_product"):
        return "vector_ip_ops"
    return "vector_cosine_ops"


def _get_distance_operator(distance: str) -> str:
    d = (distance or "cosine").lower()
    if d in ("cos", "cosine"):
        return "<=>"
    if d in ("l2", "euclidean"):
        return "<->"
    if d in ("ip", "inner", "inner_product"):
        return "<#>"
    return "<=>"


if TYPE_CHECKING:
    from .parent_child_chunker import ChildChunk  # for type checking only


class PGVectorChildStore:
    """Child vector store backed by Postgres + pgvector.

    Environment configuration (libpq-style or DSN):
    - CHILD_VECTOR_TABLE (default: child_embeddings)
    - PGVECTOR_DIM (default: 384)
    - PGVECTOR_DISTANCE (cosine|euclidean|ip, default: cosine)
    - DATABASE_URL or libpq vars (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)
    - PGVECTOR_LISTS (ivfflat lists; default: 100)
    """

    def __init__(self, table: Optional[str] = None, dim: int = 384, distance: str = "cosine", dsn: Optional[str] = None):
        try:
            import psycopg  # type: ignore
            from pgvector.psycopg import register_vector  # type: ignore
        except Exception as e:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "psycopg and pgvector are required. Add 'psycopg[binary]' and 'pgvector' to requirements and install."
            ) from e

        self._psycopg = psycopg
        self._register_vector = register_vector

        self.table = table or os.getenv("CHILD_VECTOR_TABLE", "child_embeddings")
        self.dim = int(os.getenv("PGVECTOR_DIM", str(dim)))
        self.distance = os.getenv("PGVECTOR_DISTANCE", distance)
        self._opclass = _get_distance_ops(self.distance)
        self._operator = _get_distance_operator(self.distance)
        self._lists = int(os.getenv("PGVECTOR_LISTS", "100"))
        self._dsn = dsn or os.getenv("DATABASE_URL")  # falls back to libpq env if None

        self._conn = self._connect()
        self._ensure_schema()

    def _connect(self):
        if self._dsn:
            conn = self._psycopg.connect(self._dsn)
        else:
            # Leverage libpq environment variables
            conn = self._psycopg.connect()
        self._register_vector(conn)
        conn.execute("SET application_name = 'financial-rag-pgvector'")
        conn.commit()
        return conn

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # Create extension, table, and index if not present
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                child_id BIGINT PRIMARY KEY,
                parent_id BIGINT,
                embedding vector({self.dim}),
                snippet TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
            ON {self.table}
            USING ivfflat (embedding {self._opclass}) WITH (lists = {self._lists})
            """
        )
        self._conn.commit()
        cur.close()

    def upsert_children(self, children: List["ChildChunk"]) -> bool:
        # Local import to avoid circular
        from .parent_child_chunker import ChildChunk

        if not children:
            return True
        rows = []
        for c in children:
            if not c.embedding:
                continue
            rows.append((int(c.child_id), int(c.parent_id), c.embedding, c.content[:256]))
        if not rows:
            return True
        cur = self._conn.cursor()
        cur.executemany(
            f"""
            INSERT INTO {self.table} (child_id, parent_id, embedding, snippet)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (child_id) DO UPDATE
            SET parent_id = EXCLUDED.parent_id,
                embedding = EXCLUDED.embedding,
                snippet = EXCLUDED.snippet
            """,
            rows,
        )
        self._conn.commit()
        cur.close()
        return True

    def search(self, text_vector: List[float], top_k: int = 6) -> List[Dict[str, Any]]:
        op = self._operator
        cur = self._conn.cursor(row_factory=self._psycopg.rows.dict_row)
    # Return distance as score to align with other vector backends
        q = (
            f"SELECT child_id, parent_id, snippet, embedding {op} %s AS score "
            f"FROM {self.table} ORDER BY embedding {op} %s LIMIT %s"
        )
        cur.execute(q, (text_vector, text_vector, int(top_k)))
        rows = cur.fetchall()
        cur.close()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "score": r["score"],
                    "child_id": r["child_id"],
                    "payload": {"parent_id": str(r["parent_id"]), "snippet": r["snippet"]},
                }
            )
        return out
