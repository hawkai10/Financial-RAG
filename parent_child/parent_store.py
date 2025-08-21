import json
import os
import sqlite3
from dataclasses import asdict
from typing import List, Optional
from .parent_child_chunker import ParentChunk


class ParentStore:
    """Compact parent storage using SQLite by default (can be swapped to Postgres).
    Schema: parents(parent_id BIGINT PRIMARY KEY, document_id TEXT, page_start INT, page_end INT, content TEXT)
    """

    def __init__(self, db_path: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(base, 'parents.db')
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parents (
                parent_id INTEGER PRIMARY KEY,
                document_id TEXT,
                page_start INTEGER,
                page_end INTEGER,
                content TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert_parents(self, parents: List[ParentChunk]):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO parents(parent_id, document_id, page_start, page_end, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(p.parent_id, p.document_id, p.page_start, p.page_end, p.content) for p in parents]
        )
        conn.commit()
        conn.close()

    def get_parents_by_ids(self, ids: List[int]) -> List[ParentChunk]:
        if not ids:
            return []
        conn = sqlite3.connect(self.db_path)
        qmarks = ','.join('?' for _ in ids)
        rows = conn.execute(
            f"SELECT parent_id, document_id, page_start, page_end, content FROM parents WHERE parent_id IN ({qmarks})",
            ids,
        ).fetchall()
        conn.close()
        # Preserve the order of the provided IDs (important for UI relevance ordering)
        order_index = {pid: i for i, pid in enumerate(ids)}
        rows.sort(key=lambda r: order_index.get(r[0], len(order_index)))

        out: List[ParentChunk] = []
        for r in rows:
            out.append(
                ParentChunk(
                    parent_id=r[0],
                    document_id=r[1],
                    page_start=r[2],
                    page_end=r[3],
                    content=r[4],
                )
            )
        return out
