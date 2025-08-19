from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .vector_store_factory import get_child_vector_store
from .parent_store import ParentStore


class ParentContextRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vec = get_child_vector_store()
        self.parents = ParentStore()

    def query(self, text: str, top_k: int = 6, dedup_parents: int = 4) -> Dict:
        qv = self.embedder.encode(text).tolist()
        child_hits = self.vec.search(qv, top_k=top_k)
        parent_ids: List[int] = []
        for h in child_hits:
            try:
                pid = int(h["payload"]["parent_id"])
                parent_ids.append(pid)
            except Exception:
                continue
        # preserve order, dedup
        seen = set()
        ordered = []
        for pid in parent_ids:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
            if len(ordered) >= dedup_parents:
                break
        parents = self.parents.get_parents_by_ids(ordered)
        return {
            "child_hits": child_hits,
            "parent_contexts": [
                {
                    "parent_id": p.parent_id,
                    "document_id": p.document_id,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "content": p.content,
                }
                for p in parents
            ],
        }
