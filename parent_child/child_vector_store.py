from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from .parent_child_chunker import ChildChunk


class ChildVectorStore:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "parent_child_children", dim: int = 384):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        cols = self.client.get_collections().collections
        if not any(c.name == self.collection for c in cols):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def upsert_children(self, children: List[ChildChunk]) -> bool:
        points: List[PointStruct] = []
        for c in children:
            if not c.embedding:
                continue
            payload = {
                "child_id": str(c.child_id),
                "parent_id": str(c.parent_id),
                "snippet": c.content[:256]
            }
            points.append(PointStruct(id=str(c.child_id), vector=c.embedding, payload=payload))
        if not points:
            return True
        self.client.upsert(collection_name=self.collection, points=points)
        return True

    def search(self, text_vector: List[float], top_k: int = 6) -> List[Dict[str, Any]]:
        res = self.client.search(collection_name=self.collection, query_vector=text_vector, limit=top_k)
        return [{"score": r.score, "child_id": r.id, "payload": r.payload} for r in res]
