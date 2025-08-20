from __future__ import annotations

import os
from typing import List, Dict, Any

import chromadb


class ChromaChildStore:
    """Child vector store backed by ChromaDB.

    Uses a persistent directory so index survives restarts.
    Env vars:
      - CHROMA_CHILD_PERSIST_DIR (default: ./.chroma_children)
      - CHILD_VECTOR_COLLECTION (default: parent_child_children)
    """

    def __init__(self, persist_dir: str | None = None, collection: str | None = None):
        # Use a stable default under the project root to avoid CWD-dependent paths
        if persist_dir:
            self.persist_dir = persist_dir
        else:
            env_dir = os.getenv("CHROMA_CHILD_PERSIST_DIR")
            if env_dir:
                self.persist_dir = env_dir
            else:
                # Project root = parent of this file's directory (../)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.persist_dir = os.path.join(project_root, ".chroma_children")
        self.collection_name = collection or os.getenv("CHILD_VECTOR_COLLECTION", "parent_child_children")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # We pass embeddings explicitly, so no embedding_function is needed
        self.col = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def upsert_children(self, children) -> bool:
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []
        for c in children:
            if not getattr(c, "embedding", None):
                continue
            ids.append(str(c.child_id))
            # Store full child content as snippet (no truncation)
            meta = {"parent_id": str(c.parent_id), "snippet": c.content}
            if getattr(c, "context", None):
                meta["context"] = c.context
            metadatas.append(meta)
            embeddings.append(c.embedding)
        if not ids:
            return True
        # Chroma upsert is via add with same ids; it will append if missing and error if duplicate.
        # Use upsert if available; else delete + add.
        try:
            self.col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        except AttributeError:
            # Fallback for older versions
            self.col.delete(ids=ids)
            self.col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        return True

    def search(self, text_vector: List[float], top_k: int = 6):
        res = self.col.query(query_embeddings=[text_vector], n_results=top_k, include=["metadatas", "distances"])
        out: List[Dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i in range(len(ids)):
            meta = metas[i] or {}
            dist = dists[i]
            # Convert distance to score similar to cosine similarity
            score = 1.0 - float(dist) if dist is not None else None
            out.append({"score": score, "child_id": ids[i], "payload": meta})
        return out

    def count(self) -> int:
        try:
            return int(self.col.count())
        except Exception:
            return -1
