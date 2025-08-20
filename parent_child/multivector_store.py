from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import chromadb  # Uses Chroma as the default multi-vector backend
import torch

# Conditional HF imports: avoid transformers in forced-local mode
_force_local = os.getenv("FORCE_LOCAL_EMBEDDER", "false").lower() == "true"
_hf_available = False
if not _force_local:
    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        _hf_available = True
    except Exception:
        _hf_available = False

# Optional local embedder for token-level outputs
_local_embedder_available = True
try:
    from local_embedder import create_local_embedder  # provides .model and .tokenizer
except Exception:
    _local_embedder_available = False


class MultiVectorChildStore:
    """Multi-vector (ColBERT-style) child token store backed by ChromaDB.

    Design (practical and simple):
    - Each token embedding of a child chunk is stored as a separate point
      with id f"{child_id}:{token_idx}" in a dedicated collection.
    - Payload stores child_id, parent_id, token_idx, and snippet (full child text)
      so retrieval can aggregate back to child-level.
    - Query: embed tokens for the query text, search per query-token, and aggregate
      with MaxSim per token and sum across tokens per child.

    Env vars:
      - CHILD_MULTI_COLLECTION (default: parent_child_child_tokens)
      - CHROMA_CHILD_PERSIST_DIR (shared with single-vector store)
      - MULTIVECTOR_MODEL (default: bert-base-uncased)
      - MULTIVECTOR_MAX_TOKENS (default: 128 for children, 16 for queries)
      - MULTIVECTOR_QUERY_TOKENS (default: 16)
      - MULTIVECTOR_TOPK_PER_TOKEN (default: 10)
    """

    def __init__(self, persist_dir: str | None = None, collection: str | None = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_CHILD_PERSIST_DIR", os.path.join(os.getcwd(), ".chroma_children"))
        self.collection_name = collection or os.getenv("CHILD_MULTI_COLLECTION", "parent_child_child_tokens")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.col = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        # Model setup: prefer local path when forced-local or provided
        self.child_max_tokens = int(os.getenv("MULTIVECTOR_MAX_TOKENS", "128"))
        self.query_max_tokens = int(os.getenv("MULTIVECTOR_QUERY_TOKENS", "16"))
        self.topk_per_token = int(os.getenv("MULTIVECTOR_TOPK_PER_TOKEN", "10"))

        self._use_local = False
        self._disabled_reason: Optional[str] = None

        local_path = os.getenv("MULTIVECTOR_MODEL_PATH", "").strip()
        if local_path and os.path.exists(local_path) and _local_embedder_available:
            try:
                self.local = create_local_embedder(local_path)
                self._use_local = True
            except Exception as e:
                self._use_local = False
                self._disabled_reason = f"Failed to load local multi-vector model: {e}"
        elif _hf_available and not _force_local:
            model_name = os.getenv("MULTIVECTOR_MODEL", "bert-base-uncased")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self._use_local = False
            except Exception as e:
                self._disabled_reason = f"Failed to load transformers model '{model_name}': {e}"
        else:
            if _force_local and not local_path:
                self._disabled_reason = "FORCE_LOCAL_EMBEDDER=true but MULTIVECTOR_MODEL_PATH is not set"
            else:
                self._disabled_reason = "No suitable model available for multi-vector store"

        if self._disabled_reason:
            # Soft-disable: operations will no-op gracefully
            pass

    def _embed_tokens(self, text: str, max_tokens: int) -> List[List[float]]:
        if not text:
            return []
        if self._disabled_reason:
            return []
        with torch.no_grad():
            if getattr(self, "_use_local", False):
                # Use local embedder's tokenizer/model directly
                toks = self.local.tokenizer.encode(text, max_length=max_tokens)
                outputs = self.local.model(**toks)
                hidden = outputs["last_hidden_state"].squeeze(0)
            else:
                toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
                outputs = self.model(**toks)
                hidden = outputs.last_hidden_state.squeeze(0)

            # Remove CLS/SEP if present by dropping first/last token for BERT-style models
            vecs = hidden
            if hidden.size(0) >= 2:
                vecs = hidden[1:-1]
            # L2-normalize token vectors for cosine-friendly behavior
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            return vecs.cpu().tolist()

    def upsert_child_tokens(self, children: List[Any]) -> bool:
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []
        for c in children:
            text = getattr(c, "content", None)
            if not text:
                continue
            token_vecs = self._embed_tokens(text, self.child_max_tokens)
            if not token_vecs:
                continue
            child_id = str(getattr(c, "child_id"))
            parent_id = str(getattr(c, "parent_id"))
            for idx, v in enumerate(token_vecs):
                ids.append(f"{child_id}:{idx}")
                metas.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "token_idx": idx,
                    "snippet": text,
                })
                embs.append(v)
        if not ids:
            return True
        try:
            self.col.upsert(ids=ids, embeddings=embs, metadatas=metas)
        except AttributeError:
            self.col.delete(ids=ids)
            self.col.add(ids=ids, embeddings=embs, metadatas=metas)
        return True

    def search_aggregate(self, query_text: str, top_k_children: int = 24) -> List[Dict[str, Any]]:
        # Embed query tokens
        qvecs = self._embed_tokens(query_text, self.query_max_tokens)
        if not qvecs:
            return []
        # For each query token, retrieve nearest token vectors and aggregate per child via max-sim
        child_scores: Dict[str, float] = {}
        child_payload_any: Dict[str, Dict[str, Any]] = {}
        for qv in qvecs:
            res = self.col.query(query_embeddings=[qv], n_results=self.topk_per_token, include=["metadatas", "distances", "ids"])
            ids = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            # compute per-child max(sim) for this token
            local_best: Dict[str, float] = {}
            for i in range(len(ids)):
                meta = metas[i] or {}
                dist = float(dists[i]) if dists and dists[i] is not None else 1.0
                score = 1.0 - dist
                cid = str(meta.get("child_id") or "")
                if not cid:
                    continue
                if cid not in local_best or score > local_best[cid]:
                    local_best[cid] = score
                    child_payload_any[cid] = {
                        "child_id": cid,
                        "parent_id": meta.get("parent_id"),
                        "snippet": meta.get("snippet", ""),
                    }
            # sum across query tokens
            for cid, s in local_best.items():
                child_scores[cid] = child_scores.get(cid, 0.0) + s
        # Build hits list
        ranked = sorted(child_scores.items(), key=lambda it: it[1], reverse=True)[:top_k_children]
        out: List[Dict[str, Any]] = []
        for cid, sc in ranked:
            meta = child_payload_any.get(cid, {})
            out.append({
                "score": float(sc),
                "child_id": cid,
                "payload": {
                    "parent_id": meta.get("parent_id"),
                    "snippet": meta.get("snippet", ""),
                },
            })
        return out
