from typing import List, Dict
import os
from pathlib import Path

# Avoid importing HF stack if forcing local
_force_local_import = os.getenv('FORCE_LOCAL_EMBEDDER', 'false').lower() == 'true'
if not _force_local_import:
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from local_embedder import SentenceTransformerWrapper as SentenceTransformer
else:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from local_embedder import SentenceTransformerWrapper as SentenceTransformer

from .vector_store_factory import get_child_vector_store
from .parent_store import ParentStore


class ParentContextRetriever:
    def __init__(self):
        # Initialize dual embedding models: BAAI and GTE
        # Prefer local paths when provided to avoid huggingface downloads
        baai_path = os.getenv('EMBED_BAAI_PATH', '').strip()
        gte_path = os.getenv('EMBED_GTE_PATH', '').strip()
        baai_model = baai_path if baai_path and Path(baai_path).exists() else os.getenv('EMBED_BAAI_NAME', 'BAAI/bge-small-en-v1.5')
        gte_model = gte_path if gte_path and Path(gte_path).exists() else os.getenv('EMBED_GTE_NAME', 'thenlper/gte-small')
        force_local = os.getenv('FORCE_LOCAL_EMBEDDER', 'false').lower() == 'true'
        def _build(model_spec: str):
            if force_local:
                from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                return _STW(model_spec)
            try:
                return SentenceTransformer(model_spec)
            except Exception:
                from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                return _STW(model_spec)

        self.embedders = {
            'baai': _build(baai_model),
            'gte': _build(gte_model)
        }
        # Vector stores: per-model collections matching ingestion
        def _default_coll(name: str) -> str:
            import re
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            return f"children_{slug}"
        self.vecs = {
            'baai': get_child_vector_store(collection=_default_coll('BAAI/bge-small-en-v1.5')),
            'gte': get_child_vector_store(collection=_default_coll('thenlper/gte-small')),
        }
        self.parents = ParentStore()

    def _encode_query_dual(self, text: str):
        """Encode query using both BAAI and GTE models, then combine."""
        import numpy as np
        
        # Get embeddings from both models
        baai_vec = self.embedders['baai'].encode(text, convert_to_numpy=True)
        gte_vec = self.embedders['gte'].encode(text, convert_to_numpy=True)
        
        # Normalize vectors before combining
        baai_norm = baai_vec / np.linalg.norm(baai_vec)
        gte_norm = gte_vec / np.linalg.norm(gte_vec)
        
        # Weighted combination: 0.6 * BAAI + 0.4 * GTE
        combined_vec = 0.6 * baai_norm + 0.4 * gte_norm
        
        # Renormalize the combined vector
        final_vec = combined_vec / np.linalg.norm(combined_vec)
        
        return final_vec.tolist()

    def query(self, text: str, top_k: int = 6, dedup_parents: int = 4) -> Dict:
        # Encode per model and search in respective collections; fuse with RRF
        def _rrf(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank)
        ranked_lists: List[List[Dict]] = []
        # Per-model queries
        for name, vec in self.vecs.items():
            qv = self.embedders[name].encode(text, convert_to_numpy=False)
            res = vec.search(qv, top_k=top_k)
            for i, r in enumerate(res):
                r['rank'] = i + 1
                r['encoder'] = name
            ranked_lists.append(res)
        # RRF fuse
        agg: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}
        for lst in ranked_lists:
            for r in lst:
                cid = str(r.get('child_id') or (r.get('payload', {}) or {}).get('child_id') or '')
                if not cid:
                    continue
                agg[cid] = agg.get(cid, 0.0) + _rrf(r.get('rank', 1))
                if cid not in payloads:
                    payloads[cid] = r
        child_hits = sorted(
            [{'child_id': cid, 'score': sc, 'payload': (payloads[cid].get('payload') or {})} for cid, sc in agg.items()],
            key=lambda x: x['score'], reverse=True
        )[:top_k]
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
