import json
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from .parent_child_chunker import ParentChildChunker, ParentChunk, ChildChunk
from .parent_store import ParentStore
from .vector_store_factory import get_child_vector_store
# Optional multi-vector index is imported lazily inside __init__ when enabled


class ParentChildPipeline:
    def __init__(self):
        self.chunker = ParentChildChunker()
        self.parents = ParentStore()
        # Default child store (may not be used directly when indexing per model)
        self.children = get_child_vector_store()
        # Optional multi-vector store (ColBERT-style). Disabled by default; flip to True to enable in code.
        self.mv_enabled = False
        self.children_mv = None
        if self.mv_enabled:
            try:
                from .multivector_store import MultiVectorChildStore  # type: ignore
                self.children_mv = MultiVectorChildStore()
            except Exception:
                # Disable if dependencies are missing or initialization fails
                self.mv_enabled = False
                self.children_mv = None

    def ingest_extracted_json(self, extraction_json_path: str, document_id: str) -> dict:
        # The Marker chunks JSON may be a list of pages/blocks or list of docs; accept a flat blocks list under 'blocks' too
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        blocks: List[dict] = []
        # Accept multiple Marker formats
        if isinstance(data, dict):
            # v1: flat blocks list
            if 'blocks' in data and isinstance(data['blocks'], list):
                blocks = data['blocks']
            # Marker JSON with top-level children and nested structure
            elif 'children' in data and isinstance(data['children'], list):
                def walk(node, page_hint: int | None = None):
                    if not isinstance(node, dict):
                        return
                    # capture any html/content-bearing node
                    html = node.get('html') or node.get('content')
                    if isinstance(html, str) and html.strip():
                        blocks.append({'content': html, 'page': node.get('page', page_hint or 0)})
                    # recurse into children
                    kids = node.get('children')
                    if isinstance(kids, list):
                        for ch in kids:
                            walk(ch, page_hint=node.get('page', page_hint))
                walk(data)
        elif isinstance(data, list) and data:
            # v2: list of docs with pages -> blocks
            if isinstance(data[0], dict) and 'pages' in data[0]:
                for doc in data:
                    for page in doc.get('pages', []):
                        for b in page.get('blocks', []):
                            blocks.append({
                                'content': b.get('content') or b.get('html') or '',
                                'page': page.get('page_number') or b.get('page') or 0
                            })
            # v3: list of blocks already
            elif 'page' in (data[0] or {}):
                blocks = data

        # Filter empties
        blocks = [b for b in blocks if isinstance(b, dict) and (b.get('content') or b.get('html'))]
        logger = logging.getLogger(__name__)
        if not blocks:
            logger.warning(f"[PIPELINE] No blocks parsed from {extraction_json_path}; skipping.")
            return {'parents': 0, 'children': 0, 'log_path': None}

        parents = self.chunker.make_parents(blocks, document_id=document_id)
        if not parents:
            logger.warning(f"[PIPELINE] No parent chunks created for {document_id}; skipping.")
            return {'parents': 0, 'children': 0, 'log_path': None}
        self.parents.upsert_parents(parents)

        # Always dual-encoder ingestion (BAAI + GTE) with per-model collections
        children = self.chunker.make_children(parents)
        if not children:
            logger.warning(f"[PIPELINE] No child chunks created for {document_id}; skipping embeddings.")
            return {'parents': len(parents), 'children': 0, 'log_path': None}

        # Generate succinct context for each child (improves retrieval); best-effort, does not fail ingestion
        try:
            from .api_adapter import call_gemini_enhanced  # type: ignore
            for c in children:
                try:
                    prompt = (
                        "Please give a short succinct context for the purposes of improving search retrieval of the chunk. "
                        "Answer only with the succinct context and nothing else.\n\n"
                        f"<chunk>\n{c.content[:2000]}\n</chunk>"
                    )
                    ctx = asyncio.run(call_gemini_enhanced(prompt))
                    c.context = (ctx or '').strip()[:300] if ctx else None
                except Exception:
                    c.context = None
        except Exception:
            # LLM unavailable; proceed without contexts
            pass

        texts = [c.content for c in children]

        # Fixed models and local preferred paths
        project_root = Path(__file__).resolve().parents[1]
        model_specs = [
            ("BAAI/bge-small-en-v1.5", project_root / "local_models" / "BAAI-bge-small-en-v1.5"),
            ("thenlper/gte-small", project_root / "local_models" / "thenlper-gte-small"),
        ]

        def _default_coll(name: str) -> str:
            import re
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            return f"children_{slug}"

        # Embed and upsert into each per-model collection
        for model_name, local_path in model_specs:
            # Build embedder preferring local path; fallback to model name if needed
            try:
                lp = Path(local_path)
                target = str(lp) if lp.exists() else model_name
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    model = SentenceTransformer(target)
                except Exception:
                    from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                    model = _STW(target)
            except Exception:
                # If both load paths fail, skip this model
                continue
            coll = _default_coll(model_name)
            vec = get_child_vector_store(collection=coll)
            if not texts:
                continue
            embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            for idx, vec_np in enumerate(embs):
                children[idx].embedding = vec_np.tolist()
            vec.upsert_children(children)

        if self.mv_enabled and self.children_mv:
            # Upsert child token vectors (best-effort; errors won’t break ingestion)
            try:
                self.children_mv.upsert_child_tokens(children)
            except Exception:
                pass

        # Write JSON log of chunks (excluding embeddings for size)
        try:
            # Default to project_root/chunk_logs
            project_root = str(Path(__file__).resolve().parents[1])
            log_dir = os.path.join(project_root, 'chunk_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{document_id}_parent_child_chunks.json")
            # Compute token counts
            parent_token_counts = [self.chunker._token_len(p.content) for p in parents]
            child_token_counts = [self.chunker._token_len(c.content) for c in children]

            log_obj = {
                'document_id': document_id,
                'source_json': os.path.relpath(extraction_json_path, start=log_dir) if os.path.commonpath([log_dir, os.path.dirname(extraction_json_path)]) else extraction_json_path,
                'parents_count': len(parents),
                'children_count': len(children),
                'parents_total_tokens': sum(parent_token_counts) if parent_token_counts else 0,
                'children_total_tokens': sum(child_token_counts) if child_token_counts else 0,
                'parents': [
                    {
                        'parent_id': p.parent_id,
                        'document_id': p.document_id,
                        'page_start': p.page_start,
                        'page_end': p.page_end,
                        'content': p.content,
                        'tokens': parent_token_counts[idx],
                    }
                    for idx, p in enumerate(parents)
                ],
                'children': [
                    {
                        'child_id': c.child_id,
                        'parent_id': c.parent_id,
                        'content': c.content,
                        'tokens': child_token_counts[idx],
                    }
                    for idx, c in enumerate(children)
                ],
            }
            with open(log_path, 'w', encoding='utf-8') as lf:
                json.dump(log_obj, lf, ensure_ascii=False, indent=2)
        except Exception:
            # Logging should never break ingestion
            log_path = None

        return {
            'parents': len(parents),
            'children': len(children),
            'log_path': log_path,
        }

    def ingest_directory(self, base_dir: str) -> Dict[str, int]:
        """Ingest all JSON files found under base_dir recursively.

        Returns a summary dict with totals.
        """
        base = Path(base_dir)
        files = sorted(
            [p for p in base.glob("**/*.json") if p.is_file() and not p.name.endswith("_meta.json")]
        )
        total_parents = 0
        total_children = 0
        for jf in files:
            doc_id = jf.stem
            try:
                res = self.ingest_extracted_json(str(jf), document_id=doc_id)
                total_parents += res.get('parents', 0)
                total_children += res.get('children', 0)
            except Exception:
                # Skip files that fail to ingest; continue with others
                continue
        return {"parents": total_parents, "children": total_children}
