import json
import os
from pathlib import Path
from typing import List, Dict
from .parent_child_chunker import ParentChildChunker, ParentChunk, ChildChunk
from .parent_store import ParentStore
from .vector_store_factory import get_child_vector_store


class ParentChildPipeline:
    def __init__(self):
        self.chunker = ParentChildChunker()
        self.parents = ParentStore()
        self.children = get_child_vector_store()

    def ingest_extracted_json(self, extraction_json_path: str, document_id: str) -> dict:
        # The Marker chunks JSON may be a list of pages/blocks or list of docs; accept a flat blocks list under 'blocks' too
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        blocks: List[dict] = []
        # try common shapes
        if isinstance(data, dict) and 'blocks' in data:
            blocks = data['blocks']
        elif isinstance(data, list) and len(data) and isinstance(data[0], dict) and 'pages' in data[0]:
            # our earlier extractor format
            for doc in data:
                for page in doc.get('pages', []):
                    for b in page.get('blocks', []):
                        blocks.append({
                            'content': b.get('content') or b.get('html') or '',
                            'page': page.get('page_number') or b.get('page') or 0
                        })
        elif isinstance(data, list) and len(data) and 'page' in (data[0] or {}):
            blocks = data
        else:
            # last resort, attempt to find nested 'html' or 'content'
            pass
        parents = self.chunker.make_parents(blocks, document_id=document_id)
        children = self.chunker.make_children_with_embeddings(parents)
        self.parents.upsert_parents(parents)
        self.children.upsert_children(children)
        # Write JSON log of chunks (excluding embeddings for size)
        try:
            # Default to project_root/chunk_logs unless CHUNK_LOG_DIR is set
            if os.getenv('CHUNK_LOG_DIR'):
                log_dir = os.getenv('CHUNK_LOG_DIR')
            else:
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
        files = sorted([p for p in base.glob("**/*.json") if p.is_file()])
        total_parents = 0
        total_children = 0
        for jf in files:
            doc_id = jf.stem
            try:
                res = self.ingest_extracted_json(str(jf), document_id=doc_id)
                total_parents += res.get('parents', 0)
                total_children += res.get('children', 0)
            except Exception:
                continue
        return {"parents": total_parents, "children": total_children}
