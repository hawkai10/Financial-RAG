import json
from typing import List
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
        return {
            'parents': len(parents),
            'children': len(children)
        }
