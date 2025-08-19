import re
import html as htmlmod
from dataclasses import dataclass
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .snowflake_id import SnowflakeGenerator


@dataclass
class ParentChunk:
    parent_id: int
    document_id: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ChildChunk:
    child_id: int
    parent_id: int
    content: str
    embedding: Optional[List[float]]


class ParentChildChunker:
    """Two-tier chunker: parent (large logical section) then child (small precise).
    Embeddings are generated only for child chunks.
    """

    def __init__(self, parent_max_tokens: int = 1500, child_max_tokens: int = 400, child_overlap: int = 40):
        self.parent_max_tokens = parent_max_tokens
        self.child_max_tokens = child_max_tokens
        self.child_overlap = child_overlap
        self.id_gen = SnowflakeGenerator(worker_id=1)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _normalize_text(self, text: str) -> str:
        # If HTML-like, strip tags and unescape entities; collapse whitespace
        if '<' in text and '>' in text:
            # basic breaks to spaces
            t = re.sub(r'<\s*br\s*/?>', '\n', text, flags=re.IGNORECASE)
            # add line breaks for some block tags
            t = re.sub(r'</\s*(p|div|tr|table|h\d)\s*>', '\n', t, flags=re.IGNORECASE)
            t = re.sub(r'<[^>]+>', ' ', t)
            t = htmlmod.unescape(t)
        else:
            t = text
        # collapse whitespace
        t = re.sub(r'[ \t\r\f]+', ' ', t)
        t = re.sub(r'\n\s*\n+', '\n', t)
        return t.strip()

    def _token_len(self, text: str) -> int:
        # heuristic token length ~ word count * 1.3
        return int(len(re.findall(r"\w+", text)) * 1.3)

    def make_parents(self, blocks: List[dict], document_id: str) -> List[ParentChunk]:
        # Concatenate blocks by page with soft boundaries until parent_max_tokens
        parents: List[ParentChunk] = []
        buf = []
        page_start = None
        page_end = None
        acc_tokens = 0
        for b in blocks:
            raw = b.get('content') or b.get('html') or ''
            text = self._normalize_text(raw)
            if not text:
                continue
            page = int(b.get('page', 0))
            tlen = self._token_len(text)
            if page_start is None:
                page_start = page
            if acc_tokens + tlen > self.parent_max_tokens and buf:
                parent_text = '\n'.join(buf).strip()
                parents.append(ParentChunk(
                    parent_id=self.id_gen.next_id(),
                    document_id=document_id,
                    content=parent_text,
                    page_start=page_start,
                    page_end=page_end if page_end is not None else page_start,
                ))
                buf = [text]
                acc_tokens = tlen
                page_start = page
                page_end = page
            else:
                buf.append(text)
                acc_tokens += tlen
                page_end = page
        if buf:
            parent_text = '\n'.join(buf).strip()
            parents.append(ParentChunk(
                parent_id=self.id_gen.next_id(),
                document_id=document_id,
                content=parent_text,
                page_start=page_start or 0,
                page_end=page_end or (page_start or 0),
            ))
        return parents

    def _split_child(self, text: str) -> List[str]:
        # sentence-based split with overlap
        sentences = re.split(r'[.!?]\s+', text)
        chunks: List[str] = []
        cur = ''
        cur_tokens = 0
        for s in sentences:
            if not s:
                continue
            t = s if cur == '' else cur + ' ' + s
            tlen = self._token_len(t)
            if tlen > self.child_max_tokens and cur:
                # emit cur
                chunks.append(cur.strip())
                # overlap
                words = cur.split()
                overlap = ' '.join(words[-self.child_overlap:]) if len(words) > self.child_overlap else cur
                cur = overlap + ' ' + s
                cur_tokens = self._token_len(cur)
            else:
                cur = t
                cur_tokens = tlen
        if cur:
            chunks.append(cur.strip())
        return [c for c in chunks if c]

    def make_children_with_embeddings(self, parents: List[ParentChunk]) -> List[ChildChunk]:
        children: List[ChildChunk] = []
        texts: List[str] = []
        index_map: List[tuple[int, int]] = []  # (parent_idx, child_local_idx)
        for p_idx, p in enumerate(parents):
            child_texts = self._split_child(p.content)
            for c_idx, ct in enumerate(child_texts):
                cid = self.id_gen.next_id()
                children.append(ChildChunk(child_id=cid, parent_id=p.parent_id, content=ct, embedding=None))
                texts.append(ct)
                index_map.append((p_idx, c_idx))
        if texts:
            embs = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            for i, vec in enumerate(embs):
                children[i].embedding = vec.tolist()
        return children
