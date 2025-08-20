import os
from pathlib import Path
import re
import html as htmlmod
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Conditional import to prevent pulling in HF stack when forced local
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
    context: Optional[str] = None


class ParentChildChunker:
    """Two-tier chunker: parent (large logical section) then child (small precise).
    Embeddings are generated only for child chunks.
    """

    def __init__(self, parent_max_tokens: int = 1500, child_max_tokens: int = 300, child_overlap: int = 80):
        # Allow environment overrides for easier tuning without code changes
        try:
            self.parent_max_tokens = int(os.getenv('PARENT_MAX_TOKENS', str(parent_max_tokens)))
        except Exception:
            self.parent_max_tokens = parent_max_tokens
        try:
            self.child_max_tokens = int(os.getenv('CHILD_MAX_TOKENS', str(child_max_tokens)))
        except Exception:
            self.child_max_tokens = child_max_tokens
        try:
            self.child_overlap = int(os.getenv('CHILD_OVERLAP', str(child_overlap)))
        except Exception:
            self.child_overlap = child_overlap
        # Sanity: keep overlap less than max tokens
        if self.child_overlap >= self.child_max_tokens:
            self.child_overlap = max(self.child_max_tokens // 5, 10)
        self.id_gen = SnowflakeGenerator(worker_id=1)
        
        # Initialize dual embedding models: prefer local paths when provided
        baai_path = os.getenv('EMBED_BAAI_PATH', '').strip()
        gte_path = os.getenv('EMBED_GTE_PATH', '').strip()
        baai_model = baai_path if baai_path and Path(baai_path).exists() else os.getenv('EMBED_BAAI_NAME', 'BAAI/bge-small-en-v1.5')
        gte_model = gte_path if gte_path and Path(gte_path).exists() else os.getenv('EMBED_GTE_NAME', 'thenlper/gte-small')
        # Instantiate embedders with robust fallback to local wrapper
        force_local = os.getenv('FORCE_LOCAL_EMBEDDER', 'false').lower() == 'true'
        def _build(model_spec: str):
            if force_local:
                # Force use of local wrapper
                from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                return _STW(model_spec)
            try:
                return SentenceTransformer(model_spec)
            except Exception:
                # Fallback to local wrapper if HF stack is unavailable or broken
                from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                return _STW(model_spec)

        self.embedders = {}
        for name, spec in (('baai', baai_model), ('gte', gte_model)):
            try:
                self.embedders[name] = _build(spec)
            except Exception:
                # Skip missing/broken model (e.g., local path not found)
                pass
        if not self.embedders:
            raise RuntimeError("No embedding model available. Set EMBED_BAAI_PATH and/or EMBED_GTE_PATH to local model folders or install sentence-transformers.")
        # Keep single embedder for backward compatibility: prefer BAAI, else first available
        self.embedder = self.embedders.get('baai') or next(iter(self.embedders.values()))

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
        """Split child chunks using a hybrid strategy.

        1) Try sentence-based accumulation with overlap (good for prose).
        2) If that yields too few chunks (e.g., tables/lists), fall back to
           line/window-based splitting to ensure multiple children for retrieval.
        """
        # First pass: sentence-based
        sentences = re.split(r'[.!?]\s+', text)
        chunks: List[str] = []
        cur = ''
        for s in sentences:
            if not s:
                continue
            t = s if cur == '' else cur + ' ' + s
            tlen = self._token_len(t)
            if tlen > self.child_max_tokens and cur:
                # emit current chunk
                chunks.append(cur.strip())
                # build next with overlap
                words = cur.split()
                overlap_words = words[-self.child_overlap:] if len(words) > self.child_overlap else words
                cur = (' '.join(overlap_words) + ' ' + s).strip()
            else:
                cur = t
        if cur:
            chunks.append(cur.strip())
        chunks = [c for c in chunks if c]

        # If only one chunk or oversized chunk, use a line/window-based fallback
        if len(chunks) <= 1 or max(self._token_len(c) for c in chunks) > int(self.child_max_tokens * 0.9):
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            win_chunks: List[str] = []
            buf: List[str] = []
            buf_tokens = 0
            for ln in lines:
                tlen = self._token_len(ln)
                if buf_tokens + tlen > self.child_max_tokens and buf:
                    win = '\n'.join(buf).strip()
                    win_chunks.append(win)
                    # overlap in tokens (approx by words from the end of buffer)
                    words = win.split()
                    overlap_words = words[-self.child_overlap:] if len(words) > self.child_overlap else words
                    buf = [' '.join(overlap_words), ln]
                    buf_tokens = self._token_len(buf[0]) + tlen
                else:
                    buf.append(ln)
                    buf_tokens += tlen
            if buf:
                win_chunks.append('\n'.join(buf).strip())
            # If still nothing, fallback to hard windows over words
            if not win_chunks:
                words = text.split()
                step = max(self.child_max_tokens - self.child_overlap, 1)
                for i in range(0, len(words), step):
                    seg = ' '.join(words[i:i + self.child_max_tokens])
                    if seg:
                        win_chunks.append(seg)
            chunks = [c for c in win_chunks if c]

        return chunks

    def make_children(self, parents: List[ParentChunk]) -> List[ChildChunk]:
        """Create child chunks without embeddings (for multi-embedder pipelines)."""
        children: List[ChildChunk] = []
        for p in parents:
            child_texts = self._split_child(p.content)
            for ct in child_texts:
                cid = self.id_gen.next_id()
                children.append(ChildChunk(child_id=cid, parent_id=p.parent_id, content=ct, embedding=None, context=None))
        return children

    def make_children_with_embeddings(self, parents: List[ParentChunk]) -> List[ChildChunk]:
        children: List[ChildChunk] = []
        texts: List[str] = []
        index_map: List[tuple[int, int]] = []  # (parent_idx, child_local_idx)
        for p_idx, p in enumerate(parents):
            child_texts = self._split_child(p.content)
            for c_idx, ct in enumerate(child_texts):
                cid = self.id_gen.next_id()
                children.append(ChildChunk(child_id=cid, parent_id=p.parent_id, content=ct, embedding=None, context=None))
                texts.append(ct)
                index_map.append((p_idx, c_idx))
        if texts:
            # If both models available, compute and combine; else use the single available model
            if 'baai' in self.embedders and 'gte' in self.embedders:
                baai_embs = self.embedders['baai'].encode(texts, show_progress_bar=False, convert_to_numpy=True)
                gte_embs = self.embedders['gte'].encode(texts, show_progress_bar=False, convert_to_numpy=True)
                for i, (baai_vec, gte_vec) in enumerate(zip(baai_embs, gte_embs)):
                    baai_norm = baai_vec / np.linalg.norm(baai_vec)
                    gte_norm = gte_vec / np.linalg.norm(gte_vec)
                    combined_vec = 0.6 * baai_norm + 0.4 * gte_norm
                    final_vec = combined_vec / np.linalg.norm(combined_vec)
                    children[i].embedding = final_vec.tolist()
            else:
                # Use whichever embedder is available
                key = 'baai' if 'baai' in self.embedders else next(iter(self.embedders.keys()))
                single_embs = self.embedders[key].encode(texts, show_progress_bar=False, convert_to_numpy=True)
                for i, vec in enumerate(single_embs):
                    norm = vec / np.linalg.norm(vec)
                    children[i].embedding = norm.tolist()
        return children
