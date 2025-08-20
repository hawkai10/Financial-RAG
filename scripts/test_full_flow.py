import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import sys

# Ensure project root is on sys.path for imports when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
import rag_backend as rb
from document_reranker import EnhancedDocumentReranker  # type: ignore
from parent_child.parent_store import ParentStore  # type: ignore


QUESTIONS = [
    "What is the rent for the 1st year?",
    "what is the lock in period according to the rent agreement?",
    "what is the name of the lessor?",
    "what is amount being invoiced to Jay Shree Solvex ?",
    "what is the GST amount on the invoice to Seetharama Oil Industies?",
]


def _token_len(text: str) -> int:
    """Heuristic token length ~ word count * 1.3 (keeps us dependency-free)."""
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def _expand_snippet_in_parent(snippet: str, parent_text: str, target_tokens: int = 400) -> str:
    """Approximate the full child chunk by expanding around the snippet within the parent text.
    If the snippet isn't found, fallback to the first target_tokens of the parent.
    """
    if not parent_text:
        return snippet or ""
    # Use a robust but simple locate strategy
    needle = (snippet or "").strip()
    if not needle:
        # take the first window from parent
        # approximate chars per token ~ 4
        max_chars = target_tokens * 4
        return parent_text[:max_chars]
    # Try to find the full snippet, else a prefix
    idx = parent_text.find(needle)
    if idx < 0 and len(needle) > 60:
        idx = parent_text.find(needle[:60])
    # Choose a window around the found index
    max_chars = target_tokens * 4  # rough mapping
    if idx >= 0:
        start = max(0, idx - max_chars // 2)
        end = min(len(parent_text), idx + len(needle) + max_chars // 2)
        return parent_text[start:end]
    # Fallback: take the most relevant-looking slice (start of parent)
    return parent_text[:max_chars]


def ensure_logs_dir() -> Path:
    logs = Path("test_logs")
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "query"


def build_prompt(question: str, parent_chunks: List[Dict[str, Any]], related_queries: List[str]) -> str:
    # Mirrors rag_backend.synthesize_answer_simple
    ctx_lines: List[str] = []
    for i, pc in enumerate(parent_chunks[:5], 1):
        name = pc.get("document_name", f"Doc {i}")
        txt = pc.get("chunk_text", pc.get("text", ""))
        ctx_lines.append(f"[Source {i}: {name}]\n{txt}\n")
    context = "\n".join(ctx_lines)
    rq_block = ""
    if related_queries:
        rq_lines = "\n".join([f"- {q}" for q in related_queries[:3]])
        rq_block = f"\n\nRELATED QUERIES:\n{rq_lines}\n"
    prompt = (
        "You are an assistant answering from financial documents. If uncertain, say you don't know.\n\n"
        f"Question: {question}\n"
        f"{rq_block}\n"
        f"Context:\n{context}\n"
        "Answer concisely and cite facts from the context."
    )
    return prompt


async def run_one(question: str, logs_dir: Path):
    # 1) Retrieve children (includes LLM normalization -> multi-queries)
    child_chunks, child_to_parent, related_queries = await rb._retrieve_children_hybrid(question, max_children=24)  # type: ignore[attr-defined]

    # 2) Rerank children
    reranked_children = child_chunks
    try:
        rr = EnhancedDocumentReranker()
        reranked_children, _ = rr.rerank_chunks(question, child_chunks, strategy="Simple")
    except Exception as e:
        rb.logger.warning(f"Reranking failed; using merged scores: {e}")

    def child_score(c: Dict[str, Any]) -> float:
        return float(c.get("final_rerank_score", c.get("retrieval_score", 0.0)))

    top_children = sorted(reranked_children, key=child_score, reverse=True)[:3]

    # 3) Map to parents
    parent_ids: List[int] = []
    seen = set()
    for c in top_children:
        cid = str(c.get("child_id") or str(c.get("chunk_id", ""))[6:])
        pid = child_to_parent.get(cid)
        if pid is None:
            continue
        if pid not in seen:
            seen.add(pid)
            parent_ids.append(pid)
        if len(parent_ids) >= 3:
            break

    parents = ParentStore().get_parents_by_ids(parent_ids)
    parent_chunks: List[Dict[str, Any]] = []
    for p in parents:
        parent_chunks.append({
            "chunk_id": f"parent_{p.parent_id}",
            "chunk_text": p.content,
            "text": p.content,
            "document_name": str(p.document_id),
            "page_start": p.page_start,
            "page_end": p.page_end,
            "retrieval_score": 1.0,
            "retrieval_method": "parent_from_top_children",
        })

    # Build mapping for child->parent content (to reconstruct full child text)
    parent_by_id: Dict[int, Dict[str, Any]] = {}
    for pc in parent_chunks:
        # chunk_id looks like parent_{id}
        try:
            pid = int(str(pc.get("chunk_id", "")).split("_")[1])
            parent_by_id[pid] = pc
        except Exception:
            continue

    # 4) Build prompt and call LLM
    prompt = build_prompt(question, parent_chunks, related_queries)
    answer = await rb.call_gemini_enhanced(prompt)

    # 5) Prepare log payload
    # Enrich top_children with full text and token counts
    enriched_top_children = []
    for c in top_children:
        cid = str(c.get("child_id") or "")
        snippet = c.get("chunk_text") or c.get("text", "")
        pid = None
        try:
            pid = child_to_parent.get(cid)
        except Exception:
            pid = None
        full_text = snippet
        if pid is not None and pid in parent_by_id:
            full_text = _expand_snippet_in_parent(snippet, parent_by_id[pid].get("text", ""), target_tokens=400)
        enriched_top_children.append({
            "child_id": c.get("child_id"),
            "retrieval_score": c.get("retrieval_score"),
            "final_rerank_score": c.get("final_rerank_score"),
            "snippet": snippet,
            "full_text": full_text,
            "tokens": _token_len(full_text),
        })

    # Add token counts to parent chunks as well
    for pc in parent_chunks:
        pc["tokens"] = _token_len(pc.get("text", ""))

    log = {
        "question": question,
        "multiqueries": related_queries,
        "top_children": enriched_top_children,
        "parent_chunks": parent_chunks,
        "llm_prompt": prompt,
        "final_answer": answer,
        "token_summary": {
            "children_total_tokens": sum(tc["tokens"] for tc in enriched_top_children) if enriched_top_children else 0,
            "parents_total_tokens": sum(pc.get("tokens", 0) for pc in parent_chunks) if parent_chunks else 0,
        },
    }

    # 6) Write log file
    fname = logs_dir / f"fullflow_{slugify(question)}.json"
    with fname.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"✔ Wrote log: {fname}")


async def main():
    logs_dir = ensure_logs_dir()
    # Ensure single strategy path is in effect (rag_backend already simplified)
    os.environ["BACKEND_USE_PARENT_CHILD"] = "true"
    for q in QUESTIONS:
        await run_one(q, logs_dir)


if __name__ == "__main__":
    asyncio.run(main())
