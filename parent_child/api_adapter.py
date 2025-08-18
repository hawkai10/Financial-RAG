from __future__ import annotations

from typing import Dict, List, Any

from .retriever import ParentContextRetriever

try:
    # Reuse existing LLM caller for answer generation if configured
    from rag_backend import call_gemini_enhanced  # type: ignore
except Exception:  # pragma: no cover
    async def call_gemini_enhanced(prompt: str, **kwargs) -> str:  # type: ignore
        return ""


def build_answer_prompt(question: str, parent_contexts: List[Dict[str, Any]]) -> str:
    ctxs = []
    for i, pc in enumerate(parent_contexts[:6], start=1):
        ctxs.append(
            f"[Context {i}] Document: {pc.get('document_id')} | ParentID: {pc.get('parent_id')} | Pages: {pc.get('page_start','?')}-{pc.get('page_end','?')}\n{pc.get('content','')[:3000]}"
        )
    ctx_block = "\n\n".join(ctxs) if ctxs else "No context available."
    prompt = (
        "You are an assistant answering from financial documents. Use the contexts to answer concisely. "
        "If uncertain, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Contexts:\n{ctx_block}\n\n"
        "Answer:"
    )
    return prompt


async def pc_search(question: str, top_k: int = 6, dedup_parents: int = 4) -> Dict[str, Any]:
    retriever = ParentContextRetriever()
    result = retriever.query(question, top_k=top_k, dedup_parents=dedup_parents)
    parent_contexts = result.get("parent_contexts", [])

    # Build a simple documents list for UI (title + snippet)
    documents: List[Dict[str, Any]] = []
    for pc in parent_contexts:
        documents.append(
            {
                "title": f"Doc {pc.get('document_id')} • Parent {pc.get('parent_id')}",
                "content": pc.get("content", ""),
                "source": str(pc.get("parent_id")),
            }
        )

    # Optional LLM answer
    answer = ""
    try:
        prompt = build_answer_prompt(question, parent_contexts)
        answer = await call_gemini_enhanced(prompt)
    except Exception:
        # Graceful degrade: return concatenated summary
        joined = "\n\n".join([pc.get("content", "")[:600] for pc in parent_contexts])
        answer = joined[:1500]

    return {
        "answer": answer,
        "documents": documents,
        "chunks": [],  # parent-child path returns parent contexts directly
        "child_hits": result.get("child_hits", []),
    }
