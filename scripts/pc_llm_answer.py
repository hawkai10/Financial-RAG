"""
Parent–child LLM answering script.

Auto-detects the most relevant extraction JSON under 'New folder/', ingests
parents/children, retrieves contexts for three invoice questions, and calls the
configured LLM to produce concise answers.

Requires the existing rag_backend.call_gemini_enhanced to be configured via env
(e.g., model key). If LLM fails, prints context-only fallback.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Path setup for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parent_child.pipeline import ParentChildPipeline
from parent_child.retriever import ParentContextRetriever
from parent_child.api_adapter import build_answer_prompt

try:
    from rag_backend import call_gemini_enhanced  # type: ignore
except Exception:
    async def call_gemini_enhanced(prompt: str, **kwargs) -> str:  # type: ignore
        return ""


def run_extraction() -> bool:
    """Run extraction.py to generate/refresh JSONs under 'New folder/'."""
    script = ROOT / "extraction.py"
    if not script.exists():
        print("❌ extraction.py not found at repo root")
        return False
    print("\n📥 Running extraction.py (Marker) …")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.returncode != 0:
        print("❌ extraction.py failed")
        if proc.stderr:
            print(proc.stderr.strip())
        return False
    return True


def auto_select_json() -> Path | None:
    base = ROOT / "New folder"
    if not base.exists():
        return None
    cands = sorted([p for p in base.glob("**/*.json") if p.is_file()])
    if not cands:
        return None

    def score(p: Path) -> int:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return 0
        texts: List[str] = []
        try:
            if isinstance(data, dict) and "blocks" in data:
                for b in data["blocks"]:
                    t = (b.get("content") or b.get("html") or "").lower()
                    if t:
                        texts.append(t)
            elif isinstance(data, list) and data and isinstance(data[0], dict) and "pages" in data[0]:
                for doc in data:
                    for page in doc.get("pages", []):
                        for b in page.get("blocks", []):
                            t = (b.get("content") or b.get("html") or "").lower()
                            if t:
                                texts.append(t)
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                for b in data:
                    t = (b.get("content") or b.get("html") or "").lower()
                    if t:
                        texts.append(t)
        except Exception:
            pass
        joined = "\n".join(texts[:300])
        sc = 0
        for k in ["invoice", "gst", "taxable", "billed to", "bill to", "party"]:
            if k in joined:
                sc += 2
        return sc

    ranked = sorted([(score(p), p) for p in cands], key=lambda x: x[0], reverse=True)
    return ranked[0][1] if ranked else None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Parent–child LLM answering")
    parser.add_argument("--all", action="store_true", help="Ingest all JSONs under 'New folder' before answering")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip running extraction.py before ingesting")
    parser.add_argument(
        "--questions",
        nargs="*",
        help="Questions to answer. If omitted, uses three invoice defaults.",
    )
    args = parser.parse_args()

    # Step 0: Extraction
    if not args.skip_extraction:
        ok = run_extraction()
        if not ok:
            return 1

    pc = ParentChildPipeline()

    if args.all:
        base = ROOT / "New folder"
        if not base.exists():
            print("❌ No 'New folder' directory found.")
            return 1
        res = pc.ingest_directory(str(base))
        print(f"📥 Ingested ALL: parents={res['parents']} children={res['children']}")
    else:
        target = auto_select_json()
        if not target:
            print("❌ No JSON found under 'New folder/'.")
            return 1
        print(f"📄 Using {target.relative_to(ROOT)}")
        res = pc.ingest_extracted_json(str(target), document_id=target.stem)
        print(f"✅ Ingested parents={res['parents']} children={res['children']}")

    retr = ParentContextRetriever()

    # Prepare LLM payload log
    def _token_len(text: str) -> int:
        return int(len(re.findall(r"\w+", text)) * 1.3)
    log_dir = os.getenv("QA_LOG_DIR") or str(ROOT / "chunk_logs")
    os.makedirs(log_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"qa_llm_payload_{run_stamp}.json")
    qa_log = {
        "run_started": run_stamp,
        "ingest_all": bool(args.all),
        "questions": [],
    }
    questions = args.questions or [
        "What is the rent for the 1st year?",
        "what is the lock in period according to the rent agreement?",
        "what is the name of the lessor?",
        "what is amount being invoiced to Jay Shree Solvex ?",
        "what is the GST amount on the invoice to Seetharama Oil Industies?",
    ]

    for q in questions:
        out = retr.query(q, top_k=8, dedup_parents=4)
        parent_contexts: List[Dict[str, Any]] = out.get("parent_contexts", [])
        # The prompt uses up to 6 parents; mirror that selection for logging
        selected_parents = parent_contexts[:6]
        prompt = build_answer_prompt(q, selected_parents)
        try:
            ans = await call_gemini_enhanced(prompt)
        except Exception:
            ans = (parent_contexts[0].get("content", "")[:600] if parent_contexts else "")
        print(f"\nQ: {q}\nA: {ans.strip()}")

        # Build per-question payload log
        sel_parent_ids = {int(p.get("parent_id")) for p in selected_parents}
        # Child hits filtered to selected parents
        child_hits = out.get("child_hits", [])
        logged_child_hits = []
        for h in child_hits:
            try:
                pid = int(h.get("payload", {}).get("parent_id"))
            except Exception:
                continue
            if pid in sel_parent_ids:
                logged_child_hits.append({
                    "child_id": h.get("child_id"),
                    "parent_id": pid,
                    "score": h.get("score"),
                    "snippet": h.get("payload", {}).get("snippet", ""),
                })
        qa_log["questions"].append({
            "question": q,
            "parents": [
                {
                    "parent_id": p.get("parent_id"),
                    "document_id": p.get("document_id"),
                    "page_start": p.get("page_start"),
                    "page_end": p.get("page_end"),
                    "tokens": _token_len(p.get("content", "")),
                    "content": p.get("content", ""),
                }
                for p in selected_parents
            ],
            "child_hits": logged_child_hits,
        })

    # Write the LLM payload log
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(qa_log, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved LLM payload log: {os.path.relpath(log_path, ROOT)}")
    except Exception as e:
        print(f"⚠️ Failed to write LLM payload log: {e}")

    return 0


if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main()))
