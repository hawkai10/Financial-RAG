"""
Parent–child retrieval smoke test.

Usage:
  python scripts/pc_retrieval_smoke.py --files "New folder/cn 21-22/cn 21-22.json"

Ingests specified extraction JSON(s) into the parent–child pipeline, then runs
three retrieval questions and prints top context snippets plus light heuristics
for answers (no external LLM needed).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import json
from typing import List, Optional
import sys

# Ensure project root is on sys.path for imports when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parent_child.pipeline import ParentChildPipeline
from parent_child.retriever import ParentContextRetriever


def ingest_files(files: List[Path]) -> tuple[int, int]:
    pc = ParentChildPipeline()
    total_parents = 0
    total_children = 0
    for jf in files:
        try:
            res = pc.ingest_extracted_json(str(jf), document_id=jf.stem)
            total_parents += res.get("parents", 0)
            total_children += res.get("children", 0)
        except Exception as e:
            print(f"⚠️ Ingest skipped for {jf.name}: {e}")
    return total_parents, total_children


def best_parent_text(retriever: ParentContextRetriever, question: str) -> str:
    out = retriever.query(question, top_k=8, dedup_parents=4)
    pcs = out.get("parent_contexts", [])
    return pcs[0].get("content", "") if pcs else ""


def _find_number(line: str) -> Optional[str]:
    m = re.search(r"([₹$]?)\s?([0-9][0-9,]*)(?:\.[0-9]{1,2})?", line)
    if m:
        cur = m.group(0)
        return cur.strip()
    return None


def extract_party_name(text: str) -> Optional[str]:
    # Look for common labels
    labels = [
        r"invoice\s*to\b",
        r"billed\s*to\b",
        r"bill\s*to\b",
        r"party\s*name\b",
        r"customer\b",
        r"client\b",
        r"consignee\b",
    ]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        low = l.lower()
        for lbl in labels:
            if re.search(lbl, low):
                # Try after ':' or entire line minus label
                after = re.split(r":|-", l, maxsplit=1)
                if len(after) > 1:
                    cand = after[1].strip()
                else:
                    cand = re.sub(lbl, "", low, flags=re.IGNORECASE).strip()
                # Clean overly long names
                cand = re.sub(r"[^\w\s\.&,-]", " ", cand).strip()
                if cand:
                    return cand[:120]
    return None


def extract_taxable_value(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if re.search(r"taxable\s*value", l, flags=re.IGNORECASE):
            n = _find_number(l)
            if n:
                return n
    return None


def extract_gst_amount(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Prefer explicit GST Amount label
    for l in lines:
        if re.search(r"gst\s*amount", l, flags=re.IGNORECASE):
            n = _find_number(l)
            if n:
                return n
    # Fallback: look for CGST/SGST/IGST lines and return sum-like heuristic (largest near labels)
    candidates: List[str] = []
    for l in lines:
        if re.search(r"\b(cgst|sgst|igst|gst)\b", l, flags=re.IGNORECASE):
            n = _find_number(l)
            if n:
                candidates.append(n)
    # Return the last/most recent candidate
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Parent–child retrieval smoke test")
    parser.add_argument(
        "--files",
        nargs="*",
        help="Paths to extraction JSON files. If omitted, will scan 'New folder/'",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidate_files: List[Path] = []
    if args.files:
        for f in args.files:
            p = (root / f) if not os.path.isabs(f) else Path(f)
            if p.exists() and p.is_file():
                candidate_files.append(p)
    else:
        default_dir = root / "New folder"
        if default_dir.exists():
            candidate_files = sorted([p for p in default_dir.glob("**/*.json") if p.is_file()])

    if not candidate_files:
        print("❌ No JSON files found to consider.")
        return 1

    # Auto-detect the best JSON for the invoice-style questions
    questions = [
        "What is the party's name to whom the bill has been invoiced?",
        "what is the GST amount in the bill?",
        "what is the taxable value according to the invoice?",
    ]

    def extract_blocks_for_scoring(p: Path) -> List[str]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        texts: List[str] = []
        try:
            if isinstance(data, dict) and "blocks" in data:
                for b in data["blocks"]:
                    t = (b.get("content") or b.get("html") or "").strip()
                    if t:
                        texts.append(t)
            elif isinstance(data, list) and data and isinstance(data[0], dict) and "pages" in data[0]:
                for doc in data:
                    for page in doc.get("pages", []):
                        for b in page.get("blocks", []):
                            t = (b.get("content") or b.get("html") or "").strip()
                            if t:
                                texts.append(t)
            elif isinstance(data, list) and data and isinstance(data[0], dict) and ("page" in data[0] or "content" in data[0] or "html" in data[0]):
                for b in data:
                    t = (b.get("content") or b.get("html") or "").strip()
                    if t:
                        texts.append(t)
        except Exception:
            pass
        return texts

    def score_texts(texts: List[str]) -> int:
        # Keyword-based scoring tailored to the provided questions
        kw_invoice = ["invoice", "billed to", "bill to", "party", "customer", "client", "consignee"]
        kw_gst = ["gst", "cgst", "sgst", "igst", "gst amount"]
        kw_taxable = ["taxable value", "taxable"]
        score = 0
        sample = "\n".join(texts[:300])  # limit for speed
        low = sample.lower()
        for k in kw_invoice:
            if k in low:
                score += 2 if k in ("invoice", "billed to", "bill to") else 1
        for k in kw_gst:
            if k in low:
                score += 2
        for k in kw_taxable:
            if k in low:
                score += 2 if k == "taxable value" else 1
        return score

    scored: list[tuple[int, Path]] = []
    for p in candidate_files:
        texts = extract_blocks_for_scoring(p)
        s = score_texts(texts)
        scored.append((s, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    if best_score == 0:
        print("⚠️ Could not confidently detect an invoice-like JSON. Proceeding with the first candidate.")
        best_path = candidate_files[0]

    print(f"📄 Auto-selected JSON: {best_path.relative_to(root)} (score={best_score})")

    print(f"📥 Ingesting 1 file into parent–child…")
    parents, children = ingest_files([best_path])
    if children == 0:
        print("❌ No child vectors indexed. Check JSON format.")
        return 1
    print(f"✅ Ingested: parents={parents}, children={children}")

    retr = ParentContextRetriever()

    print("\n🔎 Retrieval results:")
    for q in questions:
        ctx = best_parent_text(retr, q)
        snippet = ctx[:400].replace("\n", " ") if ctx else "<no context>"
        print(f"\nQ: {q}")
        print(f"- Top context: {snippet}")
        # Heuristic answers
        ans: Optional[str] = None
        lowq = q.lower()
        if "party" in lowq or "invoiced" in lowq:
            ans = extract_party_name(ctx)
        elif "gst" in lowq:
            ans = extract_gst_amount(ctx)
        elif "taxable value" in lowq:
            ans = extract_taxable_value(ctx)
        print(f"- Heuristic answer: {ans or '<not found in context>'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
