"""
End-to-end runner: extraction -> ingestion (chunking + embeddings) -> optional retrieval.

Usage examples (PowerShell):
  # Full pipeline using defaults (Source_Documents -> New folder)
  python scripts/run_end_to_end.py

  # Specify custom paths and skip extraction (if JSON already exists)
  python scripts/run_end_to_end.py --source "C:/path/to/pdfs" --output "C:/path/to/out" --skip-extract

  # Run a quick retrieval smoke test at the end
  python scripts/run_end_to_end.py --quick-retrieval "What is the GST amount?"

Notes:
 - Extraction uses Marker via extraction.run_marker() and respects MARKER_* env vars.
 - Ingestion indexes into dual per-model Chroma collections and parents.db.
 - Retrieval smoke uses the ParentContextRetriever (dual dense RRF) by default.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional


# Ensure project root on path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def step(msg: str):
    print(f"\n=== {msg} ===")


def do_extraction(source: Path, output: Path, output_format: str = "chunks") -> None:
    """Run Marker extraction via extraction.run_marker().

    Config is passed through environment variables that extraction.py reads.
    """
    # Set env for extraction
    os.environ["MARKER_INPUT_PATH"] = str(source)
    os.environ["MARKER_OUTPUT_DIR"] = str(output)
    os.environ["MARKER_OUTPUT_FORMAT"] = output_format

    try:
        from extraction import run_marker  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import extraction.run_marker: {e}")

    step(f"Extraction: Source={source} -> Output={output} (format={output_format})")
    run_marker()


def find_jsons(base_dir: Path) -> List[Path]:
    """Recursively find JSON files under a directory."""
    return sorted([p for p in base_dir.glob("**/*.json") if p.is_file()])


def do_ingestion(json_base: Path) -> tuple[int, int]:
    """Ingest all extraction JSONs under json_base using the parent–child pipeline."""
    from parent_child.pipeline import ParentChildPipeline  # type: ignore

    pc = ParentChildPipeline()
    step(f"Ingestion: Scanning JSONs under {json_base}")
    res = pc.ingest_directory(str(json_base))
    parents = int(res.get("parents", 0))
    children = int(res.get("children", 0))
    print(f"Ingested totals -> parents={parents} children={children}")
    return parents, children


def report_vector_counts() -> None:
    """Print per-model Chroma collection sizes for quick verification."""
    try:
        from parent_child.vector_store_factory import get_child_vector_store  # type: ignore
    except Exception:
        print("(skip) Could not import vector store to report counts.")
        return

    def coll(name: str) -> str:
        import re
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"children_{slug}"

    specs = [
        "BAAI/bge-small-en-v1.5",
        "thenlper/gte-small",
    ]

    for spec in specs:
        c = coll(spec)
        try:
            store = get_child_vector_store(collection=c)
            n = store.count()
            print(f"Collection {c}: {n} vectors")
        except Exception as e:
            print(f"Collection {c}: error -> {e}")


def do_retrieval_smoke(question: str, top_k: int = 5) -> None:
    """Run a quick retrieval smoke test and print top contexts."""
    try:
        from parent_child.retriever import ParentContextRetriever  # type: ignore
    except Exception as e:
        print(f"(skip) Could not load retriever: {e}")
        return

    step(f"Retrieval smoke: '{question}'")
    retr = ParentContextRetriever()
    out = retr.query(question, top_k=top_k, dedup_parents=min(3, top_k))
    pcs = out.get("parent_contexts", [])
    if not pcs:
        print("No contexts returned.")
        return
    for i, pc in enumerate(pcs[:top_k], 1):
        doc = str(pc.get("document_id", pc.get("document_name", f"Doc {i}")))
        txt = pc.get("content") or pc.get("text") or ""
        print(f"-- Parent {i} | {doc} | chars={len(txt)}")
        print(txt[:400].replace("\n", " ") + ("…" if len(txt) > 400 else ""))


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline: extract -> ingest -> (optional) retrieve")
    parser.add_argument("--source", type=str, default=str(ROOT / "Source_Documents"), help="Folder or file to extract from")
    parser.add_argument("--output", type=str, default=str(ROOT / "New folder"), help="Output directory for extracted JSON")
    parser.add_argument("--output-format", type=str, default="chunks", help="Marker output format (default: chunks)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion step")
    parser.add_argument("--quick-retrieval", type=str, default=None, help="Run a retrieval smoke test with this question")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    # Step 1: Extraction
    if not args.skip_extract:
        if not source.exists():
            print(f"❌ Source path does not exist: {source}")
            sys.exit(1)
        do_extraction(source, output, args.output_format)
    else:
        step("Skip extraction")

    # Step 2: Ingestion
    if not args.skip_ingest:
        if not output.exists():
            print(f"❌ Output directory not found for ingestion: {output}")
            sys.exit(1)
        parents, children = do_ingestion(output)
        if children == 0:
            print("⚠️ No child vectors indexed; check that your JSON format matches expected shapes.")
    else:
        step("Skip ingestion")

    # Report per-model collection counts
    report_vector_counts()

    # Optional Step 3: Retrieval smoke test
    if args.quick_retrieval:
        do_retrieval_smoke(args.quick_retrieval)


if __name__ == "__main__":
    main()
