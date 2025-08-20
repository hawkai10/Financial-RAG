#!/usr/bin/env python3
"""Simple script to ingest all extraction JSONs into parent-child pipeline."""

import os
import re
from pathlib import Path
from parent_child.pipeline import ParentChildPipeline


def _coll_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"children_{slug}"


def main():
    print("Starting ingestion of all files...")

    # Resolve base directory robustly (project root / "New folder")
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / "New folder"
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return 1

    # Create pipeline
    p = ParentChildPipeline()

    # Ingest all files from 'New folder' directory (recursive)
    result = p.ingest_directory(str(base_dir))

    print("✅ Ingestion complete!")
    print(f"   - Parents: {result['parents']}")
    print(f"   - Children: {result['children']}")

    # Verify by checking per-model collection counts (BAAI + GTE)
    try:
        from parent_child.chroma_child_store import ChromaChildStore
        baai = _coll_name("BAAI/bge-small-en-v1.5")
        gte = _coll_name("thenlper/gte-small")
        c_baai = ChromaChildStore(collection=baai).count()
        c_gte = ChromaChildStore(collection=gte).count()
        print("   - Chroma collections:")
        print(f"     • {baai}: {c_baai}")
        print(f"     • {gte}: {c_gte}")
    except Exception as e:
        print(f"   - Could not verify per-model vector counts: {e}")

    # Show how many chunk log files were written
    try:
        log_dir = project_root / "chunk_logs"
        if log_dir.exists():
            files = list(log_dir.glob("*_parent_child_chunks.json"))
            print(f"   - Chunk logs: {len(files)} file(s) in {log_dir}")
        else:
            print("   - Chunk logs: directory not found (will be created on first successful ingest)")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
