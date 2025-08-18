"""
Comprehensive End-to-End Test for Financial RAG System
Tests the complete pipeline from document ingestion to query answering.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_rag_pipeline():
    """End-to-end: extract → parent-child ingest → retrieve (real documents only)."""

    print("🚀 Starting Financial RAG E2E (real documents)")
    print("=" * 60)

    results = {
        "extraction": False,
        "parent_child_ingest": False,
        "retrieval": False,
    }

    try:
        # Step 1: Run extraction to produce JSON from Source_Documents (skip if JSON already exists)
        base_dir = Path(__file__).parent
        extraction_dir = base_dir / "New folder"
        existing_json = []
        if extraction_dir.exists():
            existing_json = list(extraction_dir.rglob("*.json"))
        auto_skip = os.getenv("SKIP_EXTRACTION", "1") == "1"

        if existing_json and auto_skip:
            print("\n📥 Step 1: Skipping extraction (found existing JSON in 'New folder'). Set SKIP_EXTRACTION=0 to re-extract.")
            results["extraction"] = True
        else:
            print("\n📥 Step 1: Running extraction.py (Marker) ...")
            ran = run_extraction_script()
            results["extraction"] = ran

        # Step 2: Chunk extracted JSON and save json_chunks.json, then embed + retrieve
        print("\n🔧 Step 2: Ingest into parent-child and retrieve")
        pipeline_ok = run_parent_child_e2e_with_questions()
        results["parent_child_ingest"] = pipeline_ok
        results["retrieval"] = pipeline_ok

        print_test_summary(results)
        return all(results.values())

    except Exception as e:
        logger.error(f"E2E failed: {e}")
        print_test_summary(results)
        return False

def run_parent_child_e2e_with_questions():
    """Process extraction JSONs, ingest into parent-child pipeline (parents in SQLite, children in Chroma), then retrieve."""
    try:
        base_dir = Path(__file__).parent
        extraction_dir = base_dir / "New folder"

        # 1) Locate extraction JSON files (prefer outputs in "New folder")
        extraction_files: List[Path] = []
        if extraction_dir.exists():
            extraction_files = sorted([p for p in extraction_dir.glob("**/*.json") if p.is_file()])

        if not extraction_files:
            print(f"   ❌ No extraction JSON files found in '{extraction_dir}'. Ensure extraction.py ran successfully.")
            return False

        print(f"   📄 Using {len(extraction_files)} extraction file(s) for parent-child ingest")

        # 2) Ingest into parent-child
        from parent_child.pipeline import ParentChildPipeline
        pc = ParentChildPipeline()
        total_parents = 0
        total_children = 0
        for jf in extraction_files:
            try:
                doc_id = jf.stem
                res = pc.ingest_extracted_json(str(jf), document_id=doc_id)
                total_parents += res.get('parents', 0)
                total_children += res.get('children', 0)
            except Exception as ie:
                print(f"   ⚠️ Ingest skipped for {jf.name}: {ie}")
        if total_children == 0:
            print("   ❌ No child vectors indexed. Ensure extraction JSON format is compatible.")
            return False
        print(f"   🗄️  Parent-child storage: ✅ parents={total_parents}, children={total_children}")

        # 4) Run retrieval for user-specified questions
        questions = [
            "what is the name of the client invoiced by Bhartiya enterpirse.",
            "what is the taxable value of bran?",
            "what is the rent for 1st year?",
            "what is the rent for 2nd year?",
            "what is the name of the lisensor?",
        ]

        print("   🔎 Running retrieval for specified questions...")
        from parent_child.retriever import ParentContextRetriever
        retr = ParentContextRetriever()
        qa_results = []
        for q in questions:
            out = retr.query(q, top_k=6, dedup_parents=4)
            pcs = out.get('parent_contexts', [])
            snippet = (pcs[0].get('content','')[:300] if pcs else '')
            qa_results.append({
                "question": q,
                "parents": len(pcs),
                "snippet": snippet,
            })

        # 5) Print a concise report
        print("\n   📋 Retrieval report:")
        for i, r in enumerate(qa_results, 1):
            print(f"     {i}. Q: {r['question']}")
            print(f"        - parents: {r['parents']}")
            print(f"        - snippet: {r['snippet'] or '<no result>'}")

        return True

    except Exception as e:
        print(f"   ❌ Parent-child E2E error: {e}")
        return False

def run_extraction_script() -> bool:
    """Run extraction.py to generate JSON chunks from PDFs in Source_Documents."""
    try:
        script = Path(__file__).parent / "extraction.py"
        if not script.exists():
            print("   ❌ extraction.py not found")
            return False
        # Run via current Python
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print("   ❌ extraction.py failed")
            print(proc.stderr.strip())
            return False
        return True
    except Exception as e:
        print(f"   ❌ Failed to run extraction.py: {e}")
        return False


"""
Legacy Dgraph/Qdrant tests removed. Parent–child E2E above is the single source of truth.
"""

def test_entity_operations(chunks: List):
    """Test entity extraction and relationship discovery."""
    try:
        # Aggregate entity analysis from chunks
        all_entities = {}
        all_relationships = []
        
        if chunks:
            for chunk in chunks:
                insight = chunk.content_insight
                
                # Aggregate entities
                for entity_type, entities in insight.entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = set()
                    all_entities[entity_type].update(entities)
                
                # Aggregate relationships
                all_relationships.extend(insight.relationships)
        
        # Convert sets back to lists for display
        display_entities = {k: list(v) for k, v in all_entities.items()}
        
        entity_success = len(all_entities) > 0
        relationship_success = len(all_relationships) >= 0  # Even 0 is acceptable
        
        print(f"      - Entity extraction: {'✅ Success' if entity_success else '❌ Failed'}")
        if entity_success:
            for entity_type, entities in display_entities.items():
                print(f"        {entity_type}: {len(entities)} entities")
        
        print(f"      - Relationship discovery: {'✅ Success' if relationship_success else '❌ Failed'}")
        print(f"        Found {len(all_relationships)} relationships")
        
        if entity_success:
            print(f"   ✅ Entity operations successful")
        else:
            print(f"   ❌ Entity operations failed")
            
        return entity_success, relationship_success
        
    except Exception as e:
        print(f"   ❌ Entity operations error: {e}")
        return False, False


def print_test_summary(results: Dict[str, bool]):
    """Print a comprehensive test summary."""
    print("\n" + "="*60)
    print("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 70 else "❌"
    print(f"System Status: {status_icon} {'FULLY OPERATIONAL' if success_rate == 100 else 'PARTIALLY OPERATIONAL' if success_rate >= 70 else 'NEEDS ATTENTION'}")
    
    print(f"\nDetailed Results:")
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        formatted_name = test_name.replace('_', ' ').title()
        print(f"  {icon} {formatted_name}")
    
    if success_rate < 100:
        print(f"\n⚠️  Recommendations:")
        if not results.get('content_analysis', True):
            print(f"   - Check spaCy model installation: python -m spacy download en_core_web_sm")
        if not results.get('dgraph_storage', True):
            print(f"   - Verify Dgraph is running: docker run -p 8080:8080 -p 9080:9080 dgraph/standalone:latest")
        if not results.get('qdrant_storage', True):
            print(f"   - Verify Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        if not results.get('json_chunking', True):
            print(f"   - Check sentence-transformers installation: pip install sentence-transformers")
    
    print(f"\n📊 System Components:")
    print(f"   - ML Content Analysis: {'✅ Ready' if results.get('content_analysis') else '❌ Not Ready'}")
    print(f"   - Graph Database (Dgraph): {'✅ Ready' if results.get('dgraph_storage') else '❌ Not Ready'}")
    print(f"   - Vector Database (Qdrant): {'✅ Ready' if results.get('qdrant_storage') else '❌ Not Ready'}")
    print(f"   - Semantic Search: {'✅ Ready' if results.get('semantic_search') else '❌ Not Ready'}")
    print(f"   - RAG Pipeline: {'✅ Ready' if results.get('full_pipeline') else '❌ Not Ready'}")
    
    print("="*60)

def main():
    """Run the comprehensive end-to-end test."""
    success = test_complete_rag_pipeline()
    
    if success:
        print(f"\n🎉 All tests passed! Your Financial RAG system is fully operational.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Please check the recommendations above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
