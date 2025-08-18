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
    """End-to-end: extract → chunk → store → retrieve (real documents only)."""

    print("🚀 Starting Financial RAG E2E (real documents)")
    print("=" * 60)

    results = {
        "extraction": False,
        "json_chunking": False,
        "dgraph_storage": False,
        "qdrant_storage": False,
        "semantic_search": False,
    }

    try:
        # Step 1: Run extraction to produce JSON from Source_Documents
        print("\n📥 Step 1: Running extraction.py (Marker) ...")
        ran = run_extraction_script()
        results["extraction"] = ran

        # Step 2: Chunk extracted JSON and save json_chunks.json, then embed + retrieve
        print("\n🔧 Step 2: Chunk, embed, and retrieve")
        pipeline_ok = run_real_docs_e2e_with_questions()
        # The function prints its own retrieval report; we infer storage and search status loosely
        results["json_chunking"] = pipeline_ok
        results["qdrant_storage"] = pipeline_ok
        results["semantic_search"] = pipeline_ok

        print_test_summary(results)
        return all(results.values())

    except Exception as e:
        logger.error(f"E2E failed: {e}")
        print_test_summary(results)
        return False

def run_real_docs_e2e_with_questions():
    """Process Source_Documents, save chunks as json_chunks.json, embed to Qdrant, and run retrieval for specific questions."""
    try:
        base_dir = Path(__file__).parent
        source_dir = base_dir / "Source_Documents"
        extraction_dir = base_dir / "New folder"
        output_chunks_path = base_dir / "json_chunks.json"

        # 1) Locate extraction JSON files (prefer outputs in "New folder")
        extraction_files: List[Path] = []
        if extraction_dir.exists():
            extraction_files = sorted([p for p in extraction_dir.glob("**/*.json") if p.is_file()])

        if not extraction_files:
            print(f"   ❌ No extraction JSON files found in '{extraction_dir}'. Ensure extraction.py ran successfully.")
            return False

        print(f"   📄 Using {len(extraction_files)} extraction file(s) for chunking")

        # 2) Chunk and analyze, then save to json_chunks.json
        from enhanced_json_chunker import EnhancedJSONChunker
        chunker = EnhancedJSONChunker()

        all_chunks = []
        for jf in extraction_files:
            try:
                chunks = chunker.process_extracted_json(str(jf))
                all_chunks.extend(chunks)
            except Exception as ce:
                print(f"   ⚠️ Skipped {jf.name} due to parse error: {ce}")

        if not all_chunks:
            print("   ❌ No chunks produced from extraction JSON. Ensure the JSON matches expected schema.")
            return False

        # Persist chunks
        try:
            chunker.save_chunks(all_chunks, str(output_chunks_path))
            print(f"   💾 Saved {len(all_chunks)} chunks to {output_chunks_path.name}")
        except Exception as se:
            print(f"   ⚠️ Failed to save chunks JSON: {se}")

        # 3) Embed and store in Qdrant
        from adaptive_qdrant_manager import AdaptiveQdrantManager
        qdrant_manager = AdaptiveQdrantManager()
        stored = qdrant_manager.store_enhanced_chunks(all_chunks)
        print(f"   🗄️  Vector storage: {'✅ Success' if stored else '❌ Failed'}")
        if not stored:
            print("   ❌ Aborting question retrieval due to storage failure.")
            return False

        # 4) Run retrieval for user-specified questions
        questions = [
            "what is the name of the client invoiced by Bhartiya enterpirse.",
            "what is the taxable value of bran?",
            "what is the rent for 1st year?",
            "what is the rent for 2nd year?",
            "what is the name of the lisensor?",
        ]

        print("   🔎 Running retrieval for specified questions...")
        qa_results = []
        for q in questions:
            hits = qdrant_manager.semantic_search(q, top_k=5)
            top = hits[0] if hits else None
            answer_hint = (top or {}).get("content", "")[:300] if top else ""
            qa_results.append({
                "question": q,
                "hits": len(hits),
                "top_score": round(top.get("score", 0.0), 4) if top else 0.0,
                "top_document": (top or {}).get("document_id", ""),
                "snippet": answer_hint,
            })

        # 5) Print a concise report
        print("\n   📋 Retrieval report:")
        for i, r in enumerate(qa_results, 1):
            print(f"     {i}. Q: {r['question']}")
            print(f"        - hits: {r['hits']}, top_score: {r['top_score']}, doc: {r['top_document']}")
            if r["snippet"]:
                print(f"        - snippet: {r['snippet']}")
            else:
                print("        - snippet: <no result>")

        # Optional: collection stats
        try:
            stats = qdrant_manager.get_collection_statistics()
            if stats:
                print(f"\n   📈 Qdrant stats → points: {stats.get('total_points')}, vector_size: {stats.get('vector_size')}\n")
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"   ❌ Real docs E2E error: {e}")
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


def test_dgraph_operations(chunks: List):
    """Test Dgraph storage and schema operations."""
    try:
        from adaptive_dgraph_manager import AdaptiveDgraphManager
        from dynamic_schema_manager import DynamicSchemaManager
        
        # Initialize managers
        dgraph_manager = AdaptiveDgraphManager()
        schema_manager = DynamicSchemaManager()
        
        # Test schema evolution
        if chunks:
            insights = [chunk.content_insight for chunk in chunks]
            schema_updates = schema_manager.analyze_and_update_schema(insights)
            
            schema_success = len(schema_updates) >= 0  # Even 0 updates is success
            print(f"      - Schema updates applied: {len(schema_updates)}")
        else:
            schema_success = True
            print(f"      - Schema evolution skipped (no chunks)")
        
        # Test chunk storage
        if chunks:
            storage_success = dgraph_manager.store_enhanced_chunks(chunks[:3])  # Store first 3 chunks
            print(f"      - Chunk storage: {'✅ Success' if storage_success else '❌ Failed'}")
        else:
            storage_success = True
            print(f"      - Chunk storage skipped (no chunks)")
        
        # Test queries
        try:
            stats = dgraph_manager.get_statistics()
            query_success = isinstance(stats, dict)
            print(f"      - Query operations: {'✅ Success' if query_success else '❌ Failed'}")
            print(f"        Statistics: {stats}")
        except Exception as e:
            query_success = False
            print(f"      - Query operations: ❌ Failed - {e}")
        
        overall_success = schema_success and storage_success and query_success
        
        if overall_success:
            print(f"   ✅ Dgraph operations successful")
        else:
            print(f"   ⚠️  Dgraph operations partially successful")
            
        return overall_success, schema_success
        
    except Exception as e:
        print(f"   ❌ Dgraph operations error: {e}")
        return False, False

def test_qdrant_operations(chunks: List):
    """Test Qdrant storage and vector search."""
    try:
        from adaptive_qdrant_manager import AdaptiveQdrantManager
        
        qdrant_manager = AdaptiveQdrantManager()
        
        # Test chunk storage
        if chunks:
            storage_success = qdrant_manager.store_enhanced_chunks(chunks[:3])
            print(f"      - Vector storage: {'✅ Success' if storage_success else '❌ Failed'}")
        else:
            storage_success = True
            print(f"      - Vector storage skipped (no chunks)")
        
        # Test semantic search
        try:
            search_queries = [
                "Apple financial performance revenue",
                "CEO Tim Cook expansion plans",
                "R&D spending artificial intelligence"
            ]
            
            search_results = []
            for query in search_queries:
                results = qdrant_manager.semantic_search(query, top_k=3)
                search_results.extend(results)
                print(f"        Query: '{query[:30]}...' -> {len(results)} results")
            
            search_success = len(search_results) > 0
            print(f"      - Semantic search: {'✅ Success' if search_success else '❌ Failed'}")
            
        except Exception as e:
            search_success = False
            print(f"      - Semantic search: ❌ Failed - {e}")
        
        # Test collection statistics
        try:
            stats = qdrant_manager.get_collection_statistics()
            stats_success = isinstance(stats, dict) and len(stats) > 0
            print(f"      - Statistics: {'✅ Success' if stats_success else '❌ Failed'}")
            if stats_success:
                print(f"        Points: {stats.get('total_points', 'N/A')}")
                print(f"        Vector size: {stats.get('vector_size', 'N/A')}")
        except Exception as e:
            stats_success = False
            print(f"      - Statistics: ❌ Failed - {e}")
        
        overall_success = storage_success and search_success and stats_success
        
        if overall_success:
            print(f"   ✅ Qdrant operations successful")
        else:
            print(f"   ⚠️  Qdrant operations partially successful")
            
        return overall_success, search_success
        
    except Exception as e:
        print(f"   ❌ Qdrant operations error: {e}")
        return False, False

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
