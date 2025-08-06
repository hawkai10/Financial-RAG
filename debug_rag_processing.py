"""
Debug script to test unified query processor in RAG context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from unified_query_processor import unified_processor
from rag_backend import rag_query_enhanced
from txtai import Embeddings
from utils import logger

async def debug_rag_processing():
    """Debug the RAG processing to see where the issue is."""
    
    # Load embeddings
    try:
        embeddings = Embeddings()
        embeddings.load("business-docs-index")
        logger.info("Embeddings loaded successfully")
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return
    
    test_query = "List all the parties being issued an invoice by Bhartiya Enterprises?"
    
    print("🔍 Debugging RAG Processing\n")
    print(f"Query: {test_query}")
    print("-" * 60)
    
    # Step 1: Test unified processor directly
    print("Step 1: Testing unified processor directly...")
    try:
        classification = unified_processor.process_query_unified(test_query)
        print(f"  ✅ Direct classification successful:")
        print(f"    Intent: {classification.get('intent')}")
        print(f"    Confidence: {classification.get('confidence')}")
        print(f"    Aggregation Type: {classification.get('aggregation_type')}")
        print(f"    Reasoning: {classification.get('reasoning')}")
    except Exception as e:
        print(f"  ❌ Direct classification failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 60)
    
    # Step 2: Test RAG pipeline
    print("Step 2: Testing RAG pipeline...")
    try:
        result = await rag_query_enhanced(
            question=test_query,
            embeddings=embeddings,
            topn=5
        )
        
        classification_from_rag = result.get('classification', {})
        agent_used = result.get('agent_used', 'Unknown')
        
        print(f"  ✅ RAG processing completed:")
        print(f"    Agent Used: {agent_used}")
        print(f"    Classification Intent: {classification_from_rag.get('intent')}")
        print(f"    Classification Confidence: {classification_from_rag.get('confidence')}")
        print(f"    Classification Reasoning: {classification_from_rag.get('reasoning')}")
        
        # Check if fallback occurred
        if classification_from_rag.get('reasoning') == 'Fallback processing':
            print(f"  ⚠️  ISSUE: Unified processor fell back to basic processing!")
        
    except Exception as e:
        print(f"  ❌ RAG processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_rag_processing())
