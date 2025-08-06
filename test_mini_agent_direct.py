#!/usr/bin/env python3
"""
Direct test of Mini-Agent functionality
"""

import asyncio
import sys
import traceback
from txtai import Embeddings

def test_mini_agent_direct():
    """Test Mini-Agent directly without going through the full RAG pipeline."""
    
    print("🧪 Testing Mini-Agent Direct Functionality")
    print("=" * 60)
    
    try:
        # Import dependencies
        print("📦 Importing dependencies...")
        from mini_agent import mini_agent, initialize_mini_agent, PatternExtractor
        from chunk_manager import ChunkManager
        from progressive_retrieval import ProgressiveRetriever
        from config import config
        print("✅ All imports successful")
        
        # Initialize embeddings
        print("\n🔍 Loading embeddings...")
        embeddings = Embeddings()
        embeddings.load("business-docs-index")
        print("✅ Embeddings loaded")
        
        # Initialize Mini-Agent components
        print("\n🤖 Initializing Mini-Agent...")
        chunk_manager = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
        progressive_retriever = ProgressiveRetriever(embeddings)
        mini_agent_instance = initialize_mini_agent(chunk_manager, progressive_retriever)
        print("✅ Mini-Agent initialized")
        
        # Test queries - only the invoice recipients query
        test_queries = [
            ("List all the parties being issued an invoice by Bhartiya Enterprises?", "invoice_recipients")
        ]
        
        print("\n🔍 Testing Mini-Agent queries...")
        
        async def run_test():
            for query, expected_type in test_queries:
                print(f"\n📝 Query: {query}")
                print(f"🎯 Expected Type: {expected_type}")
                
                try:
                    result = await mini_agent_instance.process_aggregation_query(query, expected_type)
                    
                    print(f"✅ Success: {result.get('success', False)}")
                    print(f"📊 Answer Length: {len(result.get('answer', ''))}")
                    print(f"🎯 Agent Used: {result.get('strategy_used', 'Unknown')}")
                    
                    if result.get('should_fallback', False):
                        print("⚠️  Mini-Agent recommends fallback")
                    
                    # Show the complete answer
                    answer = result.get('answer', '')
                    if answer:
                        print(f"\n📄 Complete Answer:\n{answer}")
                    else:
                        print("📄 No answer provided")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    traceback.print_exc()
                    print(f"❌ Error: {str(e)}")
                    traceback.print_exc()
                
                print("-" * 40)
        
        # Run async tests
        asyncio.run(run_test())
        
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
        traceback.print_exc()
        return False
    
    print("\n✅ Mini-Agent direct test completed!")
    return True

if __name__ == "__main__":
    success = test_mini_agent_direct()
    sys.exit(0 if success else 1)
