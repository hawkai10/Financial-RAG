#!/usr/bin/env python3
"""
Test script to verify Mini-Agent functionality
"""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import logger
from config import config
from chunk_manager import ChunkManager
from txtai import Embeddings

async def test_mini_agent():
    """Test Mini-Agent directly"""
    
    print("🔧 Testing Mini-Agent functionality...")
    
    try:
        # Initialize required components
        print("1. Initializing ChunkManager...")
        chunk_manager = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
        print(f"   ✅ ChunkManager initialized with {len(chunk_manager._index)} chunks")
        
        print("2. Loading embeddings...")
        embeddings = Embeddings()
        embeddings.load(config.EMBEDDINGS_PATH)
        print("   ✅ Embeddings loaded successfully")
        
        # Import and initialize Mini-Agent
        print("3. Importing Mini-Agent...")
        from mini_agent import initialize_mini_agent, mini_agent
        from progressive_retrieval import ProgressiveRetriever
        
        print("4. Initializing Mini-Agent...")
        progressive_retriever = ProgressiveRetriever(embeddings)
        initialize_mini_agent(chunk_manager, progressive_retriever)
        print("   ✅ Mini-Agent initialized successfully")
        
        # Test aggregation query
        test_query = "List all the parties being issued an invoice by Bhartiya Enterprises?"
        aggregation_type = "invoice_recipients"
        
        print(f"5. Testing aggregation query...")
        print(f"   Query: {test_query}")
        print(f"   Type: {aggregation_type}")
        
        result = await mini_agent.process_aggregation_query(test_query, aggregation_type)
        
        print("6. Results:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Should Fallback: {result.get('should_fallback', False)}")
        print(f"   Answer Length: {len(result.get('answer', ''))}")
        
        if result.get('answer'):
            print(f"   Answer Preview: {result['answer'][:200]}...")
        
        if result.get('extracted_data'):
            print(f"   Extracted Data: {len(result['extracted_data'])} items")
            for i, item in enumerate(result['extracted_data'][:3]):
                print(f"     {i+1}. {item}")
        
        print("\n✅ Mini-Agent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Mini-Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_mini_agent())
