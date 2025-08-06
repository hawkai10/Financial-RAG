#!/usr/bin/env python3
"""
Debug script to see what patterns the Mini-Agent is actually finding
"""

import asyncio
import sys
from txtai import Embeddings
from mini_agent import MiniAgent
from chunk_manager import ChunkManager

async def debug_mini_agent():
    print("🔍 Debug Mini-Agent Pattern Matching")
    print("=" * 60)
    
    # Load embeddings
    print("📦 Loading embeddings...")
    embeddings = Embeddings()
    embeddings.load("business-docs-index")
    print("✅ Embeddings loaded")
    
    # Initialize Mini-Agent
    print("🤖 Initializing Mini-Agent...")
    chunk_manager = ChunkManager("contextualized_chunks.json")
    mini_agent = MiniAgent(embeddings, chunk_manager)
    
    # Test query
    query = "List all the parties being issued an invoice by Bhartiya Enterprises?"
    print(f"📝 Query: {query}")
    
    # Get chunks directly
    from progressive_retrieval import ProgressiveRetriever
    retriever = ProgressiveRetriever(embeddings)
    chunks, info = await retriever.retrieve_progressively(
        queries=[query], 
        strategy="Aggregation", 
        confidence=0.3
    )
    
    print(f"\n📊 Retrieved {len(chunks)} chunks")
    
    # Show chunk contents and test patterns
    from mini_agent import InvoiceRecipientsExtractor
    extractor = InvoiceRecipientsExtractor()
    
    print("\n🔍 Checking each chunk for patterns:")
    for i, chunk in enumerate(chunks[:5]):  # Check first 5 chunks
        print(f"\n--- Chunk {i+1} ---")
        content = chunk.get('chunk_text', chunk.get('content', ''))
        print(f"📄 Content Preview: {content[:200]}...")
        
        # Test pattern matching
        results = extractor.extract([chunk])
        print(f"🎯 Pattern matches: {len(results)} found")
        if results:
            for j, result in enumerate(results[:3]):  # Show first 3 matches
                print(f"  - Match {j+1}: {result}")
        
        # Also check for common business terms
        business_terms = ['invoice', 'recipient', 'company', 'enterprise', 'client', 'customer']
        found_terms = [term for term in business_terms if term.lower() in content.lower()]
        print(f"💼 Business terms found: {found_terms}")

if __name__ == "__main__":
    asyncio.run(debug_mini_agent())
