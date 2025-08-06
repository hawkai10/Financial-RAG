#!/usr/bin/env python3
"""
Debug script to test Full Agent functionality directly
"""

import asyncio
import sys
import traceback
from txtai import Embeddings

async def debug_full_agent():
    """Test Full Agent directly to identify issues."""
    
    print("🔍 Debug Full Agent Functionality")
    print("=" * 60)
    
    try:
        # Import dependencies
        print("📦 Importing dependencies...")
        from full_agent import FullAgent, initialize_full_agent
        from chunk_manager import ChunkManager
        from progressive_retrieval import ProgressiveRetriever
        from rag_backend import call_gemini_enhanced
        from config import config
        print("✅ All imports successful")
        
        # Initialize embeddings
        print("\n🔍 Loading embeddings...")
        embeddings = Embeddings()
        embeddings.load("business-docs-index")
        print("✅ Embeddings loaded")
        
        # Initialize Full Agent components
        print("\n🤖 Initializing Full Agent...")
        chunk_manager = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
        progressive_retriever = ProgressiveRetriever(embeddings)
        full_agent_instance = initialize_full_agent(chunk_manager, progressive_retriever, call_gemini_enhanced)
        print("✅ Full Agent initialized")
        
        # Test queries that should be routed to Full Agent
        test_queries = [
            "What is the total revenue from Bhartiya Enterprises invoices and how does it compare to their operational costs?",
            "Analyze the payment patterns of Bhartiya Enterprises' top 3 clients and identify any potential cash flow issues",
            "Compare the financial performance across different quarters and explain the key factors driving revenue changes",
            "What are the main business relationships shown in the documents and how do they impact financial performance?"
        ]
        
        print("\n🔍 Testing Full Agent queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"📝 Test {i}: {query}")
            print("="*60)
            
            try:
                # Test Full Agent processing
                result = await full_agent_instance.process_complex_query(query)
                
                print(f"✅ Success: {result.get('success', False)}")
                print(f"📊 Answer Length: {len(result.get('answer', ''))}")
                print(f"🎯 Agent Used: {result.get('strategy_used', 'Unknown')}")
                print(f"🔧 Steps Executed: {result.get('steps_executed', 0)}")
                
                if result.get('should_fallback', False):
                    print("⚠️  Full Agent recommends fallback")
                
                # Show execution plan if available
                plan = result.get('execution_plan', [])
                if plan:
                    print(f"\n📋 Execution Plan ({len(plan)} steps):")
                    for j, step in enumerate(plan, 1):
                        print(f"  {j}. {step}")
                
                # Show the complete answer
                answer = result.get('answer', '')
                if answer:
                    print(f"\n📄 Complete Answer:\n{answer}")
                else:
                    print("📄 No answer provided")
                
                # Show any errors or debug info
                if 'error' in result:
                    print(f"❌ Error in result: {result['error']}")
                    
                if 'debug_info' in result:
                    print(f"🐛 Debug info: {result['debug_info']}")
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                traceback.print_exc()
            
            print("-" * 60)
        
        # Test individual components
        print(f"\n{'='*60}")
        print("🔧 Testing Individual Full Agent Components")
        print("="*60)
        
        try:
            # Test plan generation
            print("\n📋 Testing plan generation...")
            test_query = "What is the total revenue from all invoices?"
            plan = await full_agent_instance._generate_plan(test_query)
            print(f"✅ Plan generated: {len(plan)} steps")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step}")
                
        except Exception as e:
            print(f"❌ Plan generation failed: {str(e)}")
            traceback.print_exc()
        
        try:
            # Test step execution
            print("\n⚙️ Testing step execution...")
            test_step = "Retrieve all invoice documents"
            step_result = await full_agent_instance._retrieve_chunks_for_step(test_step)
            print(f"✅ Step execution result: {len(step_result)} chunks retrieved")
            if step_result:
                print(f"📄 First chunk preview: {str(step_result[0])[:200]}...")
            else:
                print("📄 No chunks retrieved")
                
        except Exception as e:
            print(f"❌ Step execution failed: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
        traceback.print_exc()
        return False
    
    print("\n✅ Full Agent debug completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(debug_full_agent())
    sys.exit(0 if success else 1)
