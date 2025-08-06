"""
Test script for the new Hybrid Query System
Tests Mini-Agent, Full Agent, and Standard RAG routing
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_backend import rag_query_enhanced
from txtai import Embeddings
from utils import logger

def load_embeddings():
    """Load the embeddings index."""
    try:
        embeddings = Embeddings()
        embeddings.load("business-docs-index")
        logger.info("Embeddings loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None

async def test_hybrid_queries():
    """Test different types of queries with the hybrid system."""
    
    # Load embeddings
    embeddings = load_embeddings()
    if not embeddings:
        print("❌ Cannot run tests - embeddings not loaded")
        return
    
    print("🚀 Testing Hybrid Query System\n")
    
    # Test queries for different agents
    test_queries = [
        # Standard RAG test (Simple fact-finding)
        {
            "query": "What is the rent for the first year according to the rent agreement?",
            "expected_agent": "Standard-RAG",
            "type": "Standard - Specific Information"
        },
        
        # Mini-Agent test (Aggregation)
        {
            "query": "List all the parties being issued an invoice by Bhartiya Enterprises?",
            "expected_agent": "Mini-Agent",
            "type": "Aggregation - Invoice Recipients"
        },
        
        # Full Agent test (Complex Analysis)
        {
            "query": "Please analyze the impact of using nifty 500 index for the research paper",
            "expected_agent": "Full-Agent",
            "type": "Analysis - Impact Assessment"
        }
    ]
    
    results = []
    
    # Create a detailed log file for UI-style output
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'hybrid_test_detailed_log_{timestamp}.txt'
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("HYBRID QUERY SYSTEM - DETAILED TEST LOG\n")
        log_file.write(f"Test Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for i, test in enumerate(test_queries, 1):
            print(f"📝 Test {i}: {test['type']}")
            print(f"Query: {test['query']}")
            print("Processing...")
            
            # Log test header
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"TEST {i}: {test['type']}\n")
            log_file.write(f"{'='*60}\n")
            log_file.write(f"Query: {test['query']}\n")
            log_file.write(f"Expected Agent: {test['expected_agent']}\n")
            log_file.write("-" * 60 + "\n")
            
            try:
                # Run the query
                result = await rag_query_enhanced(
                    question=test['query'],
                    embeddings=embeddings,
                    topn=5
                )
                
                # Extract detailed information
                agent_used = result.get('agent_used', 'Unknown')
                strategy_used = result.get('strategy_used', result.get('query_strategy', 'Unknown'))
                processing_time = result.get('processing_time', 0)
                success = result.get('success', True)
                answer = result.get('answer', '')
                classification = result.get('classification', {})
                
                # Log classification details
                log_file.write("CLASSIFICATION DETAILS:\n")
                log_file.write(f"  Intent: {classification.get('intent', 'N/A')}\n")
                log_file.write(f"  Confidence: {classification.get('confidence', 'N/A')}\n")
                log_file.write(f"  Aggregation Type: {classification.get('aggregation_type', 'N/A')}\n")
                log_file.write(f"  Complexity Level: {classification.get('complexity_level', 'N/A')}\n")
                log_file.write(f"  Requires Multi-step: {classification.get('requires_multi_step', 'N/A')}\n")
                log_file.write(f"  Corrected Query: {classification.get('corrected_query', 'N/A')}\n")
                log_file.write("\n")
                
                # Log processing details
                log_file.write("PROCESSING DETAILS:\n")
                log_file.write(f"  Agent Used: {agent_used}\n")
                log_file.write(f"  Strategy: {strategy_used}\n")
                log_file.write(f"  Processing Time: {processing_time:.2f} seconds\n")
                log_file.write(f"  Success: {success}\n")
                log_file.write("\n")
                
                # Log the full answer as it would appear in UI
                log_file.write("FULL ANSWER (AS DISPLAYED IN UI):\n")
                log_file.write("-" * 40 + "\n")
                log_file.write(answer)
                log_file.write("\n" + "-" * 40 + "\n")
                
                print(f"✅ Agent Used: {agent_used}")
                print(f"📊 Strategy: {strategy_used}")
                print(f"⏱️  Time: {processing_time:.2f}s")
                print(f"🎯 Success: {success}")
                
                # Check if correct agent was used
                if agent_used == test['expected_agent']:
                    print(f"✅ Correct routing! Expected {test['expected_agent']}, got {agent_used}")
                    log_file.write(f"✅ ROUTING: Correct! Expected {test['expected_agent']}, got {agent_used}\n")
                else:
                    print(f"⚠️  Routing mismatch. Expected {test['expected_agent']}, got {agent_used}")
                    log_file.write(f"⚠️ ROUTING: Mismatch! Expected {test['expected_agent']}, got {agent_used}\n")
                
                # Show answer preview in console
                answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"💬 Answer Preview: {answer_preview}")
                
                # Log additional metadata if available
                if 'retrieval_info' in result:
                    log_file.write(f"\nRETRIEVAL INFO: {result['retrieval_info']}\n")
                
                if 'chunks' in result and result['chunks']:
                    log_file.write(f"\nCHUNKS USED: {len(result['chunks'])} chunks\n")
                    for idx, chunk in enumerate(result['chunks'][:3], 1):  # Show first 3 chunks
                        chunk_preview = chunk.get('chunk_text', chunk.get('text', ''))[:100]
                        log_file.write(f"  Chunk {idx}: {chunk_preview}...\n")
                
                log_file.write("\n" + "="*60 + "\n")
                
                results.append({
                    'test': test['type'],
                    'query': test['query'],
                    'expected_agent': test['expected_agent'],
                    'actual_agent': agent_used,
                    'strategy': strategy_used,
                    'success': success,
                    'processing_time': processing_time,
                    'routing_correct': agent_used == test['expected_agent'],
                    'classification': classification,
                    'answer_length': len(answer)
                })
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Test failed: {error_msg}")
                
                log_file.write(f"❌ ERROR: {error_msg}\n")
                log_file.write(f"Full traceback:\n")
                import traceback
                log_file.write(traceback.format_exc())
                log_file.write("\n" + "="*60 + "\n")
                
                results.append({
                    'test': test['type'],
                    'query': test['query'], 
                    'expected_agent': test['expected_agent'],
                    'actual_agent': 'Error',
                    'success': False,
                    'error': error_msg,
                    'routing_correct': False
                })
            
            print("-" * 80)
    
    print(f"\n📄 Detailed log saved to: {log_filename}")
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    correct_routing = sum(1 for r in results if r.get('routing_correct', False))
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}/{total_tests}")
    print(f"Correct Routing: {correct_routing}/{total_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"Routing Accuracy: {(correct_routing/total_tests)*100:.1f}%")
    
    # Agent usage breakdown
    agent_usage = {}
    for result in results:
        agent = result.get('actual_agent', 'Unknown')
        agent_usage[agent] = agent_usage.get(agent, 0) + 1
    
    print(f"\n🤖 Agent Usage:")
    for agent, count in agent_usage.items():
        print(f"  {agent}: {count} queries")
    
    # Show any failed tests
    failed_tests = [r for r in results if not r.get('success', False)]
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test['test']}: {test.get('error', 'Unknown error')}")
    
    return results

def main():
    """Main test function."""
    print("🧪 Starting Hybrid Query System Tests")
    print("="*60)
    
    try:
        # Run async tests
        results = asyncio.run(test_hybrid_queries())
        
        print("\n✅ Testing completed!")
        
        # Save results to file
        import json
        with open('hybrid_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("📄 Results saved to hybrid_test_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
