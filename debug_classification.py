"""
Debug script to test query classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_query_processor import unified_processor

def test_classification():
    """Test the classification of specific queries."""
    
    test_queries = [
        "What is the rent for the first year according to the rent agreement?",
        "List all the parties being issued an invoice by Bhartiya Enterprises?",
        "Please analyze the impact of using nifty 500 index for the research paper"
    ]
    
    print("🔍 Testing Query Classification\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        try:
            result = unified_processor.process_query_unified(query)
            
            print(f"  Intent: {result.get('intent', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Aggregation Type: {result.get('aggregation_type', 'N/A')}")
            print(f"  Complexity Level: {result.get('complexity_level', 'N/A')}")
            print(f"  Requires Multi-step: {result.get('requires_multi_step', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
            print(f"  Corrected Query: {result.get('corrected_query', 'N/A')}")
            
            # Determine expected routing
            intent = result.get('intent')
            aggregation_type = result.get('aggregation_type', 'none')
            complexity_level = result.get('complexity_level', 'simple')
            requires_multi_step = result.get('requires_multi_step', False)
            
            if intent == "Aggregation" and aggregation_type != 'none':
                expected_agent = "Mini-Agent"
            elif intent == "Analyse" and (complexity_level in ['complex'] or requires_multi_step):
                expected_agent = "Full-Agent"
            else:
                expected_agent = "Standard-RAG"
            
            print(f"  Expected Agent: {expected_agent}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_classification()
