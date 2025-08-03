#!/usr/bin/env python3
"""
Diagnostic script to check each component step by step.
"""

def test_imports():
    """Test all imports."""
    print("🔍 Testing imports...")
    
    try:
        import os
        print("✅ os imported")
        
        from txtai import Embeddings
        print("✅ txtai imported")
        
        from config import Config
        print("✅ config imported")
        
        from rag_backend import rag_query_enhanced
        print("✅ rag_backend imported")
        
        from hierarchical_processor import HierarchicalProcessor
        print("✅ hierarchical_processor imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test config loading."""
    print("\n🔍 Testing config...")
    
    try:
        from config import Config
        config = Config()
        print(f"✅ Config loaded")
        print(f"   INDEX_PATH: {config.INDEX_PATH}")
        print(f"   PROGRESSIVE_RETRIEVAL_ENABLED: {config.PROGRESSIVE_RETRIEVAL_ENABLED}")
        print(f"   SAMPLING_AGGREGATION_ENABLED: {config.SAMPLING_AGGREGATION_ENABLED}")
        print(f"   HYBRID_ALPHA: {config.HYBRID_ALPHA}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hierarchical_processor():
    """Test hierarchical processor initialization."""
    print("\n🔍 Testing HierarchicalProcessor...")
    
    try:
        from hierarchical_processor import HierarchicalProcessor
        
        def mock_llm(prompt, **kwargs):
            return "Mock response"
        
        processor = HierarchicalProcessor(mock_llm)
        print("✅ HierarchicalProcessor created")
        
        # Test if methods exist
        methods_to_check = [
            '_assess_completeness',
            '_analyze_completeness', 
            '_create_smart_batches',
            '_estimate_total_tokens_with_prompt',
            '_estimate_chunk_tokens',
            '_detect_conflicts',
            '_combine_batch_results'
        ]
        
        for method in methods_to_check:
            if hasattr(processor, method):
                print(f"✅ {method} exists")
            else:
                print(f"❌ {method} missing")
                
        return True
        
    except Exception as e:
        print(f"❌ HierarchicalProcessor error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embeddings loading."""
    print("\n🔍 Testing embeddings...")
    
    try:
        import os
        from txtai import Embeddings
        from config import Config
        
        config = Config()
        embeddings = Embeddings()
        index_path = config.INDEX_PATH
        
        if os.path.exists(index_path):
            embeddings.load(index_path)
            print(f"✅ Embeddings loaded from {index_path}")
            return True
        else:
            print(f"❌ Index not found at {index_path}")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("🚀 Running Diagnostic Tests for New RAG System\n")
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),  
        ("HierarchicalProcessor", test_hierarchical_processor),
        ("Embeddings", test_embeddings)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
        
        print("-" * 60)
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All diagnostic tests passed! System should be working.")
    else:
        print("⚠️ Some tests failed. Issues need to be fixed.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()
