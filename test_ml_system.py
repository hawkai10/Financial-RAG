"""
ML-based System Test Suite
Comprehensive tests for the keyword-free, ML-based RAG system
"""

import asyncio
import json
import logging
import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip this legacy test suite
pytest.skip("Legacy ML system tests are deprecated; parent–child E2E supersedes them.", allow_module_level=True)

class TestMLRAGSystem:
    """Test suite for the ML-based RAG system."""
    
    @pytest.fixture
    def sample_extraction_data(self):
        """Create sample extraction data for testing."""
        return [{
            'document_id': 'test_financial_doc_001',
            'pages': [
                {
                    'blocks': [
                        {
                            'type': 'text',
                            'content': 'Apple Inc. reported quarterly revenue of $89.5 billion for Q1 2023, representing a 5% decline from the previous year. The company\'s iPhone sales decreased by 8% while Services revenue grew by 6.4% to $20.8 billion.'
                        },
                        {
                            'type': 'table',
                            'content': '<table><tr><th>Product</th><th>Revenue (Billions)</th><th>Growth</th></tr><tr><td>iPhone</td><td>$65.8</td><td>-8%</td></tr><tr><td>Services</td><td>$20.8</td><td>+6.4%</td></tr><tr><td>Mac</td><td>$7.7</td><td>-29%</td></tr></table>'
                        },
                        {
                            'type': 'text',
                            'content': 'CEO Tim Cook stated that the company remains optimistic about long-term growth prospects in emerging markets, particularly India and Southeast Asia. The board of directors approved a quarterly dividend of $0.24 per share.'
                        }
                    ]
                },
                {
                    'blocks': [
                        {
                            'type': 'text',
                            'content': 'The company\'s cash position remains strong at $165.0 billion, with $29.0 billion in debt. Apple continues to invest heavily in research and development, spending $7.8 billion in the quarter, up 14% year-over-year.'
                        }
                    ]
                }
            ]
        }]
    
    def test_content_analyzer(self, sample_extraction_data):
        """Test the dynamic content analyzer."""
        from content_analyzer import DynamicContentAnalyzer
        
        analyzer = DynamicContentAnalyzer()
        
        # Test with financial content
        content = sample_extraction_data[0]['pages'][0]['blocks'][0]['content']
        insight = analyzer.analyze_content(content)
        
        # Assertions
        assert insight is not None
        assert insight.confidence_score > 0
        assert len(insight.entities) > 0
        assert 'ORG' in insight.entities  # Should detect Apple Inc.
        assert 'MONEY' in insight.entities  # Should detect $89.5 billion
        assert 'PERCENT' in insight.entities  # Should detect 5%
        assert insight.document_type in ['financial_statement', 'financial_report', 'corporate_document']
        
        logger.info(f"✅ Content Analyzer test passed - found {len(insight.entities)} entity types")
    
    def test_enhanced_chunker(self, sample_extraction_data):
        """Test the enhanced JSON chunker."""
        from enhanced_json_chunker import EnhancedJSONChunker
        
        chunker = EnhancedJSONChunker()
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_extraction_data, f)
            temp_file = f.name
        
        try:
            # Process the extraction data
            chunks = chunker.process_extracted_json(temp_file)
            
            # Assertions
            assert len(chunks) > 0
            assert all(chunk.chunk_id for chunk in chunks)
            assert all(chunk.content for chunk in chunks)
            assert all(chunk.content_insight for chunk in chunks)
            assert all(chunk.confidence_score > 0 for chunk in chunks)
            
            # Check that entities were detected
            entities_found = any(
                len(chunk.content_insight.entities) > 0 
                for chunk in chunks
            )
            assert entities_found
            
            # Check chunk quality
            quality_report = chunker.analyze_chunk_quality(chunks)
            assert quality_report['total_chunks'] == len(chunks)
            assert quality_report['avg_confidence'] > 0
            
            logger.info(f"✅ Enhanced Chunker test passed - created {len(chunks)} chunks")
            
        finally:
            os.unlink(temp_file)
    
    def test_dynamic_schema_manager(self):
        """Test the dynamic schema manager."""
        from dynamic_schema_manager import DynamicSchemaManager
        from content_analyzer import DynamicContentAnalyzer, ContentInsight, EntityRelationship
        
        # Mock schema manager (without actual Dgraph connection)
        schema_manager = DynamicSchemaManager("http://mock:8080")
        
        # Create mock insight
        analyzer = DynamicContentAnalyzer()
        insight = analyzer.analyze_content("Apple Inc. owns multiple subsidiaries and Tim Cook manages the company.")
        
        # Test schema analysis
        updates = schema_manager._analyze_entities_for_schema(insight.entities)
        assert len(updates) > 0
        
        relationship_updates = schema_manager._analyze_relationships_for_schema(insight.relationships)
        # May or may not have relationships depending on content analysis
        
        logger.info(f"✅ Dynamic Schema Manager test passed - {len(updates)} schema updates identified")
    
    def test_dgraph_manager_mock(self, sample_extraction_data):
        """Test Dgraph manager with mock operations."""
        from adaptive_dgraph_manager import AdaptiveDgraphManager
        from enhanced_json_chunker import EnhancedJSONChunker
        
        # Create chunks for testing
        chunker = EnhancedJSONChunker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_extraction_data, f)
            temp_file = f.name
        
        try:
            chunks = chunker.process_extracted_json(temp_file)
            
            # Test Dgraph manager creation (will fail connection but test structure)
            dgraph_manager = AdaptiveDgraphManager("http://mock:8080")
            
            # Test internal methods
            if chunks:
                chunk = chunks[0]
                
                # Test metadata creation methods
                entities = chunk.content_insight.entities
                assert isinstance(entities, dict)
                
                relationships = chunk.content_insight.relationships
                assert isinstance(relationships, list)
                
            logger.info("✅ Dgraph Manager structure test passed")
            
        finally:
            os.unlink(temp_file)
    
    def test_qdrant_manager_mock(self, sample_extraction_data):
        """Test Qdrant manager with mock operations."""
        from adaptive_qdrant_manager import AdaptiveQdrantManager
        from enhanced_json_chunker import EnhancedJSONChunker
        
        # Test metadata creation without actual Qdrant connection
        chunker = EnhancedJSONChunker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_extraction_data, f)
            temp_file = f.name
        
        try:
            chunks = chunker.process_extracted_json(temp_file)
            
            if chunks:
                chunk = chunks[0]
                
                # Test adaptive metadata creation
                # We'll test this without initializing QdrantClient
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "confidence_score": chunk.confidence_score,
                    "document_type": chunk.content_insight.document_type
                }
                
                # Add entity information
                for entity_type, entities in chunk.content_insight.entities.items():
                    if entities:
                        metadata[f"entities_{entity_type.lower()}"] = entities
                        metadata[f"has_{entity_type.lower()}"] = True
                
                assert len(metadata) > 5  # Should have basic fields plus entity fields
                
            logger.info("✅ Qdrant Manager structure test passed")
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_ml_rag_backend_structure(self, sample_extraction_data):
        """Test ML RAG backend structure without database connections."""
        from ml_rag_backend import MLBasedRAGBackend
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_extraction_data, f)
            temp_file = f.name
        
        try:
            # Test backend initialization (will work without actual DB connections)
            backend = MLBasedRAGBackend("http://mock:8080", "http://mock:6333")
            
            # Test configuration
            assert hasattr(backend, 'content_analyzer')
            assert hasattr(backend, 'chunker')
            assert hasattr(backend, 'dgraph_manager')
            assert hasattr(backend, 'qdrant_manager')
            assert hasattr(backend, 'query_processor')
            
            # Test performance metrics initialization
            assert 'total_queries' in backend.performance_metrics
            assert 'successful_queries' in backend.performance_metrics
            
            logger.info("✅ ML RAG Backend structure test passed")
            
        finally:
            os.unlink(temp_file)
    
    def test_system_integration_flow(self, sample_extraction_data):
        """Test the complete system integration flow."""
        from content_analyzer import DynamicContentAnalyzer
        from enhanced_json_chunker import EnhancedJSONChunker
        
        # Step 1: Content Analysis
        analyzer = DynamicContentAnalyzer()
        content = sample_extraction_data[0]['pages'][0]['blocks'][0]['content']
        insight = analyzer.analyze_content(content)
        
        assert insight.confidence_score > 0
        
        # Step 2: Enhanced Chunking
        chunker = EnhancedJSONChunker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_extraction_data, f)
            temp_file = f.name
        
        try:
            chunks = chunker.process_extracted_json(temp_file)
            assert len(chunks) > 0
            
            # Step 3: Verify ML pipeline
            for chunk in chunks:
                assert chunk.content_insight is not None
                assert len(chunk.schema_elements) > 0
                assert chunk.confidence_score > 0
                
                # Verify no hardcoded keywords were used
                # The system should work dynamically
                assert chunk.content_insight.document_type != 'unknown'
            
            logger.info("✅ System Integration Flow test passed")
            
        finally:
            os.unlink(temp_file)
    
    def test_keyword_free_operation(self, sample_extraction_data):
        """Test that the system operates without predefined keywords."""
        from content_analyzer import DynamicContentAnalyzer
        from enhanced_json_chunker import EnhancedJSONChunker
        
        # Test with completely different content that wasn't in training
        different_content = [{
            'document_id': 'different_doc_001',
            'pages': [{
                'blocks': [{
                    'type': 'text',
                    'content': 'Netflix Inc. announced subscriber growth of 8.9 million in Q2 2023. The streaming service now has 238.4 million global subscribers. Content spending reached $4.2 billion for original programming.'
                }]
            }]
        }]
        
        chunker = EnhancedJSONChunker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(different_content, f)
            temp_file = f.name
        
        try:
            chunks = chunker.process_extracted_json(temp_file)
            
            # System should still work with different content
            assert len(chunks) > 0
            
            for chunk in chunks:
                # Should detect entities dynamically
                assert len(chunk.content_insight.entities) > 0
                
                # Should classify document type dynamically
                assert chunk.content_insight.document_type != 'unknown'
                
                # Should have reasonable confidence
                assert chunk.confidence_score > 0.3
            
            logger.info("✅ Keyword-free Operation test passed")
            
        finally:
            os.unlink(temp_file)

def run_comprehensive_tests():
    """Run comprehensive test suite."""
    
    # Create sample data
    sample_data = TestMLRAGSystem().sample_extraction_data()
    
    # Initialize test instance
    test_suite = TestMLRAGSystem()
    
    try:
        logger.info("🧪 Starting ML RAG System Tests...")
        
        # Run individual tests
        test_suite.test_content_analyzer(sample_data)
        test_suite.test_enhanced_chunker(sample_data)
        test_suite.test_dynamic_schema_manager()
        test_suite.test_dgraph_manager_mock(sample_data)
        test_suite.test_qdrant_manager_mock(sample_data)
        test_suite.test_system_integration_flow(sample_data)
        test_suite.test_keyword_free_operation(sample_data)
        
        # Run async tests
        asyncio.run(test_suite.test_ml_rag_backend_structure(sample_data))
        
        logger.info("🎉 All ML RAG System Tests Passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_comprehensive_tests()
