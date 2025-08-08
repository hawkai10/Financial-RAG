"""
Comprehensive Parser Comparison Test
Compares Marker vs Docling parsers on the same PDF file and evaluates:
1. Chunking quality and table extraction
2. Embedding and retrieval performance  
3. Answer accuracy for specific financial questions

Test Questions:
1. What is the GSTIN of Krishna Prabhash Agro Oil?
2. What is the total amount chargeable by Bhartiya Enterprise to Krishna Prabhash Agro Oil Pvt?
3. What is the GST in the invoice to Krishna Prabhash Agro Oil Pvt?
4. What is the GST Rate?
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import os
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import both parsers
import subprocess
import tempfile
import shutil

# Import RAG components  
from chunk_manager import ChunkManager
from progressive_retrieval import ProgressiveRetriever
from document_reranker import EnhancedDocumentReranker
from unified_query_processor import UnifiedQueryProcessor
from init_chunks_db import create_chunks_database

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParserComparison:
    """Comprehensive comparison of Marker vs Docling parsers."""
    
    def __init__(self, test_file_path: str):
        self.test_file = Path(test_file_path)
        self.test_questions = [
            "What is the GSTIN of Krishna Prabhash Agro Oil?",
            "What is the total amount chargeable by Bhartiya Enterprise to Krishna Prabhash Agro Oil Pvt?", 
            "What is the GST in the invoice to Krishna Prabhash Agro Oil Pvt?",
            "What is the GST Rate?"
        ]
        
        # Output files for each parser
        self.marker_output = "chunks_marked_test.json"
        self.docling_output = "chunks_docling_test.json"
        self.comparison_results = "parser_comparison_results.json"
        
    def test_marker_parser(self) -> Dict[str, Any]:
        """Test Marker parser on the specified file."""
        logger.info("=" * 60)
        logger.info("TESTING MARKED.PY PARSER")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Import and use marked.py processing
            from marked import process_single_document as marked_process
            from docling.document_converter import DocumentConverter
            
            # Initialize converter
            converter = DocumentConverter()
            
            # Process the single document using marked.py
            chunks = marked_process(self.test_file, converter)
            
            if not chunks:
                logger.error("Marked.py parser failed to produce chunks")
                return {"success": False, "error": "No chunks produced"}
            
            processing_time = time.time() - start_time
            
            # Save chunks
            with open(self.marker_output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Analyze results
            analysis = self._analyze_chunks(chunks, "Marked")
            analysis['processing_time'] = processing_time
            analysis['success'] = True
            
            logger.info(f"✅ Marked.py processing complete: {len(chunks)} chunks in {processing_time:.2f}s")
            logger.info(f"📊 Tables found: {analysis['table_chunks']}")
            logger.info(f"📝 Text chunks: {analysis['text_chunks']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Marked.py parser failed: {e}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}
    
    def test_docling_parser(self) -> Dict[str, Any]:
        """Test Docling parser on the specified file."""
        logger.info("=" * 60)
        logger.info("TESTING DOCLING AUTO_PARSE_FOLDER PARSER")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Import docling processing function from your local file
            sys.path.append(str(Path(__file__).parent / "docling"))
            from auto_parse_folder import process_single_document as docling_process
            from docling.document_converter import DocumentConverter
            
            # Initialize converter
            converter = DocumentConverter()
            
            # Process the single document
            chunks = docling_process(self.test_file, converter)
            
            if not chunks:
                logger.error("Docling parser failed to produce chunks")
                return {"success": False, "error": "No chunks produced"}
            
            processing_time = time.time() - start_time
            
            # Save chunks
            with open(self.docling_output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Analyze results
            analysis = self._analyze_chunks(chunks, "Docling")
            analysis['processing_time'] = processing_time
            analysis['success'] = True
            
            logger.info(f"✅ Docling processing complete: {len(chunks)} chunks in {processing_time:.2f}s")
            logger.info(f"📊 Tables found: {analysis['table_chunks']}")
            logger.info(f"📝 Text chunks: {analysis['text_chunks']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Docling parser failed: {e}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}
    
    def _analyze_chunks(self, chunks: List[Dict], parser_name: str) -> Dict[str, Any]:
        """Analyze chunk characteristics."""
        total_chunks = len(chunks)
        table_chunks = sum(1 for chunk in chunks if chunk.get('is_table', False))
        text_chunks = total_chunks - table_chunks
        
        # Calculate token statistics
        token_counts = [chunk.get('num_tokens', 0) for chunk in chunks if chunk.get('num_tokens')]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        # Find chunks with financial keywords
        financial_keywords = ['gstin', 'gst', 'invoice', 'amount', 'total', 'krishna', 'bhartiya', 'enterprise']
        relevant_chunks = []
        
        for chunk in chunks:
            text = chunk.get('chunk_text', '').lower()
            if any(keyword in text for keyword in financial_keywords):
                relevant_chunks.append({
                    'chunk_id': chunk.get('chunk_id'),
                    'is_table': chunk.get('is_table', False),
                    'text_preview': chunk.get('chunk_text', '')[:200] + "..." if len(chunk.get('chunk_text', '')) > 200 else chunk.get('chunk_text', ''),
                    'context': chunk.get('context', ''),
                })
        
        return {
            'parser': parser_name,
            'total_chunks': total_chunks,
            'table_chunks': table_chunks,
            'text_chunks': text_chunks,
            'avg_tokens': avg_tokens,
            'relevant_chunks_count': len(relevant_chunks),
            'relevant_chunks': relevant_chunks[:5],  # Show first 5 relevant chunks
        }
    
    def test_retrieval_performance(self, chunks_file: str, parser_name: str) -> Dict[str, Any]:
        """Test retrieval performance using the RAG system."""
        logger.info(f"Testing retrieval performance for {parser_name}")
        
        try:
            # Load chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Create temporary chunks file for testing
            temp_chunks_file = f"temp_chunks_{parser_name.lower()}.json"
            with open(temp_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Initialize RAG components with temporary chunks
            try:
                # Initialize ChunkManager with the temporary file
                chunk_manager = ChunkManager(temp_chunks_file)
                
                # Initialize embeddings (required for ProgressiveRetriever)
                from txtai import Embeddings
                embeddings = Embeddings()
                embeddings.load("business-docs-index")
                
                # Initialize other components with correct parameters
                retriever = ProgressiveRetriever(embeddings)
                reranker = EnhancedDocumentReranker()
                query_processor = UnifiedQueryProcessor()
                
                # Test each question
                results = {}
                import asyncio
                
                for question in self.test_questions:
                    logger.info(f"Testing question: {question}")
                    
                    start_time = time.time()
                    
                    # Process query using correct method name
                    query_result = query_processor.process_query_unified(question)
                    strategy = query_result.get('intent', 'Standard')
                    processed_query = query_result.get('corrected_query', question)
                    confidence = query_result.get('confidence', 0.8)
                    
                    # Retrieve chunks using async method
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        retrieved_chunks, retrieval_info = loop.run_until_complete(
                            retriever.retrieve_progressively([processed_query], strategy, confidence)
                        )
                    finally:
                        loop.close()
                    
                    # Rerank if needed
                    if strategy != "Aggregation":
                        reranked_chunks, rerank_info = reranker.rerank_chunks(
                            query=processed_query,
                            chunks=retrieved_chunks,
                            strategy=strategy
                        )
                    else:
                        reranked_chunks = retrieved_chunks
                        rerank_info = {"message": "Skipped for aggregation"}
                    
                    retrieval_time = time.time() - start_time
                    
                    # Analyze retrieved chunks
                    chunk_analysis = self._analyze_retrieved_chunks(reranked_chunks, question)
                    
                    results[question] = {
                        'strategy': strategy,
                        'retrieval_time': retrieval_time,
                        'chunks_retrieved': len(retrieved_chunks),
                        'chunks_reranked': len(reranked_chunks),
                        'top_chunks': [
                            {
                                'chunk_id': chunk.get('chunk_id'),
                                'score': chunk.get('final_rerank_score', chunk.get('retrieval_score', 0)),
                                'is_table': chunk.get('is_table', False),
                                'text_preview': chunk.get('chunk_text', '')[:150] + "..." if len(chunk.get('chunk_text', '')) > 150 else chunk.get('chunk_text', ''),
                            }
                            for chunk in reranked_chunks[:3]  # Top 3 chunks
                        ],
                        'analysis': chunk_analysis
                    }
                
                # Cleanup
                if os.path.exists(temp_chunks_file):
                    os.remove(temp_chunks_file)
                
                return {
                    'success': True,
                    'parser': parser_name,
                    'results': results
                }
                
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                if os.path.exists(temp_chunks_file):
                    os.remove(temp_chunks_file)
                return {'success': False, 'error': str(e)}
                
        except Exception as e:
            logger.error(f"Error testing retrieval for {parser_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_retrieved_chunks(self, chunks: List[Dict], question: str) -> Dict[str, Any]:
        """Analyze quality of retrieved chunks for a specific question."""
        if not chunks:
            return {'relevance_score': 0, 'has_financial_data': False, 'table_chunks': 0}
        
        # Define question-specific keywords
        question_keywords = {
            'gstin': ['gstin', 'gst identification', 'tax id', 'krishna prabhash'],
            'total amount': ['total', 'amount', 'chargeable', 'bhartiya enterprise', 'krishna prabhash'],
            'gst': ['gst', 'tax', 'sgst', 'cgst', 'igst'],
            'gst rate': ['rate', 'gst rate', '%', 'percent', 'tax rate']
        }
        
        # Determine question type
        question_lower = question.lower()
        relevant_keywords = []
        if 'gstin' in question_lower:
            relevant_keywords = question_keywords['gstin']
        elif 'total amount' in question_lower:
            relevant_keywords = question_keywords['total amount']
        elif 'gst rate' in question_lower:
            relevant_keywords = question_keywords['gst rate']
        elif 'gst' in question_lower:
            relevant_keywords = question_keywords['gst']
        
        # Calculate relevance
        relevance_scores = []
        table_chunks = 0
        has_financial_data = False
        
        for chunk in chunks:
            text = chunk.get('chunk_text', '').lower()
            
            if chunk.get('is_table', False):
                table_chunks += 1
            
            # Check for financial indicators
            financial_indicators = ['₹', 'rs.', 'amount', 'total', 'gst', 'tax', 'invoice']
            if any(indicator in text for indicator in financial_indicators):
                has_financial_data = True
            
            # Calculate keyword relevance
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in text)
            relevance_score = keyword_matches / len(relevant_keywords) if relevant_keywords else 0
            relevance_scores.append(relevance_score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return {
            'relevance_score': avg_relevance,
            'has_financial_data': has_financial_data,
            'table_chunks': table_chunks,
            'total_chunks': len(chunks)
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete comparison test."""
        logger.info("🚀 Starting comprehensive parser comparison test")
        logger.info(f"📄 Test file: {self.test_file}")
        
        if not self.test_file.exists():
            logger.error(f"Test file not found: {self.test_file}")
            return {"error": "Test file not found"}
        
        results = {
            'test_file': str(self.test_file),
            'test_questions': self.test_questions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Test Marker parser
        marker_results = self.test_marker_parser()
        results['marked'] = marker_results
        
        # Test Docling parser
        docling_results = self.test_docling_parser()
        results['docling'] = docling_results
        
        # Test retrieval performance if both parsers succeeded
        if marker_results.get('success') and os.path.exists(self.marker_output):
            marker_retrieval = self.test_retrieval_performance(self.marker_output, "Marked")
            results['marked']['retrieval'] = marker_retrieval
        
        if docling_results.get('success') and os.path.exists(self.docling_output):
            docling_retrieval = self.test_retrieval_performance(self.docling_output, "Docling")
            results['docling']['retrieval'] = docling_retrieval
        
        # Generate comparison summary
        results['comparison'] = self._generate_comparison_summary(results)
        
        # Save results
        with open(self.comparison_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self._print_summary(results)
        
        return results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary comparison of both parsers."""
        summary = {}
        
        # Processing comparison
        marker = results.get('marked', {})
        docling = results.get('docling', {})
        
        if marker.get('success') and docling.get('success'):
            summary['processing'] = {
                'marker_time': marker.get('processing_time', 0),
                'docling_time': docling.get('processing_time', 0),
                'marker_chunks': marker.get('total_chunks', 0),
                'docling_chunks': docling.get('total_chunks', 0),
                'marker_tables': marker.get('table_chunks', 0),
                'docling_tables': docling.get('table_chunks', 0),
            }
            
            # Determine winner for each category
            summary['processing']['faster_parser'] = 'Marked' if marker.get('processing_time', float('inf')) < docling.get('processing_time', float('inf')) else 'Docling'
            summary['processing']['more_chunks'] = 'Marked' if marker.get('total_chunks', 0) > docling.get('total_chunks', 0) else 'Docling'
            summary['processing']['more_tables'] = 'Marked' if marker.get('table_chunks', 0) > docling.get('table_chunks', 0) else 'Docling'
        
        # Retrieval comparison
        marker_retrieval = marker.get('retrieval', {})
        docling_retrieval = docling.get('retrieval', {})
        
        if marker_retrieval.get('success') and docling_retrieval.get('success'):
            summary['retrieval'] = {}
            
            for question in self.test_questions:
                m_result = marker_retrieval.get('results', {}).get(question, {})
                d_result = docling_retrieval.get('results', {}).get(question, {})
                
                m_analysis = m_result.get('analysis', {})
                d_analysis = d_result.get('analysis', {})
                
                summary['retrieval'][question] = {
                    'marked_relevance': m_analysis.get('relevance_score', 0),
                    'docling_relevance': d_analysis.get('relevance_score', 0),
                    'marked_tables': m_analysis.get('table_chunks', 0),
                    'docling_tables': d_analysis.get('table_chunks', 0),
                    'better_parser': 'Marked' if m_analysis.get('relevance_score', 0) > d_analysis.get('relevance_score', 0) else 'Docling'
                }
        
        return summary
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of the comparison results."""
        logger.info("\n" + "=" * 80)
        logger.info("🏆 PARSER COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        marker = results.get('marker', {})
        docling = results.get('docling', {})
        comparison = results.get('comparison', {})
        
        # Processing Summary
        if marker.get('success') and docling.get('success'):
            logger.info("\n📊 PROCESSING PERFORMANCE:")
            logger.info(f"⏱️  Processing Time: Marked {marker.get('processing_time', 0):.2f}s vs Docling {docling.get('processing_time', 0):.2f}s")
            logger.info(f"📝 Total Chunks: Marked {marker.get('total_chunks', 0)} vs Docling {docling.get('total_chunks', 0)}")
            logger.info(f"📊 Table Chunks: Marked {marker.get('table_chunks', 0)} vs Docling {docling.get('table_chunks', 0)}")
            
            processing = comparison.get('processing', {})
            logger.info(f"🏃 Faster: {processing.get('faster_parser', 'Unknown')}")
            logger.info(f"📈 More Chunks: {processing.get('more_chunks', 'Unknown')}")
            logger.info(f"🗃️  Better Table Detection: {processing.get('more_tables', 'Unknown')}")
        
        # Retrieval Summary
        retrieval_summary = comparison.get('retrieval', {})
        if retrieval_summary:
            logger.info("\n🎯 RETRIEVAL QUALITY:")
            for question in self.test_questions:
                q_result = retrieval_summary.get(question, {})
                logger.info(f"\n❓ {question}")
                logger.info(f"   Marked relevance: {q_result.get('marked_relevance', 0):.2f}")
                logger.info(f"   Docling relevance: {q_result.get('docling_relevance', 0):.2f}")
                logger.info(f"   Better: {q_result.get('better_parser', 'Unknown')}")
        
        logger.info(f"\n💾 Detailed results saved to: {self.comparison_results}")
        logger.info("=" * 80)

def main():
    """Main function to run the parser comparison."""
    test_file = r"C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 19-20.pdf"
    
    comparison = ParserComparison(test_file)
    results = comparison.run_comprehensive_test()
    
    return results

if __name__ == "__main__":
    results = main()
