"""
Unified ML-based RAG Backend 
Orchestrates the new ML-based RAG system without keywords, using Dgraph + Qdrant architecture.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our new ML-based components
from content_analyzer import DynamicContentAnalyzer
from enhanced_json_chunker import EnhancedJSONChunker
from adaptive_dgraph_manager import AdaptiveDgraphManager
from adaptive_qdrant_manager import AdaptiveQdrantManager
from dynamic_schema_manager import DynamicSchemaManager

# Import existing components we still need
from unified_query_processor import UnifiedQueryProcessor
from config import get_config

config = get_config()

class MLBasedRAGBackend:
    """
    Main orchestrator for the ML-based RAG system.
    Coordinates content analysis, graph storage, vector search, and answer generation.
    """
    
    def __init__(self, 
                 dgraph_url: str = "http://localhost:8080",
                 qdrant_url: str = "http://localhost:6333"):
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML-based components
        self.content_analyzer = DynamicContentAnalyzer()
        self.chunker = EnhancedJSONChunker()
        self.dgraph_manager = AdaptiveDgraphManager(dgraph_url)
        self.qdrant_manager = AdaptiveQdrantManager(qdrant_url)
        self.schema_manager = DynamicSchemaManager(dgraph_url)
        
        # Keep query processor for strategy detection
        self.query_processor = UnifiedQueryProcessor()
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'avg_processing_time': 0,
            'successful_queries': 0
        }
        
        self.logger.info("ML-based RAG backend initialized")
    
    async def process_documents(self, json_file_path: str) -> Dict[str, Any]:
        """
        Process extracted JSON documents through the ML pipeline.
        
        Args:
            json_file_path: Path to JSON extraction file
            
        Returns:
            Processing results and statistics
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing documents from {json_file_path}")
            
            # Step 1: Enhanced chunking with ML analysis
            chunks = self.chunker.process_extracted_json(json_file_path)
            
            if not chunks:
                return {
                    'success': False,
                    'message': 'No chunks created from input file',
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Store in graph database (Dgraph)
            graph_success = self.dgraph_manager.store_enhanced_chunks(chunks)
            
            # Step 3: Store in vector database (Qdrant)  
            vector_success = self.qdrant_manager.store_enhanced_chunks(chunks)
            
            # Step 4: Analyze chunk quality
            quality_report = self.chunker.analyze_chunk_quality(chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': graph_success and vector_success,
                'chunks_created': len(chunks),
                'graph_stored': graph_success,
                'vector_stored': vector_success,
                'quality_report': quality_report,
                'processing_time': processing_time,
                'message': 'Documents processed successfully'
            }
        
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            return {
                'success': False,
                'message': f'Error processing documents: {str(e)}',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def answer_query(self, query: str, context_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer a query using the ML-based RAG system.
        
        Args:
            query: User query
            context_params: Optional parameters for context retrieval
            
        Returns:
            Query results with answer and metadata
        """
        start_time = datetime.now()
        query_id = f"query_{int(datetime.now().timestamp())}"
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Analyze query strategy
            query_analysis = self.query_processor.process_query(query)
            strategy = query_analysis.get('strategy', 'Standard')
            
            # Step 2: Multi-modal retrieval based on strategy
            retrieval_results = await self._retrieve_relevant_content(
                query, strategy, context_params or {}
            )
            
            if not retrieval_results['chunks']:
                return {
                    'query_id': query_id,
                    'answer': 'I could not find relevant information to answer your query.',
                    'confidence': 0.0,
                    'sources': [],
                    'strategy': strategy,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Generate answer based on retrieved content
            answer_result = await self._generate_answer(
                query, retrieval_results['chunks'], strategy
            )
            
            # Step 4: Enhance with graph relationships
            enhanced_answer = await self._enhance_with_graph_context(
                answer_result, retrieval_results['graph_context']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            return {
                'query_id': query_id,
                'answer': enhanced_answer['answer'],
                'confidence': enhanced_answer['confidence'],
                'sources': retrieval_results['sources'],
                'strategy': strategy,
                'retrieval_method': retrieval_results['method'],
                'entities_found': retrieval_results['entities'],
                'relationships_found': retrieval_results['relationships'],
                'processing_time': processing_time,
                'metadata': {
                    'chunks_used': len(retrieval_results['chunks']),
                    'graph_context_used': len(retrieval_results['graph_context']),
                    'query_analysis': query_analysis
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self._update_performance_metrics((datetime.now() - start_time).total_seconds(), False)
            
            return {
                'query_id': query_id,
                'answer': f'I encountered an error while processing your query: {str(e)}',
                'confidence': 0.0,
                'sources': [],
                'strategy': 'Error',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _retrieve_relevant_content(self, query: str, strategy: str, 
                                       context_params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant content using multiple approaches."""
        
        # Determine retrieval parameters based on strategy
        if strategy == "Aggregation":
            top_k = 20  # More chunks for aggregation
            use_graph = True
        elif strategy == "Analyse":
            top_k = 10  # Focused analysis
            use_graph = True
        else:  # Standard
            top_k = 5   # Quick answers
            use_graph = False
        
        # Analyze query content first
        query_insight = self.content_analyzer.analyze_content(query)
        
        # Parallel retrieval from vector and graph databases
        async with asyncio.TaskGroup() as tg:
            # Vector search task
            vector_task = tg.create_task(
                self._vector_search_async(query, top_k, query_insight)
            )
            
            # Graph search task (if needed)
            if use_graph:
                graph_task = tg.create_task(
                    self._graph_search_async(query_insight, top_k)
                )
            else:
                graph_task = None
        
        vector_results = vector_task.result()
        graph_results = graph_task.result() if graph_task else {'chunks': [], 'context': []}
        
        # Combine and deduplicate results
        combined_chunks = self._combine_retrieval_results(vector_results, graph_results, top_k)
        
        return {
            'chunks': combined_chunks,
            'sources': [chunk.get('document_id', 'Unknown') for chunk in combined_chunks],
            'entities': list(query_insight.entities.keys()),
            'relationships': [rel.relationship_type for rel in query_insight.relationships],
            'graph_context': graph_results.get('context', []),
            'method': 'hybrid_ml' if use_graph else 'vector_ml'
        }
    
    async def _vector_search_async(self, query: str, top_k: int, 
                                  query_insight) -> List[Dict[str, Any]]:
        """Async wrapper for vector search."""
        loop = asyncio.get_event_loop()
        
        # Build filters based on query analysis
        filters = {}
        
        # Filter by document type if detected
        if query_insight.document_type != 'unknown':
            filters['document_type'] = query_insight.document_type
        
        # Filter by entities if present
        for entity_type, entities in query_insight.entities.items():
            if entities:
                filters[f'has_{entity_type.lower()}'] = True
        
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor,
                self.qdrant_manager.semantic_search,
                query, None, top_k, filters if filters else None
            )
        
        return results
    
    async def _graph_search_async(self, query_insight, top_k: int) -> Dict[str, Any]:
        """Async wrapper for graph search using improved reverse edge queries."""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            # Search for entities mentioned in query using comprehensive occurrence search
            all_chunks = []
            all_context = []
            
            # Get unique entities across all types
            all_entities = set()
            for entity_type, entities in query_insight.entities.items():
                for entity in entities[:2]:  # Limit to first 2 entities per type for performance
                    all_entities.add(entity)
            
            # Use the new comprehensive entity occurrence query for each entity
            for entity in all_entities:
                entity_data = await loop.run_in_executor(
                    executor,
                    self.dgraph_manager.query_entity_occurrences,
                    entity
                )
                
                # Extract chunks from comprehensive results
                if entity_data['related_chunks']:
                    all_chunks.extend(entity_data['related_chunks'])
                
                # Extract relationship context
                if entity_data['relationships']:
                    all_context.extend(entity_data['relationships'])
                
                # Add entity info for context
                if entity_data['entity_info']:
                    all_context.extend([{
                        'type': 'entity_info',
                        'data': info
                    } for info in entity_data['entity_info']])
        
        # Remove duplicates based on chunk_id
        seen_chunks = set()
        unique_chunks = []
        for chunk in all_chunks:
            chunk_id = chunk.get('chunk_id', chunk.get('uid'))
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_chunks.append(chunk)
        
        return {
            'chunks': unique_chunks[:top_k],
            'context': all_context
        }
    
    def _combine_retrieval_results(self, vector_results: List[Dict], 
                                  graph_results: Dict[str, Any], 
                                  top_k: int) -> List[Dict[str, Any]]:
        """Combine and deduplicate results from vector and graph search."""
        
        combined_chunks = []
        seen_chunk_ids = set()
        
        # Add vector results first (usually higher quality)
        for result in vector_results:
            chunk_id = result.get('chunk_id', result.get('id', ''))
            if chunk_id not in seen_chunk_ids:
                combined_chunks.append(result)
                seen_chunk_ids.add(chunk_id)
        
        # Add graph results
        for chunk in graph_results.get('chunks', []):
            chunk_id = chunk.get('chunk_id', '')
            if chunk_id not in seen_chunk_ids:
                # Convert graph format to standard format
                standardized_chunk = {
                    'chunk_id': chunk_id,
                    'content': chunk.get('content', ''),
                    'document_id': chunk.get('document_id', ''),
                    'confidence_score': chunk.get('confidence_score', 0.5),
                    'score': 0.7,  # Default relevance score for graph results
                    'source': 'graph'
                }
                combined_chunks.append(standardized_chunk)
                seen_chunk_ids.add(chunk_id)
        
        # Sort by relevance score and return top_k
        combined_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        return combined_chunks[:top_k]
    
    async def _generate_answer(self, query: str, chunks: List[Dict[str, Any]], 
                              strategy: str) -> Dict[str, Any]:
        """Generate answer based on retrieved chunks and strategy."""
        
        if not chunks:
            return {
                'answer': 'No relevant information found.',
                'confidence': 0.0
            }
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            source = chunk.get('document_id', 'Unknown')
            
            context_parts.append(f"Source {i+1} ({source}): {content}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt based on strategy
        if strategy == "Aggregation":
            system_prompt = """You are a financial data aggregation assistant. 
            Analyze the provided information and create comprehensive summaries, 
            identify patterns, and aggregate data points. Focus on quantitative analysis."""
        
        elif strategy == "Analyse":
            system_prompt = """You are a financial analysis assistant. 
            Provide detailed analysis of the information, including trends, 
            implications, and insights. Support your analysis with specific data points."""
        
        else:  # Standard
            system_prompt = """You are a helpful financial assistant. 
            Provide accurate, concise answers based on the provided information. 
            Cite specific sources when possible."""
        
        user_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context information above."""
        
        # Here you would call your LLM API (Gemini, OpenAI, etc.)
        # For now, we'll return a placeholder
        answer = await self._call_llm_api(system_prompt, user_prompt)
        
        # Calculate confidence based on chunk relevance and content quality
        confidence = self._calculate_answer_confidence(chunks, answer)
        
        return {
            'answer': answer,
            'confidence': confidence
        }
    
    async def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API to generate answer."""
        # This would integrate with your chosen LLM API
        # For demonstration, returning a template response
        
        return f"""Based on the provided information, I can help answer your query. 
        This response would be generated by your chosen LLM (Gemini, OpenAI, etc.) 
        using the system prompt for strategy-specific behavior and the user prompt 
        with context from the retrieved chunks.
        
        The ML-based content analysis ensures that relevant information is retrieved 
        without relying on predefined keywords, making the system adaptable to 
        dynamic content types."""
    
    async def _enhance_with_graph_context(self, answer_result: Dict[str, Any], 
                                        graph_context: List[Dict]) -> Dict[str, Any]:
        """Enhance answer with additional graph relationship context."""
        
        answer = answer_result['answer']
        confidence = answer_result['confidence']
        
        if graph_context:
            # Add relationship insights
            relationship_info = []
            for context in graph_context[:3]:  # Limit to top 3 relationships
                if 'entity_value' in context:
                    relationship_info.append(f"Related: {context['entity_value']}")
            
            if relationship_info:
                enhancement = f"\n\nAdditional Context: {'; '.join(relationship_info)}"
                answer += enhancement
                confidence = min(1.0, confidence + 0.1)  # Slight confidence boost
        
        return {
            'answer': answer,
            'confidence': confidence
        }
    
    def _calculate_answer_confidence(self, chunks: List[Dict[str, Any]], 
                                   answer: str) -> float:
        """Calculate confidence score for generated answer."""
        if not chunks:
            return 0.0
        
        # Base confidence from chunk scores
        chunk_scores = [chunk.get('score', 0.5) for chunk in chunks]
        avg_chunk_score = sum(chunk_scores) / len(chunk_scores)
        
        # Content quality score
        content_quality = min(1.0, len(answer) / 200)  # Prefer substantial answers
        
        # Combine scores
        confidence = (avg_chunk_score * 0.7) + (content_quality * 0.3)
        
        return round(confidence, 3)
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update system performance metrics."""
        self.performance_metrics['total_queries'] += 1
        
        if success:
            self.performance_metrics['successful_queries'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['avg_processing_time']
        total_queries = self.performance_metrics['total_queries']
        
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self.performance_metrics['avg_processing_time'] = new_avg
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        
        # Get Dgraph statistics
        dgraph_stats = self.dgraph_manager.get_statistics()
        
        # Get Qdrant statistics  
        qdrant_stats = self.qdrant_manager.get_collection_statistics()
        
        # Get schema evolution history
        schema_history = self.schema_manager.get_schema_evolution_history()
        
        return {
            'performance_metrics': self.performance_metrics,
            'dgraph_statistics': dgraph_stats,
            'qdrant_statistics': qdrant_stats,
            'schema_evolution': {
                'total_updates': len(schema_history),
                'recent_updates': schema_history[-5:] if schema_history else []
            },
            'system_health': {
                'dgraph_connected': bool(dgraph_stats),
                'qdrant_connected': bool(qdrant_stats),
                'success_rate': (
                    self.performance_metrics['successful_queries'] / 
                    max(1, self.performance_metrics['total_queries'])
                )
            }
        }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Run system optimization routines."""
        
        optimization_results = {
            'schema_optimizations': [],
            'collection_optimizations': [],
            'performance_improvements': []
        }
        
        try:
            # Get schema optimization suggestions
            schema_suggestions = self.schema_manager.suggest_schema_optimizations()
            optimization_results['schema_optimizations'] = schema_suggestions
            
            # Analyze collection performance
            qdrant_stats = self.qdrant_manager.get_collection_statistics()
            if qdrant_stats.get('total_points', 0) > 10000:
                optimization_results['collection_optimizations'].append(
                    "Consider creating specialized collections for different document types"
                )
            
            # Performance analysis
            success_rate = (
                self.performance_metrics['successful_queries'] / 
                max(1, self.performance_metrics['total_queries'])
            )
            
            if success_rate < 0.9:
                optimization_results['performance_improvements'].append(
                    "Consider improving retrieval algorithms or expanding training data"
                )
            
            return optimization_results
        
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            return optimization_results

def test_ml_rag_backend():
    """Test the ML-based RAG backend."""
    import asyncio
    
    backend = MLBasedRAGBackend()
    
    # Test document processing
    print("Testing document processing...")
    
    # Create sample JSON data
    sample_data = [{
        'document_id': 'test_doc_001',
        'pages': [{
            'blocks': [{
                'type': 'text',
                'content': 'Apple Inc. reported revenue of $365.8 billion in 2021, representing a 33% increase from the previous year. The company\'s CEO Tim Cook announced expansion plans for international markets.'
            }]
        }]
    }]
    
    with open('test_ml_extraction.json', 'w') as f:
        json.dump(sample_data, f)
    
    # Process documents
    async def run_test():
        processing_result = await backend.process_documents('test_ml_extraction.json')
        print(f"Document processing result: {processing_result}")
        
        # Test query
        query_result = await backend.answer_query("What was Apple's revenue in 2021?")
        print(f"Query result: {query_result}")
        
        # Get system statistics
        stats = backend.get_system_statistics()
        print(f"System statistics: {stats}")
    
    asyncio.run(run_test())

if __name__ == "__main__":
    test_ml_rag_backend()
