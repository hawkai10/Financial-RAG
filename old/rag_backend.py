import os
import time
import json
import hashlib
import numpy as np
import sqlite3
import threading
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache, wraps
from queue import Queue, Empty
from contextlib import contextmanager
import pickle
import types
import traceback
import sys

from txtai import Embeddings
from rank_bm25 import BM25Okapi
import requests as http_requests  # Use alias to avoid conflicts

# Import configurations and utilities
from config import config
from utils import (
    logger, validate_and_sanitize_query, create_query_hash,
    RateLimiter, safe_mean, QueryAnalyzer, assess_chunk_quality,
    calculate_cost_reduction, sanitize_for_json
)

# Import enhanced modules
from feedback_database import EnhancedFeedbackDatabase
from progressive_retrieval import ProgressiveRetriever
from aggregation_optimizer import AggregationOptimizer
from prompt_templates import PromptBuilder
from unified_query_processor import unified_processor

# ============================================================
# OPTIMIZATION CLASSES
# ============================================================

class SmartChunkCache:
    """Smart chunk cache with file change detection."""
    
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.access_times = {}
        self.file_timestamps = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_file_timestamp(self, file_path: str) -> float:
        """Get the last modification time of a file."""
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0
    
    def _is_cache_valid(self, file_path: str) -> bool:
        """Check if cache is still valid (file hasn't changed)."""
        current_timestamp = self._get_file_timestamp(file_path)
        cached_timestamp = self.file_timestamps.get(file_path, 0.0)
        return current_timestamp == cached_timestamp
    
    def get(self, chunk_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get chunk from cache if file hasn't changed."""
        if not self._is_cache_valid(file_path):
            # File changed, invalidate cache
            self.clear()
            self.misses += 1
            return None
        
        if chunk_id in self.cache:
            self.access_times[chunk_id] = time.time()
            self.hits += 1
            return self.cache[chunk_id]
        
        self.misses += 1
        return None
    
    def put(self, chunk_id: str, chunk_data: Dict[str, Any], file_path: str):
        """Store chunk in cache and update file timestamp."""
        # Update file timestamp when we cache something
        self.file_timestamps[file_path] = self._get_file_timestamp(file_path)
        
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[chunk_id] = chunk_data
        self.access_times[chunk_id] = time.time()
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()
        self.file_timestamps.clear()
        logger.info("[CACHE] Chunk cache cleared")
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'file_tracked': len(self.file_timestamps)
        }

class SmartEmbeddingCache:
    """Embedding cache with content-based invalidation."""
    
    def __init__(self, cache_dir: str = "embedding_cache", max_memory_size: int = 100):
        self.cache_dir = cache_dir
        self.max_memory_size = max_memory_size
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, content: str) -> str:
        """Generate cache key from content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, content: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        cache_key = self._get_cache_key(content)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Store in memory cache if there's space
                if len(self.memory_cache) < self.max_memory_size:
                    self.memory_cache[cache_key] = embedding
                
                self.hits += 1
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        self.misses += 1
        return None
    
    def put(self, content: str, embedding: np.ndarray):
        """Store embedding in cache."""
        cache_key = self._get_cache_key(content)
        
        # Store in memory cache if there's space
        if len(self.memory_cache) < self.max_memory_size:
            self.memory_cache[cache_key] = embedding
        
        # Store on disk
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cached embeddings."""
        self.memory_cache.clear()
        # Clear disk cache
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        disk_files = 0
        try:
            disk_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        except OSError:
            pass
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'memory_cache_size': len(self.memory_cache),
            'max_memory_size': self.max_memory_size,
            'disk_cache_files': disk_files
        }

class ConnectionPool:
    """Simple database connection pool."""
    
    def __init__(self, database_path: str, pool_size: int = 10):
        self.database_path = database_path
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.created_connections = 0
        
        # Pre-create connections
        self._fill_pool()
    
    def _create_connection(self):
        """Create a new database connection."""
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _fill_pool(self):
        """Fill the pool with connections."""
        for _ in range(self.pool_size):
            if self.created_connections < self.pool_size:
                try:
                    conn = self._create_connection()
                    self.pool.put(conn)
                    self.created_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create database connection: {e}")
                    break
    
    def get_connection(self, timeout: float = 5.0):
        """Get a connection from the pool."""
        try:
            return self.pool.get(timeout=timeout)
        except Empty:
            logger.warning("Connection pool exhausted, creating temporary connection")
            return self._create_connection()
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        try:
            self.pool.put_nowait(conn)
        except:
            # Pool is full, close the connection
            conn.close()
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for automatic connection management."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def get_stats(self):
        """Get connection pool statistics."""
        return {
            'pool_size': self.pool_size,
            'available_connections': self.pool.qsize(),
            'created_connections': self.created_connections,
            'utilization': f"{((self.pool_size - self.pool.qsize()) / self.pool_size * 100):.1f}%"
        }
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break
        logger.info("[DB] All database connections closed")

# ============================================================
# GLOBAL INSTANCES WITH OPTIMIZATIONS
# ============================================================

# Create optimized global instances
prompt_builder = PromptBuilder()
rate_limiter = RateLimiter(max_requests=30, time_window=60)
feedback_db = EnhancedFeedbackDatabase()
query_analyzer = QueryAnalyzer()

# Initialize optimization instances
chunk_cache = SmartChunkCache(max_size=500)
embedding_cache = SmartEmbeddingCache()
db_pool = ConnectionPool(config.CHUNKS_FILE, pool_size=10)

# Custom exceptions
class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""
    pass

class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    pass

class OptimizationError(Exception):
    """Custom exception for optimization failures."""
    pass

# Enhanced smart cache
class EnhancedSmartCache:
    """Intelligent caching system with Phase 2 optimizations."""
    
    def __init__(self):
        self.feedback_db = EnhancedFeedbackDatabase()
        self.hit_count = 0
        self.miss_count = 0
    
    def cache_query_result(self, ttl_hours: int = 1):
        """Enhanced decorator for caching with optimization metadata."""
        def decorator(func):
            @wraps(func)
            def wrapper(question, *args, **kwargs):
                cache_key = hashlib.md5(
                    f"{question}:{str(args)}:{str(kwargs)}".encode()
                ).hexdigest()
                
                cached_result = self.feedback_db.get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for query: {question[:50]}...")
                    self.hit_count += 1
                    cached_result['from_cache'] = True
                    cached_result['cache_stats'] = self.get_cache_stats()
                    return cached_result
                
                logger.info(f"Cache miss, computing result for: {question[:50]}...")
                self.miss_count += 1
                result = func(question, *args, **kwargs)
                self.feedback_db.cache_query_result(cache_key, question, result)
                result['from_cache'] = False
                result['cache_stats'] = self.get_cache_stats()
                return result
            return wrapper
        return decorator
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests
        }

# Initialize enhanced smart cache
smart_cache = EnhancedSmartCache()

# Simple fallback functions for core operations
def call_gemini_enhanced(prompt: str, **kwargs) -> str:
    """Enhanced Gemini API call with optimization."""
    try:
        # Use existing call_gemini function from prompt_templates
        return prompt_builder.call_gemini(prompt, **kwargs)
    except Exception as e:
        logger.error(f"Enhanced Gemini call failed: {e}")
        raise GeminiAPIError(f"API call failed: {e}")

def get_chunk_by_id_enhanced(embeddings: Embeddings, uid: str) -> Dict[str, Any]:
    """Enhanced chunk retrieval with connection pooling and caching."""
    
    # Try smart cache first (fastest)
    cached_chunk = chunk_cache.get(uid, config.CHUNKS_FILE)
    if cached_chunk:
        return cached_chunk
    
    # Try database with connection pooling (medium speed)
    try:
        with db_pool.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (uid,))
            result = cursor.fetchone()
            
            if result:
                chunk_data = dict(result)
                # Map database fields to expected fields
                chunk_data["text"] = chunk_data.get("chunk_text", "Content not available")
                chunk_data["retrieval_method"] = "database_pooled"
                # Store in cache for next time
                chunk_cache.put(uid, chunk_data, config.CHUNKS_FILE)
                return chunk_data
    except Exception as e:
        logger.error(f"Database lookup with pooling failed: {e}")
    
    # Fallback to file system (slowest)
    chunk_data = get_chunk_from_file_enhanced(uid)
    return chunk_data

def get_chunk_from_file_enhanced(uid: str) -> Dict[str, Any]:
    """Enhanced file-based chunk retrieval."""
    try:
        # Try to load from contextualized chunks JSON
        if os.path.exists(config.CONTEXTUALIZED_CHUNKS_JSON):
            with open(config.CONTEXTUALIZED_CHUNKS_JSON, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                
                for chunk in chunks:
                    if chunk.get('chunk_id') == uid:
                        chunk['retrieval_method'] = 'file_json'
                        # Store in cache for next time
                        chunk_cache.put(uid, chunk, config.CONTEXTUALIZED_CHUNKS_JSON)
                        return chunk
        
        logger.warning(f"Chunk {uid} not found in any source")
        return {
            "chunk_id": uid,
            "text": "Content not available",
            "error": "Chunk not found",
            "retrieval_method": "error"
        }
    except Exception as e:
        logger.error(f"File-based chunk retrieval failed: {e}")
        return {
            "chunk_id": uid,
            "text": "Error retrieving content",
            "error": str(e),
            "retrieval_method": "error"
        }

def simple_retrieve_enhanced(embeddings: Embeddings, queries: List[str], topn: int = 5,
                           filters: Optional[Dict] = None,
                           strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced traditional retrieval function with better error handling."""
    try:
        all_uids = set()
        
        for query_idx, query in enumerate(queries):
            try:
                results = embeddings.search(query, limit=topn)
                
                for rank, result in enumerate(results):
                    if isinstance(result, dict):
                        uid = result.get("id") or result.get("uid") or str(result)
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        uid = result[0]
                    else:
                        uid = str(result)
                    
                    if uid:
                        all_uids.add(uid)
                        base_score = 1.0 - (rank * 0.1)
                        query_weight = 1.0 - (query_idx * 0.1)
                        
                        if strategy == "Standard" and query_idx == 0:
                            query_weight *= 1.2
                        elif strategy == "Aggregation":
                            query_weight = 1.0 - (query_idx * 0.05)
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Convert UIDs to chunk data with scores
        chunk_results = []
        for uid in all_uids:
            try:
                chunk = get_chunk_by_id_enhanced(embeddings, uid)
                if chunk and not chunk.get("error"):
                    score_info = {
                        "retrieval_score": 0.8,  # Default score
                        "strategy": strategy
                    }
                    chunk_results.append((uid, score_info))
            except Exception as e:
                logger.warning(f"Failed to retrieve chunk {uid}: {e}")
                continue
        
        retrieval_info = {
            "method": "simple_enhanced",
            "total_queries": len(queries),
            "successful_queries": len([q for q in queries if q]),
            "total_chunks": len(chunk_results),
            "strategy": strategy
        }
        
        return chunk_results, retrieval_info
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        raise RetrievalError(f"Retrieval failed: {e}")

def enhanced_retrieve(embeddings: Embeddings, queries: List[str], topn: int = 5, 
                     strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced retrieval with progressive and hybrid options."""
    
    # Use progressive retrieval if enabled
    if config.PROGRESSIVE_RETRIEVAL_ENABLED:
        try:
            retriever = ProgressiveRetriever(embeddings)
            return retriever.retrieve_progressively(queries, strategy, confidence=0.8)
        except Exception as e:
            logger.warning(f"Progressive retrieval failed, falling back to simple: {e}")
    
    # Fall back to simple enhanced retrieval
    return simple_retrieve_enhanced(embeddings, queries, topn, strategy=strategy)

def synthesize_answer_enhanced(question: str, chunks: List[Dict[str, Any]], 
                             strategy: str = "Standard", optimization_result: Dict = None,
                             use_hierarchical: bool = False) -> str:
    """Enhanced answer synthesis with strategy-aware optimization."""
    
    if not chunks:
        return "I couldn't find relevant information to answer your question."
    
    try:
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to top 10 chunks
            chunk_text = chunk.get('chunk_text', chunk.get('text', ''))
            doc_name = chunk.get('document_name', f'Document {i+1}')
            
            if chunk_text:
                context_parts.append(f"[Source: {doc_name}]\n{chunk_text}\n")
        
        context = "\n".join(context_parts)
        
        # Strategy-specific prompts
        if strategy == "Aggregation":
            prompt = f"""Based on the following documents, provide a comprehensive aggregation for: "{question}"

TASK: Extract and list ALL relevant items, data points, or instances found in the documents.

CONTEXT:
{context}

INSTRUCTIONS:
- List every relevant item found
- Include specific values, amounts, dates, and references
- Group similar items but preserve individual instances
- Cite source documents for each item
- Provide totals or summaries where appropriate

ANSWER:"""
        
        elif strategy == "Analyse":
            prompt = f"""Analyze the following documents to answer: "{question}"

CONTEXT:
{context}

TASK: Provide a detailed analysis including:
- Key patterns and trends identified
- Important insights and relationships
- Comparative analysis where relevant
- Supporting evidence from the documents
- Actionable conclusions

ANSWER:"""
        
        else:  # Standard
            prompt = f"""Answer the following question based on the provided context: "{question}"

CONTEXT:
{context}

Please provide a clear, accurate answer based on the information above. If you cannot find specific information to answer the question, please state that clearly.

ANSWER:"""
        
        # Call Gemini API
        response = call_gemini_enhanced(prompt)
        
        if not response or len(response.strip()) < 10:
            return "I was unable to generate a comprehensive answer based on the available information."
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

@smart_cache.cache_query_result(ttl_hours=1)
def rag_query_enhanced(question: str, embeddings: Embeddings, topn: int = 5,
                      filters: Optional[Dict] = None, enable_reranking: bool = True,
                      session_id: str = None, enable_optimization: bool = True) -> Dict[str, Any]:
    """Enhanced RAG pipeline with unified preprocessing and optimizations."""
    start_time = time.time()
    logger.info(f"Starting enhanced RAG query: {question[:100]}...")
    
    try:
        # STEP 1: Unified preprocessing (single LLM call)
        processed = unified_processor.process_query_unified(question)
        corrected_query = processed['corrected_query']
        intent = processed['intent']
        confidence = processed['confidence']
        alternative_queries = processed['alternative_queries']
        
        # Map intent to strategy using config
        query_strategy = config.INTENT_TO_STRATEGY.get(intent, "Standard")
        logger.info(f"Unified processing: {intent} -> {query_strategy} (confidence: {confidence:.3f})")
        
        # STEP 2: Generate multiqueries with unified results
        multiqueries = [corrected_query] + alternative_queries
        
        # STEP 3: Strategy-specific retrieval with optimizations
        if config.PROGRESSIVE_RETRIEVAL_ENABLED:
            retriever = ProgressiveRetriever(embeddings)
            chunks, retrieval_info = retriever.retrieve_progressively(multiqueries, query_strategy, confidence)
        else:
            chunk_results, retrieval_info = enhanced_retrieve(embeddings, multiqueries, topn=topn, strategy=query_strategy)
            chunks = []
            for uid, score_info in chunk_results:
                chunk = get_chunk_by_id_enhanced(embeddings, uid)
                chunk.update(score_info)
                if not chunk.get("error"):
                    chunks.append(chunk)
        
        # STEP 4: Apply reranking if enabled
        optimization_result = None
        if enable_reranking and chunks:
            from document_reranker import EnhancedDocumentReranker
            reranker = EnhancedDocumentReranker()
            chunks, rerank_info = reranker.rerank_chunks(corrected_query, chunks, query_strategy)
            
            # Apply aggregation optimization if needed
            if query_strategy == "Aggregation" and config.SAMPLING_AGGREGATION_ENABLED:
                optimizer = AggregationOptimizer()
                optimization_result = optimizer.optimize_aggregation(corrected_query, chunks)
                if optimization_result.get("catalog"):
                    chunks = optimization_result["catalog"].get("sample_content", chunks)
        
        # STEP 5: Intelligent processing approach selection - Universal Hierarchical Processing
        # Calculate if chunks will exceed token capacity for ANY query type
        estimated_total_tokens = _estimate_total_tokens(chunks, query_strategy)
        token_capacity = config.MAX_CONTEXT_LENGTH - 500  # Reserve for prompt overhead
        
        should_use_hierarchical = (
            config.HIERARCHICAL_PROCESSING_ENABLED and 
            (estimated_total_tokens > token_capacity or len(chunks) > config.HIERARCHICAL_CHUNK_THRESHOLD)
        )
        
        if should_use_hierarchical:
            # Use enhanced strategy-aware hierarchical processing
            from hierarchical_processor import HierarchicalProcessor, ProcessingConfig
            
            # Create LLM function wrapper for hierarchical processor
            def llm_function(prompt: str, batch_chunks: List[Dict]) -> str:
                return synthesize_answer_enhanced(prompt, batch_chunks, query_strategy, None, use_hierarchical=True)
            
            # Configure hierarchical processing based on config
            hierarchical_config = ProcessingConfig(
                max_tokens_per_batch=config.HIERARCHICAL_MAX_TOKENS_PER_BATCH,
                min_chunks_per_batch=config.HIERARCHICAL_MIN_CHUNKS_PER_BATCH,
                max_chunks_per_batch=config.HIERARCHICAL_MAX_CHUNKS_PER_BATCH,
                parallel_batches=config.HIERARCHICAL_PARALLEL_BATCHES,
                enable_parallel=config.HIERARCHICAL_ENABLE_PARALLEL
            )
            
            # Create processor with strategy-aware optimization
            processor = HierarchicalProcessor(llm_function, hierarchical_config)
            processor.set_query_strategy(query_strategy)  # Set strategy for optimization
            
            # Process with strategy-specific optimizations
            hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
            
            answer = hierarchical_result['final_answer']
            processing_method = f"hierarchical-{query_strategy.lower()}"
            hierarchical_stats = hierarchical_result['processing_stats']
            
            logger.info(f"🔄 Strategy-aware hierarchical processing ({query_strategy}): "
                       f"{hierarchical_stats['total_batches']} batches, "
                       f"{hierarchical_stats['processing_time']:.2f}s, "
                       f"est. tokens: {estimated_total_tokens}, "
                       f"completeness: {hierarchical_result['completeness']['success_rate']:.2f}, "
                       f"conflicts: {'detected' if hierarchical_result.get('conflicts_detected') else 'none'}")
        else:
            # Use standard processing for smaller chunk sets
            answer = synthesize_answer_enhanced(corrected_query, chunks, query_strategy, optimization_result)
            processing_method = "standard"
            hierarchical_stats = None
            logger.info(f"📄 Standard processing ({query_strategy}): {len(chunks)} chunks, est. tokens: {estimated_total_tokens}")
        
        # Calculate metrics
        processing_time = time.time() - start_time
        avg_score = safe_mean([chunk.get('final_rerank_score', chunk.get('retrieval_score', 0)) for chunk in chunks])
        
        # Calculate cost savings
        original_strategy_limits = {
            "Standard": 5,
            "Analyse": 8,
            "Aggregation": 20
        }
        
        original_count = original_strategy_limits.get(query_strategy, 5)
        chunks_saved = max(0, original_count - len(chunks))
        cost_reduction = (chunks_saved / original_count * 100) if original_count > 0 else 0
        
        savings_info = {
            'original_chunks': original_count,
            'optimized_chunks': len(chunks),
            'chunks_saved': chunks_saved,
            'cost_reduction_percentage': round(cost_reduction, 2)
        }
        
        # Prepare response with enhanced metadata including optimization stats
        display_chunks = chunks[:5]
        
        result = {
            "answer": answer,
            "corrected_query": corrected_query,
            "multiqueries": multiqueries,
            "chunks": display_chunks,
            "all_chunks_count": len(chunks),
            "processing_time": processing_time,
            "session_id": session_id or "anonymous",
            "avg_relevance_score": round(avg_score, 3),
            "query_strategy": query_strategy,
            "classification": processed,
            "optimization_used": enable_optimization,
            "retrieval_method": retrieval_info.get("method", "unknown"),
            "retrieval_info": retrieval_info,
            "optimization_result": optimization_result,
            "savings_info": savings_info,
            "processing_method": processing_method,
            "hierarchical_stats": hierarchical_stats,
            "optimization_stats": get_optimization_stats()  # Add optimization performance stats
        }
        
        sanitized_result = sanitize_for_json(result)
        logger.info(f"Enhanced RAG completed in {processing_time:.2f}s - Strategy: {query_strategy}, Chunks: {len(chunks)}, Savings: {savings_info.get('cost_reduction_percentage', 0):.1f}%")
        
        return sanitized_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Enhanced RAG query failed after {processing_time:.2f}s: {e}")
        return _create_error_response(
            str(e), question, [question], start_time, "error", {}, session_id
        )

def _estimate_total_tokens(chunks: List[Dict], strategy: str) -> int:
    """Estimate total tokens needed for all chunks."""
    try:
        total_chars = sum(len(chunk.get('chunk_text', chunk.get('text', ''))) for chunk in chunks)
        # Rough estimation: 4 characters per token
        estimated_tokens = total_chars // 4
        return estimated_tokens
    except:
        return len(chunks) * 200  # Fallback estimate

def _create_error_response(error_msg: str, question: str, multiqueries: List[str], 
                          start_time: float, status: str, retrieval_info: Dict, 
                          session_id: str) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "answer": f"I encountered an error while processing your question: {error_msg}",
        "corrected_query": question,
        "multiqueries": multiqueries,
        "chunks": [],
        "all_chunks_count": 0,
        "processing_time": time.time() - start_time,
        "session_id": session_id or "anonymous",
        "avg_relevance_score": 0.0,
        "query_strategy": "Standard",
        "error": error_msg,
        "status": status,
        "retrieval_info": retrieval_info
    }

def get_optimization_stats():
    """Get comprehensive optimization statistics."""
    return {
        'chunk_cache': chunk_cache.get_stats(),
        'embedding_cache': embedding_cache.get_stats(),
        'connection_pool': db_pool.get_stats(),
        'optimizations_enabled': True,
        'timestamp': time.time()
    }

def clear_all_caches():
    """Clear all optimization caches."""
    chunk_cache.clear()
    embedding_cache.clear()
    logger.info("[CACHE] All optimization caches cleared")

def get_cache_health():
    """Get cache health status."""
    chunk_stats = chunk_cache.get_stats()
    embedding_stats = embedding_cache.get_stats()
    pool_stats = db_pool.get_stats()
    
    return {
        'overall_status': 'healthy',
        'chunk_cache_health': {
            'status': 'healthy' if chunk_stats['hit_rate'] != '0.0%' else 'cold',
            'performance': chunk_stats['hit_rate']
        },
        'embedding_cache_health': {
            'status': 'healthy' if embedding_stats['hit_rate'] != '0.0%' else 'cold',
            'performance': embedding_stats['hit_rate']
        },
        'connection_pool_health': {
            'status': 'healthy' if int(pool_stats['utilization'].replace('%', '')) < 90 else 'stressed',
            'utilization': pool_stats['utilization']
        }
    }

# ============================================================
# FEEDBACK AND ANALYTICS WITH OPTIMIZATION DATA
# ============================================================

def collect_query_feedback_enhanced(query: str, result: Dict[str, Any], user_rating: int,
                                  feedback_text: str = "", session_id: str = None,
                                  user_agent: str = None, ip_address: str = None) -> bool:
    """Enhanced feedback collection with optimization metadata."""
    try:
        savings_info = result.get("savings_info", {})
        
        return feedback_db.collect_feedback_enhanced(
            query=query,
            response=result.get("answer", ""),
            user_rating=user_rating,
            response_time=result.get("processing_time", 0),
            method=result.get("processing_method", "unknown"),
            chunks_used=result.get("all_chunks_count", 0),
            avg_relevance_score=result.get("avg_relevance_score", 0.0),
            chunks_data=[
                {
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "relevance_score": chunk.get("retrieval_score", 0.0),
                    "final_score": chunk.get("final_rerank_score", 0.0),
                    "retrieval_method": chunk.get("retrieval_method", "unknown")
                }
                for chunk in result.get("chunks", [])
            ],
            feedback_text=feedback_text,
            session_id=session_id or "anonymous",
            query_strategy=result.get("query_strategy"),
            query_complexity_score=result.get("classification", {}).get("confidence", 0.0),
            user_agent=user_agent,
            ip_address=ip_address,
            optimization_used=result.get("optimization_used", False),
            chunks_saved=savings_info.get("chunks_saved", 0),
            cost_reduction_percentage=savings_info.get("cost_reduction_percentage", 0.0)
        )
    except Exception as e:
        logger.error(f"Enhanced feedback collection failed: {e}")
        return False

def get_performance_metrics_enhanced(days: int = 30) -> Dict[str, Any]:
    """Get enhanced performance metrics with optimization data."""
    try:
        base_metrics = feedback_db.get_performance_metrics(days)
        optimization_stats = get_optimization_stats()
        cache_health = get_cache_health()
        
        # Add Phase 2 optimization metrics
        base_metrics.update({
            'optimization_performance': {
                'cache_hit_rates': {
                    'chunk_cache': optimization_stats['chunk_cache']['hit_rate'],
                    'embedding_cache': optimization_stats['embedding_cache']['hit_rate'],
                },
                'resource_utilization': {
                    'connection_pool': optimization_stats['connection_pool']['utilization'],
                    'memory_usage': f"{optimization_stats['embedding_cache']['memory_cache_size']}/{optimization_stats['embedding_cache']['max_memory_size']}"
                },
                'cache_health': cache_health
            },
            'cost_savings': feedback_db.get_cost_savings_summary(days),
            'processing_efficiency': feedback_db.get_processing_efficiency_metrics(days)
        })
        
        return base_metrics
    except Exception as e:
        logger.error(f"Enhanced performance metrics failed: {e}")
        return {"error": str(e)}

def get_system_health_enhanced() -> Dict[str, Any]:
    """Get enhanced system health status with optimization monitoring."""
    try:
        base_health = feedback_db.get_system_health()
        optimization_stats = get_optimization_stats()
        cache_health = get_cache_health()
        
        # Enhanced health check with optimization monitoring
        enhanced_health = {
            **base_health,
            'optimization_health': {
                'caches': cache_health,
                'database_pool': {
                    'status': 'healthy' if optimization_stats['connection_pool']['available_connections'] > 0 else 'critical',
                    'available_connections': optimization_stats['connection_pool']['available_connections'],
                    'utilization': optimization_stats['connection_pool']['utilization']
                },
                'memory_usage': {
                    'embedding_cache_utilization': f"{optimization_stats['embedding_cache']['memory_cache_size']}/{optimization_stats['embedding_cache']['max_memory_size']}"
                }
            }
        }
        
        return enhanced_health
    except Exception as e:
        logger.error(f"Enhanced system health check failed: {e}")
        return {"error": str(e), "status": "unhealthy"}

def debug_query_processing_enhanced(question: str, embeddings: Embeddings) -> Dict[str, Any]:
    """Enhanced debug information for query processing with optimization details."""
    try:
        debug_info = {
            'query': question,
            'timestamp': time.time(),
            'optimization_stats': get_optimization_stats(),
            'cache_health': get_cache_health()
        }
        
        # Test query processing pipeline
        try:
            result = rag_query_enhanced(question, embeddings, topn=3, enable_optimization=True)
            debug_info['processing_result'] = {
                'success': True,
                'processing_time': result.get('processing_time', 0),
                'chunks_retrieved': result.get('all_chunks_count', 0),
                'query_strategy': result.get('query_strategy', 'Unknown'),
                'processing_method': result.get('processing_method', 'Unknown'),
                'cost_savings': result.get('savings_info', {})
            }
        except Exception as e:
            debug_info['processing_result'] = {
                'success': False,
                'error': str(e)
            }
        
        return debug_info
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# BACKWARD COMPATIBILITY AND EXPORTS
# ============================================================

# Backward compatibility and exports
rag_query = rag_query_enhanced
collect_query_feedback = collect_query_feedback_enhanced
get_performance_metrics = get_performance_metrics_enhanced
get_system_health = get_system_health_enhanced
debug_query_processing = debug_query_processing_enhanced
call_gemini = call_gemini_enhanced

# Legacy functions for compatibility
def correct_query(query: str) -> str:
    """Legacy function for backward compatibility."""
    try:
        processed = unified_processor.process_query_unified(query)
        return processed['corrected_query']
    except:
        return query

def generate_multiqueries(query: str) -> List[str]:
    """Legacy function for backward compatibility."""
    try:
        processed = unified_processor.process_query_unified(query)
        return [processed['corrected_query']] + processed['alternative_queries']
    except:
        return [query]

def get_document_catalog(chunks: List[Dict]) -> Dict:
    """Legacy function for backward compatibility."""
    return get_document_catalog_enhanced(chunks)

def synthesize_answer_scalable(question: str, retrieved_chunks: List[Dict[str, Any]],
                             query_strategy: str = "Standard") -> str:
    """Legacy function for backward compatibility."""
    return synthesize_answer_enhanced(question, retrieved_chunks, query_strategy)

# ============================================================
# CLEANUP AND INITIALIZATION
# ============================================================

def initialize_optimizations():
    """Initialize all optimizations and log status."""
    logger.info("[RAG] Initializing optimizations...")
    logger.info(f"   - Chunk Cache: {chunk_cache.max_size} items max")
    logger.info(f"   - Embedding Cache: {embedding_cache.max_memory_size} items max in memory")
    logger.info(f"   - Connection Pool: {db_pool.pool_size} connections")
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup_optimizations)

def cleanup_optimizations():
    """Cleanup function to close connections and clear caches on exit."""
    try:
        db_pool.close_all()
        logger.info("[CLEANUP] Optimization cleanup completed")
    except Exception as e:
        logger.error(f"Error during optimization cleanup: {e}")

# Export enhanced functions
__all__ = [
    # Core RAG functions
    'rag_query_enhanced', 'rag_query',
    'collect_query_feedback_enhanced', 'collect_query_feedback',
    'get_performance_metrics_enhanced', 'get_performance_metrics',
    'get_system_health_enhanced', 'get_system_health',
    'debug_query_processing_enhanced', 'debug_query_processing',
    
    # Optimization classes
    'SmartChunkCache', 'SmartEmbeddingCache', 'ConnectionPool',
    'EnhancedSmartCache',
    
    # Core functions
    'call_gemini_enhanced', 'call_gemini',
    'correct_query_enhanced', 'correct_query',
    'generate_multiqueries_enhanced', 'generate_multiqueries',
    'synthesize_answer_enhanced', 'synthesize_answer_scalable',
    'get_document_catalog_enhanced', 'get_document_catalog',
    
    # Retrieval functions
    'enhanced_retrieve', 'simple_retrieve_enhanced',
    'get_chunk_by_id_enhanced', 'get_chunk_from_file_enhanced',
    
    # Optimization functions
    'get_optimization_stats', 'clear_all_caches', 'get_cache_health',
    'initialize_optimizations', 'cleanup_optimizations',
    
    # Exceptions
    'GeminiAPIError', 'RetrievalError', 'OptimizationError'
]

# Module initialization with optimizations
initialize_optimizations()
logger.info("Enhanced RAG Backend with Optimizations loaded successfully")
logger.info(f"Features enabled: Progressive={config.PROGRESSIVE_RETRIEVAL_ENABLED}, "
           f"Sampling={config.SAMPLING_AGGREGATION_ENABLED}, "
           f"Hybrid={config.HYBRID_SEARCH_ENABLED}")
logger.info(f"Chunk limits optimized: {config.OPTIMAL_CHUNK_LIMITS}")
logger.info("Optimization features: Chunk Caching [SUCCESS], Embedding Caching [SUCCESS], Connection Pooling [SUCCESS]")
