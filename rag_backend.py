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
        # Check if source file has been modified
        if not self._is_cache_valid(file_path):
            logger.info("[CACHE] Source file changed, clearing chunk cache")
            self.cache.clear()
            self.access_times.clear()
            self.file_timestamps.clear()
            self.misses += 1
            return None
        
        # Normal cache lookup
        if chunk_id in self.cache:
            self.hits += 1
            self.access_times[chunk_id] = time.time()
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
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_size = 1000
        self.hits = 0
        self.misses = 0
    
    def _get_content_hash(self, text: str) -> str:
        """Create hash of text content for change detection."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding using content hash."""
        cache_key = self._get_content_hash(text)
        
        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                    # Store in memory for next time (if space available)
                    if len(self.memory_cache) < self.max_memory_size:
                        self.memory_cache[cache_key] = embedding
                    self.hits += 1
                    return embedding
            except Exception as e:
                logger.error(f"Error loading cached embedding: {e}")
        
        self.misses += 1
        return None
    
    def store_embedding(self, text: str, embedding: List[float]):
        """Store embedding with content-based key."""
        cache_key = self._get_content_hash(text)
        
        # Store in memory (if space available)
        if len(self.memory_cache) < self.max_memory_size:
            self.memory_cache[cache_key] = embedding
        
        # Store on disk for persistence
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Error saving embedding to cache: {e}")
    
    def clear(self):
        """Clear memory cache and optionally disk cache."""
        self.memory_cache.clear()
        logger.info("[CACHE] Embedding memory cache cleared")
    
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
db_pool = ConnectionPool("chunks.db", pool_size=10)

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

smart_cache = EnhancedSmartCache()

# Simplified query classifier
class SimplifiedQueryClassifier:
    """Simplified classifier that trusts LLM results with minimal fallback."""
    
    def __init__(self):
        # Keep only minimal emergency patterns for complete LLM failure
        self.emergency_patterns = {
            'how many': 'Aggregation',
            'count': 'Aggregation',
            'list all': 'Aggregation',
            'analyze': 'Analyse',
            'compare': 'Analyse',
            'what is': 'Standard'
        }
    
    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Simplified classification - mainly for emergency fallback."""
        query_lower = query.lower()
        
        # Only use emergency patterns if absolutely needed
        for pattern, strategy in self.emergency_patterns.items():
            if pattern in query_lower:
                return {
                    "strategy": strategy,
                    "confidence": 0.6,
                    "reasoning": f"Emergency fallback classification: {pattern}",
                    "method": "emergency_keywords"
                }
        
        # Default fallback
        return {
            "strategy": "Standard",
            "confidence": 0.5,
            "reasoning": "Default classification when LLM unavailable",
            "method": "emergency_keywords"
        }

# Hybrid retrieval system
class HybridRetriever:
    """Enhanced hybrid retrieval combining dense and sparse methods."""
    
    def __init__(self, embeddings: Embeddings, chunk_text: List[str]):
        """Initialize hybrid retriever with both dense and sparse components."""
        self.embeddings = embeddings
        self.chunk_text = chunk_text
        
        try:
            logger.info("Initializing BM25 sparse retriever...")
            tokenized_corpus = [doc.lower().split() for doc in chunk_text]
            self.bm25 = BM25Okapi(
                tokenized_corpus,
                k1=config.BM25_K1,
                b=config.BM25_B
            )
            self.has_bm25 = True
            logger.info(f"BM25 initialized with {len(chunk_text)} documents")
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25 = None
            self.has_bm25 = False
    
    def hybrid_search(self, query: str, top_k: int = 20, alpha: float = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Perform hybrid search combining dense and sparse retrieval."""
        if alpha is None:
            alpha = config.HYBRID_ALPHA
        
        try:
            # Dense retrieval using txtai
            logger.info(f"Performing dense retrieval for: {query[:50]}...")
            dense_results = self.embeddings.search(query, top_k * 2)
            dense_scores = {}
            
            for i, result in enumerate(dense_results):
                doc_id = self._extract_doc_id(result)
                if doc_id:
                    score = 1.0 - (i / len(dense_results))
                    dense_scores[doc_id] = score
            
            logger.info(f"Dense retrieval found {len(dense_scores)} results")
            
            # Sparse retrieval using BM25
            sparse_scores = {}
            if self.has_bm25:
                logger.info(f"Performing sparse retrieval (BM25)...")
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                if len(bm25_scores) > 0:
                    max_bm25_score = float(np.max(bm25_scores))
                    for i, score in enumerate(bm25_scores):
                        score_val = float(score)
                        if score_val > 0:
                            doc_id = str(i)
                            normalized_score = score_val / max_bm25_score if max_bm25_score > 0 else 0
                            sparse_scores[doc_id] = normalized_score
                
                logger.info(f"Sparse retrieval found {len(sparse_scores)} results")
            else:
                logger.warning("BM25 not available, using dense-only retrieval")
            
            # Combine scores
            combined_scores = {}
            all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            
            for doc_id in all_doc_ids:
                dense_score = dense_scores.get(doc_id, 0.0)
                sparse_score = sparse_scores.get(doc_id, 0.0)
                combined_score = alpha * dense_score + (1 - alpha) * sparse_score
                combined_scores[doc_id] = combined_score
            
            # Sort by combined score and return top_k
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            logger.info(f"Hybrid retrieval returned {len(sorted_results)} combined results")
            
            enhanced_results = []
            for doc_id, score in sorted_results:
                enhanced_results.append((doc_id, {
                    'combined_score': score,
                    'dense_score': dense_scores.get(doc_id, 0.0),
                    'sparse_score': sparse_scores.get(doc_id, 0.0),
                    'retrieval_method': 'hybrid'
                }))
            
            return sanitize_for_json(enhanced_results)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._fallback_dense_search(query, top_k)
    
    def _extract_doc_id(self, result) -> Optional[str]:
        """Extract document ID from txtai search result."""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("id") or result.get("uid") or str(result)
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            return str(result[0])
        else:
            return str(result)
    
    def _fallback_dense_search(self, query: str, top_k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Fallback to dense-only search if hybrid fails."""
        try:
            results = self.embeddings.search(query, top_k)
            enhanced_results = []
            
            for i, result in enumerate(results):
                doc_id = self._extract_doc_id(result)
                if doc_id:
                    score = 1.0 - (i / len(results))
                    enhanced_results.append((doc_id, {
                        'combined_score': score,
                        'dense_score': score,
                        'sparse_score': 0.0,
                        'retrieval_method': 'dense_only'
                    }))
            
            return sanitize_for_json(enhanced_results)
            
        except Exception as e:
            logger.error(f"Fallback dense search also failed: {e}")
            return []

# Enhanced document reranker
class EnhancedDocumentReranker:
    """Enhanced document reranker with optimization controls."""
    
    def __init__(self):
        """Initialize the reranker with cross-encoder."""
        self.cross_encoder = None
        self.has_cross_encoder = False
        
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.has_cross_encoder = True
            logger.info("Cross-encoder reranker loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.has_cross_encoder = False
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}. Using fallback scoring.")
            self.has_cross_encoder = False
    
    def rerank_chunks(self, query: str, chunks: List[Dict], strategy: str = "Standard",
                     top_k: int = 5) -> Tuple[List[Dict], Dict[str, Any]]:
        """Enhanced reranking with strategy-aware optimization."""
        if not chunks:
            return chunks, {"reranking_applied": False, "reason": "no_chunks"}
        
        rerank_info = {
            "original_count": len(chunks),
            "strategy": strategy,
            "reranking_applied": False,
            "method": "none"
        }
        
        # Strategy-specific reranking decisions
        if strategy == "Aggregation":
            logger.info("Skipping reranking for aggregation query to preserve all results")
            rerank_info.update({
                "reason": "aggregation_skip",
                "final_count": len(chunks)
            })
            return chunks[:top_k], rerank_info
        
        if not self.has_cross_encoder:
            sorted_chunks = sorted(chunks,
                                 key=lambda x: x.get('combined_score', x.get('retrieval_score', 0)),
                                 reverse=True)
            rerank_info.update({
                "reranking_applied": True,
                "method": "retrieval_score_sort",
                "final_count": min(top_k, len(sorted_chunks))
            })
            return sorted_chunks[:top_k], rerank_info
        
        # Apply cross-encoder reranking
        try:
            reranked_chunks = self._cross_encoder_rerank(query, chunks)
            final_chunks = reranked_chunks[:top_k]
            
            rerank_info.update({
                "reranking_applied": True,
                "method": "cross_encoder",
                "final_count": len(final_chunks),
                "cross_encoder_available": True
            })
            
            logger.info(f"Cross-encoder reranking: {len(chunks)} -> {len(final_chunks)} chunks")
            return sanitize_for_json(final_chunks), sanitize_for_json(rerank_info)
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            sorted_chunks = sorted(chunks,
                                 key=lambda x: x.get('combined_score', x.get('retrieval_score', 0)),
                                 reverse=True)
            rerank_info.update({
                "reranking_applied": True,
                "method": "fallback_sort",
                "error": str(e),
                "final_count": min(top_k, len(sorted_chunks))
            })
            return sorted_chunks[:top_k], rerank_info
    
    def _cross_encoder_rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Apply cross-encoder reranking with optimization."""
        if not chunks:
            return chunks
        
        pairs = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if len(text) > 512:
                text = text[:512]
            pairs.append([query, text])
        
        cross_scores = self.cross_encoder.predict(pairs)
        
        for i, chunk in enumerate(chunks):
            chunk["cross_encoder_score"] = float(cross_scores[i])
            retrieval_score = chunk.get('combined_score', chunk.get('retrieval_score', 0.0))
            chunk["final_rerank_score"] = float((cross_scores[i] * 0.7) + (retrieval_score * 0.3))
        
        reranked_chunks = sorted(chunks, key=lambda x: x["final_rerank_score"], reverse=True)
        return reranked_chunks

# ============================================================
# CORE API FUNCTIONS WITH OPTIMIZATIONS
# ============================================================

def call_gemini_enhanced(prompt: str, max_retries: int = None, strategy: str = "Standard") -> str:
    """Enhanced Gemini API call with strategy-specific optimizations."""
    max_retries = max_retries or config.MAX_RETRIES
    
    if not rate_limiter.is_allowed("gemini_api"):
        raise GeminiAPIError("Rate limit exceeded. Please try again later.")
    
    # Strategy-specific prompt optimization
    if strategy == "Standard":
        prompt = f"{prompt}\n\nPlease provide a direct, comprehensive answer."
    elif strategy == "Aggregation":
        prompt = f"{prompt}\n\nPlease provide exact counts and list items clearly."
    elif strategy == "Analyse":
        prompt = f"{prompt}\n\nPlease provide a thorough analysis with insights and patterns."
    
    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            params = {"key": config.GEMINI_API_KEY}
            data = {
                "contents": [
                    {"role": "user", "parts": [{"text": prompt}]}
                ]
            }
            
            # LLM LOGGING DISABLED: Detailed prompt logging for debugging
            # log_llm_interaction(
            #     phase="PROMPT_TO_GEMINI",
            #     content=prompt,
            #     strategy=strategy,
            #     attempt=f"{attempt + 1}/{max_retries}",
            #     character_count=len(prompt),
            #     word_count=len(prompt.split())
            # )
            
            # Use aliased import to avoid module confusion
            response = http_requests.post(
                config.GEMINI_API_URL,
                headers=headers,
                params=params,
                json=data,
                timeout=config.REQUEST_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            if ("candidates" not in result or
                not result["candidates"] or
                "content" not in result["candidates"][0] or
                "parts" not in result["candidates"][0]["content"]):
                raise ValueError("Invalid response structure from Gemini API")
            
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # LLM LOGGING DISABLED: Detailed response logging for debugging
            # log_llm_interaction(
            #     phase="RESPONSE_FROM_GEMINI",
            #     content=content,
            #     strategy=strategy,
            #     character_count=len(content),
            #     word_count=len(content.split()),
            #     status="success"
            # )
            
            return content
            
        except http_requests.exceptions.RequestException as e:
            logger.warning(f"Gemini API request failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise GeminiAPIError(f"Gemini API failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Invalid Gemini API response format: {e}")
            raise GeminiAPIError(f"Unexpected response format from Gemini API: {e}")

def correct_query_enhanced(query: str) -> Tuple[str, Dict[str, Any]]:
    """Enhanced query correction with metadata tracking."""
    correction_info = {
        "correction_applied": False,
        "original_query": query,
        "method": "none"
    }
    
    try:
        sanitized_query = validate_and_sanitize_query(query)
        
        if (len(sanitized_query.split()) <= 2 and
            sanitized_query.replace(' ', '').replace('?', '').isalnum()):
            correction_info["method"] = "validation_only"
            return sanitized_query, correction_info
        
        prompt = (
            f"Correct the spelling and grammar of this search query for business documents. "
            f"Keep the corrected query concise and focused. Only fix obvious errors: {sanitized_query}"
        )
        
        corrected = call_gemini_enhanced(prompt, strategy="Standard").strip()
        
        if corrected.lower() != sanitized_query.lower():
            correction_info.update({
                "correction_applied": True,
                "method": "llm_correction",
                "corrected_query": corrected
            })
            logger.info(f"Query corrected: '{query}' -> '{corrected}'")
            return corrected, correction_info
        else:
            correction_info["method"] = "no_correction_needed"
            return sanitized_query, correction_info
            
    except Exception as e:
        logger.warning(f"Query correction failed: {e}. Using original query.")
        correction_info.update({
            "method": "error_fallback",
            "error": str(e)
        })
        return query, correction_info

@lru_cache(maxsize=100)
def generate_multiqueries_enhanced(query_hash: str, original_query: str,
                                 strategy: str = "Standard") -> List[str]:
    """Enhanced multiquery generation with strategy-specific optimization."""
    if strategy == "Standard":
        variations = [original_query]
        if "what is" in original_query.lower():
            variations.append(original_query.replace("what is", "show me"))
        return variations[:2]
    
    elif strategy == "Aggregation":
        prompt = (
            f"Given this counting/listing query: '{original_query}', "
            f"generate 2 natural language variations that would help find ALL relevant documents. "
            f"Focus on different ways to express the same counting/listing need. "
            f"Return each query on a new line without numbering."
        )
    else:  # Standard and Analyse
        prompt = (
            f"Given the user query about business documents: '{original_query}', "
            f"generate 2 diverse, rephrased queries that could retrieve relevant information "
            f"from a database containing invoices, contracts, and financial documents. "
            f"Return each query on a new line without numbering."
        )
    
    try:
        response = call_gemini_enhanced(prompt, strategy=strategy)
        queries = [line.strip() for line in response.splitlines() if line.strip()]
        
        natural_queries = []
        for query in queries:
            if not any(sql_word in query.upper() for sql_word in ['SELECT', 'COUNT(', 'WHERE', 'FROM']):
                if len(query) > 5 and query != original_query:
                    natural_queries.append(query)
        
        if original_query not in natural_queries:
            natural_queries.insert(0, original_query)
        
        max_queries = 2 if strategy == "Standard" else 3
        final_queries = natural_queries[:max_queries]
        
        logger.info(f"Generated {len(final_queries)} enhanced multiqueries for {strategy}")
        return final_queries
        
    except Exception as e:
        logger.warning(f"Enhanced multiquery generation failed: {e}. Using original query only.")
        return [original_query]

# ============================================================
# OPTIMIZED RETRIEVAL FUNCTIONS
# ============================================================

def load_chunk_text_for_bm25() -> List[str]:
    """Load chunk text for BM25 initialization."""
    try:
        with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        chunk_text = []
        for chunk in chunks:
            text = chunk.get("chunk_text", "")
            if text.strip():
                chunk_text.append(text)
        
        logger.info(f"Loaded {len(chunk_text)} chunk text for BM25")
        return chunk_text
        
    except Exception as e:
        logger.error(f"Failed to load chunk text for BM25: {e}")
        return []

def get_embeddings_cached(texts: List[str], embeddings_model) -> List[List[float]]:
    """Get embeddings with smart caching."""
    if not texts:
        return []
    
    results = []
    texts_to_compute = []
    text_indices = []
    
    # Check cache for each text
    for i, text in enumerate(texts):
        cached_embedding = embedding_cache.get_embedding(text)
        if cached_embedding:
            results.append(cached_embedding)
        else:
            results.append(None)  # Placeholder
            texts_to_compute.append(text)
            text_indices.append(i)
    
    # Compute missing embeddings
    if texts_to_compute:
        logger.info(f"[EMBED] Computing {len(texts_to_compute)} new embeddings")
        try:
            # Use your existing embedding model
            new_embeddings = embeddings_model.embed(texts_to_compute)
            
            # Store new embeddings in cache and results
            for text, embedding, idx in zip(texts_to_compute, new_embeddings, text_indices):
                embedding_cache.store_embedding(text, embedding)
                results[idx] = embedding
                
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            # Fill with empty embeddings for failed computations
            for idx in text_indices:
                if results[idx] is None:
                    results[idx] = []
    
    return results

def get_chunk_from_file_enhanced(uid: str) -> Dict[str, Any]:
    """Enhanced chunk retrieval with smart caching."""
    
    # Try smart cache first (fastest)
    cached_chunk = chunk_cache.get(uid, config.CHUNKS_FILE)
    if cached_chunk:
        logger.info(f"[SUCCESS] Cache HIT for chunk {uid}")
        return cached_chunk
    
    # Cache miss - load from file
    logger.info(f"[CACHE] Cache MISS for chunk {uid}, loading from disk")
    
    try:
        with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id == uid:
                chunk_data = {
                    "text": chunk.get("chunk_text", "Content not available"),
                    "chunk_id": uid,
                    "document_name": chunk.get("document_name", "Unknown"),
                    "context": chunk.get("context", ""),
                    "is_table": chunk.get("is_table", False),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "start_token": chunk.get("start_token", 0),
                    "end_token": chunk.get("end_token", 0),
                    "num_tokens": chunk.get("num_tokens", 0),
                    "num_rows": chunk.get("num_rows"),
                    "num_cols": chunk.get("num_cols"),
                    "retrieval_method": "file_lookup_cached"
                }
                
                # Store in cache for future use
                chunk_cache.put(uid, chunk_data, config.CHUNKS_FILE)
                return chunk_data
        
        # Chunk not found
        error_data = {
            "text": f"Chunk {uid} not found",
            "chunk_id": uid,
            "document_name": "Unknown",
            "context": "",
            "is_table": False,
            "chunk_index": 0,
            "error": "Not found in source file"
        }
        return error_data
        
    except Exception as e:
        logger.error(f"Enhanced file lookup failed for {uid}: {e}")
        return {
            "text": "Error loading chunk content",
            "chunk_id": uid,
            "error": str(e),
            "document_name": "Error",
            "context": "",
            "is_table": False,
            "chunk_index": 0
        }

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

def enhanced_retrieve(embeddings: Embeddings, queries: List[str], topn: int = 5,
                     filters: Optional[Dict] = None, use_hybrid: bool = True,
                     strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced retrieval function with hybrid search support."""
    retrieval_info = {
        "method": "unknown",
        "queries_processed": 0,
        "total_results": 0,
        "strategy": strategy
    }
    
    if use_hybrid and config.HYBRID_SEARCH_ENABLED:
        chunk_text = load_chunk_text_for_bm25()
        if chunk_text:
            try:
                hybrid_retriever = HybridRetriever(embeddings, chunk_text)
                all_results = {}
                
                for query_idx, query in enumerate(queries):
                    logger.info(f"Hybrid search for query {query_idx + 1}: {query}")
                    results = hybrid_retriever.hybrid_search(query, topn * 2)
                    retrieval_info["queries_processed"] += 1
                    
                    for doc_id, score_info in results:
                        if doc_id not in all_results or score_info['combined_score'] > all_results[doc_id]['combined_score']:
                            score_info['source_query'] = query
                            score_info['query_index'] = query_idx
                            all_results[doc_id] = score_info
                
                sorted_results = sorted(
                    all_results.items(),
                    key=lambda x: x[1]['combined_score'],
                    reverse=True
                )[:topn]
                
                retrieval_info.update({
                    "method": "hybrid",
                    "total_results": len(sorted_results)
                })
                
                logger.info(f"Hybrid retrieval returned {len(sorted_results)} results")
                return sanitize_for_json(sorted_results), sanitize_for_json(retrieval_info)
                
            except Exception as e:
                logger.error(f"Hybrid retrieval failed: {e}, falling back to dense search")
                return simple_retrieve_enhanced(embeddings, queries, topn, filters, strategy)
    
    # Fallback to simple retrieval
    return simple_retrieve_enhanced(embeddings, queries, topn, filters, strategy)

def simple_retrieve_enhanced(embeddings: Embeddings, queries: List[str], topn: int = 5,
                           filters: Optional[Dict] = None,
                           strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced traditional retrieval function with better error handling."""
    all_uids = set()
    uid_scores = {}
    retrieval_info = {
        "method": "dense_only",
        "queries_processed": 0,
        "total_results": 0,
        "strategy": strategy,
        "filters_applied": bool(filters)
    }
    
    try:
        for query_idx, query in enumerate(queries):
            try:
                logger.info(f"Dense search for: {query}")
                results = embeddings.search(query, topn)
                retrieval_info["queries_processed"] += 1
                
                for rank, result in enumerate(results):
                    uid = None
                    if isinstance(result, str):
                        uid = result
                    elif isinstance(result, dict):
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
                        
                        final_score = base_score * query_weight
                        
                        if uid not in uid_scores or final_score > uid_scores[uid]['combined_score']:
                            uid_scores[uid] = {
                                'combined_score': final_score,
                                'dense_score': final_score,
                                'sparse_score': 0.0,
                                'retrieval_method': 'dense_only',
                                'source_query': query,
                                'query_index': query_idx
                            }
                
                logger.info(f"Found {len(results)} results for query: {query}")
                retrieval_info["total_results"] += len(results)
                
            except Exception as e:
                logger.warning(f"Dense search failed for query '{query}': {e}")
                continue
        
        scored_results = [(uid, uid_scores[uid]) for uid in all_uids]
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for uid, score_info in scored_results:
                try:
                    chunk_data = get_chunk_from_file_enhanced(uid)
                    if chunk_data and not chunk_data.get("error"):
                        filter_match = all(
                            chunk_data.get(k) == v for k, v in filters.items()
                        )
                        if filter_match:
                            filtered_results.append((uid, score_info))
                except Exception as e:
                    logger.warning(f"Filter check failed for {uid}: {e}")
                    continue
            
            scored_results = filtered_results
            retrieval_info["filtered_count"] = len(scored_results)
        
        scored_results = sorted(scored_results, key=lambda x: x[1]['combined_score'], reverse=True)
        final_results = scored_results[:topn]
        retrieval_info["final_count"] = len(final_results)
        
        logger.info(f"Dense retrieval returned {len(final_results)} results")
        return sanitize_for_json(final_results), sanitize_for_json(retrieval_info)
        
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        retrieval_info["error"] = str(e)
        return [], retrieval_info

def get_document_catalog_enhanced(chunks: List[Dict]) -> Dict[str, Any]:
    """Create enhanced document catalog from chunks."""
    if not chunks:
        return {"documents": [], "total_count": 0, "document_types": {}}
    
    documents = []
    doc_types = {"tables": 0, "text": 0}
    
    for chunk in chunks:
        doc_name = chunk.get('document_name', 'Unknown')
        if isinstance(doc_name, str):
            doc_name = doc_name.split('/')[-1]
        
        doc_type = "table" if chunk.get('is_table') else "text"
        doc_types[f"{doc_type}s"] += 1
        
        documents.append({
            "name": doc_name,
            "type": doc_type,
            "relevance": chunk.get('final_rerank_score', chunk.get('retrieval_score', 0))
        })
    
    return {
        "documents": documents,
        "total_count": len(chunks),
        "document_types": doc_types
    }

def synthesize_answer_enhanced(question: str, chunks: List[Dict],
                             strategy: str = "Standard",
                             optimization_result: Optional[Dict] = None,
                             use_hierarchical: bool = False) -> str:
    """Enhanced answer synthesis using centralized prompt templates."""
    
    if not chunks:
        return "No relevant information found in the documents."
    
    try:
        # Handle hierarchical processing mode (for batch processing)
        if use_hierarchical:
            # In hierarchical mode, question is actually the batch prompt
            # and chunks are the batch chunks - use simplified processing
            prompt = question  # Already formatted prompt from hierarchical processor
        else:
            # Handle aggregation optimization results
            if optimization_result and optimization_result.get("method") == "statistical_sampling":
                context_chunks = chunks.copy()
                sampling_note = (
                    f"\n\nNote: This analysis is based on a representative sample of "
                    f"{optimization_result['sample_size']} out of {optimization_result['total_population']} "
                    f"documents with {optimization_result['confidence_level']*100:.0f}% confidence."
                )
            else:
                context_chunks = chunks
                sampling_note = ""
            
            # Build prompt using centralized system
            prompt = prompt_builder.build_prompt(strategy, question, context_chunks)
            
            # Add sampling note for aggregation
            if sampling_note:
                prompt += sampling_note
        
        # Call Gemini with strategy-specific optimization
        answer = call_gemini_enhanced(prompt, strategy=strategy)
        
        # LLM LOGGING DISABLED: Answer synthesis result
        # log_llm_interaction(
        #     phase="ANSWER_SYNTHESIS_RESULT",
        #     content=answer,
        #     strategy=strategy,
        #     optimization_applied=bool(optimization_result),
        #     character_count=len(answer),
        #     word_count=len(answer.split())
        # )
        
        # Post-process answer based on strategy
        if strategy == "Aggregation" and optimization_result:
            answer = _enhance_aggregation_answer(answer, optimization_result)
            # LLM LOGGING DISABLED: Answer after enhancement
            # log_llm_interaction(
            #     phase="ANSWER_AFTER_ENHANCEMENT",
            #     content=answer,
            #     strategy=strategy,
            #     enhancement_type="aggregation",
            #     character_count=len(answer),
            #     word_count=len(answer.split())
            # )
        
        return answer.strip()
        
    except Exception as e:
        # Enhanced debugging - Show exactly which module is not callable
        logger.error(f"Enhanced answer synthesis failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        
        # Full traceback
        import traceback
        traceback.print_exc()

        # Check if this is a NameError (variable not found)
        if isinstance(e, NameError):
            logger.error(f"NameError detected: {e}")
            # This is likely a variable name issue
            error_str = str(e)
            if "'" in error_str:
                var_name = error_str.split("'")[1]
                return f"I encountered a system error while processing your question. The variable '{var_name}' was not found. Please try rephrasing your query."
            else:
                return f"I encountered a system error while processing your question. A variable was not found. Please try rephrasing your query."
        
        return f"I encountered an error while processing your question: {str(e)}. Please try again or try rephrasing your query."

def _enhance_aggregation_answer(answer: str, optimization_result: Dict) -> str:
    """Enhance aggregation answers with optimization metadata."""
    if optimization_result.get("sampling_used"):
        confidence = optimization_result.get("confidence_level", 0.85)
        sample_size = optimization_result.get("sample_size", 0)
        total_pop = optimization_result.get("total_population", 0)
        
        enhancement = (
            f"\n\n📊 **Statistical Analysis Summary:**\n"
            f"- Sample analyzed: {sample_size} of {total_pop} documents\n"
            f"- Confidence level: {confidence*100:.0f}%\n"
            f"- Method: Stratified sampling with quality-based selection"
        )
        
        return answer + enhancement
    
    return answer

def _estimate_total_tokens(chunks: List[Dict], strategy: str) -> int:
    """Estimate total tokens needed for all chunks plus prompt overhead."""
    total_chars = 0
    
    # Calculate characters in all chunk content
    for chunk in chunks:
        text = chunk.get('text', '') or chunk.get('chunk_text', '')
        doc_name = chunk.get('document_name', 'Unknown')
        # Account for chunk formatting: "Document X (filename):\ntext\n---\n"
        chunk_overhead = len(f"Document X ({doc_name}):\n\n---\n")
        total_chars += len(text) + chunk_overhead
    
    # Estimate tokens (rough conversion: 4 chars per token)
    content_tokens = total_chars // 4
    
    # Add strategy-specific prompt overhead
    prompt_overhead = {
        "Standard": 200,      # Simple prompt
        "Analyse": 300,       # More complex analysis prompt  
        "Aggregation": 250    # Aggregation-specific prompt
    }
    
    total_tokens = content_tokens + prompt_overhead.get(strategy, 200)
    
    logger.debug(f"Token estimation: {len(chunks)} chunks, {total_chars} chars, "
                f"~{total_tokens} tokens (strategy: {strategy})")
    
    return total_tokens

# ============================================================
# MAIN RAG FUNCTION WITH OPTIMIZATIONS
# ============================================================

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

def _create_error_response(error_msg: str, query: str, queries: List[str],
                          start_time: float, strategy: str, classification: Dict,
                          session_id: str) -> Dict[str, Any]:
    """Create standardized error response."""
    error_response = {
        "answer": f"I encountered an error while processing your question: {error_msg}. Please try again or try rephrasing your query.",
        "corrected_query": query,
        "multiqueries": queries,
        "chunks": [],
        "processing_time": time.time() - start_time,
        "error": error_msg,
        "query_strategy": strategy,
        "classification": classification,
        "session_id": session_id,
        "optimization_used": False,
        "savings_info": {"cost_reduction_percentage": 0}
    }
    
    return sanitize_for_json(error_response)

# ============================================================
# OPTIMIZATION MONITORING AND MANAGEMENT
# ============================================================

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
        optimization_result = result.get("optimization_result", {})
        optimization_stats = result.get("optimization_stats", {})
        
        feedback_data = {
            "query": query,
            "answer": result.get("answer", ""),
            "rating": user_rating,
            "retrieval_score": result.get("avg_relevance_score", 0.0),
            "processing_time": result.get("processing_time", 0.0),
            "chunks_used": len(result.get("chunks", [])),
            "chunks_data": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_name": chunk.get("document_name"),
                    "is_table": chunk.get("is_table", False),
                    "final_score": chunk.get("final_rerank_score", 0.0),
                    "retrieval_method": chunk.get("retrieval_method", "unknown")
                }
                for chunk in result.get("chunks", [])
            ],
            "feedback_text": feedback_text,
            "session_id": session_id or "anonymous",
            "query_strategy": result.get("query_strategy"),
            "query_complexity_score": result.get("classification", {}).get("confidence", 0.0),
            "user_agent": user_agent,
            "ip_address": ip_address,
            "optimization_used": result.get("optimization_used", False),
            "chunks_saved": savings_info.get("chunks_saved", 0),
            "cost_reduction_percentage": savings_info.get("cost_reduction_percentage", 0.0),
            "optimization_performance": optimization_stats
        }
        
        sanitized_feedback_data = sanitize_for_json(feedback_data)
        feedback_db.store_feedback(sanitized_feedback_data)
        
        logger.info(f"Enhanced feedback collected: {user_rating}/5 stars for query: {query[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"Failed to collect enhanced feedback: {e}")
        return False

def get_performance_metrics_enhanced(days: int = 30) -> Dict[str, Any]:
    """Get enhanced performance metrics with optimization data."""
    try:
        standard_metrics = feedback_db.get_performance_metrics(days)
        optimization_analytics = feedback_db.get_optimization_analytics(days)
        cache_stats = smart_cache.get_cache_stats()
        optimization_stats = get_optimization_stats()
        cache_health = get_cache_health()
        
        enhanced_metrics = {
            **standard_metrics,
            "optimization_analytics": optimization_analytics,
            "rate_limiter_stats": rate_limiter.get_usage_stats(),
            "cache_performance": cache_stats,
            "optimization_performance": optimization_stats,
            "cache_health": cache_health
        }
        
        return sanitize_for_json(enhanced_metrics)
        
    except Exception as e:
        logger.error(f"Failed to get enhanced performance metrics: {e}")
        return {"error": str(e)}

def get_system_health_enhanced() -> Dict[str, Any]:
    """Get enhanced system health status with optimization monitoring."""
    try:
        optimization_stats = get_optimization_stats()
        cache_health = get_cache_health()
        
        health_info = {
            "status": "healthy",
            "components": {
                "embeddings": True,
                "gemini_api": bool(config.GEMINI_API_KEY and config.GEMINI_API_KEY != "your_gemini_api_key_here"),
                "database": True,
                "rate_limiter": True,
                "cache": True,
                "optimizations": True
            },
            "optimization_features": {
                "progressive_retrieval": config.PROGRESSIVE_RETRIEVAL_ENABLED,
                "sampling_aggregation": config.SAMPLING_AGGREGATION_ENABLED,
                "hybrid_search": config.HYBRID_SEARCH_ENABLED,
                "chunk_caching": True,
                "embedding_caching": True,
                "connection_pooling": True
            },
            "cache_stats": smart_cache.get_cache_stats(),
            "rate_limiter_stats": rate_limiter.get_usage_stats(),
            "optimization_stats": optimization_stats,
            "cache_health": cache_health
        }
        
        # Check if any critical components are down
        if not all(health_info["components"].values()):
            health_info["status"] = "degraded"
        
        # Check cache health
        if cache_health['overall_status'] != 'healthy':
            health_info["status"] = "degraded"
            health_info["warnings"] = ["Some optimization caches are underperforming"]
        
        return sanitize_for_json(health_info)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {},
            "optimization_features": {}
        }

def debug_query_processing_enhanced(question: str, embeddings: Embeddings) -> Dict[str, Any]:
    """Enhanced debug information for query processing with optimization details."""
    debug_info = {
        "original_query": question,
        "preprocessing": {},
        "retrieval": {},
        "synthesis": {},
        "optimizations": {},
        "errors": []
    }
    
    try:
        # Test preprocessing
        processed = unified_processor.process_query_unified(question)
        debug_info["preprocessing"] = {
            "corrected_query": processed.get("corrected_query"),
            "intent": processed.get("intent"),
            "confidence": processed.get("confidence"),
            "alternative_queries": processed.get("alternative_queries"),
            "method": processed.get("method", "unknown")
        }
        
        # Test retrieval
        multiqueries = [processed["corrected_query"]] + processed.get("alternative_queries", [])
        strategy = config.INTENT_TO_STRATEGY.get(processed["intent"], "Standard")
        chunk_results, retrieval_info = enhanced_retrieve(embeddings, multiqueries, topn=3, strategy=strategy)
        
        debug_info["retrieval"] = {
            "strategy": strategy,
            "multiqueries": multiqueries,
            "chunks_found": len(chunk_results),
            "retrieval_method": retrieval_info.get("method"),
            "queries_processed": retrieval_info.get("queries_processed")
        }
        
        # Test synthesis
        if chunk_results:
            chunks = []
            for uid, score_info in chunk_results[:2]:  # Test with 2 chunks
                chunk = get_chunk_by_id_enhanced(embeddings, uid)
                chunk.update(score_info)
                if not chunk.get("error"):
                    chunks.append(chunk)
            
            if chunks:
                test_answer = synthesize_answer_enhanced(processed["corrected_query"], chunks, strategy)
                debug_info["synthesis"] = {
                    "chunks_used": len(chunks),
                    "answer_length": len(test_answer),
                    "strategy": strategy,
                    "success": True
                }
            else:
                debug_info["synthesis"] = {"success": False, "reason": "No valid chunks"}
        else:
            debug_info["synthesis"] = {"success": False, "reason": "No chunks retrieved"}
        
        # Add optimization debug info
        debug_info["optimizations"] = get_optimization_stats()
        debug_info["overall_status"] = "success"
        
    except Exception as e:
        debug_info["errors"].append(str(e))
        debug_info["overall_status"] = "failed"
        logger.error(f"Debug query processing failed: {e}")
    
    return sanitize_for_json(debug_info)

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
    corrected, _ = correct_query_enhanced(query)
    return corrected

def generate_multiqueries(query: str) -> List[str]:
    """Legacy function for backward compatibility."""
    query_hash = create_query_hash(query)
    return generate_multiqueries_enhanced(query_hash, query)

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
    'SimplifiedQueryClassifier', 'HybridRetriever', 'EnhancedDocumentReranker', 'EnhancedSmartCache',
    
    # Core functions
    'call_gemini_enhanced', 'call_gemini',
    'correct_query_enhanced', 'correct_query',
    'generate_multiqueries_enhanced', 'generate_multiqueries',
    'synthesize_answer_enhanced', 'synthesize_answer_scalable',
    'get_document_catalog_enhanced', 'get_document_catalog',
    
    # Retrieval functions
    'enhanced_retrieve', 'simple_retrieve_enhanced',
    'get_chunk_by_id_enhanced', 'get_chunk_from_file_enhanced',
    'get_embeddings_cached',
    
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
