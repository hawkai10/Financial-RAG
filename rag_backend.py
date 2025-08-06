import os
import time
import json
import hashlib
import numpy as np
import re
import threading
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache, wraps
from contextlib import contextmanager
import pickle
import types
import traceback
import sys
import asyncio
import aiohttp
import aiosqlite

from txtai import Embeddings
from rank_bm25 import BM25Okapi

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
from chunk_manager import ChunkManager

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

class AsyncConnectionPool:
    """Asynchronous database connection pool."""
    
    def __init__(self, database_path: str, pool_size: int = 10):
        self.database_path = database_path
        self.pool_size = pool_size
        self._pool = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self.created_connections = 0

    async def _create_connection(self):
        """Create a new aiosqlite database connection."""
        conn = await aiosqlite.connect(self.database_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def _fill_pool(self):
        """Fill the pool with connections."""
        async with self._lock:
            while self.created_connections < self.pool_size and not self._pool.full():
                try:
                    conn = await self._create_connection()
                    await self._pool.put(conn)
                    self.created_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create async database connection: {e}")
                    break

    async def get_connection(self, timeout: float = 5.0):
        """Get a connection from the pool."""
        if self._pool.empty() and self.created_connections < self.pool_size:
            await self._fill_pool()
            
        try:
            return await asyncio.wait_for(self._pool.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Async connection pool exhausted, creating temporary connection")
            return await self._create_connection()

    async def return_connection(self, conn):
        """Return a connection to the pool."""
        if self._pool.full():
            await conn.close()
        else:
            await self._pool.put(conn)

    async def __aenter__(self):
        """Async context manager entry."""
        self._current_conn = await self.get_connection()
        return self._current_conn
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, '_current_conn'):
            await self.return_connection(self._current_conn)
            delattr(self, '_current_conn')

    def get_connection_context(self):
        """Get async context manager for connection management."""
        return self

    def get_stats(self):
        """Get connection pool statistics."""
        return {
            'pool_size': self.pool_size,
            'available_connections': self._pool.qsize(),
            'created_connections': self.created_connections,
            'utilization': f"{((self.pool_size - self._pool.qsize()) / self.pool_size * 100) if self.pool_size > 0 else 0:.1f}%"
        }

    async def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break
        logger.info("[DB] All async database connections closed")

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
db_pool = AsyncConnectionPool(config.CHUNKS_FILE, pool_size=20) # Increased pool size

# Initialize ChunkManager
try:
    chunk_manager = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
    logger.info("ChunkManager initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChunkManager: {e}", exc_info=True)
    chunk_manager = None

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
            async def wrapper(question, *args, **kwargs):
                # Check if we're in a Flask/threading context to avoid database threading issues
                import threading
                current_thread = threading.current_thread()
                is_main_thread = current_thread is threading.main_thread()
                
                # Skip caching if we're in a Flask worker thread to avoid threading issues
                if not is_main_thread or 'werkzeug' in current_thread.name.lower():
                    logger.info(f"Skipping cache in thread context for: {question[:50]}...")
                    result = await func(question, *args, **kwargs)
                    return result
                
                cache_key = hashlib.md5(
                    f"{question}:{str(args)}:{str(kwargs)}".encode()
                ).hexdigest()
                
                try:
                    cached_result = await self.feedback_db.get_cached_result(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for query: {question[:50]}...")
                        self.hit_count += 1
                        cached_result['from_cache'] = True
                        cached_result['cache_stats'] = self.get_cache_stats()
                        return cached_result
                except Exception as cache_error:
                    logger.warning(f"Cache access failed: {cache_error}, proceeding without cache")
                
                logger.info(f"Cache miss, computing result for: {question[:50]}...")
                self.miss_count += 1
                result = await func(question, *args, **kwargs)
                
                # Only try to cache if we're in main thread context
                if is_main_thread and 'werkzeug' not in current_thread.name.lower():
                    try:
                        await self.feedback_db.cache_query_result(cache_key, question, result)
                    except Exception as cache_error:
                        logger.warning(f"Cache storage failed: {cache_error}")
                
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
async def call_gemini_enhanced(prompt: str, **kwargs) -> str:
    """Enhanced Gemini API call with optimization."""
    try:
        from config import config
        
        # Direct API call
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                params = {"key": config.GEMINI_API_KEY}
                data = {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        config.GEMINI_API_URL,
                        headers=headers,
                        params=params,
                        json=data,
                        timeout=30
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                
                if ("candidates" in result and 
                    result["candidates"] and
                    "content" in result["candidates"][0] and
                    "parts" in result["candidates"][0]["content"]):
                    
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    raise Exception("Invalid response format from Gemini API")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        
    except Exception as e:
        logger.error(f"Enhanced Gemini call failed: {e}")
        raise GeminiAPIError(f"API call failed: {e}")

async def get_chunk_by_id_enhanced(embeddings: Embeddings, uid: str) -> Dict[str, Any]:
    """Enhanced chunk retrieval with connection pooling and caching."""
    
    # Try smart cache first (fastest)
    cached_chunk = chunk_cache.get(uid, config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
    if cached_chunk:
        return cached_chunk

    # Use ChunkManager as the primary source
    if chunk_manager:
        try:
            chunk_data = chunk_manager.get_chunk(uid)
            if chunk_data:
                chunk_data["retrieval_method"] = "chunk_manager"
                chunk_cache.put(uid, chunk_data, config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
                return chunk_data
        except Exception as e:
            logger.error(f"ChunkManager retrieval failed for {uid}: {e}")

    # Fallback to database if ChunkManager fails or is not available
    try:
        async with db_pool.get_connection_context() as conn:
            cursor = await conn.cursor()
            await cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (uid,))
            result = await cursor.fetchone()
            
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
    
    logger.warning(f"Chunk {uid} not found in any source")
    return {
        "chunk_id": uid,
        "text": "Content not available",
        "error": "Chunk not found",
        "retrieval_method": "error"
    }

def get_chunk_from_file_enhanced(uid: str) -> Dict[str, Any]:
    """
    Retrieve a specific chunk by its UID using the ChunkManager.
    This function is now primarily a fallback or for specific cases.
    """
    if not chunk_manager:
        logger.error("ChunkManager is not available.")
        return {"error": "ChunkManager not initialized."}
    try:
        chunk = chunk_manager.get_chunk(uid)
        if chunk:
            return chunk
        else:
            logger.warning(f"Chunk with UID {uid} not found by ChunkManager.")
            return {
                "chunk_id": uid,
                "text": "Content not available",
                "error": "Chunk not found",
                "retrieval_method": "error"
            }
    except Exception as e:
        logger.error(f"Failed to retrieve chunk {uid} via ChunkManager: {e}", exc_info=True)
        return {"error": f"Failed to retrieve chunk {uid}", "details": str(e)}

async def simple_retrieve_enhanced(embeddings: Embeddings, queries: List[str], topn: int = 5,
                           filters: Optional[Dict] = None,
                           strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced traditional retrieval function with better error handling."""
    try:
        all_uids = set()
        
        for query_idx, query in enumerate(queries):
            try:
                # This part remains synchronous as txtai search is not async
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
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Concurrently retrieve all unique chunks
        chunk_tasks = [get_chunk_by_id_enhanced(embeddings, uid) for uid in all_uids]
        retrieved_chunks = await asyncio.gather(*chunk_tasks)

        # Convert UIDs to chunk data with scores
        chunk_results = []
        for chunk in retrieved_chunks:
            if chunk and not chunk.get("error"):
                uid = chunk.get("chunk_id")
                score_info = {
                    "retrieval_score": 0.8,  # Default score
                    "strategy": strategy
                }
                chunk_results.append((uid, score_info))

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

async def enhanced_retrieve(embeddings: Embeddings, queries: List[str], topn: int = 5, 
                     strategy: str = "Standard") -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Enhanced retrieval with progressive and hybrid options."""
    
    # Use progressive retrieval if enabled
    if config.PROGRESSIVE_RETRIEVAL_ENABLED:
        try:
            retriever = ProgressiveRetriever(embeddings)
            return await retriever.retrieve_progressively(queries, strategy, confidence=0.8)
        except Exception as e:
            logger.warning(f"Progressive retrieval failed, falling back to simple: {e}")
    
    # Fall back to simple enhanced retrieval
    return await simple_retrieve_enhanced(embeddings, queries, topn, strategy=strategy)

async def synthesize_answer_enhanced(question: str, chunks: List[Dict[str, Any]], 
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
        
        # Strategy-specific prompts with natural language formatting
        if strategy == "Aggregation":
            prompt = f"""Based on the following documents, provide a comprehensive aggregation for: "{question}"

TASK: Extract and list ALL relevant items, data points, or instances found in the documents.

CONTEXT:
{context}

INSTRUCTIONS:
- List every relevant item found
- Include specific values, amounts, dates, and references  
- Group similar items but preserve individual instances
- Provide totals or summaries where appropriate
- Use clear, natural language formatting

FORMATTING GUIDELINES:
- Use clear headings and sections
- Present data in well-structured tables when appropriate
- Use bullet points for lists
- Write in a professional, readable format
- Do NOT use HTML tags or markup
- Provide clean, properly spaced text output

Please provide a clear, well-structured answer in natural language:"""
        
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

FORMATTING GUIDELINES:
- Use clear section headings
- Write in well-structured paragraphs
- Use bullet points for key findings
- Present data clearly and professionally
- Do NOT use HTML tags or markup
- Provide clean, properly spaced text output

Please provide a clear, well-structured analysis in natural language:"""
        
        else:  # Standard
            prompt = f"""Answer the following question based on the provided context: "{question}"

CONTEXT:
{context}

Please provide a clear, accurate answer based on the information above. If you cannot find specific information to answer the question, please state that clearly.

FORMATTING GUIDELINES:
- Write in clear, natural language
- Use proper spacing and structure
- Present information in a readable format
- Do NOT use HTML tags or markup
- Provide clean, well-formatted text

Please provide a clear answer in natural language:"""
        
        # Call Gemini API
        response = await call_gemini_enhanced(prompt)
        
        if not response or len(response.strip()) < 10:
            return "I was unable to generate a comprehensive answer based on the available information."
        
        # Clean and format the HTML response for UI display
        formatted_response = _format_html_response(response.strip(), strategy)
        return formatted_response
        
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

def _format_html_response(response: str, strategy: str) -> str:
    """Format the LLM response for proper HTML display in UI."""
    
    # Clean up any HTML artifacts that might be causing display issues
    cleaned_response = response
    
    # Remove HTML code block markers
    cleaned_response = cleaned_response.replace('```html', '')
    cleaned_response = cleaned_response.replace('```', '')
    
    # Remove extra paragraph tags that are being displayed
    cleaned_response = cleaned_response.replace('<p><p>', '<p>')
    cleaned_response = cleaned_response.replace('</p></p>', '</p>')
    
    # Remove any standalone HTML comments or artifacts
    import re
    cleaned_response = re.sub(r'<p>\s*</p>', '', cleaned_response)  # Remove empty paragraphs
    cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response)  # Remove multiple newlines
    
    # If response already contains HTML tags, just clean and return
    if any(tag in cleaned_response for tag in ['<table>', '<td>', '<h1>', '<h2>', '<h3>', '<p>', '<ul>', '<li>']):
        return cleaned_response.strip()
    
    # For natural language responses, convert to clean HTML
    lines = cleaned_response.split('\n')
    formatted_lines = []
    in_table_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('<br>')
            continue
            
        # Check if this looks like a table header or data
        if ('|' in line or 
            line.lower().startswith(('document', 'invoice', 'amount', 'date', 'company')) and 
            ('no.' in line.lower() or 'amount' in line.lower() or 'date' in line.lower())):
            
            # Start table formatting for structured data
            if not in_table_section:
                formatted_lines.append('<table class="table table-striped">')
                in_table_section = True
            
            # Format as table row
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                formatted_lines.append('<tr>')
                for cell in cells:
                    if cell:  # Skip empty cells
                        formatted_lines.append(f'<td>{cell}</td>')
                formatted_lines.append('</tr>')
            else:
                # Single line data, treat as table row
                formatted_lines.append(f'<tr><td colspan="4">{line}</td></tr>')
        
        else:
            # End table if we were in one
            if in_table_section:
                formatted_lines.append('</table>')
                in_table_section = False
            
            # Format other content types
            if (line.lower().startswith(('total', 'summary', 'conclusion')) or 
                '**' in line or line.isupper()):
                # Important information - make it stand out
                formatted_lines.append(f'<p><strong>{line.replace("**", "")}</strong></p>')
            elif line.endswith(':') and len(line) < 100:
                # Looks like a heading
                formatted_lines.append(f'<h4>{line}</h4>')
            elif line.startswith(('- ', '• ', '* ')):
                # List item
                formatted_lines.append(f'<li>{line[2:].strip()}</li>')
            elif line.startswith(tuple(str(i) + '.' for i in range(1, 10))):
                # Numbered list
                formatted_lines.append(f'<li>{line[2:].strip()}</li>')
            else:
                # Regular paragraph
                formatted_lines.append(f'<p>{line}</p>')
    
    # Close table if still open
    if in_table_section:
        formatted_lines.append('</table>')
    
    # Wrap list items in ul tags
    result = '\n'.join(formatted_lines)
    result = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', result, flags=re.DOTALL)
    
    return result

@smart_cache.cache_query_result(ttl_hours=1)
async def rag_query_enhanced(question: str, embeddings: Embeddings, topn: int = 5,
                      filters: Optional[Dict] = None, enable_reranking: bool = True,
                      session_id: str = None, enable_optimization: bool = True) -> Dict[str, Any]:
    """Enhanced RAG pipeline with hybrid query routing and unified preprocessing."""
    start_time = time.time()
    logger.info(f"Starting hybrid RAG query: {question[:100]}...")
    
    try:
        # STEP 1: Enhanced unified preprocessing with hybrid classification
        processed = unified_processor.process_query_unified(question)
        corrected_query = processed['corrected_query']
        intent = processed['intent']
        confidence = processed['confidence']
        alternative_queries = processed['alternative_queries']
        aggregation_type = processed.get('aggregation_type', 'none')
        complexity_level = processed.get('complexity_level', 'simple')
        requires_multi_step = processed.get('requires_multi_step', False)
        
        logger.info(f"Hybrid classification: {intent} | Aggregation: {aggregation_type} | Complexity: {complexity_level}")
        
        # STEP 2: Route to appropriate agent based on classification
        if intent == "Aggregation" and aggregation_type != 'none':
            # Route to Mini-Agent for pattern-based extraction
            logger.info(f"Routing to Mini-Agent: {aggregation_type}")
            return await route_to_mini_agent(corrected_query, aggregation_type, embeddings, start_time)
        
        elif intent == "Analyse" and (complexity_level in ['moderate', 'complex'] or requires_multi_step):
            # Route to Full Agent for complex reasoning
            logger.info(f"Routing to Full-Agent: {complexity_level}")
            return await route_to_full_agent(corrected_query, complexity_level, embeddings, start_time)
        
        else:
            # Continue with Standard RAG pipeline (existing logic)
            logger.info(f"Routing to Standard RAG: {intent}")
            return await execute_standard_rag(
                corrected_query, intent, confidence, alternative_queries, 
                embeddings, topn, filters, enable_reranking, session_id, 
                enable_optimization, start_time, processed  # Pass classification data
            )
            
    except Exception as e:
        logger.error(f"Hybrid RAG query failed: {e}")
        return {
            "answer": f"I encountered an error processing your query: {str(e)}",
            "chunks": [],
            "strategy": "Error",
            "success": False,
            "processing_time": time.time() - start_time
        }

async def route_to_mini_agent(query: str, aggregation_type: str, embeddings: Embeddings, start_time: float) -> Dict[str, Any]:
    """Route query to Mini-Agent for pattern-based extraction."""
    
    try:
        # Initialize Mini-Agent if not already done
        from mini_agent import mini_agent, initialize_mini_agent
        from progressive_retrieval import ProgressiveRetriever
        
        if mini_agent is None:
            chunk_manager_instance = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
            progressive_retriever = ProgressiveRetriever(embeddings)
            initialize_mini_agent(chunk_manager_instance, progressive_retriever)
        
        # Process with Mini-Agent
        result = await mini_agent.process_aggregation_query(query, aggregation_type)
        
        # Check if fallback is needed
        if result.get('should_fallback', False):
            logger.info("Mini-Agent recommends fallback to Standard RAG")
            return await fallback_to_standard_rag(query, embeddings, start_time)
        
        # Add standard metadata
        result.update({
            "processing_time": time.time() - start_time,
            "query": query,
            "agent_used": "Mini-Agent",
            "chunks": []  # Mini-agent doesn't return chunks in standard format
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Mini-Agent routing failed: {e}")
        return await fallback_to_standard_rag(query, embeddings, start_time)

async def route_to_full_agent(query: str, complexity_level: str, embeddings: Embeddings, start_time: float) -> Dict[str, Any]:
    """Route query to Full Agent for complex reasoning."""
    
    try:
        # Initialize Full Agent if not already done
        from full_agent import full_agent, initialize_full_agent
        from progressive_retrieval import ProgressiveRetriever
        
        if full_agent is None:
            chunk_manager_instance = ChunkManager(config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
            progressive_retriever = ProgressiveRetriever(embeddings)
            initialize_full_agent(chunk_manager_instance, progressive_retriever, call_gemini_enhanced)
        
        # Process with Full Agent
        result = await full_agent.process_complex_query(query, complexity_level)
        
        # Check if fallback is needed
        if result.get('should_fallback', False):
            logger.info("Full Agent recommends fallback to Standard RAG")
            return await fallback_to_standard_rag(query, embeddings, start_time)
        
        # Add standard metadata
        result.update({
            "processing_time": time.time() - start_time,
            "query": query,
            "agent_used": "Full-Agent",
            "chunks": []  # Full agent manages chunks internally
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Full Agent routing failed: {e}")
        return await fallback_to_standard_rag(query, embeddings, start_time)

async def fallback_to_standard_rag(query: str, embeddings: Embeddings, start_time: float) -> Dict[str, Any]:
    """Fallback to standard RAG when agents fail."""
    
    logger.info("Falling back to Standard RAG pipeline")
    
    # Create basic classification for fallback
    basic_classification = {
        'corrected_query': query,
        'intent': 'Standard',
        'confidence': 0.5,
        'reasoning': 'Fallback processing',
        'alternative_queries': [query],
        'aggregation_type': 'none',
        'complexity_level': 'simple',
        'requires_multi_step': False
    }
    
    # Use existing standard RAG logic
    return await execute_standard_rag(
        query, "Standard", 0.8, [query], embeddings, 
        topn=5, filters=None, enable_reranking=True, 
        session_id=None, enable_optimization=True, 
        start_time=start_time, processed=basic_classification
    )

async def execute_standard_rag(corrected_query: str, intent: str, confidence: float, 
                             alternative_queries: List[str], embeddings: Embeddings,
                             topn: int, filters: Optional[Dict], enable_reranking: bool,
                             session_id: str, enable_optimization: bool, start_time: float,
                             processed: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the standard RAG pipeline (existing logic)."""
    
    try:
        # Map intent to strategy using config
        query_strategy = config.INTENT_TO_STRATEGY.get(intent, "Standard")
        logger.info(f"Standard RAG processing: {intent} -> {query_strategy} (confidence: {confidence:.3f})")
        
        # Generate multiqueries with unified results
        multiqueries = [corrected_query] + alternative_queries
        
        # Strategy-specific retrieval with optimizations
        if config.PROGRESSIVE_RETRIEVAL_ENABLED:
            retriever = ProgressiveRetriever(embeddings)
            chunks, retrieval_info = await retriever.retrieve_progressively(multiqueries, query_strategy, confidence)
        else:
            chunk_results, retrieval_info = await enhanced_retrieve(embeddings, multiqueries, topn=topn, strategy=query_strategy)
            chunks = []
            if chunk_results:
                chunk_tasks = [get_chunk_by_id_enhanced(embeddings, uid) for uid, score_info in chunk_results]
                retrieved_chunks = await asyncio.gather(*chunk_tasks)
                
                for i, chunk in enumerate(retrieved_chunks):
                    if chunk and not chunk.get("error"):
                        chunk.update(chunk_results[i][1])
                        chunks.append(chunk)

        # Apply reranking if enabled
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
        
        # Intelligent processing approach selection - Universal Hierarchical Processing
        estimated_total_tokens = _estimate_total_tokens(chunks, query_strategy)
        token_capacity = config.MAX_CONTEXT_LENGTH - 500
        
        should_use_hierarchical = (
            config.HIERARCHICAL_PROCESSING_ENABLED and 
            (estimated_total_tokens > token_capacity or len(chunks) > config.HIERARCHICAL_CHUNK_THRESHOLD)
        )
        
        if should_use_hierarchical:
            from hierarchical_processor import HierarchicalProcessor, ProcessingConfig
            
            async def llm_function_async(prompt: str, batch_chunks: List[Dict]) -> str:
                return await synthesize_answer_enhanced(prompt, batch_chunks, query_strategy, None, use_hierarchical=True)

            hierarchical_config = ProcessingConfig(
                max_tokens_per_batch=config.HIERARCHICAL_MAX_TOKENS_PER_BATCH,
                min_chunks_per_batch=config.HIERARCHICAL_MIN_CHUNKS_PER_BATCH,
                max_chunks_per_batch=config.HIERARCHICAL_MAX_CHUNKS_PER_BATCH,
                parallel_batches=config.HIERARCHICAL_PARALLEL_BATCHES,
                enable_parallel=config.HIERARCHICAL_ENABLE_PARALLEL
            )
            
            processor = HierarchicalProcessor(llm_function_async, hierarchical_config)
            processor.set_query_strategy(query_strategy)
            
            hierarchical_result = await processor.process_large_query_async(corrected_query, chunks, query_strategy)
            
            answer = hierarchical_result['final_answer']
            processing_method = f"hierarchical-{query_strategy.lower()}"
            hierarchical_stats = hierarchical_result['processing_stats']
        else:
            answer = await synthesize_answer_enhanced(corrected_query, chunks, query_strategy, optimization_result)
            processing_method = "standard"
            hierarchical_stats = None
            logger.info(f"📄 Standard processing ({query_strategy}): {len(chunks)} chunks, est. tokens: {estimated_total_tokens}")
        
        # Calculate metrics and build response
        processing_time = time.time() - start_time
        avg_score = safe_mean([chunk.get('final_rerank_score', chunk.get('retrieval_score', 0)) for chunk in chunks])
        
        original_strategy_limits = {
            "Standard": 5, "Analyse": 8, "Aggregation": 20
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
        
        display_chunks = chunks[:5]
        
        result = {
            "answer": answer,
            "corrected_query": corrected_query,
            "multiqueries": alternative_queries,
            "chunks": display_chunks,
            "all_chunks_count": len(chunks),
            "processing_time": processing_time,
            "session_id": session_id or "anonymous",
            "avg_relevance_score": round(avg_score, 3),
            "query_strategy": query_strategy,
            "classification": processed,  # Add classification data
            "optimization_used": enable_optimization,
            "retrieval_method": retrieval_info.get("method", "unknown"),
            "retrieval_info": retrieval_info,
            "optimization_result": optimization_result,
            "savings_info": savings_info,
            "processing_method": processing_method,
            "hierarchical_stats": hierarchical_stats,
            "agent_used": "Standard-RAG"
        }
        
        sanitized_result = sanitize_for_json(result)
        logger.info(f"Standard RAG completed in {processing_time:.2f}s - Strategy: {query_strategy}, Chunks: {len(chunks)}")
        
        return sanitized_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Standard RAG pipeline failed: {e}")
        return {
            "answer": f"Standard RAG processing failed: {str(e)}",
            "chunks": [],
            "strategy": "Error",
            "success": False,
            "processing_time": processing_time,
            "agent_used": "Standard-RAG"
        }

async def _create_error_response_async(error_msg: str, question: str, multiqueries: List[str], 
                          start_time: float, status: str, retrieval_info: Dict, 
                          session_id: str) -> Dict[str, Any]:
    """Async version of create_error_response."""
    return _create_error_response(error_msg, question, multiqueries, start_time, status, retrieval_info, session_id)


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

async def collect_query_feedback_enhanced(query: str, result: Dict[str, Any], user_rating: int,
                                  feedback_text: str = "", session_id: str = None,
                                  user_agent: str = None, ip_address: str = None) -> bool:
    """Enhanced feedback collection with optimization metadata."""
    try:
        savings_info = result.get("savings_info", {})
        
        # Assuming feedback_db.collect_feedback_enhanced is also async
        return await feedback_db.collect_feedback_enhanced(
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

async def get_performance_metrics_enhanced(days: int = 30) -> Dict[str, Any]:
    """Get enhanced performance metrics with optimization data."""
    try:
        base_metrics = await feedback_db.get_performance_metrics(days)
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
            'cost_savings': await feedback_db.get_cost_savings_summary(days),
            'processing_efficiency': await feedback_db.get_processing_efficiency_metrics(days)
        })
        
        return base_metrics
    except Exception as e:
        logger.error(f"Enhanced performance metrics failed: {e}")
        return {"error": str(e)}

async def get_system_health_enhanced() -> Dict[str, Any]:
    """Get enhanced system health status with optimization monitoring."""
    try:
        base_health = await feedback_db.get_system_health()
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

async def debug_query_processing_enhanced(question: str, embeddings: Embeddings) -> Dict[str, Any]:
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
            result = await rag_query_enhanced(question, embeddings, topn=3, enable_optimization=True)
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

def get_document_catalog_enhanced(chunks: List[Dict]) -> Dict:
    """Enhanced document catalog generation."""
    if not chunks:
        return {"documents": [], "total_count": 0}
    
    doc_catalog = {}
    for chunk in chunks:
        doc_name = chunk.get('document_name', 'Unknown')
        if doc_name not in doc_catalog:
            doc_catalog[doc_name] = {
                'name': doc_name,
                'chunks': 0,
                'total_tokens': 0
            }
        doc_catalog[doc_name]['chunks'] += 1
        doc_catalog[doc_name]['total_tokens'] += chunk.get('num_tokens', 0)
    
    return {
        "documents": list(doc_catalog.values()),
        "total_count": len(doc_catalog),
        "total_chunks": len(chunks)
    }

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
    
    # NOTE: atexit cannot handle async functions.
    # The cleanup must be called from the main application's shutdown sequence.
    # For example, in a FastAPI app:
    # @app.on_event("shutdown")
    # async def shutdown_event():
    #     await cleanup_optimizations()

async def cleanup_optimizations():
    """Async cleanup function to close connections on exit."""
    try:
        await db_pool.close_all()
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





