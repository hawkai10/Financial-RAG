import os
import time
import json
import hashlib
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from functools import wraps
import pickle
import asyncio
import aiohttp
import aiosqlite

from rank_bm25 import BM25Okapi

# Import configurations and utilities
from config import config
from utils import logger, safe_mean

# Import enhanced modules
from feedback_database import EnhancedFeedbackDatabase

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
feedback_db = EnhancedFeedbackDatabase()

# Initialize optimization instances
chunk_cache = SmartChunkCache(max_size=500)
embedding_cache = SmartEmbeddingCache()
db_pool = AsyncConnectionPool(config.CHUNKS_FILE, pool_size=20) # Increased pool size

"""
Chunk retrieval no longer relies on an external ChunkManager module.
We keep a lightweight, lazy JSON index as a fallback when the SQLite DB
doesn't have the chunk, with smart cache and file timestamp invalidation.
"""

# Lazy JSON index for contextualized chunks
_chunk_file_index = None  # type: Optional[Dict[str, Dict[str, Any]]]
_chunk_file_mtime = 0.0

def _load_chunk_file_index() -> Dict[str, Dict[str, Any]]:
    global _chunk_file_index, _chunk_file_mtime
    path = config.CONTEXTUALIZED_CHUNKS_JSON_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}

    if _chunk_file_index is not None and abs(_chunk_file_mtime - mtime) < 1e-9:
        return _chunk_file_index

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Data may be a list of chunks or an object with a key containing the list
        chunks = data
        if isinstance(data, dict):
            # Common keys that might contain the list
            for key in ("chunks", "data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    chunks = data[key]
                    break
        index = {}
        if isinstance(chunks, list):
            for ch in chunks:
                if isinstance(ch, dict):
                    cid = ch.get("chunk_id") or ch.get("uid") or ch.get("id")
                    if cid:
                        index[str(cid)] = ch
        _chunk_file_index = index
        _chunk_file_mtime = mtime
        return _chunk_file_index
    except Exception as e:
        logger.warning(f"Failed to load chunk JSON index from {path}: {e}")
        return {}

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

async def get_chunk_by_id_enhanced(uid: str) -> Dict[str, Any]:
    """Enhanced chunk retrieval with connection pooling and caching (DB first, then JSON)."""
    
    # Try smart cache first (fastest)
    cached_chunk = chunk_cache.get(uid, config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
    if cached_chunk:
        return cached_chunk

    # Try database first
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

    # Fallback to JSON index
    try:
        index = _load_chunk_file_index()
        ch = index.get(str(uid))
        if ch:
            # Normalize fields
            text = ch.get("text") or ch.get("content") or ch.get("chunk_text") or "Content not available"
            chunk_data = {**ch, "text": text, "retrieval_method": "json_index"}
            chunk_cache.put(uid, chunk_data, config.CONTEXTUALIZED_CHUNKS_JSON_PATH)
            return chunk_data
    except Exception as e:
        logger.error(f"JSON index lookup failed for {uid}: {e}")
    
    logger.warning(f"Chunk {uid} not found in any source")
    return {
        "chunk_id": uid,
        "text": "Content not available",
        "error": "Chunk not found",
        "retrieval_method": "error"
    }

def get_chunk_from_file_enhanced(uid: str) -> Dict[str, Any]:
    """Retrieve a specific chunk by its UID from the lazy JSON index."""
    try:
        index = _load_chunk_file_index()
        ch = index.get(str(uid))
        if ch:
            text = ch.get("text") or ch.get("content") or ch.get("chunk_text") or "Content not available"
            return {**ch, "text": text, "retrieval_method": "json_index"}
        logger.warning(f"Chunk with UID {uid} not found in JSON index.")
        return {
            "chunk_id": uid,
            "text": "Content not available",
            "error": "Chunk not found",
            "retrieval_method": "error"
        }
    except Exception as e:
        logger.error(f"Failed to retrieve chunk {uid} from JSON index: {e}", exc_info=True)
        return {"error": f"Failed to retrieve chunk {uid}", "details": str(e)}

## simple_retrieve_enhanced removed to eliminate txtai dependency.

# Note: enhanced_retrieve removed to simplify to single-strategy path.

async def _normalize_query_with_llm(question: str) -> Tuple[str, List[str]]:
    """Ask the LLM to correct grammar and produce two similar queries. Returns (corrected, similars)."""
    try:
        prompt = (
            "You are a query reformulator.\n"
            "Task: 1) Fix grammar and make the query clearer. 2) Provide two similar alternative queries.\n"
            "Output JSON with fields: corrected, alternatives(list of 2).\n\n"
            f"Query: {question}"
        )
        raw = await call_gemini_enhanced(prompt)
        try:
            data = json.loads(raw)
            corrected = data.get("corrected") or question
            alts = data.get("alternatives") or []
            if isinstance(alts, list):
                alts = [str(a) for a in alts][:2]
            else:
                alts = []
            return corrected, alts
        except Exception:
            # Fallback: return original and no alts
            return question, []
    except Exception:
        return question, []


async def _hybrid_child_first_parent_aggregation(*args, **kwargs):
    """Deprecated: kept only for import safety if referenced elsewhere."""
    raise NotImplementedError("Deprecated: use execute_single_strategy via rag_query_enhanced")


async def _retrieve_children_hybrid(query: str, max_children: int = 24) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Return child chunks with merged dense+sparse score, a child->parent mapping, and related queries.

    Dense retrieval supports an optional multi-encoder ensemble controlled via env:
      - ENSEMBLE_ENCODERS: comma-separated SentenceTransformer model names
      - ENSEMBLE_COLLECTIONS: comma-separated collection names (Chroma/Qdrant)
      - ENSEMBLE_TABLES: comma-separated table names (pgvector)
      - ENSEMBLE_FUSION: 'rrf' or 'avg' (default: 'avg')

    Falls back to a single encoder/index when ENSEMBLE_ENCODERS is not set.
    """
    from parent_child.vector_store_factory import get_child_vector_store

    corrected, sims = await _normalize_query_with_llm(query)
    queries = [corrected] + sims

    # Always use dual encoders: BAAI + GTE with RRF fusion
    emb_names = ["BAAI/bge-small-en-v1.5", "thenlper/gte-small"]
    from pathlib import Path as _RBPath
    _proj_root = _RBPath(__file__).resolve().parent
    local_paths = [
        str(_proj_root / "local_models" / "BAAI-bge-small-en-v1.5"),
        str(_proj_root / "local_models" / "thenlper-gte-small"),
    ]
    fusion = "rrf"
    collections = []
    tables = []

    ensemble: List[Dict[str, Any]] = []

    # Helper to derive stable default collection names when not provided (must match ingestion)
    def _default_coll(name: str) -> str:
        import re
        slug = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
        return f"children_{slug}"
    # Build embedders and vector stores for both models
    # Permanently enforce local embedder
    try:
        raise ImportError("Permanently using local embedder")
    except ImportError:
        import sys
        from pathlib import Path as _PathLocal
        sys.path.insert(0, str(_PathLocal(__file__).resolve().parent))
        from local_embedder import SentenceTransformerWrapper as SentenceTransformer  # type: ignore
    from pathlib import Path as _Path
    force_local = True  # Permanently local
    for i, m in enumerate(emb_names):
        try:
            target = None
            if i < len(local_paths) and local_paths[i] and _Path(local_paths[i]).exists():
                target = local_paths[i]
            else:
                target = m
            if force_local:
                from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                embedder = _STW(target)
            else:
                try:
                    embedder = SentenceTransformer(target)
                except Exception:
                    from local_embedder import SentenceTransformerWrapper as _STW  # type: ignore
                    embedder = _STW(target)
        except Exception as e:
            logger.warning(f"Failed to load embedder '{m}': {e}; skipping")
            continue
        # Use per-model collections matching ingestion defaults
        coll = _default_coll(m)
        vec = get_child_vector_store(collection=coll, table=None)
        # Diagnostic: log vector store config if available (best effort)
        try:
            backend = "chroma"  # Permanently chroma
            if hasattr(vec, "collection_name") and hasattr(vec, "persist_dir"):
                size = vec.count() if hasattr(vec, "count") else "n/a"
                logger.info(
                    f"[retrieval] backend={backend}, collection={getattr(vec, 'collection_name', '?')} (model={m}), persist_dir={getattr(vec, 'persist_dir', '?')}, count={size}"
                )
        except Exception:
            pass
        ensemble.append({"name": m, "embedder": embedder, "vec": vec})
    if not ensemble:
        raise RuntimeError("Dual-encoder retrieval not available: failed to initialize BAAI/GTE embedders or vector stores.")

    # Collect per-list results for fusion
    # lists: List[List[Dict]] over all (query,encoder) pairs preserving rank order
    ranked_lists: List[List[Dict[str, Any]]] = []
    candidate_payloads: Dict[str, Dict[str, Any]] = {}  # cid -> example hit with payload

    use_mv = (os.getenv("CHILD_USE_MULTIVECTOR", "false").lower() == "true")
    if use_mv:
        try:
            from parent_child.multivector_store import MultiVectorChildStore
            mv = MultiVectorChildStore()
            for q in queries:
                res = mv.search_aggregate(q, top_k_children=max_children)
                for rank_idx, r in enumerate(res):
                    r["query"] = q
                    r["encoder"] = "multivector"
                    r["rank"] = rank_idx + 1
                ranked_lists.append(res)
                for r in res:
                    # child_id is returned directly by ChromaChildStore, not in payload  
                    cid = str(r.get("child_id") or "")
                    if not cid:
                        continue
                    if cid not in candidate_payloads:
                        candidate_payloads[cid] = r
        except Exception as e:
            logger.warning(f"Multi-vector retrieval disabled due to error: {e}")

    # Always also include standard dense retrieval (ensemble or single encoder)
    for q in queries:
        for member in ensemble:
            try:
                # Get a 1D embedding for the single query, regardless of backend
                import numpy as _np
                qv_any = member["embedder"].encode(q, convert_to_numpy=True)
                # SentenceTransformers returns (d,) for str; our local wrapper returns (1,d)
                if isinstance(qv_any, _np.ndarray):
                    if qv_any.ndim == 2:
                        qv_any = qv_any[0]
                    qv = qv_any.astype(float).tolist()
                else:
                    # Fallback if some backend returns list/torch tensor
                    try:
                        # If it looks like [[...]], take first row
                        if qv_any and isinstance(qv_any[0], (list, tuple)):
                            qv_any = qv_any[0]
                        qv = list(map(float, list(qv_any)))
                    except Exception:
                        # Last resort: re-encode ensuring numpy
                        qv2 = member["embedder"].encode(q, convert_to_numpy=True)
                        if isinstance(qv2, _np.ndarray) and qv2.ndim == 2:
                            qv2 = qv2[0]
                        qv = (qv2 if isinstance(qv2, list) else qv2.tolist())  # type: ignore
                res = member["vec"].search(qv, top_k=max_children)
                for rank_idx, r in enumerate(res):
                    r["query"] = q
                    r["encoder"] = member["name"]
                    r["rank"] = rank_idx + 1
                ranked_lists.append(res)
                for r in res:
                    # child_id is returned directly by ChromaChildStore, not in payload
                    cid = str(r.get("child_id") or "")
                    if not cid:
                        continue
                    if cid not in candidate_payloads:
                        candidate_payloads[cid] = r
            except Exception as e:
                logger.warning(f"Dense search failed for encoder '{member['name']}' on query variant: {e}")
                continue

    # If nothing retrieved, hard error (must ingest dual per-model collections first)
    if not ranked_lists:
        raise RuntimeError("No child hits from dual-encoder retrieval. Ensure ingestion populated per-model collections children_baai_bge_small_en_v1_5 and children_thenlper_gte_small.")

    # Fusion across all ranked lists
    combined_dense: Dict[str, float] = {}
    if fusion == "rrf":
        k_rrf = int(os.getenv("ENSEMBLE_RRF_K", "60"))
        for lst in ranked_lists:
            for r in lst:
                # child_id is returned directly by ChromaChildStore, not in payload
                cid = str(r.get("child_id") or "")
                if not cid:
                    continue
                rank = int(r.get("rank", 1))
                combined_dense[cid] = combined_dense.get(cid, 0.0) + 1.0 / (k_rrf + rank)
    else:
        # average of per-list min-max normalized scores
        for lst in ranked_lists:
            # extract scores
            scores = [float(x.get("score", 0.0) or 0.0) for x in lst]
            if not scores:
                continue
            mn, mx = min(scores), max(scores)
            for x, s in zip(lst, scores):
                # child_id is returned directly by ChromaChildStore, not in payload
                cid = str(x.get("child_id") or "")
                if not cid:
                    continue
                if mx > mn:
                    norm = (s - mn) / (mx - mn)
                else:
                    norm = 0.0
                combined_dense[cid] = combined_dense.get(cid, 0.0) + norm
        # average over number of lists
        nlists = float(len(ranked_lists))
        if nlists > 0:
            for cid in list(combined_dense.keys()):
                combined_dense[cid] /= nlists

    # Build corpus for BM25 over candidate child snippets (optionally include LLM context)
    child_docs: Dict[str, str] = {}
    child_parent: Dict[str, int] = {}
    for cid, rhit in candidate_payloads.items():
        payload = rhit.get("payload", {}) or {}
        snippet = payload.get("snippet") or ""
        ctx_extra = payload.get("context") or ""
        text_for_bm25 = (snippet + "\n" + ctx_extra).strip() if ctx_extra else snippet
        if text_for_bm25 and cid not in child_docs:
            child_docs[cid] = text_for_bm25
        try:
            pid = int(payload.get("parent_id")) if payload.get("parent_id") is not None else None
            if pid is not None:
                child_parent[cid] = pid
        except Exception:
            pass

    corpus_ids = list(child_docs.keys())
    corpus_texts = [child_docs[cid] for cid in corpus_ids]

    bm25_scores: Dict[str, float] = {}
    if corpus_texts:
        try:
            tokenized = [txt.split() for txt in corpus_texts]
            bm25 = BM25Okapi(tokenized)
            for q in queries:
                q_tokens = q.split()
                scores = bm25.get_scores(q_tokens)
                for idx, s in enumerate(scores):
                    cid = corpus_ids[idx]
                    bm25_scores[cid] = max(bm25_scores.get(cid, 0.0), float(s))
        except Exception:
            pass

    # Merge dense + sparse
    child_score_map: Dict[str, float] = {}
    for cid, dscore in combined_dense.items():
        sparse_score = bm25_scores.get(cid, 0.0)
        norm_sparse = sparse_score / (len(corpus_texts) or 1)
        child_score_map[cid] = dscore + norm_sparse

    # Build deduped child chunks
    ranked = sorted(child_score_map.items(), key=lambda it: it[1], reverse=True)[:max_children]

    # Diagnostics: pre-rerank hit@k based on substring expectation (optional)
    try:
        expect = os.getenv("HITK_EXPECT_CONTAINS", "").strip()
        k_val = int(os.getenv("HITK_K", "10"))
        hit_at_k = None
        matched_id = None
        if expect:
            top_ids = [cid for cid, _ in ranked[:k_val]]
            for cid in top_ids:
                txt = child_docs.get(cid, "")
                if expect.lower() in txt.lower():
                    hit_at_k = True
                    matched_id = cid
                    break
            if hit_at_k is None:
                hit_at_k = False
        if hit_at_k is not None:
            logger.info(f"[metrics] pre-rerank hit@{k_val}={'YES' if hit_at_k else 'NO'} expect='{expect}' matched_id={matched_id}")
    except Exception:
        pass
    child_chunks: List[Dict[str, Any]] = []
    for cid, score in ranked:
        snippet = child_docs.get(cid, "")
        child_chunks.append({
            "chunk_id": f"child_{cid}",
            "chunk_text": snippet,
            "text": snippet,
            "retrieval_score": float(score),
            "retrieval_method": "child_hybrid",
            "child_id": cid
        })

    return child_chunks, child_parent, queries


async def synthesize_answer_simple(question: str, parent_chunks: List[Dict[str, Any]], related_queries: Optional[List[str]] = None) -> str:
    """Single-strategy synthesis: answer from parent contexts with optional related queries block."""
    if not parent_chunks:
        return "I couldn't find relevant information to answer your question."
    # Build context
    ctx = []
    for i, pc in enumerate(parent_chunks[:5], 1):
        name = pc.get("document_name", f"Doc {i}")
        txt = pc.get("chunk_text", pc.get("text", ""))
        ctx.append(f"[Source {i}: {name}]\n{txt}\n")
    context = "\n".join(ctx)
    rq_block = ""
    if related_queries:
        rq_lines = "\n".join([f"- {q}" for q in related_queries[:3]])
        rq_block = f"\n\nRELATED QUERIES:\n{rq_lines}\n"
    prompt = (
        "You are an assistant answering from financial documents. If uncertain, say you don't know.\n\n"
        f"Question: {question}\n"
        f"{rq_block}\n"
        f"Context:\n{context}\n"
        "Answer concisely and cite facts from the context."
    )
    try:
        return await call_gemini_enhanced(prompt)
    except Exception as e:
        logger.error(f"Simple synthesis failed: {e}")
        return "I couldn't generate an answer at this time."


async def execute_single_strategy(question: str, top_children: int = 24, top_parents: int = 3) -> Dict[str, Any]:
    """End-to-end: normalize -> hybrid child retrieve -> dedup -> rerank -> pick top-3 children -> parents -> answer."""
    start_time = time.time()
    # 1) Children via hybrid
    child_chunks, child_to_parent, queries = await _retrieve_children_hybrid(question, max_children=top_children)
    # 2) Rerank children
    reranked_children = child_chunks
    try:
        from document_reranker import EnhancedDocumentReranker
        rr = EnhancedDocumentReranker()
        reranked_children, _ = rr.rerank_chunks(question, child_chunks, strategy="Simple")
    except Exception as e:
        logger.warning(f"Child reranking failed, using merged scores: {e}")

    # 3) Take top-3 children
    def child_score(c: Dict[str, Any]) -> float:
        return float(c.get("final_rerank_score", c.get("retrieval_score", 0.0)))

    top_children_sel = sorted(reranked_children, key=child_score, reverse=True)[:top_children]

    # 4) Get their parents (dedup)
    parent_ids: List[int] = []
    seen = set()
    for c in top_children_sel:
        cid = str(c.get("child_id") or str(c.get("chunk_id", ""))[6:])
        pid = child_to_parent.get(cid)
        if pid is None:
            continue
        if pid not in seen:
            seen.add(pid)
            parent_ids.append(pid)
        if len(parent_ids) >= top_parents:
            break

    from parent_child.parent_store import ParentStore
    parents = ParentStore().get_parents_by_ids(parent_ids)
    parent_chunks: List[Dict[str, Any]] = []
    for p in parents:
        parent_chunks.append({
            "chunk_id": f"parent_{p.parent_id}",
            "chunk_text": p.content,
            "text": p.content,
            "document_name": str(p.document_id),
            "page_start": p.page_start,
            "page_end": p.page_end,
            "retrieval_score": 1.0,
            "retrieval_method": "parent_from_top_children"
        })

    # 5) Build prompt and synthesize (also capture prompt for debugging)
    # Build context
    ctx = []
    for i, pc in enumerate(parent_chunks[:5], 1):
        name = pc.get("document_name", f"Doc {i}")
        txt = pc.get("chunk_text", pc.get("text", ""))
        ctx.append(f"[Source {i}: {name}]\n{txt}\n")
    context = "\n".join(ctx)
    rq_block = ""
    if queries:
        rq_lines = "\n".join([f"- {q}" for q in queries[:3]])
        rq_block = f"\n\nRELATED QUERIES:\n{rq_lines}\n"
    prompt = (
        "You are an assistant answering from financial documents. If uncertain, say you don't know.\n\n"
        f"Question: {question}\n"
        f"{rq_block}\n"
        f"Context:\n{context}\n"
        "Answer concisely and cite facts from the context."
    )
    try:
        answer = await call_gemini_enhanced(prompt)
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        answer = "I couldn't generate an answer at this time."

    # 5b) Structured debug trace logging
    try:
        def _serialize_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "chunk_id": c.get("chunk_id"),
                "child_id": c.get("child_id") or str(c.get("chunk_id", ""))[6:],
                "parent_id": child_to_parent.get(str(c.get("child_id") or str(c.get("chunk_id", ""))[6:])) if isinstance(child_to_parent, dict) else None,
                "retrieval_score": c.get("retrieval_score"),
                "final_rerank_score": c.get("final_rerank_score"),
                "text": c.get("chunk_text") or c.get("text") or "",
            }

        trace = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "generated_queries": queries,
            "retrieved_children": [_serialize_chunk(c) for c in top_children_sel],
            "context_prompt": prompt,
            "llm_answer": answer,
        }
        # Write to test_logs with a stable-ish filename
        out_dir = os.path.join(os.getcwd(), "test_logs")
        os.makedirs(out_dir, exist_ok=True)
        hh = hashlib.sha256((question or "").encode("utf-8")).hexdigest()[:8]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"query_trace_{ts}_{hh}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        logger.info(f"[TRACE] Query trace written to {out_path}")
        logger.info(f"[TRACE] Q='{question[:80]}...' | queries={len(queries) if queries else 0} | children={len(top_children_sel)}")
    except Exception as _e:
        logger.warning(f"[TRACE] Failed to write query trace: {_e}")
    processing_time = time.time() - start_time
    return {
        "answer": answer,
        "llm_prompt": prompt,
        "corrected_query": queries[0] if queries else question,
        "multiqueries": queries[1:] if len(queries) > 1 else [],
        "chunks": parent_chunks,
        "top_children_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "child_id": c.get("child_id"),
                "text": c.get("chunk_text", c.get("text", "")),
                "retrieval_score": c.get("retrieval_score"),
                "final_rerank_score": c.get("final_rerank_score")
            }
            for c in top_children_sel
        ],
        "all_chunks_count": len(parent_chunks),
        "processing_time": processing_time,
        "session_id": "anonymous",
        "avg_relevance_score": safe_mean([child_score(c) for c in top_children_sel]) if top_children_sel else 0.0,
        "query_strategy": "Simple",
        "retrieval_method": "single_strategy_child_parent",
        "retrieval_info": {"queries": queries, "top_children": len(child_chunks), "parents": parent_ids},
        "optimization_result": None,
        "savings_info": None,
        "processing_method": "simple",
        "hierarchical_stats": None,
        "agent_used": "Single-Strategy"
    }

@smart_cache.cache_query_result(ttl_hours=1)
async def rag_query_enhanced(question: str, topn: int = 5,
                      filters: Optional[Dict] = None, enable_reranking: bool = True,
                      session_id: str = None, enable_optimization: bool = True) -> Dict[str, Any]:
    """Single-strategy RAG pipeline: normalize -> hybrid children -> rerank -> parents -> LLM."""
    start_time = time.time()
    logger.info(f"Starting single-strategy RAG query: {question[:100]}...")
    
    try:
        # Always use the single strategy
        return await execute_single_strategy(
            question,
            top_children=max(topn * 3, 24),
            top_parents=3,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": f"I encountered an error processing your query: {str(e)}",
            "chunks": [],
            "strategy": "Error",
            "success": False,
            "processing_time": time.time() - start_time,
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

async def debug_query_processing_enhanced(question: str) -> Dict[str, Any]:
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
            result = await rag_query_enhanced(question, topn=3, enable_optimization=True)
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
    'SmartChunkCache', 'SmartEmbeddingCache', 'AsyncConnectionPool',
    'EnhancedSmartCache',

    # Core functions
    'call_gemini_enhanced', 'call_gemini',

    # Retrieval functions
    'get_chunk_by_id_enhanced', 'get_chunk_from_file_enhanced',

    # Optimization functions
    'get_optimization_stats', 'clear_all_caches', 'get_cache_health',
    'initialize_optimizations', 'cleanup_optimizations',

    # Exceptions
    'GeminiAPIError', 'RetrievalError', 'OptimizationError'
]

# Module initialization with optimizations
initialize_optimizations()
logger.info("RAG Backend (Single Strategy) loaded successfully")
logger.info("Optimization features: Chunk Caching [ON], Embedding Caching [ON], Connection Pooling [ON]")





