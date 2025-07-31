import os
import time
import pickle
import hashlib
import sqlite3
import threading
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from contextlib import contextmanager
from cachetools import TTLCache
import logging

logger = logging.getLogger("optimized_cache")

# ============================================================
# 1. CHUNK CACHING
# ============================================================

class SmartChunkCache:
    """Smart chunk cache with file change detection."""
    
    def __init__(self, max_size: int = 500):
        self.cache = TTLCache(maxsize=max_size, ttl=3600)  # 1 hour TTL
        self.file_timestamps = {}
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
            logger.info("🔄 Source file changed, clearing chunk cache")
            self.cache.clear()
            self.file_timestamps.clear()
            self.misses += 1
            return None
        
        # Normal cache lookup
        if chunk_id in self.cache:
            self.hits += 1
            return self.cache[chunk_id]
        
        self.misses += 1
        return None
    
    def put(self, chunk_id: str, chunk_data: Dict[str, Any], file_path: str):
        """Store chunk in cache and update file timestamp."""
        self.file_timestamps[file_path] = self._get_file_timestamp(file_path)
        self.cache[chunk_id] = chunk_data
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.cache),
            'max_size': self.cache.maxsize
        }

# ============================================================
# 2. EMBEDDING CACHING
# ============================================================

class SmartEmbeddingCache:
    """Embedding cache with content-based invalidation."""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = TTLCache(maxsize=1000, ttl=7200)  # 2 hours TTL
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
                    self.memory_cache[cache_key] = embedding  # Store in memory
                    self.hits += 1
                    return embedding
            except Exception as e:
                logger.error(f"Error loading cached embedding: {e}")
        
        self.misses += 1
        return None
    
    def store_embedding(self, text: str, embedding: List[float]):
        """Store embedding with content-based key."""
        cache_key = self._get_content_hash(text)
        
        # Store in memory
        self.memory_cache[cache_key] = embedding
        
        # Store on disk for persistence
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Error saving embedding to cache: {e}")
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        }

# ============================================================
# 3. CONNECTION POOLING
# ============================================================

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
                conn = self._create_connection()
                self.pool.put(conn)
                self.created_connections += 1
    
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
            'created_connections': self.created_connections
        }
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break

# ============================================================
# GLOBAL INSTANCES
# ============================================================

# Initialize global cache instances
chunk_cache = SmartChunkCache(max_size=500)
embedding_cache = SmartEmbeddingCache()
db_pool = ConnectionPool("chunks.db", pool_size=10)
