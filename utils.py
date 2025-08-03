import sys
import os
import logging

# Fix Unicode logging issues
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Reconfigure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True
)

import hashlib
import re
import numpy as np
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from time import time
from collections import defaultdict, Counter
from config import sanitize_for_json

logger = logging.getLogger(__name__)

import time
from functools import wraps
from contextlib import contextmanager
from typing import Callable, Dict

class TimingLogger:
    """Enhanced timing logger for performance monitoring."""
    
    def __init__(self, logger_name: str = "DocuChat"):
        self.logger = logging.getLogger(logger_name)
        self.timings = {}
        
    def time_function(self, description: str = None, log_args: bool = False):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                desc = description or f"{func_name}"
                
                # Log function start
                if log_args:
                    self.logger.info(f"STARTING: {desc} with args: {args[:2]}...")
                else:
                    self.logger.info(f"STARTING: {desc}")
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Store timing
                    self.timings[desc] = execution_time
                    
                    # Color-coded logging based on execution time
                    if execution_time < 1.0:
                        emoji = "⚡"  # Fast
                    elif execution_time < 3.0:
                        emoji = "🟡"  # Medium
                    else:
                        emoji = "🔴"  # Slow
                    
                    self.logger.info(f"{emoji} COMPLETED: {desc} in {execution_time:.3f}s")
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(f"❌ FAILED: {desc} after {execution_time:.3f}s - Error: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    @contextmanager
    def time_block(self, description: str):
        """Context manager for timing code blocks."""
        self.logger.info(f"STARTING: {description}")
        start_time = time.time()
        
        try:
            yield
            execution_time = time.time() - start_time
            self.timings[description] = execution_time
            
            if execution_time < 1.0:
                emoji = "⚡"
            elif execution_time < 3.0:
                emoji = "🟡"
            else:
                emoji = "🔴"
                
            self.logger.info(f"{emoji} COMPLETED: {description} in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ FAILED: {description} after {execution_time:.3f}s - Error: {e}")
            raise
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get summary of all recorded timings."""
        return self.timings.copy()
    
    def log_timing_summary(self, total_time: float = None):
        """Log a summary of all timings."""
        if not self.timings:
            return
            
        self.logger.info("TIMING SUMMARY:")
        self.logger.info("-" * 50)
        
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for process, duration in sorted_timings:
            percentage = (duration / total_time * 100) if total_time else 0
            self.logger.info(f"  {process}: {duration:.3f}s ({percentage:.1f}%)")
        
        if total_time:
            self.logger.info(f"  TOTAL PROCESS TIME: {total_time:.3f}s")
        
        self.logger.info("-" * 50)

# Global timing logger instance
timing_logger = TimingLogger("DocuChat")

# Export timing decorator for easy use
time_function = timing_logger.time_function
time_block = timing_logger.time_block

class RateLimiter:
    """Enhanced rate limiting for API calls."""
    
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        self.requests[identifier].append(now)
        return True
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        return {
            'active_users': len(self.requests),
            'total_blocked': sum(1 for reqs in self.requests.values() 
                               if len(reqs) >= self.max_requests)
        }

class QueryAnalyzer:
    """Enhanced query analysis for better classification."""
    
    def __init__(self):
        self.aggregation_keywords = [
            'how many', 'count', 'list all', 'total number', 'all the',
            'each', 'every', 'sum of', 'number of', 'show all',
            'enumerate', 'tally', 'quantity'
        ]
        
        self.simple_keywords = [
            'what is', 'who is', 'when is', 'where is',
            'what does', 'which is', 'show me', 'tell me'
        ]
        
        self.complex_keywords = [
            'analyze', 'compare', 'relationship', 'trend',
            'correlation', 'summary', 'overview', 'pattern'
        ]
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query for enhanced classification."""
        query_lower = query.lower()
        
        detected_patterns = []
        confidence_boost = 0.0
        
        # Pattern detection
        if any(kw in query_lower for kw in self.aggregation_keywords):
            detected_patterns.append('counting')
            confidence_boost += 0.1
        
        if any(kw in query_lower for kw in self.simple_keywords):
            detected_patterns.append('specific')
            confidence_boost += 0.05
        
        if any(kw in query_lower for kw in self.complex_keywords):
            detected_patterns.append('analytical')
            confidence_boost += 0.05
        
        # Entity extraction (basic)
        entities = self._extract_entities(query)
        
        return {
            'detected_patterns': detected_patterns,
            'confidence_boost': confidence_boost,
            'entities': entities,
            'query_length': len(query.split()),
            'complexity_score': self._calculate_complexity(query)
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Basic entity extraction."""
        # Simple pattern matching for business entities
        patterns = [
            r'\b[A-Z][a-zA-Z]+ (?:Enterprises|Corp|Ltd|Inc|Company)\b',
            r'\b\d{4}\b',  # Years
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return entities
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        words = query.split()
        
        # Base complexity on length and keywords
        base_score = min(len(words) / 10.0, 1.0)
        
        # Boost for complex keywords
        complex_boost = sum(0.1 for kw in self.complex_keywords if kw in query.lower())
        
        return min(base_score + complex_boost, 1.0)

def validate_and_sanitize_query(query: str) -> str:
    """Enhanced query validation and sanitization."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    sanitized = query.strip()
    
    if len(sanitized) > 1000:
        raise ValueError("Query too long (max 1000 characters)")
    
    # Remove potentially harmful patterns
    forbidden_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\('
    ]
    
    for pattern in forbidden_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized

def create_query_hash(query: str) -> str:
    """Create hash for query caching."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def safe_mean(values: List[Union[int, float]]) -> float:
    """Calculate mean with safety checks."""
    if not values:
        return 0.0
    
    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    
    if not valid_values:
        return 0.0
    
    return float(np.mean(valid_values))

def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division with zero handling."""
    if denominator == 0:
        return 0.0
    try:
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except (TypeError, ZeroDivisionError):
        return 0.0

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text for analytics."""
    if not text:
        return []
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(max_keywords)]

def calculate_cost_reduction(strategy: str, actual_chunks: int) -> float:
    """Calculate cost reduction percentage."""
    original_chunks = {
        "basic_retrieval": 3,
        "standard_rag": 5,
        "complex_analysis": 8,
        "scalable_aggregation": 20
    }
    
    original = original_chunks.get(strategy, 5)
    if original <= actual_chunks:
        return 0.0
    
    return safe_divide((original - actual_chunks), original) * 100

def assess_chunk_quality(chunks: List[Dict], query: str, threshold: float = 0.8) -> float:
    """Assess the quality of retrieved chunks."""
    if not chunks:
        return 0.0
    
    query_words = set(query.lower().split())
    total_score = 0.0
    
    for chunk in chunks:
        text = chunk.get('text', '').lower()
        
        # Calculate word overlap
        chunk_words = set(text.split())
        overlap = len(query_words.intersection(chunk_words))
        overlap_score = safe_divide(overlap, len(query_words)) if query_words else 0
        
        # Factor in retrieval score
        retrieval_score = chunk.get('retrieval_score', chunk.get('final_rerank_score', 0))
        
        # Combined score
        chunk_score = (overlap_score * 0.4) + (retrieval_score * 0.6)
        total_score += chunk_score
    
    return safe_divide(total_score, len(chunks))

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity for analytics."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return safe_divide(intersection, union)
