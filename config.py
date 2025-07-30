import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from dotenv import load_dotenv
import json
import numpy as np

def sanitize_for_json(obj):
    """Sanitize data for JSON serialization by converting numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return 0.0
    else:
        return obj

load_dotenv()

@dataclass
class Config:
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
    
    # File Paths
    INDEX_PATH: str = "business-docs-index"
    CHUNKS_FILE: str = "contextualized_chunks.json"
    
    # Processing Limits
    MAX_CONTEXT_LENGTH: int = 4000
    DEFAULT_TOP_N: int = 5
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 60
    QUERY_MAX_LENGTH: int = 1000
    
    # Updated Strategy Configuration - Fixed chunk limits
    OPTIMAL_CHUNK_LIMITS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Standard": (2, 4),           # Fixed: min < max
        "Analyse": (5, 8),            # Increased for better analysis
        "Aggregation": (8, 15)        # Unchanged for aggregation
    })
    
    # Strategy mapping from LLM intent to actual strategy
    INTENT_TO_STRATEGY: Dict[str, str] = field(default_factory=lambda: {
        "Standard": "Standard",       # General business questions
        "Analyse": "Analyse",         # Analytical questions requiring insights
        "Aggregation": "Aggregation"  # Counting/listing queries
    })
    
    CONFIDENCE_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "HIGH": 0.9,
        "MEDIUM": 0.7,
        "LOW": 0.5
    })
    
    # Feature Flags
    PROGRESSIVE_RETRIEVAL_ENABLED: bool = True
    SAMPLING_AGGREGATION_ENABLED: bool = True
    HYBRID_SEARCH_ENABLED: bool = True
    HIERARCHICAL_PROCESSING_ENABLED: bool = True
    
    # Hierarchical Processing Parameters - Strategy-Specific Optimization
    HIERARCHICAL_CHUNK_THRESHOLD: int = 8       # When to use hierarchical processing
    HIERARCHICAL_MAX_TOKENS_PER_BATCH: int = 3500  # Default token limit
    HIERARCHICAL_MIN_CHUNKS_PER_BATCH: int = 2
    HIERARCHICAL_MAX_CHUNKS_PER_BATCH: int = 6
    HIERARCHICAL_PARALLEL_BATCHES: int = 2
    HIERARCHICAL_ENABLE_PARALLEL: bool = True
    
    # Strategy-Specific Token Limits for Hierarchical Processing
    HIERARCHICAL_STRATEGY_TOKEN_LIMITS: Dict[str, int] = field(default_factory=lambda: {
        "Standard": 3200,     # Lower limit for focused responses
        "Analyse": 3800,      # Higher limit for detailed analysis
        "Aggregation": 3500   # Balanced limit for comprehensive coverage
    })
    
    # BM25 Parameters
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    HYBRID_ALPHA: float = 0.7
    
    # Sampling Parameters
    AGGREGATION_SAMPLING_THRESHOLD: int = 8
    SAMPLE_SIZE_TARGET: int = 6
    STATISTICAL_CONFIDENCE_MIN: float = 0.85
    
    # Zero-shot Classification
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.6
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Check API key
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your_gemini_api_key_here":
            errors.append("GEMINI_API_KEY is not properly configured")
        
        # Validate chunk limits
        for strategy, (min_chunks, max_chunks) in self.OPTIMAL_CHUNK_LIMITS.items():
            if min_chunks >= max_chunks:
                errors.append(f"Invalid chunk limits for {strategy}: min ({min_chunks}) >= max ({max_chunks})")
            if min_chunks < 1:
                errors.append(f"Minimum chunks for {strategy} must be >= 1")
            if max_chunks > 15:
                errors.append(f"Maximum chunks for {strategy} should be <= 15 for cost efficiency")
        
        # Validate strategy mappings
        for intent, strategy in self.INTENT_TO_STRATEGY.items():
            if strategy not in self.OPTIMAL_CHUNK_LIMITS:
                errors.append(f"Strategy '{strategy}' for intent '{intent}' not found in chunk limits")
        
        # Validate thresholds
        for threshold_name, value in self.CONFIDENCE_THRESHOLDS.items():
            if not 0 <= value <= 1:
                errors.append(f"Confidence threshold {threshold_name} must be between 0 and 1")
        
        # Check BM25 parameters
        if not 0.5 <= self.BM25_K1 <= 3.0:
            errors.append("BM25_K1 should be between 0.5 and 3.0")
        if not 0 <= self.BM25_B <= 1:
            errors.append("BM25_B should be between 0 and 1")
        
        # Check hybrid alpha
        if not 0 <= self.HYBRID_ALPHA <= 1:
            errors.append("HYBRID_ALPHA should be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))

# Create and validate config
config = Config()
