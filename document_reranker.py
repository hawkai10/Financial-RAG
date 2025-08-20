#!/usr/bin/env python3
"""
Enhanced Document Reranker Module
Implements cross-encoder based reranking with improved models and score handling
"""

import logging
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from utils import sanitize_for_json

logger = logging.getLogger(__name__)

class EnhancedDocumentReranker:
    """
    Enhanced document reranker using cross-encoder models with improved score handling.
    Implements modular architecture for better maintainability.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize the reranker with improved cross-encoder model.
        Enforces local model usage only (no remote fallback)."""
        self.cross_encoder = None
        self.has_cross_encoder = False
        # Default to the requested model; can be overridden via CROSS_ENCODER_MODEL
        self.model_name = model_name or os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._tested_model = False

        # Try to import CrossEncoder (required even for local load)
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            logger.warning("sentence-transformers not available; cross-encoder reranking disabled")
            self.has_cross_encoder = False
            return

        # Only allow local paths; do NOT attempt remote model fetch
        local_env = os.getenv("CROSS_ENCODER_PATH", "").strip()
        default_local = Path(__file__).resolve().parent / "local_models" / "cross-encoder-ms-marco-MiniLM-L-6-v2"
        load_target: Optional[str] = None
        if local_env and Path(local_env).exists():
            load_target = local_env
            logger.info(f"Loading cross-encoder from local path: {local_env} (max_length=512)")
        elif default_local.exists():
            load_target = str(default_local)
            logger.info(f"Loading cross-encoder from default local path: {default_local} (max_length=512)")
        else:
            logger.warning(
                "Local cross-encoder not found. Please run download_cross_encoder.py or set CROSS_ENCODER_PATH. "
                "Reranking will be skipped."
            )
            self.has_cross_encoder = False
            return

        # Enforce 512-token total cap at the tokenizer level
        try:
            try:
                self.cross_encoder = CrossEncoder(load_target, max_length=512)
            except TypeError:
                # Older versions may not accept max_length in constructor; set after init
                self.cross_encoder = CrossEncoder(load_target)  # type: ignore
                if hasattr(self.cross_encoder, "max_length"):
                    setattr(self.cross_encoder, "max_length", 512)
                if hasattr(self.cross_encoder, "model") and hasattr(self.cross_encoder.model, "max_seq_length"):
                    try:
                        self.cross_encoder.model.max_seq_length = 512
                    except Exception:
                        pass
            self.has_cross_encoder = True
            logger.info(f"Cross-encoder reranker loaded successfully from {load_target}")

            # Test the model with sanity check
            self._test_cross_encoder_sanity()
        except Exception as e:
            logger.warning(f"Failed to load local cross-encoder from {load_target}: {e}. Reranking disabled.")
            self.has_cross_encoder = False
    
    def _test_cross_encoder_sanity(self):
        """Test cross-encoder with known good/bad examples to verify it's working correctly."""
        if not self.has_cross_encoder or self._tested_model:
            return
        
        try:
            # Test with clearly relevant vs irrelevant pairs
            test_pairs = [
                ["financial report revenue", "The company's revenue increased by 15% this quarter"],
                ["financial report revenue", "Today is a sunny day with clear skies"],
                ["profit margin analysis", "Net profit margin improved from 8% to 12%"],
                ["profit margin analysis", "The cat sat on the mat"]
            ]
            
            test_scores = self.cross_encoder.predict(test_pairs)
            logger.info(f"Cross-encoder sanity test results for {self.model_name}:")
            
            for i, (pair, score) in enumerate(zip(test_pairs, test_scores)):
                relevance = "relevant" if i % 2 == 0 else "irrelevant"
                logger.info(f"  {relevance}: {score:.3f} - '{pair[0]}' vs '{pair[1][:50]}...'")
            
            # Analyze score quality
            relevant_scores = [test_scores[i] for i in range(0, len(test_scores), 2)]
            irrelevant_scores = [test_scores[i] for i in range(1, len(test_scores), 2)]
            
            max_score = max(test_scores)
            min_score = min(test_scores)
            score_range = max_score - min_score
            avg_relevant = np.mean(relevant_scores)
            avg_irrelevant = np.mean(irrelevant_scores)
            
            if score_range < 0.1:
                logger.warning(f"⚠️  Cross-encoder sanity test shows poor discrimination (range: {score_range:.3f})")
            elif max_score < 0:
                logger.warning(f"⚠️  Cross-encoder sanity test shows all negative scores (max: {max_score:.3f})")
            elif avg_relevant > avg_irrelevant:
                logger.info(f"✅ Cross-encoder sanity test passed (range: {score_range:.3f}, relevant avg: {avg_relevant:.3f} > irrelevant avg: {avg_irrelevant:.3f})")
            else:
                logger.warning(f"⚠️  Cross-encoder shows inverted relevance scoring (relevant: {avg_relevant:.3f} < irrelevant: {avg_irrelevant:.3f})")
            
            self._tested_model = True
            
        except Exception as e:
            logger.warning(f"Cross-encoder sanity test failed: {e}")
    
    def rerank_chunks(self, query: str, chunks: List[Dict], strategy: str = "Standard",
                     top_k: int = 5) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Enhanced reranking with strategy-aware optimization and improved score handling.
        
        Args:
            query: Search query
            chunks: List of document chunks to rerank
            strategy: Query strategy ("Standard", "Aggregation", "Analyse")
            top_k: Number of top chunks to return
            
        Returns:
            Tuple of (reranked_chunks, rerank_info)
        """
        if not chunks:
            return chunks, {"reranking_applied": False, "reason": "no_chunks"}
        
        rerank_info = {
            "original_count": len(chunks),
            "strategy": strategy,
            "reranking_applied": False,
            "method": "none",
            "model_name": self.model_name
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
            
            logger.info(f"Cross-encoder reranking ({self.model_name}): {len(chunks)} -> {len(final_chunks)} chunks")
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
        """
        Apply cross-encoder reranking with improved handling and better normalization.
        """
        if not chunks:
            return chunks
        
        # Prepare query-document pairs with improved formatting
        pairs = []
        chunk_previews = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            
            # Clean and prepare text (no manual char trim; rely on tokenizer 512 cap)
            text = text.strip()
            if not text:
                text = chunk.get("chunk_text", "")
            
            # Store preview for debugging (short preview only for logs)
            chunk_previews.append(f"Chunk {i}: {text[:50]}...")
            
            # Format the pair; tokenizer will truncate to <=512 total tokens
            clean_query = query.strip()
            pairs.append([clean_query, text])
        
        # Debug: Show what we're sending to the cross-encoder
        logger.info(f"Cross-encoder input analysis ({self.model_name}):")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Processing {len(pairs)} chunk pairs")
        
        try:
            # Get scores from cross-encoder
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Convert to numpy array for easier handling
            cross_scores = np.array(cross_scores)
            
            # Detailed score analysis
            logger.info(f"Cross-encoder raw score analysis:")
            logger.info(f"  Raw scores: min={np.min(cross_scores):.3f}, max={np.max(cross_scores):.3f}")
            logger.info(f"  Mean: {np.mean(cross_scores):.3f}, Std: {np.std(cross_scores):.3f}")
            
            # Show score distribution
            positive_scores = np.sum(cross_scores > 0)
            negative_scores = np.sum(cross_scores <= 0)
            logger.info(f"  Score distribution: {positive_scores} positive, {negative_scores} negative")
            
            # Show top 3 and bottom 3 scoring chunks for analysis
            sorted_indices = np.argsort(cross_scores)[::-1]  # Descending order
            logger.info(f"  Top 3 scoring chunks:")
            for i in range(min(3, len(sorted_indices))):
                idx = sorted_indices[i]
                logger.info(f"    Rank {i+1}: Score {cross_scores[idx]:.3f} - {chunk_previews[idx]}")
            
            # Improved normalization strategy
            score_range = np.max(cross_scores) - np.min(cross_scores)
            
            if score_range > 0.01:  # Reasonable score range
                if np.min(cross_scores) >= 0:
                    # All positive scores - use simple normalization
                    max_score = float(np.max(cross_scores))
                    normalized_scores = [float(score / max_score) for score in cross_scores]
                    normalization_method = "max-normalization"
                else:
                    # Mixed or negative scores - use min-max normalization
                    min_score = float(np.min(cross_scores))
                    max_score = float(np.max(cross_scores))
                    normalized_scores = [float((score - min_score) / (max_score - min_score)) for score in cross_scores]
                    normalization_method = "min-max-normalization"
            else:
                # Very small range - use rank-based scoring
                logger.info("Using rank-based scoring due to uniform cross-encoder scores")
                sorted_indices = np.argsort(cross_scores)[::-1]
                normalized_scores = [0.0] * len(cross_scores)
                for rank, idx in enumerate(sorted_indices):
                    # Assign scores from 1.0 (best) to 0.1 (worst)
                    normalized_scores[idx] = 1.0 - (rank / len(sorted_indices)) * 0.9
                normalization_method = "rank-based"
            
            logger.info(f"Cross-encoder normalization ({normalization_method}):")
            logger.info(f"  Normalized: min={min(normalized_scores):.3f}, max={max(normalized_scores):.3f}")
            
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback to uniform scoring
            normalized_scores = [0.5] * len(chunks)
            normalization_method = "fallback-uniform"
            cross_scores = np.array([0.0] * len(chunks))
        
        # Apply scores to chunks
        for i, chunk in enumerate(chunks):
            chunk["cross_encoder_score_raw"] = float(cross_scores[i])
            chunk["cross_encoder_score"] = float(normalized_scores[i])
            retrieval_score = chunk.get('combined_score', chunk.get('retrieval_score', 0.0))
            
            # Combine normalized cross-encoder score with retrieval score (favor cross-encoder more)
            chunk["final_rerank_score"] = float((normalized_scores[i] * 0.8) + (retrieval_score * 0.2))
            chunk["normalization_method"] = normalization_method
            chunk["reranker_model"] = self.model_name
        
        # Sort by final rerank score
        reranked_chunks = sorted(chunks, key=lambda x: x["final_rerank_score"], reverse=True)
        
        logger.info(f"Reranking complete: Final scores range {min(chunk['final_rerank_score'] for chunk in reranked_chunks):.3f} to {max(chunk['final_rerank_score'] for chunk in reranked_chunks):.3f}")
        
        return reranked_chunks

# Factory function for easy instantiation
def create_reranker(model_name: str = None) -> EnhancedDocumentReranker:
    """
    Factory function to create reranker with different models.
    
    Recommended models:
    - "cross-encoder/ms-marco-MiniLM-L-12-v2" (default, balanced)
    - "cross-encoder/ms-marco-TinyBERT-L-2-v2" (faster)
    - "cross-encoder/ms-marco-MiniLM-L-6-v2" (original)
    """
    return EnhancedDocumentReranker(model_name)
