import numpy as np
from typing import Dict, List, Tuple, Any
from txtai import Embeddings
from config import config
from utils import logger, assess_chunk_quality

class ProgressiveRetriever:
    """Implements intelligent two-stage retrieval for cost optimization."""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.quality_threshold = 0.8
        self.confidence_thresholds = config.CONFIDENCE_THRESHOLDS
        self.chunk_limits = config.OPTIMAL_CHUNK_LIMITS
    
    def retrieve_progressively(self, queries: List[str], strategy: str, 
                             confidence: float) -> Tuple[List[Dict], Dict[str, Any]]:
        """Main progressive retrieval method with detailed tracking."""
        import time
        start_time = time.time()
        # Calculate optimal chunk count based on strategy and confidence
        target_chunks = self._calculate_target_chunks(strategy, confidence)
        logger.info(f"[Timing] Started retrieve_progressively for queries: {queries}")
        logger.info(f"Progressive retrieval: strategy={strategy}, confidence={confidence:.3f}, target={target_chunks}")
        
        # Stage 1: Initial retrieval with minimal chunks
        t0 = time.time()
        initial_chunks = min(target_chunks, 3)
        stage1_results = self._initial_retrieval(queries, initial_chunks)
        t1 = time.time()
        logger.info(f"[Timing] Initial retrieval took {t1 - t0:.3f}s")

        retrieval_info = {
            'stage1_chunks': len(stage1_results),
            'target_chunks': target_chunks,
            'quality_threshold': self.quality_threshold,
            'stages_used': 1
        }

        # Quality assessment
        if stage1_results:
            t2 = time.time()
            quality_score = assess_chunk_quality(stage1_results, queries[0])
            t3 = time.time()
            logger.info(f"[Timing] Quality assessment took {t3 - t2:.3f}s")
            retrieval_info['stage1_quality'] = quality_score
            # Early termination if quality is high or we've reached target
            if quality_score >= self.quality_threshold or len(stage1_results) >= target_chunks:
                logger.info(f"Early termination: quality={quality_score:.3f}, chunks={len(stage1_results)}")
                end_time = time.time()
                elapsed = end_time - start_time
                logger.info(f"[Timing] retrieve_progressively completed in {elapsed:.3f} seconds for queries: {queries}")
                return stage1_results, retrieval_info
        
        # Stage 2: Expand retrieval if needed
        additional_needed = target_chunks - len(stage1_results)
        if additional_needed > 0:
            logger.info(f"Stage 2: retrieving {additional_needed} additional chunks")
            t4 = time.time()
            stage2_results = self._expand_retrieval(queries, additional_needed, stage1_results)
            t5 = time.time()
            logger.info(f"[Timing] Stage 2 retrieval took {t5 - t4:.3f}s")
            t6 = time.time()
            final_results = self._merge_results(stage1_results, stage2_results)
            t7 = time.time()
            logger.info(f"[Timing] Merging results took {t7 - t6:.3f}s")
            retrieval_info.update({
                'stage2_chunks': len(stage2_results),
                'total_chunks': len(final_results),
                'stages_used': 2,
                'expansion_triggered': True
            })
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"[Timing] retrieve_progressively completed in {elapsed:.3f} seconds for queries: {queries}")
            return final_results, retrieval_info
        
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Timing] retrieve_progressively completed in {elapsed:.3f} seconds for queries: {queries}")
        return stage1_results, retrieval_info
    
    def _calculate_target_chunks(self, strategy: str, confidence: float) -> int:
        """Calculate optimal chunk count based on strategy and confidence."""
        base_limits = self.chunk_limits.get(strategy, (3, 4))
        min_chunks, max_chunks = base_limits
        
        if confidence >= self.confidence_thresholds["HIGH"]:
            return min_chunks
        elif confidence >= self.confidence_thresholds["MEDIUM"]:
            return min_chunks + 1
        else:
            return min(max_chunks, min_chunks + 2)
    
    def _initial_retrieval(self, queries: List[str], chunk_count: int) -> List[Dict]:
        """Stage 1: Initial retrieval with minimal chunks."""
        try:
            # Import the enhanced functions with proper handling
            from rag_backend import simple_retrieve_enhanced, get_chunk_by_id_enhanced
            
            # Use the first query for initial retrieval
            results, retrieval_info = simple_retrieve_enhanced(
                self.embeddings, [queries[0]], topn=chunk_count
            )
            
            chunk_objects = []
            for uid, score_info in results:
                chunk = get_chunk_by_id_enhanced(self.embeddings, uid)
                # Merge score info into chunk
                chunk.update(score_info)
                chunk["retrieval_stage"] = 1
                if not chunk.get("error"):
                    chunk_objects.append(chunk)
            
            logger.info(f"Stage 1 retrieval: {len(chunk_objects)} valid chunks from {len(results)} candidates")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Stage 1 retrieval failed: {e}")
            return []
    
    def _expand_retrieval(self, queries: List[str], additional_count: int, 
                         existing_results: List[Dict]) -> List[Dict]:
        """Stage 2: Expand retrieval with additional chunks."""
        try:
            from rag_backend import simple_retrieve_enhanced, get_chunk_by_id_enhanced
            
            # Use remaining queries or generate new variations
            search_queries = queries[1:] if len(queries) > 1 else [queries[0]]
            
            existing_uids = {chunk.get('chunk_id') for chunk in existing_results}
            
            # Retrieve more chunks
            results, retrieval_info = simple_retrieve_enhanced(
                self.embeddings, search_queries, 
                topn=additional_count + len(existing_uids)
            )
            
            new_chunk_objects = []
            for uid, score_info in results:
                if uid not in existing_uids and len(new_chunk_objects) < additional_count:
                    chunk = get_chunk_by_id_enhanced(self.embeddings, uid)
                    # Merge score info into chunk
                    chunk.update(score_info)
                    chunk["retrieval_stage"] = 2
                    if not chunk.get("error"):
                        new_chunk_objects.append(chunk)
            
            logger.info(f"Stage 2 expansion: {len(new_chunk_objects)} additional chunks")
            return new_chunk_objects
            
        except Exception as e:
            logger.error(f"Stage 2 expansion failed: {e}")
            return []
    
    def _merge_results(self, stage1_results: List[Dict], stage2_results: List[Dict]) -> List[Dict]:
        """Merge results from both stages, maintaining quality order."""
        
        # Combine results
        all_results = stage1_results + stage2_results
        
        # Sort by retrieval score (descending)
        all_results.sort(key=lambda x: x.get('combined_score', x.get('retrieval_score', 0)), reverse=True)
        
        # Remove duplicates based on chunk_id
        seen_ids = set()
        unique_results = []
        
        for chunk in all_results:
            chunk_id = chunk.get('chunk_id')
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(chunk)
        
        return unique_results
    
    def get_savings_report(self, original_strategy: str, final_chunks: int) -> Dict[str, Any]:
        """Generate savings report for this retrieval."""
        original_limits = {
            "basic_retrieval": 3,
            "standard_rag": 5,
            "complex_analysis": 8,
            "scalable_aggregation": 20
        }
        
        original_count = original_limits.get(original_strategy, 5)
        chunks_saved = max(0, original_count - final_chunks)
        
        cost_reduction = (chunks_saved / original_count * 100) if original_count > 0 else 0
        
        return {
            'original_chunks': original_count,
            'optimized_chunks': final_chunks,
            'chunks_saved': chunks_saved,
            'cost_reduction_percentage': round(cost_reduction, 2),
            'method': 'progressive_retrieval'
        }
