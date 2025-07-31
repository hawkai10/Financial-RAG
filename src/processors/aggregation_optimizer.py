import random
import numpy as np
from typing import Dict, List, Any
from ..utils.config import config
from ..utils.utils import logger

class AggregationOptimizer:
    """Optimizes aggregation queries through intelligent sampling and estimation."""
    
    def __init__(self):
        self.sampling_threshold = config.AGGREGATION_SAMPLING_THRESHOLD
        self.target_sample_size = config.SAMPLE_SIZE_TARGET
        self.confidence_threshold = config.STATISTICAL_CONFIDENCE_MIN
    
    def optimize_aggregation(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Main aggregation optimization method."""
        
        if len(chunks) <= self.sampling_threshold:
            logger.info(f"Small dataset ({len(chunks)} chunks): processing all chunks")
            return self._process_all_chunks(query, chunks)
        else:
            logger.info(f"Large dataset ({len(chunks)} chunks): using statistical sampling")
            return self._process_with_sampling(query, chunks)
    
    def _process_all_chunks(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Process all chunks for small datasets."""
        # Avoid circular import by implementing catalog creation here
        catalog = self._create_document_catalog(chunks)
        
        return {
            "method": "complete_analysis",
            "catalog": catalog,
            "sample_size": len(chunks),
            "total_population": len(chunks),
            "confidence_level": 1.0,
            "sampling_used": False,
            "statistical_note": "Complete analysis of all available documents"
        }
    
    def _process_with_sampling(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Process large datasets using statistical sampling."""
        
        # Perform stratified sampling
        sample_chunks = self._stratified_sample(chunks, self.target_sample_size)
        
        # Create enhanced catalog with sampling metadata
        sample_catalog = self._create_sampled_catalog(sample_chunks, len(chunks))
        
        # Calculate statistical confidence
        confidence_level = self._calculate_statistical_confidence(sample_chunks, chunks)
        
        # Generate estimation details
        estimation_details = self._generate_estimation_details(sample_chunks, chunks)
        
        return {
            "method": "statistical_sampling",
            "catalog": sample_catalog,
            "sample_size": len(sample_chunks),
            "total_population": len(chunks),
            "confidence_level": confidence_level,
            "sampling_used": True,
            "estimation_details": estimation_details,
            "statistical_note": f"Statistical analysis based on representative sample of {len(sample_chunks)}/{len(chunks)} documents"
        }
    
    def _stratified_sample(self, chunks: List[Dict], target_size: int) -> List[Dict]:
        """Perform stratified sampling to ensure representative chunk selection."""
        
        if len(chunks) <= target_size:
            return chunks
        
        # Sort by relevance score (descending)
        sorted_chunks = sorted(chunks, 
                             key=lambda x: x.get('final_rerank_score', 
                                               x.get('retrieval_score', 0)), 
                             reverse=True)
        
        # Stratified sampling approach
        # Take top performers + random sample from different quality tiers
        
        top_tier_size = max(1, target_size // 3)  # Top 33%
        mid_tier_size = max(1, target_size // 3)  # Middle 33%
        low_tier_size = target_size - top_tier_size - mid_tier_size  # Remaining
        
        total_chunks = len(sorted_chunks)
        
        # Define tier boundaries
        top_boundary = min(top_tier_size * 2, total_chunks // 3)
        mid_boundary = min(total_chunks * 2 // 3, total_chunks)
        
        selected_chunks = []
        
        # Top tier - highest relevance
        top_tier = sorted_chunks[:top_boundary]
        selected_chunks.extend(top_tier[:top_tier_size])
        
        # Mid tier - medium relevance
        if mid_boundary > top_boundary:
            mid_tier = sorted_chunks[top_boundary:mid_boundary]
            if len(mid_tier) <= mid_tier_size:
                selected_chunks.extend(mid_tier)
            else:
                random.seed(42)  # For reproducible results
                selected_chunks.extend(random.sample(mid_tier, mid_tier_size))
        
        # Low tier - ensure diversity
        if len(selected_chunks) < target_size and mid_boundary < total_chunks:
            low_tier = sorted_chunks[mid_boundary:]
            remaining_needed = target_size - len(selected_chunks)
            if len(low_tier) <= remaining_needed:
                selected_chunks.extend(low_tier)
            else:
                random.seed(42)
                selected_chunks.extend(random.sample(low_tier, remaining_needed))
        
        logger.info(f"Stratified sampling: {len(selected_chunks)} chunks selected from {len(chunks)} total")
        return selected_chunks
    
    def _create_sampled_catalog(self, sample_chunks: List[Dict], total_population: int) -> Dict[str, Any]:
        """Create document catalog with sampling metadata."""
        
        # Analyze sample composition
        sample_docs = []
        doc_types = {"tables": 0, "text": 0}
        
        for chunk in sample_chunks:
            doc_name = chunk.get('document_name', 'Unknown')
            if isinstance(doc_name, str):
                doc_name = doc_name.split('/')[-1]  # Get filename only
            
            doc_type = "table" if chunk.get('is_table') else "text"
            doc_types[f"{doc_type}s"] += 1
            
            sample_docs.append({
                "name": doc_name,
                "type": doc_type,
                "relevance": chunk.get('final_rerank_score', chunk.get('retrieval_score', 0))
            })
        
        # Estimate total counts based on sample
        sample_size = len(sample_chunks)
        scaling_factor = total_population / sample_size if sample_size > 0 else 1
        
        estimated_total = int(sample_size * scaling_factor)
        estimated_tables = int(doc_types.get("tables", 0) * scaling_factor)
        estimated_text = int(doc_types.get("text", 0) * scaling_factor)
        
        return {
            "sample_count": sample_size,
            "estimated_total_count": estimated_total,
            "total_population": total_population,
            "documents": sample_docs,
            "sample_types": doc_types,
            "estimated_types": {
                "tables": estimated_tables,
                "text": estimated_text
            },
            "scaling_factor": round(scaling_factor, 2),
            "sample_content": sample_chunks[0].get("text", "")[:200] if sample_chunks else ""
        }
    
    def _calculate_statistical_confidence(self, sample_chunks: List[Dict], 
                                        all_chunks: List[Dict]) -> float:
        """Calculate statistical confidence level for the sample."""
        
        sample_size = len(sample_chunks)
        population_size = len(all_chunks)
        
        if sample_size >= population_size:
            return 1.0
        
        # Simple confidence calculation based on sample size and population
        # This is a simplified approach - in practice, you'd use proper statistical methods
        
        sample_ratio = sample_size / population_size
        
        # Base confidence on sample ratio and minimum thresholds
        if sample_ratio >= 0.5:
            confidence = 0.95
        elif sample_ratio >= 0.3:
            confidence = 0.90
        elif sample_ratio >= 0.2:
            confidence = 0.85
        elif sample_ratio >= 0.1:
            confidence = 0.80
        else:
            confidence = 0.75
        
        # Adjust based on sample quality distribution
        if sample_chunks:
            relevance_scores = [chunk.get('final_rerank_score', 
                                       chunk.get('retrieval_score', 0)) 
                              for chunk in sample_chunks]
            
            # Higher confidence if we have good quality scores
            avg_relevance = np.mean([score for score in relevance_scores if score > 0])
            if avg_relevance > 0.8:
                confidence = min(confidence + 0.05, 1.0)
            elif avg_relevance < 0.5:
                confidence = max(confidence - 0.05, 0.7)
        
        return round(confidence, 2)
    
    def _generate_estimation_details(self, sample_chunks: List[Dict], 
                                   all_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate detailed estimation information."""
        
        sample_size = len(sample_chunks)
        population_size = len(all_chunks)
        
        # Document type analysis
        sample_tables = sum(1 for chunk in sample_chunks if chunk.get('is_table'))
        sample_text = sample_size - sample_tables
        
        # Relevance distribution
        relevance_scores = [chunk.get('final_rerank_score', 
                                    chunk.get('retrieval_score', 0)) 
                          for chunk in sample_chunks]
        
        return {
            "sampling_method": "stratified",
            "sample_composition": {
                "total": sample_size,
                "tables": sample_tables,
                "text_documents": sample_text,
                "table_percentage": round(sample_tables / sample_size * 100, 1) if sample_size > 0 else 0,
                "text_percentage": round(sample_text / sample_size * 100, 1) if sample_size > 0 else 0
            },
            "quality_metrics": {
                "avg_relevance": round(np.mean([s for s in relevance_scores if s > 0]), 3) if relevance_scores else 0,
                "max_relevance": round(max(relevance_scores), 3) if relevance_scores else 0,
                "min_relevance": round(min(relevance_scores), 3) if relevance_scores else 0
            },
            "statistical_properties": {
                "population_size": population_size,
                "sample_size": sample_size,
                "sampling_ratio": round(sample_size / population_size * 100, 1),
                "margin_of_error": self._estimate_margin_of_error(sample_size, population_size)
            }
        }
    
    def _estimate_margin_of_error(self, sample_size: int, population_size: int) -> float:
        """Estimate margin of error for the sample."""
        if sample_size >= population_size:
            return 0.0
        
        # Simplified margin of error calculation
        # Assumes 95% confidence level
        z_score = 1.96  # For 95% confidence
        
        # Use 50% as conservative estimate for proportion
        p = 0.5
        
        # Calculate margin of error
        if population_size > sample_size:
            finite_correction = np.sqrt((population_size - sample_size) / (population_size - 1))
            margin = z_score * np.sqrt((p * (1 - p)) / sample_size) * finite_correction
        else:
            margin = 0.0
        
        return round(margin * 100, 1)  # Return as percentage
    
    def _create_document_catalog(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Create document catalog from chunks without circular import."""
        if not chunks:
            return {"documents": [], "total_count": 0, "document_types": {}, "sample_content": ""}
        
        documents = []
        doc_types = {"tables": 0, "text": 0}
        
        for chunk in chunks:
            doc_name = chunk.get('document_name', 'Unknown')
            if isinstance(doc_name, str):
                doc_name = doc_name.split('/')[-1]
            
            doc_type = "table" if chunk.get('is_table') else "text"
            # Fix: Use correct key names that match the dictionary initialization
            if doc_type == "table":
                doc_types["tables"] += 1
            else:
                doc_types["text"] += 1
            
            documents.append({
                "name": doc_name,
                "type": doc_type,
                "relevance": chunk.get('final_rerank_score', chunk.get('retrieval_score', 0))
            })
        
        return {
            "documents": documents,
            "total_count": len(chunks),
            "document_types": doc_types,
            "sample_content": chunks  # Return the chunks themselves as sample_content
        }
