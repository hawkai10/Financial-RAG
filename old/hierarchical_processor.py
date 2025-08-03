#!/usr/bin/env python3
"""
Hierarchical Map-Reduce Processing for Large Context RAG Queries
Implements a three-stage approach: Divide -> Process -> Combine
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional, Iterator
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import difflib
import re

logger = logging.getLogger(__name__)

@dataclass
class BatchResult:
    """Result from processing a single batch."""
    batch_id: int
    results: Dict[str, Any]
    chunk_count: int
    processing_time: float
    success: bool
    error: Optional[str] = None

@dataclass 
class ProcessingConfig:
    """Configuration for hierarchical processing."""
    max_tokens_per_batch: int = 3500  # Increased limit per batch
    min_chunks_per_batch: int = 2     # Minimum chunks to process together
    max_chunks_per_batch: int = 6     # Maximum chunks per batch for quality
    parallel_batches: int = 3         # How many batches to process simultaneously
    reserved_tokens: int = 800        # Reserve for prompt structure + response
    enable_parallel: bool = True      # Enable parallel processing
    max_retries: int = 2              # Maximum retry attempts for failed batches
    retry_delay: float = 1.0          # Delay between retries

class HierarchicalProcessor:
    """
    Implements hierarchical map-reduce processing for large RAG queries.
    
    Stage 1: Smart Batch Creation (token-aware division)
    Stage 2: Parallel Batch Processing (independent LLM calls)  
    Stage 3: Intelligent Result Combination (merge + deduplicate)
    """
    
    def __init__(self, llm_function, config: ProcessingConfig = None):
        self.llm_function = llm_function
        self.config = config or ProcessingConfig()
        self.processing_stats = {}
        self.query_strategy = "Standard"  # Default strategy
        
        logger.info(f"Hierarchical processor initialized with {self.config.max_chunks_per_batch} max chunks per batch")
    
    def set_query_strategy(self, strategy: str):
        """Set the query strategy and adjust configuration accordingly."""
        self.query_strategy = strategy
        
        # Adjust token limits based on strategy (if available from config)
        try:
            from config import config as app_config
            if hasattr(app_config, 'HIERARCHICAL_STRATEGY_TOKEN_LIMITS'):
                strategy_limit = app_config.HIERARCHICAL_STRATEGY_TOKEN_LIMITS.get(
                    strategy, 
                    app_config.HIERARCHICAL_MAX_TOKENS_PER_BATCH
                )
                self.config.max_tokens_per_batch = strategy_limit
                logger.info(f"Strategy '{strategy}' set with {strategy_limit} token limit")
            else:
                logger.info(f"Strategy '{strategy}' set with default token limit")
        except ImportError:
            logger.info(f"Strategy '{strategy}' set (config adjustment not available)")
        
        logger.info(f"Query strategy optimized for: {strategy}")
    
    def process_large_query(self, question: str, chunks: List[Dict], 
                          query_type: str = "Aggregation") -> Dict[str, Any]:
        """
        Main entry point for hierarchical processing with strategy-aware optimization.
        
        Returns:
            Dict containing:
            - final_answer: Combined LLM response
            - batch_results: List of individual batch results
            - processing_stats: Performance metrics
            - completeness: Analysis of data coverage
        """
        start_time = time.time()
        
        # Set the strategy for this processing run
        self.set_query_strategy(query_type)
        
        logger.info(f"Starting hierarchical processing: {len(chunks)} chunks, strategy={query_type}")
        
        # Stage 1: Strategy-Aware Smart Batch Creation
        batches = self._create_smart_batches(chunks)
        self.processing_stats['total_chunks'] = len(chunks)
        self.processing_stats['total_batches'] = len(batches)
        
        logger.info(f"Created {len(batches)} strategy-optimized batches for {query_type}")
        
        # Stage 2: Process Batches (parallel or sequential)
        if self.config.enable_parallel and len(batches) > 1:
            batch_results = self._process_batches_parallel(question, batches, query_type)
        else:
            batch_results = self._process_batches_sequential(question, batches, query_type)
        
        # Stage 3: Combine Results with Strategy-Specific Logic
        successful_results = [r for r in batch_results if r.success]
        conflicts = self._detect_conflicts(successful_results, query_type) if successful_results else {'has_conflicts': False}
        final_result = self._combine_batch_results(question, batch_results, query_type)
        
        # Calculate final stats
        self.processing_stats['processing_time'] = time.time() - start_time
        self.processing_stats['failed_batches'] = sum(1 for r in batch_results if not r.success)
        
        logger.info(f"Hierarchical processing completed in {self.processing_stats['processing_time']:.2f}s")
        
        return {
            'final_answer': final_result,
            'batch_results': batch_results,
            'processing_stats': self.processing_stats,
            'completeness': self._analyze_completeness(chunks, batch_results),
            'conflicts_detected': conflicts['has_conflicts'],
            'conflict_details': conflicts
        }
    
    def _create_smart_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """
        Stage 1: Create token-aware batches with strategy-specific optimization.
        """
        return self._create_strategy_aware_batches(chunks)
    
    def _create_strategy_aware_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """
        Create batches optimized for the specific query strategy.
        """
        if not hasattr(self, 'query_strategy'):
            # Fallback to original batching if strategy not set
            return self._create_generic_batches(chunks)
        
        strategy = getattr(self, 'query_strategy', 'Standard')
        
        if strategy == "Standard":
            return self._create_standard_batches(chunks)
        elif strategy == "Analyse":
            return self._create_analysis_batches(chunks)
        elif strategy == "Aggregation":
            return self._create_aggregation_batches(chunks)
        else:
            return self._create_generic_batches(chunks)
    
    def _create_standard_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Create batches optimized for Standard queries - prioritize relevance and narrative flow."""
        # For standard queries, maintain chunks with similar relevance scores together
        # This helps preserve context and narrative flow
        
        # Sort by relevance score (highest first)
        sorted_chunks = sorted(chunks, 
                             key=lambda x: x.get('final_rerank_score', x.get('retrieval_score', 0)), 
                             reverse=True)
        
        return self._batch_by_token_limit(sorted_chunks, preserve_order=True)
    
    def _create_analysis_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Create batches optimized for Analysis queries - group related documents."""
        # For analysis, try to group chunks from same document or similar sources
        # This helps maintain analytical depth and relationships
        
        # Group by document first, then batch by token limits
        doc_groups = {}
        for chunk in chunks:
            doc_name = chunk.get('document_name', 'unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(chunk)
        
        # Create batches trying to keep same-document chunks together
        batches = []
        current_batch = []
        current_tokens = 0
        effective_limit = self.config.max_tokens_per_batch - self.config.reserved_tokens
        
        for doc_name, doc_chunks in doc_groups.items():
            for chunk in doc_chunks:
                chunk_tokens = self._estimate_chunk_tokens(chunk)
                
                if (current_tokens + chunk_tokens > effective_limit and current_batch) or \
                   (len(current_batch) >= self.config.max_chunks_per_batch):
                    
                    if len(current_batch) >= self.config.min_chunks_per_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_tokens = 0
                
                current_batch.append(chunk)
                current_tokens += chunk_tokens
        
        # Add final batch
        if current_batch and (len(current_batch) >= self.config.min_chunks_per_batch or not batches):
            batches.append(current_batch)
        elif current_batch and batches:
            # Merge small final batch with last batch
            batches[-1].extend(current_batch)
        
        return batches
    
    def _create_aggregation_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Create batches optimized for Aggregation queries - ensure comprehensive coverage."""
        # For aggregation, we want to ensure complete coverage
        # Use smaller batches to allow more thorough processing of each chunk
        
        # Sort by document diversity to ensure each batch has varied content
        doc_count = {}
        for chunk in chunks:
            doc_name = chunk.get('document_name', 'unknown')
            doc_count[doc_name] = doc_count.get(doc_name, 0) + 1
        
        # Distribute chunks to ensure each batch has diverse document coverage
        batches = []
        current_batch = []
        current_tokens = 0
        current_docs = set()
        effective_limit = self.config.max_tokens_per_batch - self.config.reserved_tokens
        
        # Use smaller batch size for aggregation to ensure thoroughness
        max_agg_chunks = min(self.config.max_chunks_per_batch, 5)
        
        for chunk in chunks:
            chunk_tokens = self._estimate_chunk_tokens(chunk)
            doc_name = chunk.get('document_name', 'unknown')
            
            if (current_tokens + chunk_tokens > effective_limit and current_batch) or \
               (len(current_batch) >= max_agg_chunks):
                
                if len(current_batch) >= self.config.min_chunks_per_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                    current_docs = set()
            
            current_batch.append(chunk)
            current_tokens += chunk_tokens
            current_docs.add(doc_name)
        
        # Add final batch
        if current_batch and (len(current_batch) >= self.config.min_chunks_per_batch or not batches):
            batches.append(current_batch)
        elif current_batch and batches:
            batches[-1].extend(current_batch)
        
        return batches
    
    def _create_generic_batches(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Fallback generic batching method."""
        return self._batch_by_token_limit(chunks)
    
    def _batch_by_token_limit(self, chunks: List[Dict], preserve_order: bool = False) -> List[List[Dict]]:
        """Generic token-based batching."""
        batches = []
        current_batch = []
        current_tokens = 0
        effective_limit = self.config.max_tokens_per_batch - self.config.reserved_tokens
        
        for chunk in chunks:
            chunk_tokens = self._estimate_chunk_tokens(chunk)
            
            if (current_tokens + chunk_tokens > effective_limit and current_batch) or \
               (len(current_batch) >= self.config.max_chunks_per_batch):
                
                if len(current_batch) >= self.config.min_chunks_per_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(chunk)
            current_tokens += chunk_tokens
        
        if current_batch and (len(current_batch) >= self.config.min_chunks_per_batch or not batches):
            batches.append(current_batch)
        elif current_batch and batches:
            batches[-1].extend(current_batch)
        
        return batches
    
    def _estimate_chunk_tokens(self, chunk: Dict) -> int:
        """More accurate token estimation using improved character analysis."""
        text = chunk.get('chunk_text', '') or chunk.get('text', '')
        if not text:
            return 50  # Minimum token estimate for empty chunks
        
        # Improved token estimation with character type analysis
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii_chars = len(text) - ascii_chars
        
        # ASCII: ~4 chars per token, Non-ASCII: ~2 chars per token
        estimated_tokens = (ascii_chars / 4.0) + (non_ascii_chars / 2.0)
        
        # Add overhead for JSON structure and metadata
        overhead = 50 + len(chunk.get('document_name', '')) // 4
        
        return max(int(estimated_tokens + overhead), 50)
    
    def _process_batches_parallel(self, question: str, batches: List[List[Dict]], 
                                query_type: str) -> List[BatchResult]:
        """
        Stage 2: Process batches in parallel for speed.
        """
        batch_results = []
        
        # Process in waves to avoid overwhelming the system
        batch_waves = [batches[i:i + self.config.parallel_batches] 
                      for i in range(0, len(batches), self.config.parallel_batches)]
        
        for wave_idx, wave in enumerate(batch_waves):
            logger.info(f"Processing wave {wave_idx + 1}/{len(batch_waves)} ({len(wave)} batches)")
            
            with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                # Submit all batches in this wave
                future_to_batch = {
                    executor.submit(self._process_single_batch, question, batch, 
                                  len(batch_results) + i, query_type): batch
                    for i, batch in enumerate(wave)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout per batch
                        batch_results.append(result)
                        logger.info(f"Batch {result.batch_id} completed: {result.chunk_count} chunks")
                    except Exception as e:
                        batch = future_to_batch[future]
                        error_result = BatchResult(
                            batch_id=len(batch_results),
                            results={},
                            chunk_count=len(batch),
                            processing_time=0,
                            success=False,
                            error=str(e)
                        )
                        batch_results.append(error_result)
                        logger.error(f"❌ Batch {error_result.batch_id} failed: {e}")
        
        # Sort results by batch_id to maintain order
        batch_results.sort(key=lambda x: x.batch_id)
        self.processing_stats['parallel_batches'] = sum(len(wave) for wave in batch_waves)
        
        return batch_results
    
    def _process_batches_sequential(self, question: str, batches: List[List[Dict]], 
                                  query_type: str) -> List[BatchResult]:
        """
        Stage 2: Process batches sequentially (fallback or single-threaded).
        """
        batch_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"⚙️  Processing batch {i+1}/{len(batches)} ({len(batch)} chunks)")
            result = self._process_single_batch(question, batch, i, query_type)
            batch_results.append(result)
            
            if result.success:
                logger.info(f"Batch {i} completed in {result.processing_time:.2f}s")
            else:
                logger.error(f"❌ Batch {i} failed: {result.error}")
        
        return batch_results
    
    def _process_single_batch(self, question: str, batch: List[Dict], 
                            batch_id: int, query_type: str) -> BatchResult:
        """
        Process a single batch with retry and fallback strategies.
        """
        return self._process_single_batch_with_retry(question, batch, batch_id, query_type)
    
    def _process_single_batch_with_retry(self, question: str, batch: List[Dict], 
                                       batch_id: int, query_type: str, 
                                       max_retries: int = 2) -> BatchResult:
        """Process batch with retry and fallback strategies."""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # Normal processing
                    return self._process_single_batch_normal(question, batch, batch_id, query_type)
                elif attempt == 1:
                    # Retry with smaller sub-batches
                    logger.warning(f"Batch {batch_id} failed, trying smaller sub-batches")
                    return self._process_batch_with_subdivision(question, batch, batch_id, query_type)
                else:
                    # Last resort: simplified processing
                    logger.warning(f"Batch {batch_id} failed twice, using simplified processing")
                    return self._process_batch_simplified(question, batch, batch_id, query_type)
                    
            except Exception as e:
                logger.error(f"Batch {batch_id} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    # Return partial result instead of complete failure
                    return self._create_partial_result(batch, batch_id, str(e))
        
        # This should never be reached
        return self._create_empty_result(batch_id, "All retry attempts exhausted")
    
    def _process_single_batch_normal(self, question: str, batch: List[Dict], 
                                   batch_id: int, query_type: str) -> BatchResult:
        """Normal batch processing (original implementation)."""
        start_time = time.time()
        
        try:
            # Create batch-specific prompt
            batch_prompt = self._create_batch_prompt(question, query_type, batch_id, len(batch))
            
            # Call LLM function with this batch
            response = self.llm_function(batch_prompt, batch)
            
            # Parse response into structured format
            parsed_results = self._parse_batch_response(response, query_type)
            
            return BatchResult(
                batch_id=batch_id,
                results=parsed_results,
                chunk_count=len(batch),
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            raise e  # Let retry mechanism handle this
    
    def _create_batch_prompt(self, question: str, query_type: str, 
                           batch_id: int, batch_size: int) -> str:
        """
        Create adaptive, token-aware prompts for batch processing.
        """
        # Estimate available tokens
        estimated_tokens_needed = sum(self._estimate_chunk_tokens(chunk) for chunk in [{}])  # Approximate
        available_tokens = self.config.max_tokens_per_batch - estimated_tokens_needed
        
        return self._create_adaptive_batch_prompt(question, query_type, batch_id, batch_size, available_tokens)
    
    def _create_adaptive_batch_prompt(self, question: str, query_type: str, 
                                    batch_id: int, batch_size: int, 
                                    available_tokens: int) -> str:
        """Create prompts that adapt to available token space."""
        
        base_prompt_tokens = 200  # Estimated base prompt size
        available_for_content = available_tokens - base_prompt_tokens
        
        if available_for_content < 500:
            # Minimal prompt for very limited token space
            return self._create_minimal_prompt(question, query_type, batch_id)
        elif available_for_content < 1000:
            # Concise prompt
            return self._create_concise_prompt(question, query_type, batch_id, batch_size)
        else:
            # Full detailed prompt (enhanced version)
            return self._create_detailed_batch_prompt(question, query_type, batch_id, batch_size)
    
    def _create_minimal_prompt(self, question: str, query_type: str, batch_id: int) -> str:
        """Minimal prompt for token-constrained scenarios."""
        return f"Answer: {question}"
    
    def _create_concise_prompt(self, question: str, query_type: str, batch_id: int, batch_size: int) -> str:
        """Concise prompt for moderate token constraints."""
        return f"Using {batch_size} documents, answer: {question}. Be specific and complete."
    
    def _create_detailed_batch_prompt(self, question: str, query_type: str, 
                                    batch_id: int, batch_size: int) -> str:
        """Full detailed prompt (enhanced version of original)."""
        strategy = getattr(self, 'query_strategy', query_type)
        
        if strategy == "Aggregation":
            return f"""Extract ALL relevant items from batch {batch_id + 1} for: "{question}"

TASK: Find and list every qualifying item from these {batch_size} documents.

RULES:
• Extract exact values (numbers, dates, names, amounts)
• Include source document for each item
• Group similar items but list all instances
• DO NOT summarize - preserve individual entries

FORMAT:
• Structured list with categories
• Document references
• Counts per category
• Clear organization

Focus: Complete extraction from this batch only."""

        elif strategy == "Analyse":
            return f"""Analyze batch {batch_id + 1} for: "{question}"

TASK: Extract patterns, trends, and insights from these {batch_size} documents.

FRAMEWORK:
• Key facts and data points
• Patterns and relationships
• Evidence reliability
• Insights and implications
• Supporting documentation

INSTRUCTIONS:
• Examine quantitative and qualitative data
• Note correlations, anomalies, gaps
• Consider multiple perspectives
• Include specific document citations

FORMAT:
• Structured analysis with headings
• Evidence citations
• Quantitative data
• Key insights
• Data limitations

This batch will combine with others for full analysis."""

        elif strategy == "Standard":
            return f"""Answer from batch {batch_id + 1}: "{question}"

TASK: Provide direct, factual answer from these {batch_size} documents.

GUIDELINES:
• Answer directly and concisely
• Include specific facts and details
• Quote relevant information
• State if information incomplete
• Focus on most relevant content

INSTRUCTIONS:
• Maintain factual accuracy
• Present multiple viewpoints if conflicting
• Cite specific documents
• Keep response organized

FORMAT:
• Direct answer
• Supporting details
• Source references
• Logical structure

This response combines with other batches for complete answer."""

        else:  # Fallback for unknown strategy
            return f"""Answer this question using the {batch_size} documents in batch {batch_id + 1}.

QUESTION: {question}

INSTRUCTIONS:
- Provide accurate information based on these documents
- Be clear and specific in your response
- If the complete answer requires information from other documents, state this clearly
- Cite specific information found in the documents"""
    
    def _parse_batch_response(self, response: str, query_type: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format for later combination.
        """
        # Basic parsing - could be enhanced with more sophisticated extraction
        if query_type == "Aggregation":
            return {
                'items': self._extract_list_items(response),
                'counts': self._extract_numbers(response),
                'raw_response': response
            }
        else:
            return {
                'findings': response,
                'raw_response': response
            }
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text."""
        # Simple extraction of bulleted or numbered items
        lines = text.split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or re.match(r'^\d+\.', line):
                # Remove bullet/number and add to items
                clean_item = re.sub(r'^[•\-\d\.]\s*', '', line).strip()
                if clean_item:
                    items.append(clean_item)
        
        return items
    
    def _extract_numbers(self, text: str) -> Dict[str, int]:
        """Extract numerical information from text."""
        numbers = {}
        
        # Look for patterns like "Total: 5", "Count: 10", etc.
        patterns = [
            r'total[:\s]+(\d+)',
            r'count[:\s]+(\d+)',
            r'number[:\s]+(\d+)',
            r'(\d+)\s+items?',
            r'(\d+)\s+documents?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                key = pattern.split('[')[0].replace('(', '').replace('\\', '')
                numbers[key] = sum(int(match) for match in matches)
        
        return numbers
    
    def _combine_batch_results(self, question: str, batch_results: List[BatchResult], 
                             query_type: str) -> str:
        """
        Stage 3: Intelligently combine all batch results with conflict detection.
        """
        successful_results = [r for r in batch_results if r.success]
        
        if not successful_results:
            return "Error: All batch processing failed. Please try again."
        
        # Detect conflicts before combining
        conflicts = self._detect_conflicts(successful_results, query_type)
        
        if query_type == "Aggregation":
            return self._combine_aggregation_results(question, successful_results, conflicts)
        else:
            return self._combine_analysis_results(question, successful_results, conflicts)
    
    def _detect_conflicts(self, results: List[BatchResult], query_type: str) -> Dict[str, Any]:
        """Detect conflicts between batch results."""
        conflicts = {
            'has_conflicts': False,
            'numerical_conflicts': [],
            'content_conflicts': []
        }
        
        if query_type == "Aggregation":
            conflicts = self._detect_aggregation_conflicts(results)
        else:
            conflicts = self._detect_analysis_conflicts(results)
        
        return conflicts
    
    def _detect_aggregation_conflicts(self, results: List[BatchResult]) -> Dict[str, Any]:
        """Detect conflicts in aggregation results."""
        conflicts = {
            'has_conflicts': False,
            'item_conflicts': [],
            'count_conflicts': []
        }
        
        # Simple conflict detection - could be enhanced
        all_items = []
        all_counts = {}
        
        for result in results:
            items = result.results.get('items', [])
            counts = result.results.get('counts', {})
            
            all_items.extend(items)
            for key, count in counts.items():
                if key in all_counts and all_counts[key] != count:
                    conflicts['count_conflicts'].append({
                        'key': key,
                        'values': [all_counts[key], count]
                    })
                    conflicts['has_conflicts'] = True
                all_counts[key] = count
        
        return conflicts
    
    def _detect_analysis_conflicts(self, results: List[BatchResult]) -> Dict[str, Any]:
        """Detect conflicts in analysis results with improved efficiency."""
        conflicts = {
            'has_conflicts': False,
            'numerical_conflicts': []
        }
        
        # Extract numerical statements from each batch
        number_contexts = {}
        
        for i, result in enumerate(results):
            response = result.results.get('raw_response', '')
            
            # Find numbers with context
            number_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)'
            matches = re.finditer(number_pattern, response)
            
            for match in matches:
                num = float(match.group(1).replace(',', ''))
                
                # Get surrounding context
                start = max(0, match.start() - 30)
                end = min(len(response), match.end() + 30)
                context = response[start:end].strip()
                
                # Categorize by context type
                context_type = self._categorize_numerical_context(context)
                
                if context_type not in number_contexts:
                    number_contexts[context_type] = []
                
                number_contexts[context_type].append({
                    'batch': i + 1,
                    'number': num,
                    'context': context,
                    'raw_number': match.group(1)
                })
        
        # Check for conflicts within each context type
        for context_type, patterns in number_contexts.items():
            if len(patterns) > 1:
                # Look for different numbers in same context type
                unique_numbers = {}
                for pattern in patterns:
                    num = pattern['number']
                    if num not in unique_numbers:
                        unique_numbers[num] = []
                    unique_numbers[num].append(pattern)
                
                # If multiple different numbers exist for same context type
                if len(unique_numbers) > 1:
                    number_list = list(unique_numbers.keys())
                    for i, num1 in enumerate(number_list):
                        for num2 in number_list[i+1:]:
                            conflicts['numerical_conflicts'].append({
                                'context_type': context_type,
                                'batch1': unique_numbers[num1][0]['batch'],
                                'value1': unique_numbers[num1][0]['raw_number'],
                                'context1': unique_numbers[num1][0]['context'],
                                'batch2': unique_numbers[num2][0]['batch'],
                                'value2': unique_numbers[num2][0]['raw_number'],
                                'context2': unique_numbers[num2][0]['context']
                            })
                            conflicts['has_conflicts'] = True
        
        return conflicts
    
    def _categorize_numerical_context(self, context: str) -> str:
        """Categorize numerical context for conflict detection."""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['total', 'sum', 'amount']):
            return 'totals'
        elif any(word in context_lower for word in ['count', 'number', 'quantity']):
            return 'counts'
        elif any(word in context_lower for word in ['rate', 'percentage', '%']):
            return 'rates'
        else:
            return 'other'
    
    def _combine_aggregation_results(self, question: str, results: List[BatchResult], 
                                   conflicts: Dict[str, Any]) -> str:
        """Combine aggregation results with enhanced deduplication and conflict handling."""
        all_items = []
        total_counts = {}
        source_info = {}
        
        for i, result in enumerate(results):
            batch_items = result.results.get('items', [])
            batch_counts = result.results.get('counts', {})
            
            # Track source information for each item
            for item in batch_items:
                if item not in source_info:
                    source_info[item] = []
                source_info[item].append(f"Batch {i+1}")
            
            all_items.extend(batch_items)
            
            for key, count in batch_counts.items():
                total_counts[key] = total_counts.get(key, 0) + count
        
        # Enhanced deduplication with source tracking
        unique_items = list(dict.fromkeys(all_items))  # Preserves order
        
        # Create enhanced conflict notice
        conflict_notice = ""
        if conflicts['has_conflicts']:
            conflict_notice = f"\n\n⚠️ IMPORTANT - DATA CONFLICTS DETECTED:\n"
            for conflict in conflicts['numerical_conflicts']:
                conflict_notice += f"• {conflict['context1']} (Batch {conflict['batch1']})\n"
                conflict_notice += f"• {conflict['context2']} (Batch {conflict['batch2']})\n"
            conflict_notice += "\nRESOLUTION REQUIRED: Analyze these conflicts and determine the most accurate values.\n"
        
        combination_prompt = f"""You are consolidating aggregation results from {len(results)} parallel document batches.

ORIGINAL AGGREGATION QUERY: {question}

CONSOLIDATION DATA:
✓ Total unique items identified: {len(unique_items)}
✓ Items found across multiple batches: {len([item for item, sources in source_info.items() if len(sources) > 1])}
✓ Batch processing coverage: {len(results)} batches processed successfully

AGGREGATED FINDINGS:
{chr(10).join(f'• {item}' for item in unique_items[:50])}{"..." if len(unique_items) > 50 else ""}

NUMERICAL SUMMARIES:
{chr(10).join(f'• {key}: {count}' for key, count in total_counts.items())}

{conflict_notice}

FINAL TASK: Provide a comprehensive, organized response that:
1. Directly answers the original aggregation question
2. Presents all unique items in logical categories
3. Includes accurate counts and totals
4. Notes any data quality issues or potential duplicates
5. Provides clear, actionable summary

Consolidate these findings into a complete answer."""
        
        # Use LLM to create final consolidated response
        try:
            final_response = self.llm_function(combination_prompt, [])
            return final_response
        except Exception as e:
            logger.error(f"Final combination failed: {e}")
            # Fallback to simple concatenation
            fallback = f"AGGREGATION RESULTS for: {question}\n\n"
            fallback += f"Total items found: {len(unique_items)}\n"
            fallback += f"Processed {len(results)} document batches\n\n"
            
            for i, item in enumerate(unique_items[:100], 1):
                fallback += f"{i}. {item}\n"
            
            if conflicts['has_conflicts']:
                fallback += conflict_notice
            
            return fallback
    
    def _combine_analysis_results(self, question: str, results: List[BatchResult], 
                                conflicts: Dict[str, Any]) -> str:
        """Combine analysis results with enhanced synthesis and conflict resolution."""
        # Extract individual batch analyses
        batch_analyses = []
        for i, result in enumerate(results):
            analysis = {
                'batch_id': i + 1,
                'content': result.results.get('findings', result.results.get('raw_response', '')),
                'length': len(result.results.get('raw_response', ''))
            }
            batch_analyses.append(analysis)
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are synthesizing analysis results from {len(results)} parallel document processing batches.

ORIGINAL ANALYSIS QUERY: {question}

INDIVIDUAL BATCH ANALYSES:
"""
        
        for analysis in batch_analyses:
            synthesis_prompt += f"""
BATCH {analysis['batch_id']} ANALYSIS:
{analysis['content'][:1500]}{'...' if analysis['length'] > 1500 else ''}

{'─' * 80}
"""
        
        if conflicts['has_conflicts']:
            synthesis_prompt += f"""
⚠️ CRITICAL: CONFLICTING INFORMATION DETECTED
The following contradictory numerical data was found across batches:
"""
            for conflict in conflicts['numerical_conflicts']:
                synthesis_prompt += f"""
• CONFLICT: {conflict['context_type']}
  - Batch {conflict['batch1']}: {conflict['context1']}
  - Batch {conflict['batch2']}: {conflict['context2']}
"""
        
        synthesis_prompt += f"""

SYNTHESIS TASK:
1. Integrate all batch findings into a coherent analysis
2. Resolve or acknowledge conflicts explicitly
3. Provide comprehensive insights answering the original question
4. Maintain analytical rigor and evidence-based conclusions
5. Structure the response logically with clear sections

Create a unified, authoritative analysis that synthesizes all batch insights."""
        
        # Use LLM to create final synthesized response
        try:
            final_response = self.llm_function(synthesis_prompt, [])
            return final_response
        except Exception as e:
            logger.error(f"Analysis synthesis failed: {e}")
            # Fallback to structured concatenation
            fallback = f"ANALYSIS RESULTS for: {question}\n\n"
            fallback += f"Synthesized from {len(results)} document processing batches:\n\n"
            
            for i, analysis in enumerate(batch_analyses):
                fallback += f"""
BATCH {analysis['batch_id']} INSIGHTS:
{analysis['content'][:1000]}{'...' if analysis['length'] > 1000 else ''}

{'─' * 80}
"""
            
            if conflicts['has_conflicts']:
                fallback += f"""
⚠️ ANALYTICAL CONFLICTS DETECTED:
The following contradictory information was found across batches and requires manual review:
"""
                for conflict in conflicts['numerical_conflicts']:
                    fallback += f"• {conflict['context1']} vs {conflict['context2']}\n"
            
            fallback += f"""
SYNTHESIS NOTE: This analysis combines findings from {len(results)} parallel document processing batches. For optimal accuracy, conflicts should be manually reviewed and resolved.
"""
            return fallback
    
    def _analyze_completeness(self, original_chunks: List[Dict], 
                            batch_results: List[BatchResult]) -> Dict[str, Any]:
        """Analyze how complete the processing was."""
        total_chunks = len(original_chunks)
        processed_chunks = sum(r.chunk_count for r in batch_results if r.success)
        failed_chunks = sum(r.chunk_count for r in batch_results if not r.success)
        
        return {
            'total_chunks': total_chunks,
            'processed_chunks': processed_chunks,
            'failed_chunks': failed_chunks,
            'success_rate': processed_chunks / total_chunks if total_chunks > 0 else 0,
            'completeness_score': processed_chunks / total_chunks if total_chunks > 0 else 0,
            'is_complete': failed_chunks == 0
        }
    
    # Additional helper methods for batch processing with subdivision and simplified processing
    def _process_batch_with_subdivision(self, question: str, batch: List[Dict], 
                                      batch_id: int, query_type: str) -> BatchResult:
        """Process batch by subdividing into smaller chunks."""
        start_time = time.time()
        
        try:
            # Split batch into smaller sub-batches
            mid_point = len(batch) // 2
            sub_batch1 = batch[:mid_point]
            sub_batch2 = batch[mid_point:]
            
            sub_results = []
            
            # Process each sub-batch
            for i, sub_batch in enumerate([sub_batch1, sub_batch2]):
                if sub_batch:  # Only process non-empty sub-batches
                    sub_prompt = self._create_simplified_prompt(question, query_type, f"{batch_id}.{i+1}", len(sub_batch))
                    sub_response = self.llm_function(sub_prompt, sub_batch)
                    sub_results.append(sub_response)
            
            # Combine sub-results
            combined_response = self._combine_sub_results(sub_results, query_type)
            parsed_results = self._parse_batch_response(combined_response, query_type)
            
            return BatchResult(
                batch_id=batch_id,
                results=parsed_results,
                chunk_count=len(batch),
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            raise e  # Let next retry level handle this
    
    def _process_batch_simplified(self, question: str, batch: List[Dict], 
                                batch_id: int, query_type: str) -> BatchResult:
        """Simplified batch processing as last resort."""
        start_time = time.time()
        
        try:
            # Create very simple prompt
            simple_prompt = f"Based on the provided documents, answer: {question}"
            
            # Use only the first few chunks to avoid overwhelming the LLM
            reduced_batch = batch[:3]  # Use only first 3 chunks
            
            response = self.llm_function(simple_prompt, reduced_batch)
            
            # Create minimal parsed results
            parsed_results = {
                'raw_response': response,
                'simplified': True,
                'original_chunk_count': len(batch),
                'processed_chunk_count': len(reduced_batch)
            }
            
            return BatchResult(
                batch_id=batch_id,
                results=parsed_results,
                chunk_count=len(batch),
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            raise e  # This will trigger partial result creation
    
    def _create_simplified_prompt(self, question: str, query_type: str, batch_id: str, batch_size: int) -> str:
        """Create simplified prompt for subdivision processing."""
        return f"""Answer this question using the {batch_size} documents provided.

Question: {question}

Instructions:
- Be direct and concise
- Focus on the specific question asked
- Use only the information from the provided documents"""
    
    def _combine_sub_results(self, sub_results: List[str], query_type: str) -> str:
        """Combine results from sub-batch processing."""
        if not sub_results:
            return "No results to combine"
        
        if len(sub_results) == 1:
            return sub_results[0]
        
        # Simple combination strategy
        combined = f"Combined results from {len(sub_results)} sub-batches:\n\n"
        for i, result in enumerate(sub_results):
            combined += f"Sub-batch {i+1}: {result}\n\n"
        
        return combined
    
    def _create_partial_result(self, batch: List[Dict], batch_id: int, error: str) -> BatchResult:
        """Create partial result when batch processing fails completely."""
        return BatchResult(
            batch_id=batch_id,
            results={
                'raw_response': f"Batch {batch_id} processing failed: {error}",
                'partial': True,
                'error': error,
                'chunk_count': len(batch)
            },
            chunk_count=len(batch),
            processing_time=0,
            success=False,
            error=error
        )
    
    def _create_empty_result(self, batch_id: int, error: str) -> BatchResult:
        """Create empty result for completely failed batch."""
        return BatchResult(
            batch_id=batch_id,
            results={},
            chunk_count=0,
            processing_time=0,
            success=False,
            error=error
        )
