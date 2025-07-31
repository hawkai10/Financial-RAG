import re
from typing import List, Dict, Set, Any
from collections import Counter
from txtai import Embeddings
from utils import logger
import json

class KeywordAggregationRetriever:
    """Fast keyword-based retrieval for aggregation queries."""
    
    def __init__(self, embeddings: Embeddings, chunks_file: str):
        self.embeddings = embeddings
        self.chunks_file = chunks_file
        self._load_chunk_cache()
    
    def _load_chunk_cache(self):
        """Pre-load chunks for fast keyword matching."""
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.all_chunks = json.load(f)
            
            # Create keyword index for fast searching
            self.keyword_index = {}
            for i, chunk in enumerate(self.all_chunks):
                text = chunk.get('chunk_text', '').lower()
                words = re.findall(r'\b\w+\b', text)
                for word in set(words):  # Unique words only
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(i)
            
            logger.info(f"Keyword index built: {len(self.keyword_index)} unique words, {len(self.all_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load chunk cache: {e}")
            self.all_chunks = []
            self.keyword_index = {}
    
    def retrieve_aggregation_chunks(self, query: str, alternative_queries: List[str]) -> List[Dict[str, Any]]:
        """Fast keyword-based retrieval for aggregation queries."""
        
        # Extract keywords from all queries
        all_queries = [query] + alternative_queries
        query_keywords = self._extract_query_keywords(all_queries)
        
        logger.info(f"Aggregation keywords: {query_keywords}")
        
        # Find chunks with matching keywords
        matching_chunk_indices = self._find_matching_chunks(query_keywords)
        
        # Get chunks and add relevance scoring
        relevant_chunks = []
        for chunk_idx in matching_chunk_indices:
            if chunk_idx < len(self.all_chunks):
                chunk = self.all_chunks[chunk_idx].copy()
                
                # Calculate keyword relevance score
                chunk_text = chunk.get('chunk_text', '').lower()
                relevance_score = self._calculate_keyword_relevance(chunk_text, query_keywords)
                
                # Only include chunks with sufficient relevance
                if relevance_score > 0.1:  # Threshold to filter irrelevant chunks
                    chunk_data = {
                        'text': chunk.get('chunk_text', ''),
                        'chunk_id': str(chunk.get('chunk_id', f'chunk_{chunk_idx}')),
                        'document_name': chunk.get('document_name', 'Unknown'),
                        'is_table': chunk.get('is_table', False),
                        'retrieval_score': relevance_score,
                        'retrieval_method': 'keyword_aggregation',
                        'matched_keywords': self._get_matched_keywords(chunk_text, query_keywords)
                    }
                    relevant_chunks.append(chunk_data)
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['retrieval_score'], reverse=True)
        
        logger.info(f"Keyword aggregation found {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def _extract_query_keywords(self, queries: List[str]) -> Set[str]:
        """Extract meaningful keywords from queries."""
        keywords = set()
        
        # Stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'how', 'many', 'what', 'when', 'where', 'why', 'which', 'who', 'whose'
        }
        
        for query in queries:
            words = re.findall(r'\b\w+\b', query.lower())
            for word in words:
                if len(word) > 2 and word not in stop_words:
                    keywords.add(word)
        
        return keywords
    
    def _find_matching_chunks(self, keywords: Set[str]) -> List[int]:
        """Find chunk indices that match keywords."""
        chunk_scores = Counter()
        
        for keyword in keywords:
            if keyword in self.keyword_index:
                for chunk_idx in self.keyword_index[keyword]:
                    chunk_scores[chunk_idx] += 1
        
        # Return chunks with at least 1 keyword match, sorted by match count
        return [chunk_idx for chunk_idx, score in chunk_scores.most_common()]
    
    def _calculate_keyword_relevance(self, chunk_text: str, keywords: Set[str]) -> float:
        """Calculate relevance score based on keyword matches."""
        words_in_chunk = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        
        # Count exact matches
        exact_matches = len(keywords.intersection(words_in_chunk))
        
        # Bonus for partial matches (useful for variations)
        partial_matches = 0
        for keyword in keywords:
            for word in words_in_chunk:
                if keyword in word or word in keyword:
                    partial_matches += 0.5
        
        # Calculate relevance
        total_score = exact_matches + partial_matches
        max_possible = len(keywords)
        
        relevance = total_score / max_possible if max_possible > 0 else 0
        return min(relevance, 1.0)
    
    def _get_matched_keywords(self, chunk_text: str, keywords: Set[str]) -> List[str]:
        """Get list of keywords that matched in this chunk."""
        words_in_chunk = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        return list(keywords.intersection(words_in_chunk))
