"""
Mini-Agent Framework for Pattern-Based Data Extraction
Handles aggregation queries with deterministic pattern matching
"""

import re
import json
from typing import Dict, List, Any, Set
from utils import logger
from chunk_manager import ChunkManager
from progressive_retrieval import ProgressiveRetriever

class PatternExtractor:
    """Base class for pattern-based extractors."""
    
    def __init__(self, name: str):
        self.name = name
        self.patterns = []
        self.post_processors = []
    
    def extract(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract data from chunks using patterns."""
        results = set()
        
        for chunk in chunks:
            text = chunk.get('chunk_text', '')
            if not text:
                continue
                
            for pattern in self.patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if isinstance(matches[0], tuple) if matches else False:
                    # If pattern returns groups, take the first group
                    matches = [match[0] for match in matches]
                results.update(matches)
        
        # Apply post-processing
        processed_results = list(results)
        for processor in self.post_processors:
            processed_results = processor(processed_results)
        
        return processed_results
    
    def add_pattern(self, pattern: str):
        """Add a regex pattern."""
        self.patterns.append(pattern)
    
    def add_post_processor(self, func):
        """Add a post-processing function."""
        self.post_processors.append(func)

class InvoiceRecipientsExtractor(PatternExtractor):
    """Extract invoice recipients/parties."""
    
    def __init__(self):
        super().__init__("invoice_recipients")
        
        # Add patterns for invoice recipients - made more flexible to capture real-world data
        # Formal company names
        self.add_pattern(r'invoice[d]?\s+(?:to|issued\s+to|sent\s+to)\s+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'billed?\s+to\s+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'consignee[:\s]+(?:\([^)]*\))?\s*([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'buyer[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'client[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        
        # Individual names and simpler business names (more flexible)
        self.add_pattern(r'consignee[:\s]+(?:\([^)]*\))?\s*([A-Z][A-Za-z\s&\.\-]{3,30})')
        self.add_pattern(r'buyer[:\s]+([A-Z][A-Za-z\s&\.\-]{3,30})')
        self.add_pattern(r'billed?\s+to\s+([A-Z][A-Za-z\s&\.\-]{3,30})')
        self.add_pattern(r'invoice[d]?\s+(?:to|issued\s+to|sent\s+to)\s+([A-Z][A-Za-z\s&\.\-]{3,30})')
        self.add_pattern(r'ship\s+to[:\s]+([A-Z][A-Za-z\s&\.\-]{3,30})')
        
        # Add post-processors
        self.add_post_processor(self._clean_company_names)
        self.add_post_processor(self._remove_duplicates)
    
    def _clean_company_names(self, names: List[str]) -> List[str]:
        """Clean and standardize company names."""
        cleaned = []
        for name in names:
            # Remove extra whitespace
            name = ' '.join(name.split())
            # Remove trailing punctuation
            name = name.rstrip('.,;:')
            # Remove common prefixes that may be captured
            name = name.strip()
            # Filter out very short matches or common words
            if len(name) > 2 and name.lower() not in ['the', 'and', 'or', 'to', 'ship', 'bill']:
                cleaned.append(name)
        return cleaned
    
    def _remove_duplicates(self, names: List[str]) -> List[str]:
        """Remove duplicates and similar names while preserving order."""
        if not names:
            return []
        
        # Normalize and deduplicate
        normalized_names = []
        seen_normalized = set()
        
        for name in names:
            # Normalize the name for comparison
            normalized = name.lower().strip()
            
            # Check if this is a substring of an existing name or vice versa
            is_duplicate = False
            for existing in seen_normalized:
                # If one name is a substring of another, keep the longer one
                if normalized in existing or existing in normalized:
                    # If current name is longer, replace the existing one
                    if len(normalized) > len(existing):
                        # Remove the shorter version and add the longer one
                        for i, (stored_name, stored_norm) in enumerate(normalized_names):
                            if stored_norm == existing:
                                normalized_names[i] = (name, normalized)
                                seen_normalized.remove(existing)
                                seen_normalized.add(normalized)
                                break
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                normalized_names.append((name, normalized))
                seen_normalized.add(normalized)
        
        # Return only the original names
        return [name for name, _ in normalized_names]

class ProjectNamesExtractor(PatternExtractor):
    """Extract project names."""
    
    def __init__(self):
        super().__init__("project_names")
        
        # Add patterns for project names
        self.add_pattern(r'project[:\s]+([A-Z][A-Za-z\s\-0-9]+)')
        self.add_pattern(r'initiative[:\s]+([A-Z][A-Za-z\s\-0-9]+)')
        self.add_pattern(r'contract[:\s]+([A-Z][A-Za-z\s\-0-9]+)')
        self.add_pattern(r'agreement[:\s]+([A-Z][A-Za-z\s\-0-9]+)')
        
        self.add_post_processor(self._clean_project_names)
        self.add_post_processor(self._remove_duplicates)
    
    def _clean_project_names(self, names: List[str]) -> List[str]:
        """Clean project names."""
        cleaned = []
        for name in names:
            name = ' '.join(name.split())
            name = name.rstrip('.,;:')
            if len(name) > 3:
                cleaned.append(name)
        return cleaned
    
    def _remove_duplicates(self, names: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                result.append(name)
        return result

class VendorNamesExtractor(PatternExtractor):
    """Extract vendor/supplier names."""
    
    def __init__(self):
        super().__init__("vendor_names")
        
        # Add patterns for vendors
        self.add_pattern(r'vendor[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'supplier[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'contractor[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'paid\s+to\s+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        self.add_pattern(r'service\s+provider[:\s]+([A-Z][A-Za-z\s&\.\-]+(?:Ltd|Pvt|Inc|Corp|Limited|Private|Company))')
        
        self.add_post_processor(self._clean_company_names)
        self.add_post_processor(self._remove_duplicates)
    
    def _clean_company_names(self, names: List[str]) -> List[str]:
        """Clean company names."""
        cleaned = []
        for name in names:
            name = ' '.join(name.split())
            name = name.rstrip('.,;:')
            if len(name) > 3:
                cleaned.append(name)
        return cleaned
    
    def _remove_duplicates(self, names: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                result.append(name)
        return result

class PaymentDatesExtractor(PatternExtractor):
    """Extract payment dates."""
    
    def __init__(self):
        super().__init__("payment_dates")
        
        # Add patterns for dates
        self.add_pattern(r'payment\s+date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
        self.add_pattern(r'paid\s+on[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
        self.add_pattern(r'transaction\s+date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
        self.add_pattern(r'date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
        
        self.add_post_processor(self._standardize_dates)
        self.add_post_processor(self._remove_duplicates)
    
    def _standardize_dates(self, dates: List[str]) -> List[str]:
        """Standardize date formats."""
        standardized = []
        for date in dates:
            # Simple standardization - could be enhanced
            date = date.replace('/', '-')
            standardized.append(date)
        return standardized
    
    def _remove_duplicates(self, dates: List[str]) -> List[str]:
        """Remove duplicate dates."""
        return list(set(dates))

class MiniAgent:
    """Mini-Agent for pattern-based aggregation queries."""
    
    def __init__(self, chunk_manager: ChunkManager, progressive_retrieval: ProgressiveRetriever):
        self.chunk_manager = chunk_manager
        self.progressive_retrieval = progressive_retrieval
        
        # Initialize extractors
        self.extractors = {
            'invoice_recipients': InvoiceRecipientsExtractor(),
            'project_names': ProjectNamesExtractor(),
            'vendor_names': VendorNamesExtractor(),
            'payment_dates': PaymentDatesExtractor(),
            # Add more extractors as needed
        }
        
        logger.info(f"Mini-Agent initialized with {len(self.extractors)} extractors")
    
    async def process_aggregation_query(self, query: str, aggregation_type: str, **kwargs) -> Dict[str, Any]:
        """Process aggregation query using pattern extraction."""
        
        try:
            logger.info(f"Mini-Agent processing: {aggregation_type} query")
            
            # Step 1: Retrieve chunks (use more chunks for better coverage)
            chunks = await self._retrieve_chunks_for_aggregation(query, limit=50)
            
            if not chunks:
                return self._create_empty_response(aggregation_type, "No relevant documents found")
            
            # Step 2: Apply pattern extraction
            extractor = self.extractors.get(aggregation_type)
            if not extractor:
                return self._create_empty_response(aggregation_type, f"No extractor available for {aggregation_type}")
            
            # Step 3: Extract data
            extracted_data = extractor.extract(chunks)
            
            if not extracted_data:
                return self._create_empty_response(aggregation_type, "No matching patterns found")
            
            # Step 4: Format response
            return self._format_aggregation_response(extracted_data, aggregation_type, len(chunks))
            
        except Exception as e:
            logger.error(f"Mini-Agent processing failed: {e}")
            return self._create_error_response(aggregation_type, str(e))
    
    async def _retrieve_chunks_for_aggregation(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve chunks for aggregation - uses broader search."""
        
        try:
            # Use progressive retrieval but with higher limits
            chunks, retrieval_info = await self.progressive_retrieval.retrieve_progressively(
                queries=[query],  # Pass as list
                strategy="Aggregation",
                confidence=0.3  # Lower threshold for broader coverage
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk retrieval for aggregation failed: {e}")
            return []
    
    def _format_aggregation_response(self, data: List[str], aggregation_type: str, chunks_processed: int) -> Dict[str, Any]:
        """Format the aggregation response."""
        
        # Create numbered list
        formatted_items = []
        for i, item in enumerate(data[:50], 1):  # Limit to 50 items
            formatted_items.append(f"{i}. {item}")
        
        response_text = f"**{aggregation_type.replace('_', ' ').title()} Found:**\n\n"
        response_text += "\n".join(formatted_items)
        
        if len(data) > 50:
            response_text += f"\n\n*[Showing first 50 of {len(data)} items found]*"
        
        response_text += f"\n\n*Extracted from {chunks_processed} document sections*"
        
        return {
            'answer': response_text,
            'strategy_used': 'Mini-Agent',
            'aggregation_type': aggregation_type,
            'items_found': len(data),
            'chunks_processed': chunks_processed,
            'extraction_method': 'pattern_based',
            'success': True
        }
    
    def _create_empty_response(self, aggregation_type: str, reason: str) -> Dict[str, Any]:
        """Create response when no data is found."""
        
        response_text = f"**{aggregation_type.replace('_', ' ').title()}:**\n\n"
        response_text += f"No {aggregation_type.replace('_', ' ')} found in the available documents.\n\n"
        response_text += f"*Reason: {reason}*"
        
        return {
            'answer': response_text,
            'strategy_used': 'Mini-Agent',
            'aggregation_type': aggregation_type,
            'items_found': 0,
            'chunks_processed': 0,
            'extraction_method': 'pattern_based',
            'success': False,
            'should_fallback': True
        }
    
    def _create_error_response(self, aggregation_type: str, error: str) -> Dict[str, Any]:
        """Create error response."""
        
        return {
            'answer': f"Error processing {aggregation_type} query: {error}",
            'strategy_used': 'Mini-Agent',
            'aggregation_type': aggregation_type,
            'success': False,
            'error': error,
            'should_fallback': True
        }
    
    def get_available_extractors(self) -> List[str]:
        """Get list of available extractors."""
        return list(self.extractors.keys())
    
    def add_extractor(self, name: str, extractor: PatternExtractor):
        """Add a new extractor."""
        self.extractors[name] = extractor
        logger.info(f"Added extractor: {name}")

# Global instance - will be initialized in rag_backend.py
mini_agent = None

def initialize_mini_agent(chunk_manager: ChunkManager, progressive_retrieval: ProgressiveRetriever):
    """Initialize the global mini-agent instance."""
    global mini_agent
    mini_agent = MiniAgent(chunk_manager, progressive_retrieval)
    return mini_agent

__all__ = ['MiniAgent', 'PatternExtractor', 'initialize_mini_agent', 'mini_agent']
