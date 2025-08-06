"""
Unified Query Processor for DocuChat AI
Handles all query preprocessing in a single LLM call for maximum efficiency
"""

import json
import re
import time
from typing import Dict, List, Any
from utils import logger, validate_and_sanitize_query

class UnifiedQueryProcessor:
    """Single LLM call for all query preprocessing tasks."""
    
    def __init__(self):
        self.unified_prompt_template = """
You are a business document query processor. Classify this query:

USER QUERY: "{query}"

Response format (JSON only):
{{
    "corrected_query": "corrected version",
    "intent": "Standard|Analyse|Aggregation", 
    "confidence": 0.85,
    "reasoning": "brief explanation",
    "alternative_queries": ["alternative 1", "alternative 2"],
    "aggregation_type": "invoice_recipients|project_names|vendor_names|payment_dates|document_types|contract_parties|revenue_sources|expense_categories|none",
    "complexity_level": "simple|moderate|complex",
    "requires_multi_step": false
}}

INTENT RULES:
- Standard: Simple factual questions (what is, who is, show me)
- Analyse: Analytical questions (analyze, compare, trends, insights)  
- Aggregation: Counting/listing (how many, list all, count, total)

AGGREGATION TYPES:
- invoice_recipients: List companies/parties that received invoices
- project_names: List project names or titles
- vendor_names: List suppliers, vendors, or service providers
- payment_dates: List payment or transaction dates
- document_types: List types of documents available
- contract_parties: List parties involved in contracts/agreements
- revenue_sources: List sources of income or revenue
- expense_categories: List types of expenses or costs
- none: Not an aggregation query

COMPLEXITY LEVELS:
- simple: Single fact lookup or basic question
- moderate: Requires some analysis or comparison
- complex: Multi-step reasoning, deep analysis, or cross-document synthesis

Return valid JSON only.
"""
        
        # Fallback patterns for when LLM parsing fails
        self.fallback_patterns = {
            'aggregation': [
                'how many', 'count of', 'total number', 'list all', 'enumerate',
                'sum up', 'tally', 'quantity of', 'show all', 'all the',
                'each', 'every', 'sum of', 'number of', 'all bills',
                'all invoices', 'all documents'
            ],
            'analyse': [
                'analyze', 'analyse', 'compare', 'relationship', 'trend', 
                'correlation', 'summary', 'overview', 'difference', 'pattern', 
                'insights', 'breakdown', 'evaluation', 'assessment', 'analyze trend',
                'evaluate', 'assess', 'examine', 'investigate'
            ],
            'standard': [
                'what is', 'who is', 'when is', 'where is', 'what does',
                'which is', 'show me the', 'tell me', 'find the', 'get the'
            ]
        }
    
    def process_query_unified(self, query: str) -> Dict[str, Any]:
        """Single LLM call to handle all preprocessing."""
        
        try:
            # Basic validation first
            sanitized_query = validate_and_sanitize_query(query)
            
            # Create unified prompt
            prompt = self.unified_prompt_template.format(query=sanitized_query)
            
            # Import here to avoid circular import - use local import
            import requests
            import json
            import time
            from config import config
            
            # Direct API call to avoid circular import
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
                    
                    response = requests.post(
                        config.GEMINI_API_URL,
                        headers=headers,
                        params=params,
                        json=data,
                        timeout=30
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if ("candidates" in result and 
                        result["candidates"] and
                        "content" in result["candidates"][0] and
                        "parts" in result["candidates"][0]["content"]):
                        
                        llm_response = result["candidates"][0]["content"]["parts"][0]["text"]
                        break
                        
                except Exception as e:
                    logger.warning(f"Unified processing API call failed (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        # Fallback to basic processing
                        return self._fallback_processing(sanitized_query)
                    time.sleep(2 ** attempt)
            
            # Parse JSON response
            processed_data = self._parse_llm_response(llm_response, sanitized_query)
            
            logger.info(f"Unified processing: {processed_data['intent']} (confidence: {processed_data['confidence']})")
            return processed_data
            
        except Exception as e:
            logger.error(f"Unified query processing failed: {e}")
            return self._fallback_processing(query)
    
    def _parse_llm_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response."""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['corrected_query', 'intent', 'confidence', 'reasoning', 'alternative_queries']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing field: {field}")
            
            # Validate intent
            valid_intents = ['Standard', 'Analyse', 'Aggregation']
            if data['intent'] not in valid_intents:
                data['intent'] = 'Standard'  # Default fallback
            
            # Validate aggregation_type
            if 'aggregation_type' not in data:
                data['aggregation_type'] = 'none'
            
            # Validate complexity_level
            if 'complexity_level' not in data:
                data['complexity_level'] = 'simple'
            
            # Validate requires_multi_step
            if 'requires_multi_step' not in data:
                data['requires_multi_step'] = False
            
            # Ensure alternative_queries is a list
            if not isinstance(data['alternative_queries'], list):
                data['alternative_queries'] = [original_query]
            
            # Limit to 2 alternatives as requested
            data['alternative_queries'] = data['alternative_queries'][:2]
            
            return data
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_processing(original_query)
    
    def _fallback_processing(self, query: str) -> Dict[str, Any]:
        """Fallback processing when LLM parsing fails."""
        # Simple rule-based classification
        query_lower = query.lower()
        
        # Determine aggregation type
        aggregation_type = 'none'
        if 'invoice' in query_lower and ('list' in query_lower or 'who' in query_lower):
            aggregation_type = 'invoice_recipients'
        elif 'project' in query_lower and ('list' in query_lower or 'all' in query_lower):
            aggregation_type = 'project_names'
        elif any(word in query_lower for word in ['vendor', 'supplier', 'contractor']):
            aggregation_type = 'vendor_names'
        elif 'payment' in query_lower and 'date' in query_lower:
            aggregation_type = 'payment_dates'
        elif 'document' in query_lower and 'type' in query_lower:
            aggregation_type = 'document_types'
        
        # Determine intent
        if any(word in query_lower for word in self.fallback_patterns['aggregation']):
            intent = 'Aggregation'
            reasoning = 'Aggregation patterns detected'
        elif any(word in query_lower for word in self.fallback_patterns['analyse']):
            intent = 'Analyse'
            reasoning = 'Analysis patterns detected'
        elif any(word in query_lower for word in self.fallback_patterns['standard']):
            intent = 'Standard'
            reasoning = 'Standard patterns detected'
        else:
            intent = 'Standard'
            reasoning = 'Default fallback classification'
        
        # Determine complexity
        complexity_indicators = ['compare', 'analyze', 'trend', 'relationship', 'why', 'how']
        complexity_level = 'complex' if any(word in query_lower for word in complexity_indicators) else 'simple'
        requires_multi_step = 'compare' in query_lower or 'analyze' in query_lower
        
        return {
            'corrected_query': query,
            'intent': intent,
            'confidence': 0.7,
            'reasoning': reasoning,
            'alternative_queries': [query, query.replace('?', '') if query.endswith('?') else query + '?'],
            'aggregation_type': aggregation_type,
            'complexity_level': complexity_level,
            'requires_multi_step': requires_multi_step
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'available_intents': ['Standard', 'Analyse', 'Aggregation'],
            'aggregation_types': [
                'invoice_recipients', 'project_names', 'vendor_names', 
                'payment_dates', 'document_types', 'contract_parties',
                'revenue_sources', 'expense_categories', 'none'
            ],
            'complexity_levels': ['simple', 'moderate', 'complex'],
            'fallback_patterns': {k: len(v) for k, v in self.fallback_patterns.items()},
            'method': 'hybrid_query_classification'
        }

# Global instance - This is what gets imported
unified_processor = UnifiedQueryProcessor()

# Export for external use
__all__ = ['UnifiedQueryProcessor', 'unified_processor']
