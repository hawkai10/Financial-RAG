"""
Enhanced JSON Chunker with ML-based Content Analysis
Processes JSON extraction output using dynamic content analysis instead of keywords.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
import re
from datetime import datetime
from content_analyzer import DynamicContentAnalyzer, ContentInsight
from dynamic_schema_manager import DynamicSchemaManager
from config import get_config

config = get_config()

@dataclass
class EnhancedChunk:
    """Enhanced chunk with ML-based analysis results."""
    chunk_id: str
    content: str
    token_count: int
    document_id: str
    page_number: int
    block_type: str
    semantic_embedding: Optional[List[float]]
    content_insight: ContentInsight
    schema_elements: List[str]
    confidence_score: float
    created_at: str

class EnhancedJSONChunker:
    """
    Enhanced JSON chunker that uses ML-based content analysis
    instead of predefined keywords or patterns.
    """
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 500,
                 overlap_size: int = 50):
        
        self.logger = logging.getLogger(__name__)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Initialize ML components
        self.content_analyzer = DynamicContentAnalyzer()
        self.schema_manager = DynamicSchemaManager()
        
        # Initialize tokenizer and embedding model
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Initialized tokenizer and embedding model")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def process_extracted_json(self, json_file_path: str) -> List[EnhancedChunk]:
        """
        Process JSON extraction file and create enhanced chunks.
        
        Args:
            json_file_path: Path to the JSON extraction file
            
        Returns:
            List of enhanced chunks with ML analysis
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            self.logger.info(f"Loaded extraction data from {json_file_path}")
            
            # Process each document
            all_chunks = []
            for doc_data in extraction_data:
                doc_chunks = self._process_document(doc_data)
                all_chunks.extend(doc_chunks)
            
            # Analyze chunks collectively for schema updates
            self._analyze_chunks_for_schema(all_chunks)
            
            self.logger.info(f"Created {len(all_chunks)} enhanced chunks")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing JSON file: {e}")
            return []
    
    def _process_document(self, doc_data: Dict[str, Any]) -> List[EnhancedChunk]:
        """Process a single document from the extraction data."""
        chunks = []
        document_id = doc_data.get('document_id', 'unknown')
        
        # Get document pages
        pages = doc_data.get('pages', [])
        
        for page_num, page_data in enumerate(pages):
            blocks = page_data.get('blocks', [])
            
            for block in blocks:
                block_chunks = self._process_block(
                    block, document_id, page_num
                )
                chunks.extend(block_chunks)
        
        return chunks
    
    def _process_block(self, block: Dict[str, Any], 
                      document_id: str, page_number: int) -> List[EnhancedChunk]:
        """Process a single block and create chunks."""
        chunks = []
        
        # Extract and clean block content
        raw_content = block.get('content', '')
        block_type = block.get('type', 'unknown')
        
        if not raw_content:
            return chunks
        
        # Clean content based on block type
        cleaned_content = self._clean_block_content(raw_content, block_type)
        
        if not cleaned_content or len(cleaned_content.strip()) < 20:
            return chunks
        
        # Split content into appropriately sized chunks
        text_chunks = self._split_content_intelligently(cleaned_content)
        
        # Create enhanced chunks with ML analysis
        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_enhanced_chunk(
                chunk_text, document_id, page_number, block_type, i
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _clean_block_content(self, content: str, block_type: str) -> str:
        """Clean block content based on its type."""
        if not content:
            return ""
        
        # Handle HTML content (especially tables)
        if '<' in content and '>' in content:
            cleaned = self._parse_html_content(content)
        else:
            cleaned = content
        
        # General text cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\-\$\%\(\)\"\'\/]', '', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _parse_html_content(self, html_content: str) -> str:
        """Parse HTML content and extract clean text."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Handle tables specially
            if soup.find('table'):
                return self._parse_html_table(soup)
            else:
                # Regular HTML content
                return soup.get_text(separator=' ', strip=True)
        
        except Exception as e:
            self.logger.warning(f"Error parsing HTML content: {e}")
            # Fallback: strip HTML tags with regex
            return re.sub(r'<[^>]+>', ' ', html_content)
    
    def _parse_html_table(self, soup: BeautifulSoup) -> str:
        """Parse HTML table into readable text format."""
        tables = soup.find_all('table')
        table_texts = []
        
        for table in tables:
            rows = table.find_all('tr')
            row_texts = []
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                cell_texts = []
                
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    if cell_text:  # Only include non-empty cells
                        cell_texts.append(cell_text)
                
                if cell_texts:  # Only include rows with content
                    row_texts.append(' | '.join(cell_texts))
            
            if row_texts:
                table_texts.append(' '.join(row_texts))
        
        return ' '.join(table_texts)
    
    def _split_content_intelligently(self, content: str) -> List[str]:
        """Split content into chunks using intelligent boundaries."""
        if not content:
            return []
        
        # Get token count
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= self.max_chunk_size:
            return [content]
        
        # Split by sentences first
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence would exceed max size, start new chunk
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                if current_tokens >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap from previous chunk
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = len(self.tokenizer.encode(current_chunk))
                else:
                    # Current chunk too small, keep adding
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting - could be enhanced with spaCy
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk."""
        words = chunk.split()
        overlap_words = words[-self.overlap_size:] if len(words) > self.overlap_size else words
        return ' '.join(overlap_words)
    
    def _create_enhanced_chunk(self, chunk_text: str, document_id: str,
                              page_number: int, block_type: str, chunk_index: int) -> Optional[EnhancedChunk]:
        """Create an enhanced chunk with ML analysis."""
        try:
            # Generate chunk ID
            chunk_id = f"{document_id}_p{page_number}_{block_type}_{chunk_index}"
            
            # Get token count
            token_count = len(self.tokenizer.encode(chunk_text))
            
            # Generate semantic embedding
            embedding = self.embedding_model.encode(chunk_text).tolist()
            
            # Analyze content with ML
            content_insight = self.content_analyzer.analyze_content(chunk_text)
            
            # Determine schema elements needed
            schema_elements = self._determine_schema_elements(content_insight)
            
            # Calculate overall confidence
            confidence_score = self._calculate_chunk_confidence(
                content_insight, token_count, len(chunk_text)
            )
            
            return EnhancedChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                token_count=token_count,
                document_id=document_id,
                page_number=page_number,
                block_type=block_type,
                semantic_embedding=embedding,
                content_insight=content_insight,
                schema_elements=schema_elements,
                confidence_score=confidence_score,
                created_at=datetime.now().isoformat()
            )
        
        except Exception as e:
            self.logger.error(f"Error creating enhanced chunk: {e}")
            return None
    
    def _determine_schema_elements(self, insight: ContentInsight) -> List[str]:
        """Determine what schema elements are needed for this content."""
        elements = []
        
        # Add entity-based schema elements
        for entity_type in insight.entities.keys():
            elements.append(f"has_{entity_type.lower()}")
        
        # Add relationship-based schema elements
        for relationship in insight.relationships:
            elements.append(f"rel_{relationship.relationship_type.lower()}")
        
        # Add document type schema element
        elements.append("document_type")
        
        return list(set(elements))
    
    def _calculate_chunk_confidence(self, insight: ContentInsight, 
                                   token_count: int, text_length: int) -> float:
        """Calculate overall confidence score for the chunk."""
        # Base confidence from content analysis
        content_confidence = insight.confidence_score
        
        # Size-based confidence (prefer medium-sized chunks)
        size_confidence = 1.0 - abs(token_count - (self.max_chunk_size // 2)) / (self.max_chunk_size // 2)
        size_confidence = max(0.1, size_confidence)
        
        # Text quality confidence (prefer chunks with good text length)
        quality_confidence = min(1.0, text_length / 200)  # Prefer at least 200 chars
        
        # Weighted average
        final_confidence = (
            content_confidence * 0.5 +
            size_confidence * 0.3 +
            quality_confidence * 0.2
        )
        
        return round(final_confidence, 3)
    
    def _analyze_chunks_for_schema(self, chunks: List[EnhancedChunk]):
        """Analyze all chunks and update schema accordingly."""
        if not chunks:
            return
        
        # Collect all insights
        insights = [chunk.content_insight for chunk in chunks]
        
        # Update schema based on insights
        try:
            updates = self.schema_manager.analyze_and_update_schema(insights)
            self.logger.info(f"Applied {len(updates)} schema updates based on chunk analysis")
        except Exception as e:
            self.logger.error(f"Error updating schema: {e}")
    
    def save_chunks(self, chunks: List[EnhancedChunk], output_path: str):
        """Save enhanced chunks to JSON file."""
        try:
            chunks_data = []
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                # Convert ContentInsight to dict manually
                chunk_dict['content_insight'] = {
                    'entities': chunk.content_insight.entities,
                    'relationships': [asdict(rel) for rel in chunk.content_insight.relationships],
                    'document_type': chunk.content_insight.document_type,
                    'content_categories': chunk.content_insight.content_categories,
                    'confidence_score': chunk.content_insight.confidence_score
                }
                chunks_data.append(chunk_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")
    
    def load_chunks(self, input_path: str) -> List[EnhancedChunk]:
        """Load enhanced chunks from JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = []
            for chunk_dict in chunks_data:
                # Reconstruct ContentInsight
                insight_data = chunk_dict.pop('content_insight')
                # This would need full reconstruction logic for ContentInsight
                # For now, we'll skip the detailed reconstruction
                
                chunk = EnhancedChunk(**chunk_dict)
                chunks.append(chunk)
            
            self.logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error loading chunks: {e}")
            return []
    
    def analyze_chunk_quality(self, chunks: List[EnhancedChunk]) -> Dict[str, Any]:
        """Analyze the quality of created chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        confidence_scores = [chunk.confidence_score for chunk in chunks]
        
        # Collect entity distribution
        entity_distribution = {}
        for chunk in chunks:
            for entity_type, entities in chunk.content_insight.entities.items():
                if entity_type not in entity_distribution:
                    entity_distribution[entity_type] = 0
                entity_distribution[entity_type] += len(entities)
        
        return {
            'total_chunks': len(chunks),
            'avg_token_count': np.mean(token_counts),
            'min_token_count': min(token_counts),
            'max_token_count': max(token_counts),
            'avg_confidence': np.mean(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'entity_distribution': entity_distribution,
            'document_types': list(set(chunk.content_insight.document_type for chunk in chunks))
        }

def test_enhanced_chunker():
    """Test the enhanced JSON chunker."""
    chunker = EnhancedJSONChunker()
    
    # Create sample JSON data
    sample_data = [{
        'document_id': 'test_doc_001',
        'pages': [{
            'blocks': [{
                'type': 'text',
                'content': 'Apple Inc. reported revenue of $365.8 billion in 2021, representing a 33% increase from 2020. The company\'s CEO Tim Cook announced plans to expand operations in India and China.'
            }]
        }]
    }]
    
    # Save sample data
    with open('test_extraction.json', 'w') as f:
        json.dump(sample_data, f)
    
    # Process the data
    chunks = chunker.process_extracted_json('test_extraction.json')
    
    print(f"Created {len(chunks)} enhanced chunks:")
    for chunk in chunks:
        print(f"  Chunk ID: {chunk.chunk_id}")
        print(f"  Token count: {chunk.token_count}")
        print(f"  Confidence: {chunk.confidence_score}")
        print(f"  Document type: {chunk.content_insight.document_type}")
        print(f"  Entities: {chunk.content_insight.entities}")
        print(f"  Schema elements: {chunk.schema_elements}")
        print()
    
    # Analyze quality
    quality_report = chunker.analyze_chunk_quality(chunks)
    print("Quality Report:")
    for key, value in quality_report.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_enhanced_chunker()
