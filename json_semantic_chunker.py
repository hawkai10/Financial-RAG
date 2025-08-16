import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONSemanticChunker:
    def __init__(
        self,
        min_tokens: int = 100,
        max_tokens: int = 500,
        overlap_tokens: int = 100,
        similarity_threshold: float = 0.75,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        JSON-based semantic chunker that processes structured blocks from Marker extraction.
        
        Args:
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            similarity_threshold: Cosine similarity threshold for grouping
            model_name: Sentence transformer model for embeddings
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        
        # Initialize tokenizer and embedding model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info(f"Initialized JSONSemanticChunker with {model_name}")
        logger.info(f"Token limits: {min_tokens}-{max_tokens} with {overlap_tokens} overlap")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def parse_html_table(self, html: str) -> str:
        """
        Parse HTML table and convert to clean text without keyword-based extraction.
        
        Args:
            html: Raw HTML content from JSON block
            
        Returns:
            Clean text representation of the table
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all table rows
        rows = soup.find_all('tr')
        table_text = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
                
            # Extract text from cells and clean it
            cell_texts = [self.clean_text(cell.get_text(strip=True)) for cell in cells]
            non_empty_cells = [text for text in cell_texts if text]
            
            if non_empty_cells:
                # Join cells with separator for readability
                row_text = " | ".join(non_empty_cells)
                table_text.append(row_text)
                
        return "\n".join(table_text)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up common formatting issues
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        
        return text

    def is_numeric(self, text: str) -> bool:
        """Check if text contains numeric data (amounts, percentages, etc.)."""
        if not text:
            return False
        
        # Remove common non-numeric characters
        clean_text = re.sub(r'[₹,%\s,.-]', '', text)
        
        # Check if remaining text is mostly digits
        return clean_text.isdigit() or (
            len(clean_text) > 0 and 
            sum(c.isdigit() for c in clean_text) / len(clean_text) > 0.5
        )

    def process_json_block(self, block: Dict[str, Any]) -> str:
        """
        Process a single JSON block and convert to clean text.
        
        Args:
            block: Single block from JSON file
            
        Returns:
            Clean text representation of the block
        """
        block_type = block.get('block_type', 'Unknown')
        html_content = block.get('html', '')
        
        if not html_content:
            return ""
            
        if block_type == 'Form':
            # Parse as table and convert to clean text
            return self.parse_html_table(html_content)
            
        elif block_type in ['Table', 'TableHeader']:
            # Parse as structured table
            soup = BeautifulSoup(html_content, 'html.parser')
            rows = soup.find_all('tr')
            
            table_text = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                cell_texts = [self.clean_text(cell.get_text(strip=True)) for cell in cells]
                non_empty_cells = [text for text in cell_texts if text]
                
                if non_empty_cells:
                    table_text.append(" | ".join(non_empty_cells))
                    
            return "\n".join(table_text)
            
        else:
            # For other block types, extract clean text
            soup = BeautifulSoup(html_content, 'html.parser')
            return self.clean_text(soup.get_text())

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for semantic analysis."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Also split on line breaks for table data
        split_sentences = []
        for sentence in sentences:
            if '\n' in sentence:
                parts = sentence.split('\n')
                for part in parts:
                    if part.strip():
                        split_sentences.append(part.strip())
            else:
                if sentence.strip():
                    split_sentences.append(sentence.strip())
                    
        return split_sentences

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

    def group_by_similarity(self, sentences: List[str], embeddings: np.ndarray) -> List[List[str]]:
        """Group sentences by semantic similarity."""
        if len(sentences) <= 1:
            return [sentences]
        
        groups = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0:1]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i:i+1]
            
            # Calculate similarity
            group_mean_embedding = np.mean(current_embedding, axis=0, keepdims=True)
            similarity = cosine_similarity(sentence_embedding, group_mean_embedding)[0][0]
            
            if similarity >= self.similarity_threshold:
                current_group.append(sentence)
                current_embedding = np.vstack([current_embedding, sentence_embedding])
            else:
                groups.append(current_group)
                current_group = [sentence]
                current_embedding = sentence_embedding
        
        if current_group:
            groups.append(current_group)
        
        return groups

    def create_chunks_from_groups(self, groups: List[List[str]]) -> List[str]:
        """Convert sentence groups into properly sized chunks."""
        chunks = []
        
        for group in groups:
            group_text = '\n'.join(group)  # Use newlines for better readability
            group_tokens = self.count_tokens(group_text)
            
            if group_tokens <= self.max_tokens:
                if group_tokens >= self.min_tokens:
                    chunks.append(group_text)
                else:
                    # Try to combine with previous chunk
                    if chunks and self.count_tokens(chunks[-1] + '\n' + group_text) <= self.max_tokens:
                        chunks[-1] = chunks[-1] + '\n' + group_text
                    else:
                        chunks.append(group_text)
            else:
                # Split large groups
                split_chunks = self.split_large_group(group)
                chunks.extend(split_chunks)
        
        return chunks

    def split_large_group(self, sentences: List[str]) -> List[str]:
        """Split a large group into properly sized chunks."""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + ('\n' if current_chunk else '') + sentence
            test_tokens = self.count_tokens(test_chunk)
            
            if test_tokens <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def chunk_json_content(self, json_data: Dict, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Main chunking function for JSON content.
        
        Args:
            json_data: Parsed JSON data from extraction
            source_file: Source file name for metadata
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Starting JSON chunking for {source_file}")
        
        all_text_parts = []
        block_info = []
        
        # Process each block
        blocks = json_data.get('blocks', [])
        for i, block in enumerate(blocks):
            block_text = self.process_json_block(block)
            if block_text.strip():
                all_text_parts.append(block_text)
                block_info.append({
                    'block_id': block.get('id', f'block_{i}'),
                    'block_type': block.get('block_type', 'Unknown'),
                    'page': block.get('page', 0)
                })
        
        if not all_text_parts:
            logger.warning(f"No text content extracted from {source_file}")
            return []
        
        # Combine all text
        full_text = '\n\n'.join(all_text_parts)
        logger.info(f"Extracted {len(full_text)} characters from {len(blocks)} blocks")
        
        # Split into sentences
        sentences = self.split_into_sentences(full_text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        if not sentences:
            return []
        
        # Get embeddings and group
        embeddings = self.get_embeddings(sentences)
        groups = self.group_by_similarity(sentences, embeddings)
        logger.info(f"Created {len(groups)} semantic groups")
        
        # Create final chunks
        chunks = self.create_chunks_from_groups(groups)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Create chunk objects
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            token_count = self.count_tokens(chunk_text)
            chunk_objects.append({
                'chunk_id': f"{source_file}_chunk_{i}",
                'content': chunk_text,
                'token_count': token_count,
                'type': 'financial_document',
                'source_file': source_file,
                'chunk_index': i,
                'metadata': {
                    'chunking_method': 'json_semantic',
                    'similarity_threshold': self.similarity_threshold,
                    'min_tokens': self.min_tokens,
                    'max_tokens': self.max_tokens,
                    'blocks_processed': len(blocks),
                    'extraction_format': 'json'
                }
            })
        
        logger.info(f"Final output: {len(chunk_objects)} chunks")
        return chunk_objects

    def process_extracted_documents(self, input_dir: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Process all JSON files from extraction output.
        
        Args:
            input_dir: Path to directory containing extracted JSON files
            output_file: Optional path to save chunks
            
        Returns:
            List of all chunks
        """
        input_path = Path(input_dir)
        all_chunks = []
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return []
        
        # Process each subdirectory
        for subdir in input_path.iterdir():
            if not subdir.is_dir():
                continue
                
            logger.info(f"Processing directory: {subdir.name}")
            
            # Find JSON file
            json_files = list(subdir.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON file found in {subdir.name}")
                continue
            
            json_file = json_files[0]
            logger.info(f"Reading {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Chunk this document
                doc_chunks = self.chunk_json_content(json_data, source_file=subdir.name)
                all_chunks.extend(doc_chunks)
                
                logger.info(f"Processed {subdir.name}: {len(doc_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved chunks to {output_file}")
        
        return all_chunks


def main():
    """Example usage of JSONSemanticChunker."""
    chunker = JSONSemanticChunker(
        min_tokens=100,
        max_tokens=500,
        overlap_tokens=100,
        similarity_threshold=0.75
    )
    
    # Process JSON files
    base_dir = Path(__file__).parent
    input_dir = base_dir / "New folder"
    output_file = base_dir / "json_chunks.json"
    
    chunks = chunker.process_extracted_documents(
        input_dir=str(input_dir),
        output_file=str(output_file)
    )
    
    # Print summary
    if chunks:
        token_counts = [chunk['token_count'] for chunk in chunks]
        print(f"\n=== JSON CHUNKING SUMMARY ===")
        print(f"Total chunks: {len(chunks)}")
        print(f"Token count range: {min(token_counts)}-{max(token_counts)}")
        print(f"Average tokens per chunk: {sum(token_counts) / len(token_counts):.1f}")
        
        # Show sample chunk
        print(f"\n=== SAMPLE CHUNK ===")
        print(f"Content: {chunks[0]['content'][:300]}...")
        print(f"Tokens: {chunks[0]['token_count']}")
        
        print(f"\nOutput saved to: {output_file}")
    else:
        print("No chunks were created. Check your input directory and files.")


if __name__ == "__main__":
    main()
