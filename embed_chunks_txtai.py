#!/usr/bin/env python3
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any
from init_chunks_db import create_chunks_database
from utils import logger

def load_and_prepare_chunks_txtai_format(file_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks and prepare them in txtai's expected format.
    
    Returns list of dictionaries with 'text' field and metadata fields.
    """
    pass
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pass
    
    prepared_chunks = []
    skipped_count = 0
    
    for i, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            skipped_count += 1
            continue
        
        # Check for required fields
        chunk_text = chunk.get('chunk_text', '').strip()
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        
        if not chunk_text:
            skipped_count += 1
            continue
        
        # Create txtai-compatible document dictionary
        document = {
            'text': chunk_text,  # This is what txtai indexes
            'id': str(chunk_id),  # txtai will use this as the document ID
            'document_name': chunk.get('document_name', ''),
            'context': chunk.get('context', ''),
            'is_table': chunk.get('is_table', False),
            'chunk_index': chunk.get('chunk_index', 0)
        }
        
        # Add optional fields if they exist
        optional_fields = ['num_rows', 'num_cols', 'start_token', 'end_token', 'num_tokens']
        for field in optional_fields:
            if field in chunk:
                document[field] = chunk[field]
        
        prepared_chunks.append(document)
    
    pass
    if skipped_count > 0:
        pass
    
    return prepared_chunks

def create_embeddings_index_txtai(chunks: List[Dict[str, Any]], 
                                  model_path: str = "BAAI/bge-base-en-v1.5",
                                  index_name: str = "business-docs-index"):
    """Create txtai embeddings index using proper format."""
    
    try:
        from txtai import Embeddings
    except ImportError:
        raise ImportError("txtai library not found. Install with: pip install txtai")
    
    pass
    
    # Create embeddings with content storage - this is the key!
    embeddings = Embeddings(
        path=model_path,
        content=True  # Enable content storage for metadata
    )
    
    pass
    
    try:
        # Index using txtai's expected format - just pass the list of dictionaries
        embeddings.index(chunks)
        pass
        
        # Save the index
        pass
        embeddings.save(index_name)
        pass
        
        # Test the index
        pass
        results = embeddings.search("SELECT text, document_name FROM txtai WHERE similar('tax invoice') LIMIT 3")
        pass
        
        for i, result in enumerate(results):
            pass
        
    except Exception as e:
        raise Exception(f"Failed to create embeddings index: {str(e)}")

def main():
    """Main execution function."""
    input_file = "contextualized_chunks.json"
    model_name = "BAAI/bge-base-en-v1.5"
    index_name = "business-docs-index"
    
    pass
    pass
    
    try:
        # Load and prepare chunks in txtai format
        chunks = load_and_prepare_chunks_txtai_format(input_file)
        
        if not chunks:
            pass
            sys.exit(1)
        
        # Preview first chunk
        pass
        pass
        pass
        pass
        
        # Create embeddings index
        create_embeddings_index_txtai(chunks, model_name, index_name)
        
        logger.info("Embeddings index created successfully")
        
        # Automatically update chunks.db after successful embedding
        try:
            logger.info("Updating chunks.db with latest chunk data...")
            create_chunks_database("chunks.db", input_file)
            logger.info("chunks.db updated successfully")
        except Exception as db_error:
            logger.error(f"Failed to update chunks.db: {db_error}")
            # Don't fail the entire process if DB update fails
            print(f"[WARNING] chunks.db update failed: {db_error}")
        
        logger.info("Embedding pipeline completed successfully")
        
        # Usage example
        print(f"\n[INFO] Usage example:")
        print(f"from txtai import Embeddings")
        print(f"embeddings = Embeddings()")
        print(f"embeddings.load('{index_name}')")
        print(f"results = embeddings.search('your query here')")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
