#!/usr/bin/env python3
"""
Initialize the chunks database with proper table structure.
This script creates the chunks.db file and tables if they don't exist.
"""

import sqlite3
import os
import json
from utils import logger

def create_chunks_database(db_path: str = "chunks.db", chunks_file: str = "contextualized_chunks.json"):
    """Create chunks database and populate from JSON file."""
    
    logger.info(f"Creating chunks database at: {db_path}")
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_name TEXT,
                chunk_text TEXT,
                context TEXT,
                is_table BOOLEAN DEFAULT 0,
                chunk_index INTEGER DEFAULT 0,
                start_token INTEGER DEFAULT 0,
                end_token INTEGER DEFAULT 0,
                num_tokens INTEGER DEFAULT 0,
                num_rows INTEGER,
                num_cols INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        logger.info("Chunks table created successfully")
        
        # Check if chunks.json exists and populate database
        if os.path.exists(chunks_file):
            logger.info(f"Loading chunks from {chunks_file}")
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Insert chunks into database
            for chunk in chunks_data:
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, document_name, chunk_text, context, is_table,
                        chunk_index, start_token, end_token, num_tokens, num_rows, num_cols
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(chunk.get('chunk_id', '')),
                    chunk.get('document_name', ''),
                    chunk.get('chunk_text', ''),
                    chunk.get('context', ''),
                    chunk.get('is_table', False),
                    chunk.get('chunk_index', 0),
                    chunk.get('start_token', 0),
                    chunk.get('end_token', 0),
                    chunk.get('num_tokens', 0),
                    chunk.get('num_rows'),
                    chunk.get('num_cols')
                ))
            
            conn.commit()
            logger.info(f"Successfully inserted {len(chunks_data)} chunks into database")
            
        else:
            logger.warning(f"Chunks file {chunks_file} not found. Database created but empty.")
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_name ON chunks(document_name)')
        
        conn.commit()
        logger.info("Database indexes created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create chunks database: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def verify_database(db_path: str = "chunks.db"):
    """Verify the database was created correctly."""
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} does not exist")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        if not cursor.fetchone():
            logger.error("Chunks table does not exist")
            return False
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        logger.info(f"Database verification successful: {count} chunks in database")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

if __name__ == "__main__":
    # Create the database
    success = create_chunks_database()
    
    if success:
        # Verify it was created correctly
        verify_database()
        logger.info("SUCCESS: Chunks database initialization complete")
    else:
        logger.error("ERROR: Failed to initialize chunks database")
