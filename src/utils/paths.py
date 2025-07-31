"""
Centralized path configuration for the Financial RAG system.
All file paths are managed here to ensure consistency across the project.
"""

import os
from pathlib import Path

# Base project directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main folders
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data files
CHUNKS_DB = DATA_DIR / "chunks.db"
FEEDBACK_DB = DATA_DIR / "feedback.db"
SUMMARY_DB = DATA_DIR / "summary.db"
HIERARCHY_DB = DATA_DIR / "hirearchy.db"

# JSON/CSV files
CONTEXTUALIZED_CHUNKS_JSON = DATA_DIR / "contextualized_chunks.json"
CONTEXTUALIZED_CHUNKS_CSV = DATA_DIR / "contextualized_chunks.csv"
SOURCE_MANIFEST_JSON = DATA_DIR / "source_manifest.json"

# Document folders
SOURCE_DOCUMENTS = DATA_DIR / "Source_Documents"
BUSINESS_DOCS_INDEX = DATA_DIR / "business-docs-index"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDING_CACHE = DATA_DIR / "embedding_cache"

# Log files
RAG_APP_LOG = LOGS_DIR / "rag_app.log"

# Extraction logs (create if it doesn't exist)
EXTRACTION_LOGS = LOGS_DIR / "extraction_logs"

# Configuration files
CONFIG_DIR = SRC_DIR / "utils"

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, LOGS_DIR, DOCS_DIR, TESTS_DIR, SCRIPTS_DIR,
        EXTRACTION_LOGS, SOURCE_DOCUMENTS, BUSINESS_DOCS_INDEX,
        EMBEDDINGS_DIR, EMBEDDING_CACHE
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_relative_path(target_path: Path, from_path: Path = None):
    """Get relative path from one location to another."""
    if from_path is None:
        from_path = Path.cwd()
    
    try:
        return os.path.relpath(target_path, from_path)
    except ValueError:
        # If relative path can't be computed, return absolute path
        return str(target_path.absolute())

# Ensure directories exist when module is imported
ensure_directories()
