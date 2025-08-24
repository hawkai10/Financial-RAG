#!/usr/bin/env python3
"""
Simple Flask API server to bridge the React UI with the RAG backend.
This creates REST endpoints that the UI can call.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
import threading

# Fix Unicode logging issues
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Reconfigure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True
)

# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import traceback
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from typing import Dict, List, Any

# Import your existing RAG functions
from rag_backend import rag_query_enhanced
from config import config
from utils import logger, validate_and_sanitize_query

# Import the new marked pipeline orchestrator (optional)
def start_background_monitoring(*args, **kwargs):
    return False

def stop_background_monitoring(*args, **kwargs):
    return False

def is_monitoring_active() -> bool:
    return False

# Auto-ingestion settings
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() in ("1", "true", "yes")
SOURCE_DOCUMENTS_DIR = Path(os.getenv("SOURCE_DOCUMENTS_DIR", "Source_Documents")).resolve()
EXTRACTED_DIR = Path(os.getenv("EXTRACTED_DIR", "New folder")).resolve()
CHUNK_LOGS_DIR = Path("chunk_logs").resolve()
PROCESSING_STATE_FILE = Path(".processing_state.json").resolve()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".md"}

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

@dataclass
class DocumentState:
    """Track processing state of each document"""
    source_path: str
    size: int
    modified_time: float
    content_hash: str
    extracted: bool = False
    extracted_path: Optional[str] = None
    chunked: bool = False
    chunk_log_path: Optional[str] = None
    embedded: bool = False
    last_processed: Optional[float] = None
    error: Optional[str] = None

def _calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA-256 hash of file content"""
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def _load_processing_state() -> Dict[str, DocumentState]:
    """Load processing state from disk"""
    if not PROCESSING_STATE_FILE.exists():
        return {}
    
    try:
        with open(PROCESSING_STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {k: DocumentState(**v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"[STATE] Could not load processing state: {e}")
        return {}

def _save_processing_state(state: Dict[str, DocumentState]) -> None:
    """Save processing state to disk"""
    try:
        with open(PROCESSING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({k: asdict(v) for k, v in state.items()}, f, indent=2)
    except Exception as e:
        logger.error(f"[STATE] Could not save processing state: {e}")

def _scan_source_documents() -> Dict[str, DocumentState]:
    """Scan source documents and build current state"""
    documents = {}
    
    if not SOURCE_DOCUMENTS_DIR.exists():
        logger.warning(f"[SCAN] Source documents directory not found: {SOURCE_DOCUMENTS_DIR}")
        return documents
    
    for file_path in SOURCE_DOCUMENTS_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                content_hash = _calculate_file_hash(file_path)
                
                documents[str(file_path)] = DocumentState(
                    source_path=str(file_path),
                    size=stat.st_size,
                    modified_time=stat.st_mtime,
                    content_hash=content_hash
                )
            except Exception as e:
                logger.warning(f"[SCAN] Could not process {file_path}: {e}")
    
    return documents

def _check_extraction_status(doc_state: DocumentState) -> bool:
    """Check if document has been extracted"""
    source_path = Path(doc_state.source_path)
    stem = source_path.stem

    # Common candidate locations (flat and nested per Marker default layout)
    candidate_paths = [
        EXTRACTED_DIR / f"{stem}.json",
        EXTRACTED_DIR / stem / f"{stem}.json",
        EXTRACTED_DIR / f"{stem}_extracted.json",
        EXTRACTED_DIR / stem / f"{stem}_extracted.json",
    ]

    # Check direct candidates first
    for p in candidate_paths:
        if p.exists() and p.is_file():
            # Avoid selecting meta files
            if p.name.endswith("_meta.json"):
                continue
            doc_state.extracted = True
            doc_state.extracted_path = str(p)
            logger.info(f"[EXTRACT] Found existing extraction for {source_path.name} at {p}")
            return True

    # Fallback: recursive search within EXTRACTED_DIR for exact stem match
    try:
        for p in EXTRACTED_DIR.rglob("*.json"):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith("_meta.json"):
                continue
            # Accept exact matches like stem.json or stem_extracted.json
            if name == f"{stem}.json" or name == f"{stem}_extracted.json":
                doc_state.extracted = True
                doc_state.extracted_path = str(p)
                logger.info(f"[EXTRACT] Found existing extraction for {source_path.name} at {p}")
                return True
    except Exception as e:
        logger.debug(f"[EXTRACT] Error while searching extracted files recursively: {e}")

    return False

def _check_chunking_status(doc_state: DocumentState) -> bool:
    """Check if document has been chunked"""
    if not doc_state.extracted_path:
        return False
    
    # Look for chunk log file
    source_name = Path(doc_state.source_path).stem
    chunk_log_patterns = [
        f"{source_name}_parent_child_chunks.json",
        f"{source_name}_chunks.json",
        f"{source_name}.chunks.json",
    ]

    # Direct checks in the chunk logs directory
    for pattern in chunk_log_patterns:
        chunk_log_path = CHUNK_LOGS_DIR / pattern
        if chunk_log_path.exists():
            doc_state.chunked = True
            doc_state.chunk_log_path = str(chunk_log_path)
            return True

    # Fallback: recursive search in chunk_logs
    try:
        for p in CHUNK_LOGS_DIR.rglob("*.json"):
            if p.name in chunk_log_patterns:
                doc_state.chunked = True
                doc_state.chunk_log_path = str(p)
                return True
    except Exception:
        pass
    
    return False

def _check_embedding_status(doc_state: DocumentState) -> bool:
    """Check if document chunks have been embedded"""
    if not doc_state.chunked:
        return False
    
    # Check if embeddings exist in vector store
    try:
        from parent_child.chroma_child_store import ChromaChildStore
        from parent_child.parent_store import ParentStore
        
        # Check both parent and child stores
        parent_store = ParentStore()
        child_store = ChromaChildStore()
        
        # Use source file name as document ID
        doc_id = Path(doc_state.source_path).stem
        
        # Check if any chunks exist for this document
        parent_count = parent_store.count_for_document(doc_id) if hasattr(parent_store, 'count_for_document') else 0
        child_count = child_store.count_for_document(doc_id) if hasattr(child_store, 'count_for_document') else 0
        
        if parent_count > 0 or child_count > 0:
            doc_state.embedded = True
            return True
            
    except Exception as e:
        logger.debug(f"[EMBED] Could not check embedding status for {doc_state.source_path}: {e}")
    
    return False

def _cleanup_old_version(doc_state: DocumentState, old_state: DocumentState) -> None:
    """Clean up old version of document from all systems"""
    logger.info(f"[CLEANUP] Removing old version of {Path(doc_state.source_path).name}")
    
    try:
        # Remove old extracted file
        if old_state.extracted_path and Path(old_state.extracted_path).exists():
            Path(old_state.extracted_path).unlink()
            logger.info(f"[CLEANUP] Removed old extraction: {old_state.extracted_path}")
        
        # Remove old chunk log
        if old_state.chunk_log_path and Path(old_state.chunk_log_path).exists():
            Path(old_state.chunk_log_path).unlink()
            logger.info(f"[CLEANUP] Removed old chunk log: {old_state.chunk_log_path}")
        
        # Remove from vector stores
        doc_id = Path(doc_state.source_path).stem
        try:
            from parent_child.chroma_child_store import ChromaChildStore
            from parent_child.parent_store import ParentStore
            
            parent_store = ParentStore()
            child_store = ChromaChildStore()
            
            # Delete chunks for this document
            if hasattr(parent_store, 'delete_by_document_id'):
                parent_store.delete_by_document_id(doc_id)
            if hasattr(child_store, 'delete_by_document_id'):
                child_store.delete_by_document_id(doc_id)
                
            logger.info(f"[CLEANUP] Removed old embeddings for {doc_id}")
            
        except Exception as e:
            logger.warning(f"[CLEANUP] Could not clean embeddings for {doc_id}: {e}")
            
    except Exception as e:
        logger.error(f"[CLEANUP] Error cleaning up old version: {e}")

def _extract_document(doc_state: DocumentState) -> bool:
    """Extract document using Marker"""
    logger.info(f"[EXTRACT] Processing {Path(doc_state.source_path).name}")
    
    try:
        from extraction import run_marker
        import tempfile
        import shutil
        
        # Set environment variables for Marker
        old_env = os.environ.copy()
        os.environ["MARKER_INPUT_PATH"] = doc_state.source_path
        os.environ["MARKER_OUTPUT_DIR"] = str(EXTRACTED_DIR)
        os.environ["MARKER_OUTPUT_FORMAT"] = "json"
        
        # Run extraction
        run_marker()
        
        # Restore environment
        os.environ.clear()
        os.environ.update(old_env)
        
        # Check if extraction was successful
        return _check_extraction_status(doc_state)
        
    except Exception as e:
        doc_state.error = f"Extraction failed: {e}"
        logger.error(f"[EXTRACT] Failed to extract {doc_state.source_path}: {e}")
        logger.error(f"[EXTRACT] Full traceback: {traceback.format_exc()}")
        return False

def _chunk_document(doc_state: DocumentState) -> bool:
    """Chunk document using parent-child pipeline"""
    if not doc_state.extracted_path:
        return False
    
    logger.info(f"[CHUNK] Processing {Path(doc_state.source_path).name}")
    
    try:
        from parent_child.pipeline import ParentChildPipeline
        
        pipeline = ParentChildPipeline()
        doc_id = Path(doc_state.source_path).stem
        
        result = pipeline.ingest_extracted_json(doc_state.extracted_path, document_id=doc_id)
        
        if result and result.get('parents', 0) > 0:
            doc_state.chunked = True
            doc_state.embedded = True  # Pipeline also embeds
            doc_state.chunk_log_path = result.get('log_path')
            return True
            
    except Exception as e:
        doc_state.error = f"Chunking failed: {e}"
        logger.error(f"[CHUNK] Failed to chunk {doc_state.source_path}: {e}")
        logger.error(f"[CHUNK] Full traceback: {traceback.format_exc()}")
    
    return False

def _document_needs_processing(doc_state: DocumentState, old_state: Optional[DocumentState]) -> tuple[bool, str]:
    """Check if document needs processing and return reason"""
    
    # New document
    if not old_state:
        return True, "new document"
    
    # File changed
    if (doc_state.content_hash != old_state.content_hash or 
        doc_state.modified_time != old_state.modified_time or 
        doc_state.size != old_state.size):
        return True, "file changed"
    
    # Not extracted
    if not _check_extraction_status(doc_state):
        return True, "not extracted"
    
    # Not chunked  
    if not _check_chunking_status(doc_state):
        return True, "not chunked"
    
    # Not embedded
    if not _check_embedding_status(doc_state):
        return True, "not embedded"
    
    return False, "up to date"

def _comprehensive_document_processor():
    """Comprehensive document processing with state tracking"""
    try:
        logger.info("[PROCESSOR] Starting comprehensive document processing...")
        
        # Ensure directories exist
        EXTRACTED_DIR.mkdir(exist_ok=True)
        CHUNK_LOGS_DIR.mkdir(exist_ok=True)
        
        # Load previous state
        old_states = _load_processing_state()
        
        # Scan current documents
        current_documents = _scan_source_documents()
        
        if not current_documents:
            logger.info("[PROCESSOR] No source documents found")
            return
        
        # Track processing stats
        total_docs = len(current_documents)
        processed_docs = 0
        new_docs = 0
        changed_docs = 0
        up_to_date_docs = 0
        failed_docs = 0
        
        logger.info(f"[PROCESSOR] Found {total_docs} source documents")
        
        # Process each document
        for doc_path, doc_state in current_documents.items():
            old_state = old_states.get(doc_path)
            needs_processing, reason = _document_needs_processing(doc_state, old_state)
            
            doc_name = Path(doc_path).name
            
            if not needs_processing:
                logger.info(f"[PROCESSOR] ✅ {doc_name} - {reason}")
                up_to_date_docs += 1
                # Copy old state information
                if old_state:
                    doc_state.extracted = old_state.extracted
                    doc_state.extracted_path = old_state.extracted_path
                    doc_state.chunked = old_state.chunked
                    doc_state.chunk_log_path = old_state.chunk_log_path
                    doc_state.embedded = old_state.embedded
                continue
            
            logger.info(f"[PROCESSOR] 🔄 Processing {doc_name} - {reason}")
            
            # If file changed, clean up old version
            if old_state and reason == "file changed":
                _cleanup_old_version(doc_state, old_state)
                changed_docs += 1
            elif reason == "new document":
                new_docs += 1
            
            # Reset processing flags for re-processing
            doc_state.extracted = False
            doc_state.chunked = False
            doc_state.embedded = False
            doc_state.error = None
            
            success = True
            
            # Step 1: Extract if needed
            if not _check_extraction_status(doc_state):
                if not _extract_document(doc_state):
                    success = False
            
            # Step 2: Chuck and embed if extracted
            if success and doc_state.extracted and not _check_chunking_status(doc_state):
                if not _chunk_document(doc_state):
                    success = False
            
            # Update final status
            if success:
                doc_state.last_processed = time.time()
                logger.info(f"[PROCESSOR] ✅ {doc_name} - fully processed")
                processed_docs += 1
            else:
                logger.error(f"[PROCESSOR] ❌ {doc_name} - processing failed: {doc_state.error}")
                failed_docs += 1
        
        # Clean up state for removed documents
        removed_docs = []
        for old_path in old_states:
            if old_path not in current_documents:
                removed_docs.append(old_path)
                old_state = old_states[old_path]
                logger.info(f"[PROCESSOR] 🗑️ Removing deleted document: {Path(old_path).name}")
                _cleanup_old_version(DocumentState(source_path=old_path, size=0, modified_time=0, content_hash=""), old_state)
        
        # Save updated state
        _save_processing_state(current_documents)
        
        # Final summary
        logger.info("[PROCESSOR] ✅ Processing complete!")
        logger.info(f"[PROCESSOR] Summary:")
        logger.info(f"  - Total documents: {total_docs}")
        logger.info(f"  - Up to date: {up_to_date_docs}")
        logger.info(f"  - Newly processed: {processed_docs}")
        logger.info(f"  - New documents: {new_docs}")
        logger.info(f"  - Changed documents: {changed_docs}")
        logger.info(f"  - Failed: {failed_docs}")
        logger.info(f"  - Removed: {len(removed_docs)}")
        
        # Show chunk logs created
        if CHUNK_LOGS_DIR.exists():
            log_files = list(CHUNK_LOGS_DIR.glob("*.json"))
            logger.info(f"  - Chunk log files: {len(log_files)}")
            
    except Exception as e:
        logger.error(f"[PROCESSOR] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

try:
    import importlib
    _mpo = importlib.import_module('marked_pipeline_orchestrator')
    start_background_monitoring = getattr(_mpo, 'start_background_monitoring', start_background_monitoring)
    stop_background_monitoring = getattr(_mpo, 'stop_background_monitoring', stop_background_monitoring)
    is_monitoring_active = getattr(_mpo, 'is_monitoring_active', is_monitoring_active)
except Exception:
    pass

# Permanent routing: always use RAG backend (parent-child path disabled)
logger.info("[ROUTING] Using RAG backend permanently (parent-child mode disabled)")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# No global embeddings (txtai removed)

def initialize_embeddings():
    """No-op: txtai embeddings removed."""
    return False

# Lazy index of local document paths to resolve full paths from names
_DOCUMENT_PATH_INDEX = None  # type: ignore[var-annotated]
_DOCUMENT_PATH_INDEX_STEM = None  # type: ignore[var-annotated]

def _build_document_path_index(base_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    index_stem: Dict[str, str] = {}
    try:
        for root, _, files in os.walk(base_dir):
            for name in files:
                full = os.path.join(root, name)
                # Prefer first seen; avoid overriding
                index.setdefault(name, full)
                stem, _ = os.path.splitext(name)
                index_stem.setdefault(stem, full)
    except Exception as e:
        logger.warning(f"Failed to build document path index from {base_dir}: {e}")
    return index, index_stem

def _resolve_full_path_from_name(name_candidate: str) -> str:
    """Try to resolve a full path for a document given only its name or ID."""
    if not name_candidate:
        return ""
    try:
        # If it's already an existing absolute or relative path, return as-is
        if os.path.exists(name_candidate):
            return os.path.abspath(name_candidate)

        # Build index lazily
        global _DOCUMENT_PATH_INDEX, _DOCUMENT_PATH_INDEX_STEM
        if _DOCUMENT_PATH_INDEX is None or _DOCUMENT_PATH_INDEX_STEM is None:
            base_dir = os.path.join(os.getcwd(), 'Source_Documents')
            _DOCUMENT_PATH_INDEX, _DOCUMENT_PATH_INDEX_STEM = _build_document_path_index(base_dir)

        base = os.path.basename(str(name_candidate))
        stem, _ = os.path.splitext(base)

        # Exact filename match
        if base in _DOCUMENT_PATH_INDEX:
            return _DOCUMENT_PATH_INDEX[base]
        # Stem match (file name without extension)
        if stem in _DOCUMENT_PATH_INDEX_STEM:
            return _DOCUMENT_PATH_INDEX_STEM[stem]
    except Exception as e:
        logger.debug(f"Path resolution failed for {name_candidate}: {e}")
    return ""

def format_chunks_for_ui(chunks: List[Dict]) -> List[Dict]:
    """Convert backend chunk format to UI format"""
    documents = []
    
    if not chunks:
        logger.warning("No chunks provided to format_chunks_for_ui")
        return documents
    
    logger.info(f"Formatting {len(chunks)} chunks for UI")
    
    for i, chunk in enumerate(chunks):
        try:
            # Handle different chunk formats
            if isinstance(chunk, dict):
                # Extract document info from chunk (prefer child fields when present)
                chunk_id = chunk.get('child_id') or chunk.get('chunk_id') or chunk.get('id', f'doc_{i}')
                document_name = (
                    chunk.get('document_id')
                    or chunk.get('document_name')
                    or chunk.get('source')
                    or chunk.get('file')
                    or 'Unknown Document'
                )

                # Try to resolve a full file path from the document name/id when possible
                resolved_path = _resolve_full_path_from_name(str(document_name))

                # Get text content from various possible fields (prefer child text)
                text_content = chunk.get('text', chunk.get('chunk_text', chunk.get('content', '')))
                # Do not truncate snippet per requirements
                snippet = text_content
                
                # Derive file type and last modified date from source path when possible
                file_type = 'unknown'
                last_modified = 'Unknown'
                try:
                    candidates = []
                    if isinstance(document_name, (str, os.PathLike)):
                        candidates.append(str(document_name))
                    for key in ('document_path', 'source_path', 'path', 'file_path', 'source', 'file'):
                        val = chunk.get(key)
                        if isinstance(val, (str, os.PathLike)):
                            candidates.append(str(val))
                    # Prefer resolved full path if found
                    if resolved_path:
                        candidates.insert(0, resolved_path)

                    # Determine file extension from first candidate that has one
                    for c in candidates:
                        _, ext = os.path.splitext(c)
                        if ext:
                            file_type = ext.lstrip('.').lower()
                            break

                    # Determine last modified date from first existing path
                    for c in candidates:
                        if os.path.exists(c):
                            mtime = os.path.getmtime(c)
                            last_modified = datetime.fromtimestamp(mtime).strftime('%d.%m.%Y')
                            break
                except Exception:
                    # Keep safe fallbacks if anything goes wrong
                    pass

                # Prefer reranker score for ordering in UI when available
                score_val = (
                    chunk.get('final_rerank_score',
                              chunk.get('retrieval_score',
                                       chunk.get('score', 0.0)))
                )
                # Create UI-compatible document object
                doc = {
                    'id': str(chunk_id),
                    'sourceType': 'Windows Shares',  # Default for now
                    'sourcePath': str(resolved_path or document_name) if document_name else 'Unknown Path',
                    'fileType': file_type,  # Actual file extension when available
                    'title': os.path.basename(str(document_name)) if document_name else f'Document {i+1}',
                    'date': last_modified,  # Last edited date when available
                    'snippet': snippet,
                    'author': 'System',
                    'score': score_val
                }
                documents.append(doc)
                
            elif isinstance(chunk, str):
                # Handle string chunks (no truncation)
                snippet = chunk
                doc = {
                    'id': f'doc_{i}',
                    'sourceType': 'Windows Shares',
                    'sourcePath': 'Text Content',
                    'fileType': 'txt',
                    'title': f'Text Document {i+1}',
                    'date': 'Unknown',
                    'snippet': snippet,
                    'author': 'System',
                    'score': 0.0
                }
                documents.append(doc)
                
            else:
                logger.warning(f"Unknown chunk format at index {i}: {type(chunk)}")
                
        except Exception as e:
            logger.error(f"Error formatting chunk {i}: {e}")
            logger.error(f"Chunk content: {chunk}")
            continue
    
    logger.info(f"Successfully formatted {len(documents)} documents for UI")
    return documents

def format_ai_response(raw_response: str) -> Dict[str, Any]:
    """Format raw AI response into structured UI format"""
    if not raw_response or not raw_response.strip():
        return {
            'summary': 'No response generated',
            'items': []
        }
    
    try:
        import re
        
        # Clean the response
        cleaned_response = raw_response.strip()
        
        # Convert HTML to readable text with proper formatting
        # Replace HTML tags with appropriate plain text formatting
        
        # Convert paragraphs
        cleaned_response = re.sub(r'<p>(.*?)</p>', r'\1\n\n', cleaned_response, flags=re.DOTALL)
        
        # Enhanced table handling - preserve markdown table structure
        # First, handle HTML tables by converting them to markdown format
        def convert_html_table_to_markdown(match):
            table_content = match.group(1)
            rows = re.findall(r'<tr>(.*?)</tr>', table_content, flags=re.DOTALL)
            markdown_rows = []
            
            for i, row in enumerate(rows):
                cells = re.findall(r'<t[hd]>(.*?)</t[hd]>', row, flags=re.DOTALL)
                if cells:
                    # Clean cell content and join with pipes
                    clean_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                    markdown_row = '| ' + ' | '.join(clean_cells) + ' |'
                    markdown_rows.append(markdown_row)
                    
                    # Add header separator after first row
                    if i == 0:
                        separator = '| ' + ' | '.join(['---'] * len(clean_cells)) + ' |'
                        markdown_rows.append(separator)
            
            return '\n'.join(markdown_rows) + '\n\n'
        
        # Convert HTML tables to markdown
        cleaned_response = re.sub(r'<table[^>]*>(.*?)</table>', convert_html_table_to_markdown, cleaned_response, flags=re.DOTALL)
        
        # Handle table rows that aren't in full table tags
        cleaned_response = re.sub(r'<tr>(.*?)</tr>', r'\1\n', cleaned_response, flags=re.DOTALL)
        # Handle table cells with pipes - be more careful with spacing
        cleaned_response = re.sub(r'<td[^>]*>(.*?)</td>', r'| \1 ', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<th[^>]*>(.*?)</th>', r'| \1 ', cleaned_response, flags=re.DOTALL)
        
        # Convert strong/bold tags
        cleaned_response = re.sub(r'<strong>(.*?)</strong>', r'**\1**', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<b>(.*?)</b>', r'**\1**', cleaned_response, flags=re.DOTALL)
        
        # Convert lists
        cleaned_response = re.sub(r'<li>(.*?)</li>', r'• \1\n', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<ul[^>]*>(.*?)</ul>', r'\1\n', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<ol[^>]*>(.*?)</ol>', r'\1\n', cleaned_response, flags=re.DOTALL)
        
        # Remove any remaining HTML tags but preserve content
        cleaned_response = re.sub(r'<[^>]+>', '', cleaned_response)
        
        # Fix common formatting issues
        # Remove excessive whitespace but preserve table structure
        cleaned_response = re.sub(r'[ \t]+', ' ', cleaned_response)  # Multiple spaces/tabs to single space
        cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)  # Multiple newlines to double
        
        # Fix table formatting issues
        lines = cleaned_response.split('\n')
        fixed_lines = []
        for line in lines:
            if '|' in line and not line.strip().startswith('|'):
                # Fix broken table rows
                line = '| ' + line.strip() + ' |'
            elif '|' in line:
                # Clean up existing table rows
                line = re.sub(r'\|\s*\|', '| |', line)  # Fix empty cells
                line = re.sub(r'\s*\|\s*', ' | ', line)  # Standardize spacing
            fixed_lines.append(line)
        
        cleaned_response = '\n'.join(fixed_lines)
        cleaned_response = cleaned_response.strip()
        
        # Ensure we don't truncate the response
        if len(cleaned_response) > 10000:  # Only truncate if extremely long
            logger.warning("Response is very long, considering truncation")
            # Find a good break point (end of paragraph or table)
            truncate_at = 9500
            while truncate_at < len(cleaned_response) and cleaned_response[truncate_at] not in '\n\r':
                truncate_at += 1
            if truncate_at < len(cleaned_response):
                cleaned_response = cleaned_response[:truncate_at] + "\n\n[Response truncated for display...]"
        
        # Extract summary from first line or first paragraph
        lines = [line.strip() for line in cleaned_response.split('\n') if line.strip()]
        first_line = lines[0] if lines else cleaned_response[:100]
        
        # Create summary from first meaningful line
        summary = first_line
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        # Return structured response
        return {
            'summary': summary,
            'items': [{
                'title': 'Analysis Results',
                'text': cleaned_response,
                'references': []
            }]
        }
        
    except Exception as e:
        logger.error(f"Error formatting AI response: {e}")
        # Fallback - return raw response with basic HTML cleanup
        import re
        fallback_text = re.sub(r'<[^>]+>', '', raw_response)
        return {
            'summary': 'Analysis complete',
            'items': [{
                'title': 'Response',
                'text': fallback_text,
                'references': []
            }]
        }
        return {
            'summary': 'Response generated',
            'items': [{
                'title': 'AI Response',
                'text': raw_response,
                'references': []
            }]
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embeddings_loaded': False,
        'document_monitoring_active': is_monitoring_active()
    })

@app.route('/monitoring-status', methods=['GET'])
def monitoring_status():
    """Check document monitoring status"""
    return jsonify({
        'monitoring_active': is_monitoring_active(),
        'message': 'Background document monitoring is active' if is_monitoring_active() 
                  else 'Background document monitoring is not active'
    })

@app.route('/search', methods=['POST'])
def search():
    """Main search endpoint that the UI will call"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        filters = data.get('filters', {})
        
        # Validate query
        if not query or not query.strip():
            return jsonify({'error': 'Query is required'}), 400
        
        # Sanitize query
        sanitized_query = validate_and_sanitize_query(query)
        if not sanitized_query:
            return jsonify({'error': 'Invalid query'}), 400
        
    # Offline mode: no embeddings check
        
        logger.info(f"[SEARCH] UI Search request: {sanitized_query}")

    # Parent-child path removed: always use RAG
        
        # Try the full RAG pipeline first (main method)
        try:
            logger.info("[RAG] Attempting enhanced RAG search...")
            
            # Use a thread pool executor to run the async function
            import concurrent.futures
            import asyncio
            
            def run_async_rag():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(rag_query_enhanced(
                        question=sanitized_query,
                        topn=10,
                        filters=None,
                        enable_reranking=True,
                        session_id=None,
                        enable_optimization=True
                    ))
                finally:
                    loop.close()
            
            # Run in a separate thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_rag)
                try:
                    result = future.result(timeout=120)  # increased timeout for RAG
                except concurrent.futures.TimeoutError:
                    logger.error("[TIMEOUT] RAG processing exceeded 120s timeout")
                    return jsonify({'error': 'Search timed out. Please try again.'}), 504
            logger.info(f"RAG result type: {type(result)}")
            logger.info(f"RAG result keys (if dict): {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract answer and chunks from result with better handling
            if isinstance(result, dict):
                answer = result.get('answer', result.get('response', 'No answer generated'))
                # Prefer child chunks in reranker order; fallback to parent chunks
                chunks = result.get('top_children_chunks') or result.get('chunks') or result.get('retrieved_chunks', [])
                logger.info(f"Found answer: {answer[:100] if answer else 'None'}...")
                logger.info(f"Found {len(chunks)} chunks")
            else:
                # If result is just a string answer
                answer = str(result)
                chunks = []
                logger.info(f"String result: {answer[:100]}...")
            
            # Ensure chunks is a list
            if not isinstance(chunks, list):
                logger.warning(f"Chunks is not a list, type: {type(chunks)}")
                chunks = []
            
            # Format chunks for UI
            documents = format_chunks_for_ui(chunks)
            
            # Format AI response using the new formatting function
            ai_response = format_ai_response(answer)
            
            # LLM LOG DISABLED - log_llm_interaction(
            #     phase="FINAL_API_FORMATTING",
            #     content=f"RAW: {answer}\n\nFORMATTED SUMMARY: {ai_response.get('summary', 'N/A')}\n\nFORMATTED ITEMS: {len(ai_response.get('items', []))} items",
            #     method="rag_enhanced",
            #     documents_count=len(documents),
            #     summary_length=len(ai_response.get('summary', '')),
            #     items_count=len(ai_response.get('items', []))
            # )
            
            logger.info(f"[SUCCESS] Enhanced RAG search successful: {len(documents)} documents for Q='{sanitized_query[:80]}...'")
            
            return jsonify({
                'documents': documents,
                'aiResponse': ai_response,
                'query': sanitized_query,
                'status': 'success',
                'method': 'rag_enhanced'
            })
            
        except Exception as rag_error:
            logger.error(f"[ERROR] Enhanced RAG search failed: {rag_error}")
            logger.error(traceback.format_exc())
            
            return jsonify({'error': f'Enhanced RAG search failed: {str(rag_error)}'}), 500
        
    except Exception as e:
        logger.error(f"[ERROR] Search error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/filters', methods=['GET'])
def get_available_filters():
    """Get available filter options"""
    return jsonify({
        'fileTypes': ['pdf', 'word', 'excel', 'ppt', 'txt'],
        'dataSources': ['Windows Shares', 'Local Documents'],
        'timeRanges': ['all', 'week', 'month', '3months', 'year']
    })

@app.route('/pdf', methods=['GET'])
def serve_pdf():
    """Serve PDF files from Source_Documents folder"""
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'error': 'Path parameter is required'}), 400
        
        # Security: ensure the path is within Source_Documents
        base_dir = os.path.abspath(os.path.join(os.getcwd(), 'Source_Documents'))
        requested_path = os.path.abspath(os.path.join(base_dir, file_path))
        
        # Check if the requested path is within base directory
        if not requested_path.startswith(base_dir):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists
        if not os.path.isfile(requested_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Check if download parameter is set
        download = request.args.get('download', 'false').lower() == 'true'
        
        return send_file(
            requested_path,
            as_attachment=download,
            download_name=os.path.basename(requested_path) if download else None
        )
        
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recent-documents', methods=['GET'])
def get_recent_documents():
    """Get recently accessed documents by scanning Source_Documents folder"""
    try:
        base_dir = os.path.join(os.getcwd(), 'Source_Documents')

        if not os.path.isdir(base_dir):
            logger.warning(f"Source_Documents folder not found at: {base_dir}")
            return jsonify({'documents': [], 'status': 'success'})

        def map_file_type(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.pdf']:
                return 'pdf'
            if ext in ['.doc', '.docx']:
                return 'word'
            if ext in ['.xls', '.xlsx', '.csv']:
                return 'excel'
            if ext in ['.ppt', '.pptx']:
                return 'ppt'
            if ext in ['.html', '.htm']:
                return 'html'
            if ext in ['.txt', '.md']:
                return 'txt'
            return 'txt'

        files: List[Dict[str, Any]] = []
        max_files = 20

        for root, _, filenames in os.walk(base_dir):
            for name in filenames:
                full_path = os.path.join(root, name)
                try:
                    mtime = os.path.getmtime(full_path)
                except Exception:
                    continue
                files.append({
                    'path': full_path,
                    'name': name,
                    'mtime': mtime,
                    'fileType': map_file_type(full_path)
                })

        files.sort(key=lambda x: x['mtime'], reverse=True)
        top_files = files[:max_files]

        recent_docs = []
        for i, f in enumerate(top_files, start=1):
            recent_docs.append({
                'id': f'doc_{i}',
                'title': f['name'],
                'fileType': f['fileType'],
                'sourcePath': f['path'],
                'lastAccessed': datetime.fromtimestamp(f['mtime']).isoformat() + 'Z',
                'sourceType': 'Windows Shares'
            })

        return jsonify({'documents': recent_docs, 'status': 'success'})

    except Exception as e:
        logger.error(f"[ERROR] Failed to get recent documents: {e}")
        return jsonify({'error': f'Failed to get recent documents: {str(e)}'}), 500


@app.route('/search-stream', methods=['POST'])
def search_stream():
    """
    Streaming search endpoint that returns chunks first, then AI response
    """
    def generate_response():
        try:
            data = request.get_json()
            query = data.get('query', '')
            filters = data.get('filters', {})
            
            # Validate query
            if not query or not query.strip():
                yield f"data: {json.dumps({'error': 'Query is required'})}\n\n"
                return
            
            # Sanitize query
            sanitized_query = validate_and_sanitize_query(query)
            if not sanitized_query:
                yield f"data: {json.dumps({'error': 'Invalid query'})}\n\n"
                return
            
            # Offline mode: no embeddings check
            
            logger.info(f"[STREAM] Starting streaming search for: {sanitized_query}")
            
            # Parent-child streaming path removed: always use RAG

            # Try the full RAG pipeline (classic)
            try:
                logger.info("[STREAM] Getting chunks first...")
                
                # Use a thread pool executor to run the async function
                import concurrent.futures
                
                def run_async_rag():
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(rag_query_enhanced(
                            question=sanitized_query,
                            topn=10,
                            filters=None,
                            enable_reranking=True,
                            session_id=None,
                            enable_optimization=True
                        ))
                    finally:
                        loop.close()
                
                # Run in a separate thread to avoid event loop conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_rag)
                    try:
                        result = future.result(timeout=120)  # increased timeout for RAG
                    except concurrent.futures.TimeoutError:
                        logger.error("[STREAM TIMEOUT] RAG processing exceeded 120s timeout")
                        yield f"data: {json.dumps({'type': 'error', 'data': {'error': 'Streaming search timed out. Please try again.'}})}\n\n"
                        return
                
                # Extract chunks from result
                if isinstance(result, dict):
                    # Prefer child chunks in reranker order; fallback to parent chunks
                    chunks = result.get('top_children_chunks') or result.get('chunks') or result.get('retrieved_chunks', [])
                    answer = result.get('answer', result.get('response', 'No answer generated'))
                else:
                    # If result is just a string answer, we don't have chunks yet
                    chunks = []
                    answer = str(result)
                
                # Ensure chunks is a list
                if not isinstance(chunks, list):
                    logger.warning(f"Chunks is not a list, type: {type(chunks)}")
                    chunks = []
                
                # Format chunks for UI and send immediately
                documents = format_chunks_for_ui(chunks)
                
                # Send chunks first
                logger.info(f"[STREAM] Sending {len(documents)} chunks to frontend")
                yield f"data: {json.dumps({'type': 'chunks', 'data': {'documents': documents}})}\n\n"
                
                # Small delay to simulate processing time and ensure chunks are displayed
                time.sleep(0.5)
                
                # Now send the AI response
                logger.info("[STREAM] Sending AI response")
                ai_response = {
                    'summary': answer,
                    'items': [
                        {
                            'title': 'Generated Answer',
                            'text': answer,
                            'references': [{'id': i+1, 'docId': doc['id']} for i, doc in enumerate(documents[:5])]
                        }
                    ]
                }
                
                yield f"data: {json.dumps({'type': 'answer', 'data': {'aiResponse': ai_response}})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'data': {'status': 'success', 'method': 'rag_enhanced'}})}\n\n"
                
                logger.info("[STREAM] Streaming search completed successfully")
                
            except Exception as rag_error:
                logger.error(f"[STREAM] RAG search failed: {rag_error}")
                logger.error(traceback.format_exc())
                
                yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(rag_error)}})}\n\n"
        
        except Exception as e:
            logger.error(f"[STREAM] Streaming search failed completely: {e}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
    
    return Response(
        generate_response(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )


@app.route('/admin/processing-status', methods=['GET'])
def get_processing_status():
    """Get current processing status of all documents"""
    try:
        state = _load_processing_state()
        current_docs = _scan_source_documents()
        
        status_info = {
            "total_source_documents": len(current_docs),
            "tracked_documents": len(state),
            "summary": {
                "extracted": 0,
                "chunked": 0, 
                "embedded": 0,
                "failed": 0,
                "up_to_date": 0,
                "needs_processing": 0
            },
            "documents": []
        }
        
        for doc_path, current_doc in current_docs.items():
            old_state = state.get(doc_path)
            needs_processing, reason = _document_needs_processing(current_doc, old_state)
            
            # Update status from saved state if available
            if old_state:
                current_doc.extracted = old_state.extracted
                current_doc.chunked = old_state.chunked
                current_doc.embedded = old_state.embedded
                current_doc.error = old_state.error
                current_doc.last_processed = old_state.last_processed
            else:
                # Check current status for new documents
                _check_extraction_status(current_doc)
                _check_chunking_status(current_doc)
                _check_embedding_status(current_doc)
            
            doc_info = {
                "name": Path(doc_path).name,
                "path": doc_path,
                "size": current_doc.size,
                "modified": current_doc.modified_time,
                "hash": current_doc.content_hash[:16] + "...",
                "extracted": current_doc.extracted,
                "chunked": current_doc.chunked,
                "embedded": current_doc.embedded,
                "needs_processing": needs_processing,
                "reason": reason,
                "error": current_doc.error,
                "last_processed": current_doc.last_processed
            }
            
            status_info["documents"].append(doc_info)
            
            # Update summary counts
            if current_doc.error:
                status_info["summary"]["failed"] += 1
            elif not needs_processing:
                status_info["summary"]["up_to_date"] += 1
            else:
                status_info["summary"]["needs_processing"] += 1
                
            if current_doc.extracted:
                status_info["summary"]["extracted"] += 1
            if current_doc.chunked:
                status_info["summary"]["chunked"] += 1
            if current_doc.embedded:
                status_info["summary"]["embedded"] += 1
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get processing status: {e}"}), 500

@app.route('/admin/reprocess', methods=['POST'])
def trigger_reprocessing():
    """Manually trigger document reprocessing"""
    try:
        data = request.get_json() or {}
        force_all = data.get('force_all', False)
        specific_files = data.get('files', [])
        
        if force_all:
            # Clear all state to force reprocessing
            if PROCESSING_STATE_FILE.exists():
                PROCESSING_STATE_FILE.unlink()
            logger.info("[ADMIN] Forced reprocessing of all documents")
        elif specific_files:
            # Clear state for specific files
            state = _load_processing_state()
            for file_path in specific_files:
                if file_path in state:
                    del state[file_path]
                    logger.info(f"[ADMIN] Forced reprocessing of {Path(file_path).name}")
            _save_processing_state(state)
        
        # Start processing in background
        threading.Thread(target=_comprehensive_document_processor, daemon=True).start()
        
        return jsonify({
            "success": True,
            "message": "Document reprocessing started",
            "force_all": force_all,
            "files": specific_files
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to trigger reprocessing: {e}"}), 500

@app.route('/admin/cleanup', methods=['POST'])
def cleanup_orphaned_files():
    """Clean up orphaned extraction files and chunk logs"""
    try:
        current_docs = _scan_source_documents()
        current_stems = {Path(doc_path).stem for doc_path in current_docs.keys()}
        
        cleanup_stats = {
            "extracted_files_removed": 0,
            "chunk_logs_removed": 0,
            "files_removed": []
        }
        
        # Clean up orphaned extracted files
        if EXTRACTED_DIR.exists():
            for extracted_file in EXTRACTED_DIR.glob("*.json"):
                stem = extracted_file.stem
                if stem not in current_stems:
                    extracted_file.unlink()
                    cleanup_stats["extracted_files_removed"] += 1
                    cleanup_stats["files_removed"].append(str(extracted_file))
        
        # Clean up orphaned chunk logs
        if CHUNK_LOGS_DIR.exists():
            for chunk_log in CHUNK_LOGS_DIR.glob("*_parent_child_chunks.json"):
                stem = chunk_log.stem.replace("_parent_child_chunks", "")
                if stem not in current_stems:
                    chunk_log.unlink()
                    cleanup_stats["chunk_logs_removed"] += 1
                    cleanup_stats["files_removed"].append(str(chunk_log))
        
        logger.info(f"[CLEANUP] Removed {cleanup_stats['extracted_files_removed']} orphaned extracted files")
        logger.info(f"[CLEANUP] Removed {cleanup_stats['chunk_logs_removed']} orphaned chunk logs")
        
        return jsonify({
            "success": True,
            "message": "Cleanup completed",
            **cleanup_stats
        })
        
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {e}"}), 500


if __name__ == '__main__':
    print("[STARTUP] Starting RAG API Server...")
    
    # Start auto-ingestion if enabled
    if AUTO_INGEST_ON_STARTUP:
        print("[AUTO-PROCESSOR] Starting comprehensive document processing...")
        threading.Thread(target=_comprehensive_document_processor, daemon=True).start()
    else:
        print("[AUTO-PROCESSOR] Disabled (set AUTO_INGEST_ON_STARTUP=true to enable)")
    
    # Start background monitoring for document changes
    print("[MONITORING] Starting background document monitoring...")
    try:
        start_background_monitoring()
        print("[SUCCESS] Background monitoring started - will detect document changes automatically")
    except Exception as e:
        print(f"[WARNING] Could not start background monitoring: {e}")
        print("[INFO] Document changes will need to be processed manually")
    
    # No embeddings to initialize
    initialize_embeddings()
    print("[INFO] Offline mode: no txtai embeddings initialized")
    
    # Start server
    print("[SERVER] Server starting at http://localhost:5000")
    print("[INFO] UI should connect to: http://localhost:5000/search")
    print("[INFO] Background monitoring: Document changes will be processed automatically")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    finally:
        # Clean shutdown
        print("[SHUTDOWN] Stopping background monitoring...")
        stop_background_monitoring()
        print("[SHUTDOWN] Server shutdown complete")
