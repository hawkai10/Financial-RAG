#!/usr/bin/env python3
"""
Simple Flask API server to bridge the React UI with the RAG backend.
This creates REST endpoints that the UI can call.
"""

import sys
import os
import logging
import asyncio

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
from flask import Flask, request, jsonify, Response
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


if __name__ == '__main__':
    print("[STARTUP] Starting RAG API Server...")
    
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
