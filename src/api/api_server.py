#!/usr/bin/env python3
"""
Simple Flask API server to bridge the React UI with the RAG backend.
This creates REST endpoints that the UI can call.
"""

import os
# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import traceback
import random
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from typing import Dict, List, Any
from txtai import Embeddings

# Import your existing RAG functions
from ..core.rag_backend import rag_query_enhanced, call_gemini_enhanced
from ..utils.config import config
from ..utils.utils import logger, validate_and_sanitize_query

# Import pipeline orchestrator
from pipeline_orchestrator import ensure_data_pipeline_up_to_date

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global embeddings instance
embeddings = None
# Store generated example queries in memory
example_queries = []

def generate_example_queries():
    """Generate example queries from random document chunks"""
    global example_queries, embeddings
    
    if embeddings is None:
        logger.warning("Cannot generate example queries - embeddings not loaded")
        return
    
    try:
        # Get diverse documents/chunks from the embeddings using different search terms
        # Use more specific terms to get varied content
        sample_queries = [
            "standard certificate compliance regulation",
            "export import trade requirement",
            "safety protocol procedure guideline",
            "technical specification machine equipment",
            "legal contract agreement document"
        ]
        all_chunks = []
        
        for sample_query in sample_queries:
            try:
                results = embeddings.search(sample_query, limit=20)
                if results:
                    all_chunks.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{sample_query}': {e}")
                continue
        
        # If specific searches didn't work, try broader terms
        if not all_chunks:
            logger.info("Trying broader search terms...")
            broad_queries = ["document", "information", "text", "content"]
            for broad_query in broad_queries:
                try:
                    results = embeddings.search(broad_query, limit=30)
                    if results:
                        all_chunks.extend(results)
                        break  # Just need one successful search
                except:
                    continue
        
        if not all_chunks:
            logger.warning("No chunks found for example query generation")
            # Fallback to default queries
            example_queries = [
                "What are the main topics covered in the documents?",
                "Can you summarize the key information available?",
                "What important details should I know from these documents?"
            ]
            return
        
        # Remove duplicates and select chunks with meaningful content
        unique_chunks = []
        seen_texts = set()
        
        for chunk in all_chunks:
            # txtai returns (text, score) tuples or just text
            if isinstance(chunk, tuple):
                text = chunk[0]
                score = chunk[1] if len(chunk) > 1 else 0
            else:
                text = str(chunk)
                score = 0
            
            # Clean and normalize text for comparison
            clean_text = text.strip().lower()
            
            # Only add if we haven't seen this text before and it has meaningful content
            if (clean_text not in seen_texts and 
                len(text) > 100 and  # Ensure substantial content
                len(text.split()) > 20):  # At least 20 words
                unique_chunks.append(text)
                seen_texts.add(clean_text)
        
        if len(unique_chunks) < 3:
            logger.warning(f"Only found {len(unique_chunks)} unique chunks")
            # If we have some chunks, use them and pad with defaults
            if unique_chunks:
                selected_chunks = unique_chunks[:3]
                # Generate questions for available chunks
                generated_queries = []
                for i, chunk in enumerate(selected_chunks):
                    chunk_preview = chunk[:400] + "..." if len(chunk) > 400 else chunk
                    question = generate_question_from_chunk(chunk_preview, i)
                    generated_queries.append(question)
                
                # Pad with default queries if needed
                default_queries = [
                    "What are the main topics covered in the documents?",
                    "Can you summarize the key information available?",
                    "What important details should I know from these documents?"
                ]
                while len(generated_queries) < 3:
                    generated_queries.append(default_queries[len(generated_queries) - len(selected_chunks)])
                
                example_queries = generated_queries[:3]
            else:
                # No chunks found, use defaults
                example_queries = [
                    "What are the main topics covered in the documents?",
                    "Can you summarize the key information available?",
                    "What important details should I know from these documents?"
                ]
            return
        
        # Select 3 random chunks from unique ones
        selected_chunks = random.sample(unique_chunks, min(3, len(unique_chunks)))
        
        # Generate questions using Gemini
        generated_queries = []
        
        for i, chunk in enumerate(selected_chunks):
            # Use more text for better context (up to 400 chars)
            chunk_preview = chunk[:400] + "..." if len(chunk) > 400 else chunk
            
            logger.info(f"Generating question {i+1} from chunk preview: {chunk_preview[:100]}...")
            question = generate_question_from_chunk(chunk_preview, i)
            generated_queries.append(question)
        
        example_queries = generated_queries
        logger.info(f"[GENERATE] Generated {len(example_queries)} example queries: {example_queries}")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate example queries: {e}")
        logger.error(traceback.format_exc())
        # Fallback to default queries
        example_queries = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key information available?",
            "What important details should I know from these documents?"
        ]

def generate_question_from_chunk(chunk_text: str, index: int) -> str:
    """Generate a question from a chunk of text using Gemini"""
    try:
        # Try to use Gemini API for question generation
        if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY:
            try:
                prompt = f"""Based on the following document excerpt, generate a specific, relevant question that someone might ask about this content. The question should be natural and encourage exploration of the document.

Document excerpt:
{chunk_text}

Instructions:
- Generate exactly one clear, specific question
- The question should be directly related to the content shown
- Make it sound natural, as if a user would really ask this
- End with a question mark
- Keep it concise but informative

Question:"""

                response = call_gemini_enhanced(prompt, strategy="Standard")
                if response and response.strip():
                    question = response.strip()
                    # Ensure it ends with a question mark
                    if not question.endswith('?'):
                        question += '?'
                    return question
                    
            except Exception as e:
                logger.warning(f"Gemini question generation failed: {e}")
            
    except Exception as e:
        logger.warning(f"Question generation with Gemini failed: {e}")
    
    # Fallback to rule-based question generation
    chunk_lower = chunk_text.lower()
    
    # Look for key topics and generate relevant questions
    if 'standard' in chunk_lower or 'certificate' in chunk_lower or 'compliance' in chunk_lower:
        return "What standards and certifications are mentioned in the documents?"
    elif 'export' in chunk_lower or 'import' in chunk_lower or 'trade' in chunk_lower:
        return "What are the export/import requirements discussed?"
    elif 'safety' in chunk_lower or 'protocol' in chunk_lower or 'regulation' in chunk_lower:
        return "What safety protocols and regulations are covered?"
    elif 'technical' in chunk_lower or 'specification' in chunk_lower:
        return "What technical specifications are detailed?"
    elif 'legal' in chunk_lower or 'contract' in chunk_lower or 'agreement' in chunk_lower:
        return "What legal requirements or agreements are mentioned?"
    elif 'machine' in chunk_lower or 'equipment' in chunk_lower:
        return "What information about machinery and equipment is provided?"
    elif 'process' in chunk_lower or 'procedure' in chunk_lower:
        return "What processes and procedures are described?"
    elif 'cost' in chunk_lower or 'price' in chunk_lower or 'budget' in chunk_lower:
        return "What cost-related information is discussed?"
    else:
        # Generic questions based on content preview
        words = chunk_text.split()[:10]  # First 10 words for context
        context = ' '.join(words)
        return f"What information is provided about {context.lower()}?"

def initialize_embeddings():
    """Initialize embeddings on startup"""
    global embeddings
    try:
        embeddings = Embeddings()
        index_path = config.INDEX_PATH
        
        if os.path.exists(index_path):
            embeddings.load(index_path)
            logger.info(f"[GENERATE] Embeddings loaded from {index_path}")
            
            # Generate example queries after loading embeddings
            generate_example_queries()
            
            return True
        else:
            logger.error(f"[ERROR] Index not found at {index_path}")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize embeddings: {e}")
        return False

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
                # Extract document info from chunk
                chunk_id = chunk.get('chunk_id', chunk.get('id', f'doc_{i}'))
                document_name = chunk.get('document_name', chunk.get('source', chunk.get('file', 'Unknown Document')))
                
                # Get text content from various possible fields
                text_content = chunk.get('chunk_text', chunk.get('text', chunk.get('content', '')))
                snippet = text_content[:200] + '...' if len(text_content) > 200 else text_content
                
                # Create UI-compatible document object
                doc = {
                    'id': str(chunk_id),
                    'sourceType': 'Windows Shares',  # Default for now
                    'sourcePath': str(document_name) if document_name else 'Unknown Path',
                    'fileType': 'pdf',  # Default for now
                    'title': os.path.basename(str(document_name)) if document_name else f'Document {i+1}',
                    'date': '01.01.2024',  # Default date
                    'snippet': snippet,
                    'author': 'System',
                    'score': chunk.get('score', 0.0)
                }
                documents.append(doc)
                
            elif isinstance(chunk, str):
                # Handle string chunks
                snippet = chunk[:200] + '...' if len(chunk) > 200 else chunk
                doc = {
                    'id': f'doc_{i}',
                    'sourceType': 'Windows Shares',
                    'sourcePath': 'Text Content',
                    'fileType': 'txt',
                    'title': f'Text Document {i+1}',
                    'date': '01.01.2024',
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
        # Clean the response
        cleaned_response = raw_response.strip()
        
        # Remove excessive markdown formatting
        cleaned_response = cleaned_response.replace('**', '')
        cleaned_response = cleaned_response.replace('*', '')
        
        # Simple approach: use the full response as one item
        # Extract summary from first line or first 150 characters
        lines = cleaned_response.split('\n')
        first_line = lines[0].strip() if lines else cleaned_response
        
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
        # Fallback - return raw response
        return {
            'summary': 'Analysis complete',
            'items': [{
                'title': 'Response',
                'text': raw_response,
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
        'embeddings_loaded': embeddings is not None
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
        
        # Check if embeddings are loaded
        if embeddings is None:
            return jsonify({'error': 'Embeddings not loaded'}), 500
        
        logger.info(f"[SEARCH] UI Search request: {sanitized_query}")
        
        # Try the full RAG pipeline first (main method)
        try:
            logger.info("[RAG] Attempting enhanced RAG search...")
            result = rag_query_enhanced(
                question=sanitized_query,
                embeddings=embeddings,
                topn=10,
                filters=None,
                enable_reranking=True,
                session_id=None,
                enable_optimization=True
            )
            logger.info(f"RAG result type: {type(result)}")
            logger.info(f"RAG result keys (if dict): {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract answer and chunks from result with better handling
            if isinstance(result, dict):
                answer = result.get('answer', result.get('response', 'No answer generated'))
                chunks = result.get('chunks', result.get('retrieved_chunks', []))
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
            
            logger.info(f"[SUCCESS] Enhanced RAG search successful: {len(documents)} documents")
            
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
            
            # Fall back to simple txtai search
            logger.info("[FALLBACK] Falling back to simple txtai search...")
            try:
                # Direct txtai search as fallback
                simple_results = embeddings.search(sanitized_query, limit=10)
                logger.info(f"Simple txtai search returned {len(simple_results)} results")
                
                # Create simple chunks from txtai results
                simple_chunks = []
                for i, result in enumerate(simple_results):
                    if isinstance(result, tuple) and len(result) >= 2:
                        text, score = result[0], result[1]
                        chunk = {
                            'chunk_id': f'simple_{i}',
                            'text': text,
                            'chunk_text': text,
                            'document_name': f'Document_{i+1}.pdf',
                            'score': score
                        }
                        simple_chunks.append(chunk)
                    elif isinstance(result, str):
                        chunk = {
                            'chunk_id': f'simple_{i}',
                            'text': result,
                            'chunk_text': result,
                            'document_name': f'Document_{i+1}.pdf',
                            'score': 0.0
                        }
                        simple_chunks.append(chunk)
                
                # Format chunks for UI
                documents = format_chunks_for_ui(simple_chunks)
                
                # Create simple AI response using proper formatting
                if documents:
                    context_text = '\n\n'.join([chunk.get('text', '')[:200] for chunk in simple_chunks[:3]])
                    simple_answer = f"Based on the search for '{sanitized_query}', I found {len(documents)} relevant documents using simple search. Here's a summary of the key information:\n\n{context_text[:500]}..."
                else:
                    simple_answer = f"I searched for '{sanitized_query}' but couldn't find relevant documents. This might be because the content isn't indexed or the search terms don't match the available content."
                
                # Use the proper formatting function
                ai_response = format_ai_response(simple_answer)
                
                logger.info(f"[SUCCESS] Fallback simple search completed: {len(documents)} documents")
                
                return jsonify({
                    'documents': documents,
                    'aiResponse': ai_response,
                    'query': sanitized_query,
                    'status': 'success',
                    'method': 'simple_search_fallback'
                })
                
            except Exception as simple_error:
                logger.error(f"[ERROR] Simple search fallback also failed: {simple_error}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Both enhanced RAG and simple search failed. RAG: {str(rag_error)}, Simple: {str(simple_error)}'}), 500
        
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

@app.route('/example-queries', methods=['GET'])
def get_example_queries():
    """Get generated example queries"""
    global example_queries
    
    if not example_queries:
        # Generate if not already done
        generate_example_queries()
    
    return jsonify({
        'queries': example_queries,
        'status': 'success'
    })

@app.route('/recent-documents', methods=['GET'])
def get_recent_documents():
    """Get recently accessed documents"""
    try:
        # Mock recent documents - in a real app, this would come from user activity logs
        recent_docs = [
            {
                'id': 'doc_1',
                'title': 'Export_CNC_Machine_US.docx',
                'fileType': 'word',
                'sourcePath': 'C:\\Users\\arvin\\OneDrive\\Desktop\\trial\\docling\\Source_Documents\\Export_CNC_Machine_US.docx',
                'lastAccessed': '2025-07-28T10:30:00Z',
                'sourceType': 'Windows Shares'
            },
            {
                'id': 'doc_2', 
                'title': '60494300N_NOS_BE_Version_1_Installation_Handbook_Manual.pdf',
                'fileType': 'pdf',
                'sourcePath': 'C:\\Users\\arvin\\OneDrive\\Desktop\\trial\\docling\\Source_Documents\\60494300N_NOS_BE_Version_1_Installation_Handbook_Manual.pdf',
                'lastAccessed': '2025-07-28T09:15:00Z',
                'sourceType': 'Windows Shares'
            },
            {
                'id': 'doc_3',
                'title': 'EN_IH_P-4532DN_P-5032DN_P-5532DN_UT_Rev_1.pdf',
                'fileType': 'pdf', 
                'sourcePath': 'C:\\Users\\arvin\\OneDrive\\Desktop\\trial\\docling\\Source_Documents\\EN_IH_P-4532DN_P-5032DN_P-5532DN_UT_Rev_1.pdf',
                'lastAccessed': '2025-07-27T16:45:00Z',
                'sourceType': 'Windows Shares'
            }
        ]
        
        return jsonify({
            'documents': recent_docs,
            'status': 'success'
        })
        
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
            
            # Check if embeddings are loaded
            if embeddings is None:
                yield f"data: {json.dumps({'error': 'Embeddings not loaded'})}\n\n"
                return
            
            logger.info(f"[STREAM] Starting streaming search for: {sanitized_query}")
            
            # Try the full RAG pipeline
            try:
                logger.info("[STREAM] Getting chunks first...")
                
                # We need to modify rag_query_enhanced to return chunks immediately
                # For now, let's call the existing function and extract chunks
                result = rag_query_enhanced(
                    question=sanitized_query,
                    embeddings=embeddings,
                    topn=10,
                    filters=None,
                    enable_reranking=True,
                    session_id=None,
                    enable_optimization=True
                )
                
                # Extract chunks from result
                if isinstance(result, dict):
                    chunks = result.get('chunks', result.get('retrieved_chunks', []))
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
                
                # Fall back to simple txtai search
                logger.info("[STREAM] Falling back to simple search...")
                try:
                    simple_results = embeddings.search(sanitized_query, limit=10)
                    
                    # Create simple chunks from txtai results
                    simple_chunks = []
                    for i, result in enumerate(simple_results):
                        if isinstance(result, tuple) and len(result) >= 2:
                            text, score = result[0], result[1]
                            chunk = {
                                'chunk_id': f'simple_{i}',
                                'text': text,
                                'chunk_text': text,
                                'document_name': f'Document_{i+1}.pdf',
                                'score': score
                            }
                            simple_chunks.append(chunk)
                        elif isinstance(result, str):
                            chunk = {
                                'chunk_id': f'simple_{i}',
                                'text': result,
                                'chunk_text': result,
                                'document_name': f'Document_{i+1}.pdf',
                                'score': 0.0
                            }
                            simple_chunks.append(chunk)
                    
                    # Format chunks for UI and send
                    documents = format_chunks_for_ui(simple_chunks)
                    yield f"data: {json.dumps({'type': 'chunks', 'data': {'documents': documents}})}\n\n"
                    
                    # Create simple AI response
                    if documents:
                        context_text = '\n\n'.join([chunk.get('text', '')[:200] for chunk in simple_chunks[:3]])
                        simple_answer = f"Based on the search for '{sanitized_query}', I found {len(documents)} relevant documents. Here's a summary:\n\n{context_text[:500]}..."
                    else:
                        simple_answer = f"I searched for '{sanitized_query}' but couldn't find relevant documents."
                    
                    ai_response = {
                        'summary': simple_answer,
                        'items': [
                            {
                                'title': 'Fallback Search Results',
                                'text': simple_answer,
                                'references': [{'id': i+1, 'docId': doc['id']} for i, doc in enumerate(documents[:5])]
                            }
                        ]
                    }
                    
                    yield f"data: {json.dumps({'type': 'answer', 'data': {'aiResponse': ai_response}})}\n\n"
                    yield f"data: {json.dumps({'type': 'complete', 'data': {'status': 'success', 'method': 'fallback'}})}\n\n"
                    
                except Exception as fallback_error:
                    logger.error(f"[STREAM] Fallback search also failed: {fallback_error}")
                    yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(fallback_error)}})}\n\n"
        
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
    
def start_server():
    """Start the Flask API server with proper initialization."""
    # Ensure data pipeline is up-to-date before starting server
    print("[PIPELINE] Checking data pipeline status...")
    if ensure_data_pipeline_up_to_date():
        print("[SUCCESS] Data pipeline is up-to-date")
    else:
        print("[ERROR] Failed to update data pipeline. Server may serve stale data.")
        print("[WARNING] Consider running pipeline manually or check logs for errors.")
    
    # Initialize embeddings
    if initialize_embeddings():
        print("[SUCCESS] Embeddings loaded successfully")
    else:
        print("[WARNING] Embeddings not loaded - some features may not work")
    
    # Start server
    print("[SERVER] Server starting at http://localhost:5000")
    print("[INFO] UI should connect to: http://localhost:5000/search")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

if __name__ == "__main__":
    start_server()
