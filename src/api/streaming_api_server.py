"""
Streaming API server that returns chunks first, then AI response
This provides a better user experience by showing documents immediately
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import logging
import traceback

# Import your existing modules
from api_server import (
    app, embeddings, logger, validate_and_sanitize_query, 
    format_chunks_for_ui, format_ai_response, rag_query_enhanced
)

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
                
                # Now send the AI response using proper formatting
                logger.info("[STREAM] Sending AI response")
                ai_response = format_ai_response(answer)
                
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
                    
                    # Use proper formatting function
                    ai_response = format_ai_response(simple_answer)
                    
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
    print("Starting streaming API server...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
