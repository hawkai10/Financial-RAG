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
import random
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from typing import Dict, List, Any
from txtai import Embeddings

# Import your existing RAG functions
from rag_backend import rag_query_enhanced, call_gemini_enhanced
from config import config
from utils import logger, validate_and_sanitize_query

# Import the new marked pipeline orchestrator
from marked_pipeline_orchestrator import start_background_monitoring, stop_background_monitoring, is_monitoring_active

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

                response = asyncio.run(call_gemini_enhanced(prompt, strategy="Standard"))
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
        'embeddings_loaded': embeddings is not None,
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
        
        # Check if embeddings are loaded
        if embeddings is None:
            return jsonify({'error': 'Embeddings not loaded'}), 500
        
        logger.info(f"[SEARCH] UI Search request: {sanitized_query}")
        
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
                        embeddings=embeddings,
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
                result = future.result(timeout=30)  # 30 second timeout
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
                logger.info(f"[DEBUG] About to search embeddings for: {sanitized_query}")
                simple_results = embeddings.search(sanitized_query, limit=10)
                logger.info(f"[DEBUG] Simple txtai search returned {len(simple_results)} results")
                logger.info(f"First result type: {type(simple_results[0]) if simple_results else 'No results'}")
                logger.info(f"First result content: {simple_results[0] if simple_results else 'No results'}")
                
                # Create simple chunks from txtai results
                simple_chunks = []
                logger.info(f"Processing {len(simple_results)} txtai results")
                
                for i, result in enumerate(simple_results):
                    try:
                        # txtai returns a dictionary with 'text' and score
                        if isinstance(result, dict):
                            text = result.get('text', str(result))
                            score = result.get('score', 0.0)
                        elif isinstance(result, tuple) and len(result) >= 2:
                            text, score = result[0], result[1]
                        elif isinstance(result, str):
                            text = result
                            score = 0.0
                        else:
                            # Convert whatever it is to string
                            text = str(result)
                            score = 0.0
                        
                        # Create chunk in expected format
                        chunk = {
                            'chunk_id': f'simple_{i}',
                            'text': text,
                            'chunk_text': text,
                            'document_name': f'Document_{i+1}.pdf',
                            'score': float(score),
                            'source_file': f'search_result_{i+1}',
                            'page_number': 1
                        }
                        simple_chunks.append(chunk)
                        logger.info(f"Created chunk {i}: {len(text)} chars, score: {score}")
                        
                    except Exception as chunk_error:
                        logger.error(f"Error processing result {i}: {chunk_error}")
                        logger.error(f"Result content: {result}")
                        continue
                
                logger.info(f"Created {len(simple_chunks)} simple chunks")
                
                # Format chunks for UI
                documents = format_chunks_for_ui(simple_chunks)
                
                # Create simple AI response using proper formatting
                if documents:
                    # Extract relevant content and provide a meaningful answer
                    if "rent" in sanitized_query.lower() and any("rent" in chunk.get('text', '').lower() for chunk in simple_chunks):
                        # Extract rent-related information
                        rent_info = []
                        for chunk in simple_chunks[:5]:
                            text = chunk.get('text', '')
                            if 'rent' in text.lower() or 'amount' in text.lower() or 'rs.' in text.lower():
                                rent_info.append(text)
                        
                        if rent_info:
                            simple_answer = f"Based on the documents, I found information about rent:\n\n"
                            for i, info in enumerate(rent_info[:3], 1):
                                simple_answer += f"{i}. {info.strip()}\n\n"
                            simple_answer += f"\nTotal documents found: {len(documents)}"
                        else:
                            simple_answer = f"I found {len(documents)} documents but couldn't extract specific rent information. Please check the document details below."
                    else:
                        # General answer for other queries
                        key_content = []
                        for chunk in simple_chunks[:3]:
                            text = chunk.get('text', '').strip()
                            if text and len(text) > 50:
                                key_content.append(text[:300])
                        
                        simple_answer = f"I found {len(documents)} relevant documents for your query '{sanitized_query}'.\n\n"
                        if key_content:
                            simple_answer += "Key findings:\n\n"
                            for i, content in enumerate(key_content, 1):
                                simple_answer += f"{i}. {content}...\n\n"
                        simple_answer += "Please see the document details below for complete information."
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
                
                # Use a thread pool executor to run the async function
                import concurrent.futures
                
                def run_async_rag():
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(rag_query_enhanced(
                            question=sanitized_query,
                            embeddings=embeddings,
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
                    result = future.result(timeout=30)  # 30 second timeout
                
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
    
    # Start background monitoring for document changes
    print("[MONITORING] Starting background document monitoring...")
    try:
        start_background_monitoring()
        print("[SUCCESS] Background monitoring started - will detect document changes automatically")
    except Exception as e:
        print(f"[WARNING] Could not start background monitoring: {e}")
        print("[INFO] Document changes will need to be processed manually")
    
    # Initialize embeddings
    if initialize_embeddings():
        print("[SUCCESS] Embeddings loaded successfully")
    else:
        print("[WARNING] Embeddings not loaded - some features may not work")
    
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
