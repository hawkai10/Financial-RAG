import os
from pathlib import Path
from docling.document_converter import DocumentConverter
# Add tiktoken for tokenization
import tiktoken
# Add dotenv for API key loading
from dotenv import load_dotenv
import requests
import json
import csv
from typing import List, Dict, Any, Optional
import re
import camelot
import pdfplumber
import hashlib
from txtai import Embeddings
from tqdm import tqdm

# Path to the folder containing source documents - Fixed to use relative path
SOURCE_DIR = Path(__file__).parent.parent / "Source_Documents"

# Output files for contextualized chunks
OUTPUT_JSON = "contextualized_chunks.json"
OUTPUT_CSV = "contextualized_chunks.csv"

# Tokenizer setup (using cl100k_base, compatible with OpenAI/gpt-3.5/4)
ENCODING = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 50
MAX_TOKENS = 400
CHUNK_OVERLAP = 50  # Added for overlapping chunks


def clean_and_enhance_text(raw_text: str, is_table: bool = False) -> str:
    """
    Transform raw extracted text into enhanced, readable format.
    
    Args:
        raw_text: Original text extracted from PDF/OCR
        is_table: Whether this chunk represents table content
    
    Returns:
        Enhanced, cleaned text with better formatting
    """
    if not raw_text or not raw_text.strip():
        return raw_text
    
    text = raw_text.strip()
    
    if is_table:
        # Enhanced table formatting
        text = enhance_table_formatting(text)
    else:
        # Enhanced text formatting
        text = enhance_text_formatting(text)
    
    return text

def enhance_table_formatting(raw_table_text: str) -> str:
    """Convert raw table text to properly formatted markdown table"""
    try:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', raw_table_text.strip())
        
        # Detect if text contains pipe separators (common in extracted tables)
        if '|' in text:
            # Already has some table structure, enhance it
            lines = text.split('\n')
            formatted_lines = []
            
            for line in lines:
                if line.strip():
                    # Clean up pipe-separated content
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if cells:
                        formatted_line = '| ' + ' | '.join(cells) + ' |'
                        formatted_lines.append(formatted_line)
            
            if len(formatted_lines) > 1:
                # Add header separator after first row
                if formatted_lines:
                    header_sep = '|' + '---|' * (formatted_lines[0].count('|') - 1)
                    formatted_lines.insert(1, header_sep)
            
            return '\n'.join(formatted_lines)
        
        # Try to detect table patterns in unstructured text
        # Look for patterns like "Item Value Item2 Value2"
        words = text.split()
        if len(words) >= 4 and len(words) % 2 == 0:
            # Might be key-value pairs
            pairs = []
            for i in range(0, len(words), 2):
                if i + 1 < len(words):
                    pairs.append(f"| {words[i]} | {words[i+1]} |")
            
            if pairs:
                header = "| Item | Value |"
                separator = "|------|-------|"
                return '\n'.join([header, separator] + pairs)
        
        # Fallback: return cleaned text
        return text
        
    except Exception as e:
        # If enhancement fails, return cleaned original
        return re.sub(r'\s+', ' ', raw_table_text.strip())

def enhance_text_formatting(raw_text: str) -> str:
    """Enhance regular text formatting for better readability"""
    try:
        text = raw_text
        
        # Fix common OCR issues
        text = fix_common_ocr_errors(text)
        
        # Improve paragraph structure
        text = improve_paragraph_structure(text)
        
        # Enhance monetary and numeric formatting
        text = enhance_numeric_formatting(text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text
        
    except Exception as e:
        # If enhancement fails, return cleaned original
        return re.sub(r'\s+', ' ', raw_text.strip())

def fix_common_ocr_errors(text: str) -> str:
    """Fix common OCR misreadings"""
    # Common OCR corrections
    corrections = {
        r'\bl\b': 'I',       # lowercase l -> I when standalone
        r'\brn\b': 'm',      # rn -> m when it looks like OCR error
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def improve_paragraph_structure(text: str) -> str:
    """Improve paragraph breaks and structure"""
    # Add proper paragraph breaks before common section headers
    section_headers = [
        'WHEREAS', 'NOW THEREFORE', 'Period:', 'License Fee', 
        'Deposit:', 'Amount:', 'Total:', 'Summary:', 'Details:'
    ]
    
    for header in section_headers:
        pattern = f'({re.escape(header)})'
        text = re.sub(pattern, f'\n\n\\1', text)
    
    # Clean up multiple paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text

def enhance_numeric_formatting(text: str) -> str:
    """Enhance formatting of monetary amounts and numbers"""
    # Format Indian currency amounts
    text = re.sub(r'Rs\.?\s*(\d+)', r'Rs. \1', text)
    text = re.sub(r'(\d+)\s*/-', r'\1/-', text)
    
    # Format percentages
    text = re.sub(r'(\d+)\s*%', r'\1%', text)
    
    # Format dates
    text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1/\2/\3', text)
    
    return text

# ==============================================================================
# END TEXT ENHANCEMENT FUNCTIONS
# ==============================================================================

# Gemini API setup
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"  # Updated to gemini-1.5-flash-8b
GEMINI_CACHE_URL = "https://generativelanguage.googleapis.com/v1beta/cachedContents"  # For prompt caching
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PROMPT_CACHE = {}
LLM_RESPONSE_CACHE = {}
GEMINI_CACHED_CONTEXTS = {}  # Store cached content IDs for documents

# Utility to save extracted content to a log file in a separate folder
def save_extracted_log(file_path, extracted_content):
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'extraction_logs')
    os.makedirs(log_dir, exist_ok=True)
    base_name = os.path.basename(str(file_path))
    log_path = os.path.join(log_dir, base_name + ".extracted.log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(extracted_content)
    print(f"Saved extracted content to {log_path}")

# Utility to concatenate text and tables in order
def concatenate_text_and_tables(text, table_chunks):
    # For simplicity, append all tables after text (or you can interleave if you have positions)
    content = text.strip()
    for t_chunk in table_chunks:
        content += "\n\n" + t_chunk["chunk_text"].strip()
    return content

# Prompt caching utility
def get_prompt_template():
    # Cache the prompt template string
    if "template" not in PROMPT_CACHE:
        PROMPT_CACHE["template"] = (
            "<document>\n{{WHOLE_DOCUMENT}}\n{WHOLE_DOCUMENT}\n</document>\n\n"
            "Here is the chunk we want to situate within the whole document\n"
            "<chunk>\n{{CHUNK_CONTENT}}\n{CHUNK_CONTENT}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
        )
    return PROMPT_CACHE["template"]

def create_cached_context(document_text):
    """Create a cached context for a document on Gemini's servers."""
    doc_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    
    # Check if we already have a cached context for this document
    if doc_hash in GEMINI_CACHED_CONTEXTS:
        return GEMINI_CACHED_CONTEXTS[doc_hash]
    
    # Create cached content on Gemini's servers
    cache_data = {
        "model": "models/gemini-1.5-flash-8b",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"<document>\n{{{{WHOLE_DOCUMENT}}}}\n{document_text}\n</document>\n\nHere is the chunk we want to situate within the whole document\n<chunk>\n{{{{CHUNK_CONTENT}}}}\n</chunk>\n\nPlease give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
                    }
                ]
            }
        ],
        "ttl": "3600s"  # Cache for 1 hour
    }
    
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    
    try:
        response = requests.post(GEMINI_CACHE_URL, headers=headers, params=params, json=cache_data, timeout=60)
        response.raise_for_status()
        result = response.json()
        cached_content_name = result["name"]
        
            # LLM LOG DISABLED: print(f"Created cached context: {cached_content_name}")
        GEMINI_CACHED_CONTEXTS[doc_hash] = cached_content_name
        return cached_content_name
        
    except Exception as e:
            # LLM LOG DISABLED: print(f"Error creating cached context: {e}")
        return None

def get_gemini_context_with_cache(document_text, chunk_text):
    # Use a hash of (document_text, chunk_text) as cache key for our local cache
    key = hashlib.sha256((document_text + "|||" + chunk_text).encode("utf-8")).hexdigest()
    if key in LLM_RESPONSE_CACHE:
        return LLM_RESPONSE_CACHE[key]
    
    # Try to use Gemini's prompt caching
    cached_content_name = create_cached_context(document_text)
    
    if cached_content_name:
        # Use cached content
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        data = {
            "cachedContent": cached_content_name,
            "contents": [
                {
                    "role": "user", 
                    "parts": [{"text": f"<chunk>\n{chunk_text}\n</chunk>"}]
                }
            ]
        }
    else:
        # Fallback to regular prompt if caching fails
        prompt = get_prompt_template().format(WHOLE_DOCUMENT=document_text, CHUNK_CONTENT=chunk_text)
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        context = result["candidates"][0]["content"]["parts"][0]["text"]
        LLM_RESPONSE_CACHE[key] = context.strip()
        return context.strip()
    except Exception as e:
        # LLM LOG DISABLED: print(f"Error calling Gemini API: {e}")
        return ""

def extract_full_text(docling_document) -> str:
    """Extracts all text from a DoclingDocument object."""
    all_text = []
    for item, _level in docling_document.iterate_items():
        if hasattr(item, "text") and item.text:
            all_text.append(item.text)
    return "\n".join(all_text)

def chunk_text_tiktoken(text, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, overlap=CHUNK_OVERLAP, encoding=ENCODING):
    """Splits text into overlapping chunks respecting paragraph boundaries."""
    
    # Split text into paragraphs (double newlines indicate paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    start_token_pos = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Calculate tokens for this paragraph
        para_tokens = len(encoding.encode(paragraph))
        
        # If this single paragraph exceeds max_tokens, we need to split it carefully
        if para_tokens > max_tokens:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "start_token": start_token_pos,
                    "end_token": start_token_pos + current_tokens,
                    "num_tokens": current_tokens
                })
                start_token_pos += current_tokens - overlap
                current_chunk = ""
                current_tokens = 0
            
            # Split large paragraph by sentences to maintain some semantic coherence
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_tokens = len(encoding.encode(sentence))
                
                # If adding this sentence would exceed max_tokens, save current chunk
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "start_token": start_token_pos,
                        "end_token": start_token_pos + current_tokens,
                        "num_tokens": current_tokens
                    })
                    start_token_pos += current_tokens - overlap
                    current_chunk = ""
                    current_tokens = 0
                
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
                
        else:
            # Normal paragraph processing
            # If adding this paragraph would exceed max_tokens, save current chunk
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Only save if we meet minimum token requirement
                if current_tokens >= min_tokens:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "start_token": start_token_pos,
                        "end_token": start_token_pos + current_tokens,
                        "num_tokens": current_tokens
                    })
                    start_token_pos += current_tokens - overlap
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    # Get overlap text from the end of previous chunk
                    overlap_tokens = encoding.encode(current_chunk)[-overlap:]
                    overlap_text = encoding.decode(overlap_tokens) if overlap_tokens else ""
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                    current_tokens = len(encoding.encode(current_chunk))
                else:
                    current_chunk = paragraph
                    current_tokens = para_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_tokens += para_tokens + len(encoding.encode("\n\n"))
                else:
                    current_chunk = paragraph
                    current_tokens = para_tokens
    
    # Add final chunk if it has content and meets minimum requirements
    if current_chunk and current_tokens >= min_tokens:
        chunks.append({
            "chunk_text": current_chunk.strip(),
            "start_token": start_token_pos,
            "end_token": start_token_pos + current_tokens,
            "num_tokens": current_tokens
        })
    
    # Ensure we always return at least one chunk, even if below min_tokens
    if not chunks and current_chunk:
        chunks.append({
            "chunk_text": current_chunk.strip(),
            "start_token": 0,
            "end_token": current_tokens,
            "num_tokens": current_tokens
        })
    
    return chunks

def extract_table_chunks(docling_document):
    """Extract all tables from a DoclingDocument and return as single markdown chunks."""
    table_chunks = []
    for item, _level in docling_document.iterate_items():
        # For Excel, Word, PDF, etc. - look for TableItem with a 'data' attribute
        if hasattr(item, 'data'):
            data = getattr(item, 'data')
            cells = getattr(data, 'table_cells', [])
            num_rows = getattr(data, 'num_rows', 0)
            num_cols = getattr(data, 'num_cols', 0)
            if cells and num_rows > 0 and num_cols > 0:
                # Build the table as a 2D list
                grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
                for cell in cells:
                    for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                        for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                            grid[r][c] = cell.text
                # Serialize the whole table as one Markdown chunk
                header = "| " + " | ".join(grid[0]) + " |" if grid else ""
                separator = "|" + "---|" * num_cols
                body = ["| " + " | ".join(row) + " |" for row in grid[1:]]
                md_table = "\n".join([header, separator] + body)
                table_chunks.append({
                    "chunk_text": md_table,
                    "is_table": True,
                    "table_index": 0,
                    "num_rows": num_rows,
                    "num_cols": num_cols
                })
    return table_chunks

def get_gemini_context(document_text: str, chunk_text: str) -> str:
    """Send prompt to Gemini API to get context for a chunk."""
    prompt = f"""
Here is the document: {document_text}

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the context and nothing else.

{chunk_text}
"""

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        # Gemini API returns the response in a nested structure
        context = result["candidates"][0]["content"]["parts"][0]["text"]
        return context.strip()
    except requests.RequestException as e:
        # print(f"Request error calling Gemini API: {e}")
        return f"[API Error: {str(e)}]"
    except KeyError as e:
        # print(f"Unexpected API response structure: {e}")
        return "[Context unavailable: API response error]"
    except Exception as e:
        # print(f"Unexpected error calling Gemini API: {e}")
        return f"[Error: {str(e)}]"

def camelot_table_to_markdown(df):
    """Convert a Camelot table (pandas DataFrame) to markdown string."""
    header = '| ' + ' | '.join(df.iloc[0]) + ' |'
    separator = '|' + '---|' * len(df.columns)
    body = ['| ' + ' | '.join(row) + ' |' for row in df.iloc[1:].values]
    return '\n'.join([header, separator] + body)

def extract_tables_with_camelot(pdf_path, pages="all"):
    tables = camelot.read_pdf(pdf_path, pages=pages)
    return [table.df for table in tables]

def hybrid_table_extraction(docling_tables, pdf_path):
    # If Docling found tables, use them
    if docling_tables:
        return docling_tables, 'docling'
    # Otherwise, try Camelot
    camelot_tables = extract_tables_with_camelot(pdf_path)
    if camelot_tables:
        return camelot_tables, 'camelot'
    return [], None

def is_scanned_pdf(pdf_path):
    """Return True if the PDF is likely scanned (no extractable text on any page)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    return False  # At least one page has text
        return True  # No pages have extractable text
    except Exception as e:
        print(f"Error checking if PDF is scanned: {e}")
        return False  # Fallback: treat as digital if error

def merge_adjacent_tables(table_chunks):
    """Merge adjacent table chunks with the same number of columns into a single chunk."""
    merged = []
    buffer = []
    last_num_cols = None
    for chunk in table_chunks:
        if last_num_cols is not None and chunk['num_cols'] == last_num_cols:
            buffer.append(chunk)
        else:
            if buffer:
                # Merge buffered tables
                merged_text = '\n'.join(c['chunk_text'] for c in buffer)
                merged_chunk = buffer[0].copy()
                merged_chunk['chunk_text'] = merged_text
                merged_chunk['raw_chunk_text'] = merged_text
                merged_chunk['num_rows'] = sum(c['num_rows'] for c in buffer)
                merged.append(merged_chunk)
                buffer = []
            buffer = [chunk]
        last_num_cols = chunk['num_cols']
    if buffer:
        merged_text = '\n'.join(c['chunk_text'] for c in buffer)
        merged_chunk = buffer[0].copy()
        merged_chunk['chunk_text'] = merged_text
        merged_chunk['raw_chunk_text'] = merged_text
        merged_chunk['num_rows'] = sum(c['num_rows'] for c in buffer)
        merged.append(merged_chunk)
    return merged

def process_single_document(file_path: Path, converter: DocumentConverter) -> List[Dict[str, Any]]:
    """Process a single document and return its contextualized chunks."""
    result = converter.convert(file_path)
    chunks_for_doc = []
    
    # LLM LOG DISABLED: print(f"\n--- Processing: {result.input.file} ---")
    print(f"Status: {result.status}")
    
    if hasattr(result, "document") and result.document is not None:
        # 1. Extract text and tables
        full_text = extract_full_text(result.document)
        docling_tables = extract_table_chunks(result.document)
        fallback_tables, table_source = hybrid_table_extraction(docling_tables, str(result.input.file))
        table_chunks = []
        if table_source == 'docling':
            table_chunks = docling_tables
        elif table_source == 'camelot':
            for idx, df in enumerate(fallback_tables):
                md_table = camelot_table_to_markdown(df)
                table_chunks.append({
                    "chunk_text": md_table,
                    "is_table": True,
                    "table_index": idx,
                    "num_rows": len(df),
                    "num_cols": len(df.columns)
                })
        # 2. Concatenate for whole document context
        whole_doc_content = concatenate_text_and_tables(full_text, table_chunks)
        # 3. Save to log file
        save_extracted_log(result.input.file, whole_doc_content)
        
        # 4. Create cached context for this document (this will save costs for all chunks)
        # LLM LOG DISABLED: print(f"Creating cached context for document (this will reduce token costs)...")
        cached_content_name = create_cached_context(whole_doc_content)
        if cached_content_name:
            pass
        else:
            pass
        
        # 5. Chunking
        chunks = chunk_text_tiktoken(full_text)
        # LLM LOG DISABLED: print(f"Total Text Chunks: {len(chunks)} | Total Table Chunks: {len(table_chunks)}")
        # 6. Add text chunks with progress bar
        for idx, chunk in enumerate(tqdm(chunks, desc="Loading text chunks")):
            context = get_gemini_context_with_cache(whole_doc_content, chunk["chunk_text"])
            
            # Store original text and create enhanced version
            original_text = chunk["chunk_text"]
            enhanced_text = clean_and_enhance_text(original_text, is_table=False)
            
            chunk_metadata = {
                "document_name": str(result.input.file),
                "chunk_id": f"{result.input.file}_text_{idx+1}",
                "chunk_index": idx,
                "start_token": chunk["start_token"],
                "end_token": chunk["end_token"],
                "num_tokens": chunk["num_tokens"],
                "raw_chunk_text": original_text,  # Keep original extracted text
                "chunk_text": enhanced_text,     # Store enhanced, readable text
                "context": context,
                "is_table": False
            }
            chunks_for_doc.append(chunk_metadata)
        # 7. Add table chunks with progress bar
        for t_idx, t_chunk in enumerate(tqdm(table_chunks, desc="Loading table chunks")):
            context = get_gemini_context_with_cache(whole_doc_content, t_chunk["chunk_text"])
            
            # Store original table text and create enhanced version
            original_table_text = t_chunk["chunk_text"]
            enhanced_table_text = clean_and_enhance_text(original_table_text, is_table=True)
            
            chunk_metadata = {
                "document_name": str(result.input.file),
                "chunk_id": f"{result.input.file}_table_{t_idx+1}",
                "chunk_index": t_idx,
                "start_token": None,
                "end_token": None,
                "num_tokens": None,
                "raw_chunk_text": original_table_text,  # Keep original extracted table text
                "chunk_text": enhanced_table_text,      # Store enhanced, formatted table
                "context": context,
                "is_table": True,
                "num_rows": t_chunk["num_rows"],
                "num_cols": t_chunk["num_cols"]
            }
            chunks_for_doc.append(chunk_metadata)

    if hasattr(result, 'errors') and result.errors:
        print("Errors:")
        for error in result.errors:
            print(f" - {error.error_message}")
    
    return chunks_for_doc

def save_results(all_contextualized_chunks: List[Dict[str, Any]]) -> None:
    """Save results to JSON and CSV files."""
    # Save all contextualized chunks to a JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_contextualized_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_contextualized_chunks)} contextualized chunks to {OUTPUT_JSON}")

    # Save all contextualized chunks to a CSV file
    if all_contextualized_chunks:
        # Compute the union of all keys in all chunk dicts
        all_keys = set()
        for row in all_contextualized_chunks:
            all_keys.update(row.keys())
        fieldnames = list(all_keys)
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_contextualized_chunks:
                writer.writerow(row)
        print(f"Saved {len(all_contextualized_chunks)} contextualized chunks to {OUTPUT_CSV}")

def sanitize_tag_value(value):
    # Allow only str, int, float, bool, or None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def main():
    if not SOURCE_DIR.exists() or not SOURCE_DIR.is_dir():
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in .env file.")
        return

    # List all files in the source directory (ignore subdirectories)
    files = [f for f in SOURCE_DIR.iterdir() if f.is_file()]

    if not files:
        print(f"No files found in {SOURCE_DIR}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    converter = DocumentConverter()
    all_contextualized_chunks = []

    # Process files one by one to manage memory usage
    for file_path in files:
        try:
            chunks_for_doc = process_single_document(file_path, converter)
            all_contextualized_chunks.extend(chunks_for_doc)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    save_results(all_contextualized_chunks)

    # Remove the txtai embedding/indexing step from here
    # (This is now handled in index_chunks_txtai.py)
    # embeddings = Embeddings({
    #     "path": "BAAI/bge-base-en-v1.5",
    #     "content": True,
    #     "backend": "sqlite"
    # })
    # embeddings.index(txtai_data)
    # embeddings.save("business-docs-index")
    # print("Indexing complete. You can now run semantic and metadata queries with txtai!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for chunking and contextualization')
    parser.add_argument('--single-file', help='Process a single file instead of the entire folder')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        file_path = Path(args.single_file)
        if file_path.exists():
            converter = DocumentConverter()
            try:
                chunks = process_single_document(file_path, converter)
                print(f"Processed {file_path.name}: {len(chunks)} chunks generated")
                
                # Load existing data if it exists
                existing_chunks = []
                if os.path.exists(OUTPUT_JSON):
                    try:
                        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                            existing_chunks = json.load(f)
                    except Exception:
                        existing_chunks = []
                
                # Remove old chunks from this file and add new ones
                existing_chunks = [c for c in existing_chunks if c.get('source_file') != str(file_path)]
                existing_chunks.extend(chunks)
                
                # Save updated results
                save_results(existing_chunks)
                print(f"[SUCCESS] Successfully processed and saved chunks for {file_path.name}")
                
            except Exception as e:
                print(f"[ERROR] Error processing {file_path}: {e}")
                exit(1)
        else:
            print(f"[ERROR] File not found: {args.single_file}")
            exit(1)
    else:
        # Process all files in the folder (original behavior)
        main()
