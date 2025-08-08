import os
import re
import json
import csv
import hashlib
import requests
import textwrap
import atexit
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from docling.document_converter import DocumentConverter  # Your original docling import
import camelot
import pdfplumber

# -------------------
# Configuration

# Directories and output filenames
SOURCE_DIR = Path(__file__).parent / "Source_Documents"
OUTPUT_JSON = "contextualized_chunks.json"
OUTPUT_CSV = "contextualized_chunks.csv"
CACHE_FILE = "llm_response_cache.json"

# Tokenizer setup
ENCODING = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 50
MAX_TOKENS = 400          # Updated token max size
CHUNK_OVERLAP = 100       # Updated chunk overlap

# Gemini API details
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
GEMINI_CACHE_URL = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Caches
PROMPT_CACHE = {}
LLM_RESPONSE_CACHE = {}
GEMINI_CACHED_CONTEXTS = {}

# -------------------
# Utility functions for caching

def load_local_cache():
    global LLM_RESPONSE_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                LLM_RESPONSE_CACHE.update(json.load(f))
            print(f"Loaded LLM response cache with {len(LLM_RESPONSE_CACHE)} entries.")
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")

def save_local_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(LLM_RESPONSE_CACHE, f, ensure_ascii=False, indent=2)
        print(f"Saved LLM response cache with {len(LLM_RESPONSE_CACHE)} entries.")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

atexit.register(save_local_cache)
load_local_cache()

# -------------------
# Text enhancement functions

def fix_common_ocr_errors(text: str) -> str:
    corrections = {
        r'\bl\b': 'I',            # lowercase l standalone -> I
        r'\brn\b': 'm',           # rn looks like m
        r'ﬁ': 'fi',               # ligature fi
        r'ﬂ': 'fl',               # ligature fl
        r'\b0\b': 'O',            # zero to capital O in some contexts
        r'\b5\b': 'S',            # digit 5 to letter S OCR error
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return text

def improve_paragraph_structure(text: str) -> str:
    section_headers = [
        'WHEREAS', 'NOW THEREFORE', 'Period:', 'License Fee',
        'Deposit:', 'Amount:', 'Total:', 'Summary:', 'Details:'
    ]
    for header in section_headers:
        pattern = f'({re.escape(header)})'
        text = re.sub(pattern, f'\n\n\\1', text)
    # Collapse multiple paragraph breaks to max two
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text

def enhance_numeric_formatting(text: str) -> str:
    # Indian currency formatting and dates
    text = re.sub(r'Rs\.?\s*(\d+)', r'Rs. \1', text)
    text = re.sub(r'(\d+)\s*/-', r'\1/-', text)
    text = re.sub(r'(\d+)%', r'\1%', text)  # Ensure no spaces in percentages
    text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1/\2/\3', text)
    return text

def enhance_text_formatting(raw_text: str) -> str:
    try:
        text = raw_text
        text = fix_common_ocr_errors(text)
        text = improve_paragraph_structure(text)
        text = enhance_numeric_formatting(text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text
    except Exception:
        return re.sub(r'\s+', ' ', raw_text.strip())

def enhance_table_formatting(raw_table_text: str) -> str:
    try:
        text = textwrap.dedent(raw_table_text).strip()
        # Normalize pipes - ensure one space around pipe separators
        text = re.sub(r'\s*\|\s*', ' | ', text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) < 2:
            return text

        header_line = lines[0]
        second_line = lines[1]

        # Add markdown separator if missing
        if not re.match(r'(\|\s*:?-+:?\s*)+\|?', second_line):
            num_cols = header_line.count('|') - 1
            separator = '|' + ' --- |' * num_cols
            lines.insert(1, separator)

        return '\n'.join(lines)
    except Exception:
        return re.sub(r'\s+', ' ', raw_table_text.strip())

def clean_and_enhance_text(raw_text: str, is_table: bool = False) -> str:
    if not raw_text or not raw_text.strip():
        return raw_text
    text = raw_text.strip()
    if is_table:
        text = enhance_table_formatting(text)
    else:
        text = enhance_text_formatting(text)
    return text

# -------------------
# Document and chunk extraction

def extract_full_text(docling_document) -> str:
    all_text = []
    for item, _level in docling_document.iterate_items():
        if hasattr(item, "text") and item.text:
            all_text.append(item.text)
    return "\n".join(all_text)

def chunk_text_tiktoken(text: str, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, overlap=CHUNK_OVERLAP, encoding=ENCODING):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current_chunk = ""
    current_tokens = 0
    start_token_pos = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_tokens = len(encoding.encode(paragraph))

        if para_tokens > max_tokens:
            # Handle large paragraphs by sentence splitting
            if current_chunk and current_tokens >= min_tokens:
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "start_token": start_token_pos,
                    "end_token": start_token_pos + current_tokens,
                    "num_tokens": current_tokens
                })
                start_token_pos += current_tokens - overlap
                current_chunk = ""
                current_tokens = 0

            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = len(encoding.encode(sentence))
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
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        else:
            # Normal addition of paragraph
            if current_tokens + para_tokens > max_tokens and current_chunk:
                if current_tokens >= min_tokens:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "start_token": start_token_pos,
                        "end_token": start_token_pos + current_tokens,
                        "num_tokens": current_tokens
                    })
                start_token_pos += current_tokens - overlap
                if overlap > 0 and chunks:
                    overlap_tokens = encoding.encode(current_chunk)[-overlap:]
                    overlap_text = encoding.decode(overlap_tokens) if overlap_tokens else ""
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                    current_tokens = len(encoding.encode(current_chunk))
                else:
                    current_chunk = paragraph
                    current_tokens = para_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += para_tokens + len(encoding.encode("\n\n"))

    if current_chunk and current_tokens >= min_tokens:
        chunks.append({
            "chunk_text": current_chunk.strip(),
            "start_token": start_token_pos,
            "end_token": start_token_pos + current_tokens,
            "num_tokens": current_tokens
        })

    if not chunks and current_chunk:
        chunks.append({
            "chunk_text": current_chunk.strip(),
            "start_token": 0,
            "end_token": current_tokens,
            "num_tokens": current_tokens
        })

    return chunks

def extract_table_chunks(docling_document):
    table_chunks = []
    for item, _level in docling_document.iterate_items():
        if hasattr(item, 'data'):
            data = getattr(item, 'data')
            cells = getattr(data, 'table_cells', [])
            num_rows = getattr(data, 'num_rows', 0)
            num_cols = getattr(data, 'num_cols', 0)
            if cells and num_rows > 0 and num_cols > 0:
                grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
                for cell in cells:
                    for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                        for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                            grid[r][c] = cell.text
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

def extract_tables_with_camelot(pdf_path, pages="all"):
    tables = camelot.read_pdf(pdf_path, pages=pages)
    return [table.df for table in tables]

def hybrid_table_extraction(docling_tables, pdf_path):
    # Prefer Docling tables
    if docling_tables:
        return docling_tables, 'docling'
    # Fallback to Camelot tables
    camelot_tables = extract_tables_with_camelot(pdf_path)
    if camelot_tables:
        return camelot_tables, 'camelot'
    return [], None

def merge_adjacent_tables(table_chunks):
    merged = []
    buffer = []
    last_num_cols = None

    for chunk in table_chunks:
        if last_num_cols is not None and chunk['num_cols'] == last_num_cols:
            buffer.append(chunk)
        else:
            if buffer:
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

# -------------------
# Gemini LLM integration (context generation)

def get_prompt_template():
    if "template" not in PROMPT_CACHE:
        PROMPT_CACHE["template"] = (
            "You are a document assistant. Given the entire document content and a specific chunk from it, "
            "provide a concise context to situate the chunk within the whole document for better search relevance.\n\n"
            "DOCUMENT:\n{WHOLE_DOCUMENT}\n\n"
            "CHUNK:\n{CHUNK_CONTENT}\n\n"
            "Context:"
        )
    return PROMPT_CACHE["template"]

def create_cached_context(document_text):
    doc_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    if doc_hash in GEMINI_CACHED_CONTEXTS:
        return GEMINI_CACHED_CONTEXTS[doc_hash]

    cache_data = {
        "model": "models/gemini-1.5-flash-8b",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"\n{{WHOLE_DOCUMENT}}\n{document_text}\n\nHere is the chunk we want to situate within the whole document\n\n{{CHUNK_CONTENT}}\n\nPlease give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
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
        GEMINI_CACHED_CONTEXTS[doc_hash] = cached_content_name
        return cached_content_name
    except Exception as e:
        return None

def get_gemini_context_with_cache(document_text: str, chunk_text: str) -> str:
    key = hashlib.sha256((document_text + "|||" + chunk_text).encode("utf-8")).hexdigest()
    if key in LLM_RESPONSE_CACHE:
        return LLM_RESPONSE_CACHE[key]

    cached_content_name = create_cached_context(document_text)
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    if cached_content_name:
        data = {
            "cachedContent": cached_content_name,
            "contents": [
                {
                    "role": "user", "parts": [{"text": f"\n{chunk_text}\n"}]
                }
            ]
        }
    else:
        prompt = get_prompt_template().format(WHOLE_DOCUMENT=document_text, CHUNK_CONTENT=chunk_text)
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        context = result["candidates"][0]["content"]["parts"][0]["text"]
        LLM_RESPONSE_CACHE[key] = context.strip()
        return context.strip()
    except Exception:
        return ""

# -------------------
# Processing documents

def concatenate_text_and_tables(text: str, table_chunks: List[Dict[str, Any]]) -> str:
    content = text.strip()
    for t_chunk in table_chunks:
        content += "\n\n" + t_chunk["chunk_text"].strip()
    return content

def save_extracted_log(file_path: Path, extracted_content: str):
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'extraction_logs')
    os.makedirs(log_dir, exist_ok=True)
    base_name = os.path.basename(str(file_path))
    log_path = os.path.join(log_dir, base_name + ".extracted.log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(extracted_content)
    print(f"Saved extracted content to {log_path}")

def process_single_document(file_path: Path, converter: DocumentConverter) -> List[Dict[str, Any]]:
    result = converter.convert(file_path)
    chunks_for_doc = []

    if hasattr(result, "document") and result.document is not None:
        full_text = extract_full_text(result.document)
        docling_tables = extract_table_chunks(result.document)
        fallback_tables, table_source = hybrid_table_extraction(docling_tables, str(result.input.file))

        # Use Camelot tables if docling fails
        table_chunks = []
        if table_source == 'docling':
            table_chunks = docling_tables
        elif table_source == 'camelot':
            for idx, df in enumerate(fallback_tables):
                header = '| ' + ' | '.join(df.iloc[0]) + ' |'
                separator = '|' + '---|' * len(df.columns)
                body = ['| ' + ' | '.join(map(str, row)) + ' |' for row in df.iloc[1:].values]
                md_table = '\n'.join([header, separator] + body)
                table_chunks.append({
                    "chunk_text": md_table,
                    "is_table": True,
                    "table_index": idx,
                    "num_rows": len(df),
                    "num_cols": len(df.columns)
                })

        # Merge adjacent tables with same columns
        table_chunks = merge_adjacent_tables(table_chunks)

        whole_doc_content = concatenate_text_and_tables(full_text, table_chunks)
        save_extracted_log(result.input.file, whole_doc_content)
        create_cached_context(whole_doc_content)  # Cache once for document

        # Chunk text paragraphs
        chunks = chunk_text_tiktoken(full_text)
        for idx, chunk in enumerate(tqdm(chunks, desc="Loading text chunks")):
            context = get_gemini_context_with_cache(whole_doc_content, chunk["chunk_text"])
            original_text = chunk["chunk_text"]
            enhanced_text = clean_and_enhance_text(original_text, is_table=False)
            chunk_metadata = {
                "document_name": str(result.input.file),
                "chunk_id": f"{result.input.file}_text_{idx+1}",
                "chunk_index": idx,
                "start_token": chunk["start_token"],
                "end_token": chunk["end_token"],
                "num_tokens": chunk["num_tokens"],
                "raw_chunk_text": original_text,
                "chunk_text": enhanced_text,
                "context": context,
                "is_table": False
            }
            chunks_for_doc.append(chunk_metadata)

        # Process table chunks
        for t_idx, t_chunk in enumerate(tqdm(table_chunks, desc="Loading table chunks")):
            context = get_gemini_context_with_cache(whole_doc_content, t_chunk["chunk_text"])
            original_table_text = t_chunk["chunk_text"]
            enhanced_table_text = clean_and_enhance_text(original_table_text, is_table=True)
            chunk_metadata = {
                "document_name": str(result.input.file),
                "chunk_id": f"{result.input.file}_table_{t_idx+1}",
                "chunk_index": t_idx,
                "start_token": None,
                "end_token": None,
                "num_tokens": None,
                "raw_chunk_text": original_table_text,
                "chunk_text": enhanced_table_text,
                "context": context,
                "is_table": True,
                "num_rows": t_chunk.get("num_rows"),
                "num_cols": t_chunk.get("num_cols")
            }
            chunks_for_doc.append(chunk_metadata)

    if hasattr(result, 'errors') and result.errors:
        print("Errors:")
        for error in result.errors:
            print(f" - {error.error_message}")

    return chunks_for_doc

# -------------------
# Save results

def save_results(all_contextualized_chunks: List[Dict[str, Any]]) -> None:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_contextualized_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_contextualized_chunks)} contextualized chunks to {OUTPUT_JSON}")

    if all_contextualized_chunks:
        all_keys = set()
        for row in all_contextualized_chunks:
            all_keys.update(row.keys())
        fieldnames = list(all_keys)

        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_contextualized_chunks:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"Saved {len(all_contextualized_chunks)} contextualized chunks to {OUTPUT_CSV}")

# -------------------
# Main entry point

def main():
    if not SOURCE_DIR.exists() or not SOURCE_DIR.is_dir():
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables.")
        return

    files = [f for f in SOURCE_DIR.iterdir() if f.is_file()]
    if not files:
        print(f"No files found in {SOURCE_DIR}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    converter = DocumentConverter()
    all_contextualized_chunks = []

    for file_path in files:
        try:
            chunks_for_doc = process_single_document(file_path, converter)
            all_contextualized_chunks.extend(chunks_for_doc)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    save_results(all_contextualized_chunks)

# -------------------
# CLI support

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process documents for chunking and contextualization')
    parser.add_argument('--single-file', help='Process a single file instead of the entire folder')
    args = parser.parse_args()

    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {args.single_file}")
            exit(1)
        converter = DocumentConverter()
        try:
            chunks = process_single_document(file_path, converter)
            print(f"Processed {file_path.name}: {len(chunks)} chunks generated")
            existing_chunks = []
            if os.path.exists(OUTPUT_JSON):
                try:
                    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                        existing_chunks = json.load(f)
                except Exception:
                    existing_chunks = []
            existing_chunks = [c for c in existing_chunks if c.get('document_name') != str(file_path)]
            existing_chunks.extend(chunks)
            save_results(existing_chunks)
            print(f"[SUCCESS] Successfully processed and saved chunks for {file_path.name}")
        except Exception as e:
            print(f"[ERROR] Error processing {file_path}: {e}")
            exit(1)
    else:
        main()
