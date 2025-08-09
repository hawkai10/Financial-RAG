import os
import json
from pathlib import Path
from dotenv import load_dotenv
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from transformers import GPT2TokenizerFast

# 1. Load environment variables (.env should contain GEMINI_API_KEY)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# 2. Prepare marker config for JSON output, OCR, and LLM usage
config = {
    "output_format": "json",
    "ocr": True,
    "use_llm": True,
    "gemini_api_key": GEMINI_API_KEY,
    "gemini_model": "gemini-1.5-flash-8b",
    "llm_model": "gemini-1.5-flash-8b",
    "model_name": "gemini-1.5-flash-8b",
    "llm": {
        "model": "gemini-1.5-flash-8b",
        "gemini_model_name": "gemini-1.5-flash-8b"
    }
}
config_parser = ConfigParser(config)

# Set environment variables to ensure model selection
os.environ["GEMINI_MODEL"] = "gemini-1.5-flash-8b"
os.environ["LLM_MODEL"] = "gemini-1.5-flash-8b"

# 3. Create PDF converter with LLM enhancement and OCR
converter = PdfConverter(
    artifact_dict=create_model_dict(),
    config=config_parser.generate_config_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service="marker.services.gemini.GoogleGeminiService"
)

# Override model settings if possible
try:
    if hasattr(converter, 'llm_service') and converter.llm_service:
        if hasattr(converter.llm_service, 'gemini_model_name'):
            converter.llm_service.gemini_model_name = "gemini-1.5-flash-8b"
except Exception:
    pass  # Silently continue if override fails

# 4. Tokenizer for chunking logic
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def chunk_content(blocks, min_tokens=100, max_tokens=500, overlap=100):
    """
    Chunk blocks into text segments, keeping tables as atomic units.
    """
    chunks = []
    buffer = []
    buffer_types = []
    token_count = 0

    def flush_chunk():
        if buffer:
            chunk_text = "\n".join(buffer)
            chunks.append({
                "tokens": token_count,
                "content": chunk_text,
                "has_table": "table" in buffer_types
            })

    print(f"🔄 Chunking {len(blocks)} blocks...")

    for i, block in enumerate(blocks):
        # Try different ways to get block type and text
        block_type = "text"  # default
        block_text = ""
        
        # Get block type
        if hasattr(block, "block_type"):
            block_type = block.block_type
        elif hasattr(block, "type"):
            block_type = block.type
        elif hasattr(block, "__class__"):
            block_type = block.__class__.__name__.lower()
        
        # Get block text - try different methods for JSONBlockOutput
        if hasattr(block, "html") and block.html:
            # Convert HTML to text (basic)
            import re
            block_text = re.sub('<[^<]+?>', '', block.html)
        elif hasattr(block, "model_dump_json"):
            # Use the new Pydantic method instead of deprecated json()
            try:
                json_data = block.model_dump()  # Get dict directly
                # Extract text from JSON structure
                if isinstance(json_data, dict):
                    if 'text' in json_data:
                        block_text = json_data['text']
                    elif 'content' in json_data:
                        block_text = json_data['content']
                    elif 'html' in json_data:
                        import re
                        block_text = re.sub('<[^<]+?>', '', json_data['html'])
            except Exception as e:
                print(f"⚠️ Error extracting from model_dump: {e}")
        elif hasattr(block, "text"):
            block_text = block.text or ""
        elif hasattr(block, "content"):
            block_text = block.content or ""
        elif isinstance(block, str):
            block_text = block
        
        # If we still don't have text, try to inspect the block further
        if not block_text and hasattr(block, 'children'):
            # Recursively extract text from children
            child_texts = []
            for child in (block.children or []):
                child_text = extract_text_from_block(child)
                if child_text:
                    child_texts.append(child_text)
            block_text = "\n".join(child_texts)
        
        # Skip empty blocks
        if not block_text.strip():
            continue
        
        try:
            block_tokens = len(tokenizer.encode(block_text))
        except Exception as e:
            print(f"⚠️ Tokenization failed for block {i+1}: {e}")
            block_tokens = len(block_text.split())  # Fallback to word count
            
        print(f"📦 Processing block {i+1}: {len(block_text)} chars, {block_tokens} tokens")
        
        # Keep tables atomic and flush current chunk before adding
        if "table" in block_type.lower():
            if token_count > 0:
                flush_chunk()
                buffer.clear()
                buffer_types.clear()
                token_count = 0
            chunks.append({
                "tokens": block_tokens,
                "content": block_text,
                "has_table": True
            })
            print(f"📊 Added table chunk with {block_tokens} tokens")
            continue

        # Flush if adding block exceeds max_tokens
        if token_count + block_tokens > max_tokens:
            flush_chunk()
            # Overlap logic: retain last 'overlap' tokens from previous chunk
            if overlap > 0 and chunks:
                last_chunk_text = chunks[-1]["content"]
                try:
                    last_tokens = tokenizer.encode(last_chunk_text)
                    overlap_text = tokenizer.decode(last_tokens[-overlap:]) if len(last_tokens) > overlap else last_chunk_text
                except:
                    # Fallback to character-based overlap
                    overlap_text = last_chunk_text[-overlap*4:] if len(last_chunk_text) > overlap*4 else last_chunk_text
                buffer = [overlap_text]
                buffer_types = []
                try:
                    token_count = len(tokenizer.encode(overlap_text))
                except:
                    token_count = len(overlap_text.split())
            else:
                buffer.clear()
                buffer_types.clear()
                token_count = 0

        buffer.append(block_text)
        buffer_types.append(block_type)
        token_count += block_tokens

    flush_chunk()

    # Return chunks with min_tokens or containing tables
    final_chunks = []
    for c in chunks:
        if c.get("has_table", False) or c["tokens"] >= min_tokens:
            if "has_table" in c:
                c.pop("has_table")
            final_chunks.append(c)
    
    print(f"✅ Created {len(final_chunks)} final chunks")
    return final_chunks

def extract_text_from_block(block):
    """Helper function to extract text from any block type"""
    if hasattr(block, "html") and block.html:
        import re
        return re.sub('<[^<]+?>', '', block.html)
    elif hasattr(block, "text"):
        return block.text or ""
    elif hasattr(block, "content"):
        return block.content or ""
    elif isinstance(block, str):
        return block
    elif hasattr(block, 'children') and block.children:
        child_texts = []
        for child in block.children:
            child_text = extract_text_from_block(child)
            if child_text:
                child_texts.append(child_text)
        return "\n".join(child_texts)
    return ""

def process_documents(source_dir, output_path):
    """
    Process all files in source_dir, chunking their contents and saving JSON to output_path.
    """
    results = []

    for file_path in Path(source_dir).glob("*.pdf"):  # Only process PDF files
        if not file_path.is_file():
            continue
        print(f"Processing {file_path.name}...")
        try:
            rendered = converter(str(file_path))
            print(f"✅ Successfully converted {file_path.name}")
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            results.append({
                "filename": file_path.name,
                "chunks": []
            })
            continue

        # Debug: Check the structure of rendered object
        print(f"✅ Converted {file_path.name} - type: {type(rendered).__name__}")
        
        # Try different ways to access content
        blocks = []
        
        # Method 1: Try children attribute
        if hasattr(rendered, "children"):
            print(f"📄 Found {len(rendered.children)} pages")
            for page in rendered.children:
                if hasattr(page, "children"):
                    blocks.extend(page.children)
        
        # Method 2: Try markdown attribute
        elif hasattr(rendered, "markdown"):
            print(f"📄 Found markdown content: {len(rendered.markdown)} characters")
            # Simple markdown to blocks conversion
            markdown_lines = rendered.markdown.split('\n')
            for line in markdown_lines:
                if line.strip():
                    # Create a simple block object
                    block = type('Block', (), {
                        'text': line.strip(),
                        'type': 'text'
                    })()
                    blocks.append(block)
        
        print(f"📦 Extracted {len(blocks)} blocks")

        chunks = chunk_content(blocks)
        print(f"✅ Generated {len(chunks)} chunks for {file_path.name}")
        
        results.append({
            "filename": file_path.name,
            "chunks": chunks
        })

    # Save results as structured JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(results)} documents to {output_path}")

if __name__ == "__main__":
    source_dir = Path("Source_Documents")
    output_file = "parsed_chunks.json"
    
    if source_dir.exists():
        process_documents(source_dir, output_file)
        print(f"✅ Parsing complete. Chunks saved to {output_file}")
    else:
        print(f"⚠️ Source directory {source_dir} not found. Creating directory...")
        source_dir.mkdir(exist_ok=True)
        print(f"📁 Created {source_dir}. Please add PDF files and run again.")
