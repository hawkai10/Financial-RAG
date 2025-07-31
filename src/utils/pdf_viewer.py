import base64
import fitz  # PyMuPDF
import streamlit as st
from typing import Dict, List, Tuple, Optional
import json
import re

class DocumentViewer:
    """Handles PDF viewing and text highlighting functionality."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.json']
    
    def extract_pdf_text_with_positions(self, pdf_path: str) -> Dict:
        """Extract text from PDF with position information for highlighting."""
        try:
            doc = fitz.open(pdf_path)
            pages_data = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text blocks with positions
                text_blocks = page.get_text("dict")
                
                # Extract text with bounding boxes
                page_text = ""
                text_positions = []
                
                for block in text_blocks["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"]
                                bbox = span["bbox"]  # (x0, y0, x1, y1)
                                
                                if text.strip():
                                    text_positions.append({
                                        "text": text,
                                        "bbox": bbox,
                                        "start_pos": len(page_text),
                                        "end_pos": len(page_text) + len(text)
                                    })
                                    page_text += text
                
                pages_data[page_num] = {
                    "text": page_text,
                    "positions": text_positions,
                    "page_size": page.rect
                }
            
            doc.close()
            return pages_data
            
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return {}
    
    def find_text_in_pdf(self, pdf_path: str, search_text: str) -> List[Dict]:
        """Find text matches in PDF with page and position information."""
        matches = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Search for text
                text_instances = page.search_for(search_text)
                
                if text_instances:
                    for rect in text_instances:
                        matches.append({
                            "page": page_num,
                            "bbox": tuple(rect),
                            "text": search_text
                        })
            
            doc.close()
            return matches
            
        except Exception as e:
            st.error(f"Error searching PDF: {e}")
            return []
    
    def highlight_pdf_page(self, pdf_path: str, page_num: int, highlights: List[Dict]) -> bytes:
        """Create highlighted PDF page as image."""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # Add highlights
            for highlight in highlights:
                bbox = fitz.Rect(highlight["bbox"])
                # Add yellow highlight
                annot = page.add_highlight_annot(bbox)
                annot.set_colors({"stroke": [1, 1, 0], "fill": [1, 1, 0]})
                annot.update()
            
            # Render page as image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            doc.close()
            return img_data
            
        except Exception as e:
            st.error(f"Error highlighting PDF: {e}")
            return None
    
    def get_pdf_as_base64(self, pdf_path: str) -> str:
        """Convert PDF to base64 for embedding in HTML."""
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            return base64.b64encode(pdf_bytes).decode()
        except Exception as e:
            st.error(f"Error converting PDF to base64: {e}")
            return ""
    
    def create_pdf_viewer_html(self, pdf_base64: str, highlights: List[Dict] = None) -> str:
        """Create HTML for PDF viewer with highlighting."""
        html = f"""
        <div style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 5px;">
            <iframe 
                src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" 
                height="100%" 
                style="border: none;">
                <p>Your browser does not support PDFs. Please download the PDF to view it.</p>
            </iframe>
        </div>
        """
        return html

class ChunkHighlighter:
    """Handles highlighting of specific chunks within documents."""
    
    def __init__(self):
        self.doc_viewer = DocumentViewer()
    
    def find_chunk_in_document(self, chunk_text: str, document_path: str) -> Dict:
        """Find the exact location of a chunk in the source document."""
        try:
            if document_path.lower().endswith('.pdf'):
                return self._find_chunk_in_pdf(chunk_text, document_path)
            elif document_path.lower().endswith('.txt'):
                return self._find_chunk_in_text(chunk_text, document_path)
            else:
                return {"error": "Unsupported document format"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _find_chunk_in_pdf(self, chunk_text: str, pdf_path: str) -> Dict:
        """Find chunk location in PDF."""
        try:
            # Clean the chunk text for better matching
            clean_chunk = self._clean_text_for_matching(chunk_text)
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                clean_page_text = self._clean_text_for_matching(page_text)
                
                # Try to find the chunk text
                if clean_chunk[:100] in clean_page_text:  # Match first 100 chars
                    # Find exact positions
                    matches = page.search_for(chunk_text[:50])  # Search for beginning
                    
                    if matches:
                        doc.close()
                        return {
                            "found": True,
                            "page": page_num,
                            "matches": [{"bbox": tuple(rect)} for rect in matches],
                            "document_type": "pdf"
                        }
            
            doc.close()
            return {"found": False, "document_type": "pdf"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _find_chunk_in_text(self, chunk_text: str, text_path: str) -> Dict:
        """Find chunk location in text file."""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            clean_chunk = self._clean_text_for_matching(chunk_text)
            clean_full_text = self._clean_text_for_matching(full_text)
            
            start_pos = clean_full_text.find(clean_chunk[:100])
            
            if start_pos != -1:
                return {
                    "found": True,
                    "start_position": start_pos,
                    "end_position": start_pos + len(clean_chunk),
                    "document_type": "text"
                }
            else:
                return {"found": False, "document_type": "text"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _clean_text_for_matching(self, text: str) -> str:
        """Clean text for better matching between chunk and document."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\-\(\)]', '', text)
        return text.lower()
