"""
Centralized prompt template system for DocuChat AI
Eliminates code duplication and provides consistent prompt management
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .utils import logger

@dataclass
class PromptTemplate:
    """Structured prompt template with metadata."""
    instruction: str
    context_prefix: str = "Context:"
    question_prefix: str = "Question:"
    format_instructions: str = ""
    max_context_length: int = 4000

class PromptBuilder:
    """Centralized prompt builder for all RAG strategies."""
    
    def __init__(self):
        self.templates = {
            "query_correction": PromptTemplate(
                instruction="Correct the spelling and grammar of this search query for business documents. Keep the corrected query concise and focused. Only fix obvious errors:",
                context_prefix="Query to correct:",
                question_prefix="",
                format_instructions="Return only the corrected query without additional text."
            ),
            
            "multiquery_generation": PromptTemplate(
                instruction="Generate {num_queries} diverse, rephrased queries that could retrieve relevant information from a database containing invoices, contracts, and financial documents.",
                context_prefix="Original query:",
                question_prefix="",
                format_instructions="Return each query on a new line without numbering."
            ),
            
            "Standard": PromptTemplate(
                instruction="""Answer this business question using the provided context. Be direct and comprehensive while staying focused on the query. 

IMPORTANT: Pay careful attention to temporal references (first year, second year, next months, etc.) and numerical values. When multiple time periods or amounts are mentioned, ensure you answer for the specific period requested in the question.

Provide a clear, well-structured response.""",
                format_instructions="Structure your answer clearly and concisely. If multiple time periods are mentioned in the context, be specific about which period you're referencing.",
                max_context_length=3800  # Slightly reduced to ensure safety
            ),
            
            "Analyse": PromptTemplate(
                instruction="Perform a comprehensive analysis of the provided business documents to answer this analytical question. Look for patterns, trends, relationships, and provide detailed insights.",
                format_instructions="Provide a well-structured analysis with clear sections for summary, detailed findings, and key insights.",
                max_context_length=3200  # Reduced for complex analysis prompts
            ),
            
            "Aggregation": PromptTemplate(
                instruction="Count, list, or aggregate information from the provided business documents. Be precise with numbers and provide complete listings. Provide a clear, well-organized response without excessive formatting.",
                format_instructions="""Provide your response in a clear format:
- Start with a summary of totals or key findings
- Follow with detailed breakdown if needed
- List relevant items clearly and concisely""",
                max_context_length=3500  # Reduced for aggregation to ensure chunks fit
            )
        }
    
    def build_prompt(self, template_name: str, question: str, 
                    context_chunks: List[Dict], **kwargs) -> str:
        """Build a complete prompt from template and inputs."""
        
        try:
            print(f"DEBUG PromptBuilder: build_prompt called with template_name={template_name}")
            print(f"DEBUG PromptBuilder: question={question[:50]}...")
            print(f"DEBUG PromptBuilder: chunks count={len(context_chunks)}")
            
            # Check if template exists in templates
            if hasattr(self, 'templates'):
                print(f"DEBUG PromptBuilder: Available templates: {list(self.templates.keys())}")
                
                if template_name in self.templates:
                    template = self.templates[template_name]
                    print(f"DEBUG PromptBuilder: Found template for {template_name}")
                    print(f"DEBUG PromptBuilder: Template type: {type(template)}")
                    print(f"DEBUG PromptBuilder: Template: {template}")
                    
                    # Handle special cases
                    if template_name == "query_correction":
                        result = f"{template.instruction} {question}"
                        print(f"DEBUG PromptBuilder: ✅ Query correction template succeeded")
                        return result
                    
                    if template_name == "multiquery_generation":
                        num_queries = kwargs.get('num_queries', 2)
                        instruction = template.instruction.format(num_queries=num_queries)
                        result = f"{instruction}\n{template.context_prefix} {question}\n{template.format_instructions}"
                        print(f"DEBUG PromptBuilder: ✅ Multiquery generation template succeeded")
                        return result
                    
                    # Build context from chunks
                    print(f"DEBUG PromptBuilder: About to call _build_context with {len(context_chunks)} chunks")
                    context = self._build_context(context_chunks, template.max_context_length)
                    print(f"DEBUG PromptBuilder: ✅ _build_context succeeded, context length: {len(context)}")
                    
                    # LOG: Context details for debugging
                    context_summary = ""
                    for i, chunk in enumerate(context_chunks[:3]):  # Show first 3 chunks
                        text = chunk.get("text", "") or chunk.get("chunk_text", "")
                        doc_name = chunk.get("document_name", "Unknown")
                        context_summary += f"Chunk {i+1}: {doc_name} - {len(text)} chars - Preview: {text[:150]}...\n"
                    if len(context_chunks) > 3:
                        context_summary += f"... and {len(context_chunks) - 3} more chunks\n"
                    
                    # LLM LOG DISABLED - log_llm_interaction(
                    #     phase="CONTEXT_BUILDING",
                    #     content=context,
                    #     template=template_name,
                    #     chunks_count=len(context_chunks),
                    #     context_length=len(context),
                    #     chunks_summary=context_summary
                    # )
                    
                    # Construct full prompt
                    prompt_parts = [
                        template.instruction,
                        "",
                        f"{template.context_prefix}",
                        context,
                        "",
                        f"{template.question_prefix} {question}",
                        "",
                        template.format_instructions
                    ]
                    
                    result = "\n".join(part for part in prompt_parts if part)
                    print(f"DEBUG PromptBuilder: ✅ Template building succeeded, final prompt length: {len(result)}")
                    return result
                else:
                    print(f"DEBUG PromptBuilder: ❌ Template {template_name} not found in templates")
                    raise ValueError(f"Unknown template: {template_name}")
            else:
                print("DEBUG PromptBuilder: ❌ No templates attribute found")
                raise AttributeError("PromptBuilder has no templates attribute")
                
        except Exception as e:
            print(f"DEBUG PromptBuilder: ❌ build_prompt FAILED: {e}")
            print(f"DEBUG PromptBuilder: Error type: {type(e)}")
            import traceback
            print(f"DEBUG PromptBuilder: Traceback:\n{traceback.format_exc()}")
            raise e
    
    def _build_context(self, chunks: List[Dict], max_length: int) -> str:
        """Build context string from chunks with smart truncation to prevent cutting chunks in middle."""
        context_parts = []
        current_length = 0
        
        # Reserve space for prompt structure (instruction, question, formatting)
        # Estimate ~500 chars for prompt overhead
        effective_max_length = max_length - 500
        
        for i, chunk in enumerate(chunks):
            # Handle both 'text' and 'chunk_text' field names
            text = chunk.get("text", "") or chunk.get("chunk_text", "")
            doc_name = chunk.get("document_name", "Unknown")
            
            # Extract just filename for cleaner display
            if isinstance(doc_name, str) and '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            elif isinstance(doc_name, str) and '\\' in doc_name:
                doc_name = doc_name.split('\\')[-1]
            
            # Skip empty chunks
            if not text or not text.strip():
                continue
            
            chunk_text = f"Document {i+1} ({doc_name}):\n{text}\n"
            chunk_length = len(chunk_text)
            
            # Smart truncation: Only add chunk if it fits completely
            if current_length + chunk_length > effective_max_length:
                # If this is the first chunk and it's too big, truncate it smartly
                if i == 0 and not context_parts:
                    # Keep the chunk but truncate at sentence boundary
                    available_space = effective_max_length - len(f"Document {i+1} ({doc_name}):\n") - 50  # 50 for "...[truncated]"
                    
                    if available_space > 100:  # Only if we have reasonable space
                        truncated_text = self._smart_truncate(text, available_space)
                        chunk_text = f"Document {i+1} ({doc_name}):\n{truncated_text}...[truncated due to length]\n"
                        context_parts.append(chunk_text)
                        print(f"⚠️  First chunk truncated to fit within {max_length} token limit")
                    else:
                        print(f"⚠️  First chunk too large even for truncation, skipping")
                else:
                    # Stop adding more chunks - we've hit the limit
                    if context_parts:  # Only show message if we have some chunks
                        print(f"📋 Context limit reached: included {len(context_parts)}/{len(chunks)} chunks (max {max_length} chars)")
                break
                
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        if not context_parts:
            return "No relevant content found in the documents."
        
        result = "\n---\n".join(context_parts)
        
        # Final safety check
        if len(result) > effective_max_length:
            print(f"⚠️  Final context still too long ({len(result)} chars), applying emergency truncation")
            result = result[:effective_max_length] + "...[emergency truncation applied]"
        
        print(f"📊 Context built: {len(context_parts)} chunks, {len(result)} characters")
        return result
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Truncate text at sentence boundaries when possible."""
        if len(text) <= max_length:
            return text
        
        # Try to cut at sentence endings first
        sentences = text.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') > max_length:
                break
            truncated += sentence + '. '
        
        # If we got something reasonable, return it
        if len(truncated) > max_length * 0.5:  # At least half the desired length
            return truncated.rstrip()
        
        # Otherwise, just cut at word boundary
        words = text.split()
        truncated = ""
        
        for word in words:
            if len(truncated + word + ' ') > max_length:
                break
            truncated += word + ' '
        
        return truncated.rstrip() if truncated else text[:max_length]
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def validate_template(self, template_name: str) -> bool:
        """Validate if template exists."""
        return template_name in self.templates

# Global instance - This is what gets imported
prompt_builder = PromptBuilder()

# Export for external use
__all__ = ['PromptBuilder', 'PromptTemplate', 'prompt_builder']
