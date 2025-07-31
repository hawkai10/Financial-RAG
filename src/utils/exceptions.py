"""Custom exceptions for DocuChat AI with actionable error messages."""

class DocuChatError(Exception):
    """Base exception for DocuChat AI."""
    def __init__(self, message: str, remedy: str = None, error_code: str = None):
        self.message = message
        self.remedy = remedy or "Please try again or contact support."
        self.error_code = error_code or "GENERAL_ERROR"
        super().__init__(self.message)

class GeminiAPIError(DocuChatError):
    """Gemini API related errors."""
    def __init__(self, message: str, status_code: int = None):
        remedies = {
            401: "Check your GEMINI_API_KEY in the .env file.",
            429: "Rate limit exceeded. Please wait a moment and try again.",
            403: "API key doesn't have required permissions.",
            500: "Gemini service is temporarily unavailable."
        }
        
        remedy = remedies.get(status_code, "Check your API configuration and try again.")
        error_code = f"GEMINI_API_{status_code}" if status_code else "GEMINI_API_ERROR"
        
        super().__init__(message, remedy, error_code)
        self.status_code = status_code

class RetrievalError(DocuChatError):
    """Document retrieval related errors."""
    def __init__(self, message: str, retrieval_type: str = "general"):
        remedies = {
            "index": "Run 'python embed_chunks_txtai.py' to rebuild the search index.",
            "chunks": "Ensure 'contextualized_chunks.json' exists and contains valid data.",
            "hybrid": "Check BM25 initialization or disable hybrid search in config.",
            "progressive": "Fallback to standard retrieval will be used."
        }
        
        remedy = remedies.get(retrieval_type, "Check document processing and indexing.")
        super().__init__(message, remedy, f"RETRIEVAL_{retrieval_type.upper()}")

class OptimizationError(DocuChatError):
    """Optimization feature related errors."""
    def __init__(self, message: str, optimization_type: str = "general"):
        remedies = {
            "sampling": "Using complete analysis instead of sampling.",
            "progressive": "Falling back to standard retrieval method.",
            "caching": "Query will be processed without caching."
        }
        
        remedy = remedies.get(optimization_type, "Continuing with standard processing.")
        super().__init__(message, remedy, f"OPTIMIZATION_{optimization_type.upper()}")

class ValidationError(DocuChatError):
    """Input validation errors."""
    def __init__(self, message: str, field: str = None):
        remedies = {
            "query": "Please enter a valid question (3-1000 characters).",
            "session": "Session will be reset automatically.",
            "feedback": "Feedback submission failed, but your query was processed."
        }
        
        remedy = remedies.get(field, "Please check your input and try again.")
        super().__init__(message, remedy, f"VALIDATION_{field.upper()}" if field else "VALIDATION_ERROR")
