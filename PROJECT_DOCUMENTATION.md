# Financial-RAG Project Documentation

## 📊 Project Overview

**Financial-RAG** is a sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for financial document analysis and question-answering. The system processes financial documents (PDFs, invoices, reports) using advanced text extraction and semantic chunking techniques, then provides intelligent responses to user queries about the financial data.

### 🎯 Key Features
- **PDF Text Extraction** using Marker library with OCR and LLM enhancement
- **JSON-based Semantic Chunking** for financial document processing
- **Progressive Retrieval** with multi-stage relevance scoring
- **Cross-encoder Reranking** for precise context selection
- **Gemini API Integration** for high-quality response generation
- **Feedback System** for continuous improvement
- **Caching Mechanisms** for performance optimization

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Source PDFs    │ -> │   Extraction     │ -> │   JSON Blocks   │
│  (invoices,     │    │  (extraction.py) │    │  (structured)   │
│   reports, etc) │    └──────────────────┘    └─────────────────┘
└─────────────────┘                                       │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final Answer  │ <- │   RAG Backend    │ <- │   Chunking      │
│  (Gemini API)   │    │ (rag_backend.py) │    │(json_semantic_  │
└─────────────────┘    └──────────────────┘    │    chunker.py)  │
                                  │             └─────────────────┘
                                  ▼                       │
                        ┌─────────────────┐              ▼
                        │   Retrieval &   │    ┌─────────────────┐
                        │   Reranking     │ <- │  Chunk Storage  │
                        │ (progressive_   │    │ (chunks.db,     │
                        │  retrieval.py)  │    │  JSON files)    │
                        └─────────────────┘    └─────────────────┘
```

---

## 📁 Project Structure & File Documentation

### 🔧 **Core Extraction & Processing**

#### `extraction.py` ⭐ **ACTIVE**
**Purpose**: Main PDF extraction script using Marker library
- **Function**: Extracts text from PDFs in `Source_Documents` folder
- **Output**: Creates JSON files in `New folder` with structured document blocks
- **Features**:
  - API key loaded from `.env` file for security
  - Per-file processing with retry logic and error handling
  - Marker CLI integration with OCR and LLM enhancement
  - Temporary directory isolation for each file
- **Command**: `python extraction.py`

#### `json_semantic_chunker.py` ⭐ **ACTIVE**
**Purpose**: Advanced semantic chunking for JSON-structured documents
- **Function**: Processes JSON blocks from extraction into semantically meaningful chunks
- **Features**:
  - Converts HTML tables to clean, readable text
  - Semantic similarity-based grouping using sentence embeddings
  - Token-based sizing (100-500 tokens with 100-token overlap)
  - Financial document structure preservation
- **Output**: `json_chunks.json` with processed chunks
- **Command**: `python json_semantic_chunker.py`

### 🧠 **RAG System Core**

#### `rag_backend.py` ⭐ **ACTIVE**
**Purpose**: Central RAG orchestration and query processing
- **Function**: Main entry point for question-answering system
- **Components**:
  - Query preprocessing and strategy determination
  - Chunk retrieval coordination
  - Gemini API integration for answer generation
  - Context window management
  - Response formatting and citation
- **Integration**: Works with all retrieval and chunking components

#### `unified_query_processor.py` ⭐ **ACTIVE**
**Purpose**: Intelligent query analysis and strategy selection
- **Function**: Analyzes user queries to determine optimal processing strategy
- **Strategies**:
  - **Standard**: Regular Q&A queries
  - **Analyse**: Deep analysis requests
  - **Aggregation**: Summary and aggregation queries
- **Output**: Strategy recommendations that guide downstream processing

#### `progressive_retrieval.py` ⭐ **ACTIVE**
**Purpose**: Smart, multi-stage chunk retrieval system
- **Function**: Implements progressive retrieval with quality assessment
- **Features**:
  - Initial small batch retrieval
  - Quality scoring and gap analysis
  - Adaptive expansion based on content quality
  - Performance optimization over traditional top-k retrieval

#### `document_reranker.py` ⭐ **ACTIVE**
**Purpose**: Cross-encoder based relevance reranking
- **Function**: Refines chunk relevance using `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Process**:
  - Takes retrieved chunks from progressive retrieval
  - Calculates precise query-chunk relevance scores
  - Reorders chunks by final relevance score
  - Strategy-aware processing (skips reranking for aggregation queries)

### 📊 **Data Management**

#### `chunk_manager.py` ⭐ **ACTIVE**
**Purpose**: Efficient chunk storage and retrieval interface
- **Function**: Memory-mapped access to chunk data with lazy loading
- **Features**:
  - Index-based chunk access (`chunks.index.json`)
  - Memory-efficient large dataset handling
  - Caching and performance optimization
- **Critical**: DO NOT read `contextualized_chunks.json` directly; use ChunkManager

#### `config.py` ⭐ **ACTIVE**
**Purpose**: Centralized configuration management
- **Contents**:
  - File paths and directory configurations
  - Model names and API endpoints
  - Processing parameters and thresholds
  - System-wide constants
- **Usage**: Imported by all components for consistent configuration

#### `feedback_database.py` ⭐ **ACTIVE**
**Purpose**: User feedback and performance tracking
- **Function**: SQLite database for storing user interactions and feedback
- **Tables**:
  - User queries and responses
  - Feedback scores and comments
  - Performance metrics
- **File**: `feedback.db`

### 🎛️ **System Components**

#### `aggregation_optimizer.py` ⭐ **ACTIVE**
**Purpose**: Optimization for aggregation-type queries
- **Function**: Handles large-scale data aggregation efficiently
- **Features**:
  - Statistical sampling for large datasets
  - Smart chunk selection for aggregation queries
  - Performance optimization for summary operations

#### `keyword_aggregation_retriever.py` ⭐ **ACTIVE**
**Purpose**: Keyword-based retrieval for aggregation queries
- **Function**: Specialized retrieval for aggregation and summary tasks
- **Method**: Uses keyword matching and frequency analysis

#### `optimized_cache.py` ⭐ **ACTIVE**
**Purpose**: Advanced caching system for performance
- **Function**: Multi-level caching for embeddings, API calls, and processed results
- **Features**:
  - Memory and disk-based caching
  - Cache invalidation and management
  - Performance monitoring

#### `prompt_templates.py` ⭐ **ACTIVE**
**Purpose**: Standardized prompts for LLM interactions
- **Contents**:
  - Query processing prompts
  - Answer generation templates
  - Strategy-specific prompt variations
- **Integration**: Used by `rag_backend.py` and query processors

### 🌐 **API & Interface**

#### `api_server.py` ⭐ **ACTIVE**
**Purpose**: REST API server for the RAG system
- **Endpoints**:
  - `/query` - Main question-answering endpoint
  - `/health` - System health checks
  - `/feedback` - User feedback submission
- **Features**: FastAPI-based with automatic documentation

#### `streaming_api_server.py` ⭐ **ACTIVE**
**Purpose**: Streaming version of the API server
- **Function**: Real-time streaming responses for better user experience
- **Use Case**: Long-form answers and real-time interaction

#### `rag_app.py` ⭐ **ACTIVE**
**Purpose**: Main application entry point
- **Function**: Coordinates between web interface and RAG backend
- **Features**: Session management, logging, error handling

### 🛠️ **Utilities & Helpers**

#### `utils.py` ⭐ **ACTIVE**
**Purpose**: Shared utility functions
- **Functions**:
  - Text processing and cleaning
  - File I/O operations
  - Common data transformations
  - Error handling helpers

#### `exceptions.py` ⭐ **ACTIVE**
**Purpose**: Custom exception classes for error handling
- **Classes**: RAG-specific exceptions for better error management

#### `llm_logger.py` ⭐ **ACTIVE**
**Purpose**: Specialized logging for LLM interactions
- **Function**: Tracks API calls, token usage, and performance metrics
- **Output**: `rag_app.log`

### 🔄 **Maintenance & Setup**

#### `init_chunks_db.py` ⭐ **ACTIVE**
**Purpose**: Database initialization script
- **Function**: Sets up chunk database and indexes
- **Command**: `python init_chunks_db.py` (run before first use)

#### `clear_all_caches.py` ⭐ **ACTIVE**
**Purpose**: Cache management and cleanup
- **Function**: Clears all caches when changes are not reflected
- **Command**: `python clear_all_caches.py`

#### `database_migration.py` ⭐ **ACTIVE**
**Purpose**: Database schema updates and migrations
- **Function**: Handles database structure changes over time

#### `setup_system.py` ⭐ **ACTIVE**
**Purpose**: Initial system setup and configuration
- **Function**: One-time setup for new installations

### 📁 **Data Files**

#### `contextualized_chunks.json` ⭐ **DATA**
**Purpose**: Main chunk storage file
- **Contents**: All processed document chunks with metadata
- **Access**: Through `ChunkManager` only
- **Size**: Can be very large (use ChunkManager for efficiency)

#### `chunks.db` ⭐ **DATA**
**Purpose**: SQLite database for chunk content and metadata
- **Tables**: Chunks, metadata, performance data
- **Integration**: Used by `ChunkManager` and retrieval components

#### `feedback.db` ⭐ **DATA**
**Purpose**: User feedback and interaction history
- **Usage**: Performance analysis and system improvement

### 📊 **Configuration Files**

#### `.env` ⭐ **CONFIG**
**Purpose**: Environment variables and API keys
- **Contents**:
  ```
  GEMINI_API_KEY=your_api_key_here
  # Other sensitive configuration
  ```
- **Security**: Never commit this file to version control

#### `requirements.txt` ⭐ **CONFIG**
**Purpose**: Python dependencies
- **Usage**: `pip install -r requirements.txt`

#### `text_processing_config.json` ⭐ **CONFIG**
**Purpose**: Text processing parameters and configurations

### 🗂️ **Directories**

#### `Source_Documents/` ⭐ **INPUT**
**Purpose**: Input directory for PDF files to be processed
- **Usage**: Place your financial documents here
- **Process**: `extraction.py` reads from this directory

#### `New folder/` ⭐ **OUTPUT**
**Purpose**: Extraction output directory
- **Structure**: One subdirectory per source document
- **Contents**: JSON files with structured document blocks
- **Usage**: `json_semantic_chunker.py` reads from this directory

#### `embedding_cache/` ⭐ **CACHE**
**Purpose**: Cached embeddings for performance optimization

#### `extraction_logs/` ⭐ **LOGS**
**Purpose**: Detailed logs from document extraction process

---

## 🔄 Complete Workflow

### 1. **Document Ingestion**
```bash
# Place PDFs in Source_Documents/
# Run extraction
python extraction.py
```
- **Input**: PDF files in `Source_Documents/`
- **Process**: Marker extracts text with OCR and LLM enhancement
- **Output**: JSON files in `New folder/` with structured blocks

### 2. **Chunking & Processing**
```bash
# Process extracted JSON into semantic chunks
python json_semantic_chunker.py
```
- **Input**: JSON files from extraction
- **Process**: 
  - Convert HTML tables to clean text
  - Apply semantic similarity grouping
  - Enforce token limits (100-500 tokens)
  - Create overlapping chunks
- **Output**: `json_chunks.json` ready for RAG system

### 3. **System Initialization**
```bash
# Initialize chunk database (one-time setup)
python init_chunks_db.py
```
- **Function**: Sets up databases and indexes
- **Required**: Run once before first query

### 4. **Query Processing**
```bash
# Start the API server
python api_server.py
# OR use the main app
python rag_app.py
```
- **Process Flow**:
  1. **Query Analysis**: `unified_query_processor.py` determines strategy
  2. **Chunk Retrieval**: `progressive_retrieval.py` finds relevant chunks
  3. **Reranking**: `document_reranker.py` refines relevance scores
  4. **Answer Generation**: `rag_backend.py` calls Gemini API
  5. **Response**: Formatted answer with citations

### 5. **Maintenance**
```bash
# Clear caches when needed
python clear_all_caches.py

# Check system health
# Monitor rag_app.log for issues
```

---

## 🚀 Quick Start Guide

### Prerequisites
1. **Install Python 3.8+**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Install Marker**: Follow Marker library installation guide
4. **Set up API key**: Create `.env` file with `GEMINI_API_KEY=your_key`

### First Run
```bash
# 1. Place PDF documents
cp your_financial_documents.pdf Source_Documents/

# 2. Extract text from documents
python extraction.py

# 3. Process into chunks
python json_semantic_chunker.py

# 4. Initialize database
python init_chunks_db.py

# 5. Start the system
python rag_app.py
```

### Query Examples
- **Standard**: "What is the total amount on invoice 2024-001?"
- **Analysis**: "Analyze the quarterly revenue trends in these documents"
- **Aggregation**: "Summarize all tax amounts across all invoices"

---

## ⚡ Performance Optimizations

### 1. **Progressive Retrieval**
- Starts with small chunk sets
- Expands based on quality assessment
- Avoids unnecessary processing

### 2. **Multi-Level Caching**
- Embedding cache for repeated queries
- API response caching
- Chunk processing cache

### 3. **Memory-Mapped Access**
- `ChunkManager` provides efficient large file access
- Lazy loading of chunk data
- Index-based lookups

### 4. **Strategy-Aware Processing**
- Different processing paths for different query types
- Optimized for each use case
- Reduced unnecessary computation

---

## 🔍 Troubleshooting

### Common Issues

#### **"No chunks found"**
- **Cause**: Database not initialized or empty
- **Solution**: Run `python init_chunks_db.py`

#### **"Marker not found"**
- **Cause**: Marker not installed or not in PATH
- **Solution**: Install Marker and ensure it's accessible

#### **"API key error"**
- **Cause**: Missing or invalid Gemini API key
- **Solution**: Check `.env` file and API key validity

#### **"Empty extraction output"**
- **Cause**: PDF parsing issues or empty source documents
- **Solution**: Check `extraction_logs/` for detailed error messages

#### **"Poor answer quality"**
- **Cause**: Inadequate chunk retrieval or poor reranking
- **Solution**: 
  - Clear caches: `python clear_all_caches.py`
  - Check chunk quality in `json_chunks.json`
  - Adjust similarity thresholds in configuration

### Debug Tools
- **Logs**: Check `rag_app.log` for detailed operation logs
- **Cache Clearing**: `python clear_all_caches.py`
- **Database Reset**: Re-run `python init_chunks_db.py`

---

## 🔧 Configuration & Customization

### Key Parameters (in `config.py`)
- **Chunk Sizes**: `min_tokens=100, max_tokens=500, overlap=100`
- **Similarity Threshold**: `similarity_threshold=0.75`
- **Retrieval Limits**: `initial_k=5, max_k=20`
- **API Settings**: Model names, endpoints, timeout values

### Adding New Document Types
1. **Update extraction logic** in `extraction.py` if needed
2. **Modify chunking logic** in `json_semantic_chunker.py` for new formats
3. **Test with sample documents**
4. **Update configuration** as needed

### Performance Tuning
- **Chunk Sizes**: Adjust based on your document types and query patterns
- **Similarity Thresholds**: Lower for more diverse retrieval, higher for precision
- **Cache Settings**: Increase cache sizes for frequently accessed data

---

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: Extend to non-English financial documents
- **Advanced Analytics**: Built-in financial analysis capabilities
- **Real-time Processing**: Live document processing as files are added
- **Advanced Security**: Enhanced authentication and authorization

### Scalability Considerations
- **Database Optimization**: Consider PostgreSQL for larger datasets
- **Distributed Processing**: Implement for high-volume scenarios
- **Cloud Integration**: AWS/Azure deployment options

---

## 📝 Development Notes

### Code Quality
- **Type Hints**: Used throughout for better code clarity
- **Error Handling**: Comprehensive exception handling
- **Logging**: Detailed logging for debugging and monitoring
- **Documentation**: Extensive docstrings and comments

### Testing Strategy
- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing

### Version Control
- **Git Workflow**: Feature branches with pull request reviews
- **Commit Standards**: Clear, descriptive commit messages
- **Documentation**: Keep this file updated with changes

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
1. **Monitor logs** for errors and performance issues
2. **Clear caches** periodically for optimal performance  
3. **Update dependencies** and security patches
4. **Backup databases** regularly
5. **Review and optimize** chunk quality and retrieval performance

### When to Update Components
- **New document types**: Update extraction and chunking logic
- **Performance issues**: Review and optimize retrieval parameters
- **Quality problems**: Adjust chunking strategies and similarity thresholds
- **Scale requirements**: Consider architectural changes for larger datasets

---

*Last Updated: August 16, 2025*
*Project: Financial-RAG v1.0*
*Author: AI Assistant*
