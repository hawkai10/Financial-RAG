# Financial-RAG: Files Safe to Remove

This document lists all files that are no longer required by the core Financial-RAG model and can be safely removed to clean up the project.

## 🗂️ **SAFE TO REMOVE - Development & Testing Files**

### **Debug Files**
```
debug_classification.py
debug_formatting.py  
debug_full_agent.py
debug_mini_agent_patterns.py
debug_mini_agent_patterns_new.py
debug_rag_processing.py
debug_retrieval.py
diagnostic_test.py
```

### **Test Files**
```
test_formatting.py
test_hybrid_system.py
test_mini_agent.py
test_mini_agent_direct.py
```

### **Log Files**
```
hybrid_test_detailed_log_20250806_221126.txt
hybrid_test_detailed_log_20250806_222330.txt
hybrid_test_detailed_log_20250807_000332.txt
hybrid_test_detailed_log_20250807_012149.txt
hybrid_test_detailed_log_20250807_013527.txt
hybrid_test_detailed_log_20250807_013958.txt
hybrid_test_results.json
rag_app.log
log
log.md
```

### **Backup & Legacy Files**
```
rag_backend_backup.py
contextualized_chunks_backup.json
old/ (entire directory)
```

### **Analysis & Optimization Files**
```
aggregation_optimizer.py
analyze_chunk_enhancements.py
check_chunks.py
optimize_chunks_storage.py
```

### **Cache Management Files**
```
clear_all_caches.py
optimized_cache.py
```

### **Alternative/Experimental Components**
```
keyword_aggregation_retriever.py
hierarchical_processor.py
pipeline_orchestrator.py
```

### **Documentation Files (Optional)**
```
COMPREHENSIVE_PROJECT_DOCUMENTATION.md
IMPLEMENTATION_LOG.md
MAINTAINERS.md
PROMPTS.md
```

### **Database Migration**
```
database_migration.py
```

### **External Directories**
```
amber-ai-search/ (entire directory)
docling/ (entire directory)
extraction_logs/ (entire directory)
```

---

## 🔒 **CORE FILES - DO NOT REMOVE**

### **Essential Core Components**
```
rag_backend.py           # Main RAG orchestration
mini_agent.py           # Pattern-based extraction agent
full_agent.py           # Complex reasoning agent
unified_query_processor.py  # Query classification
progressive_retrieval.py    # Progressive retrieval system
document_reranker.py        # Cross-encoder reranking
chunk_manager.py           # Chunk data management
```

### **Configuration & Setup**
```
config.py               # Configuration settings
init_chunks_db.py       # Database initialization
embed_chunks_txtai.py   # Embedding generation
utils.py               # Utility functions
prompt_templates.py    # LLM prompt templates
exceptions.py          # Custom exceptions
```

### **Data & Storage**
```
contextualized_chunks.json  # Main chunk data
chunks.db                   # SQLite chunk database
chunks.index.json          # Chunk index for fast access
feedback.db                 # User feedback database
source_manifest.json       # Source document manifest
```

### **Applications & Interfaces**
```
api_server.py           # REST API server
streaming_api_server.py # Streaming API server
rag_app.py             # Main application
pdf_viewer.py          # PDF viewing functionality
```

### **Support Components**
```
feedback_database.py   # Feedback system
llm_logger.py         # LLM logging
setup_system.py      # System setup
```

### **Generated Directories**
```
business-docs-index/   # txtai embeddings index
embedding_cache/       # Embedding cache
Source_Documents/      # Original source documents
```

### **Development Environment**
```
requirements.txt       # Python dependencies
setup.py              # Package setup
Dockerfile            # Container configuration
venv/                 # Virtual environment
__pycache__/          # Python cache
```

### **Version Control & Configuration**
```
.git/                 # Git repository
.github/              # GitHub configuration
.gitignore           # Git ignore rules
.pre-commit-config.yaml # Pre-commit hooks
.env                 # Environment variables
```

### **Project Documentation**
```
README.md            # Main documentation
LICENSE              # License file
```

---

## 📊 **Summary**

**Total Files in Project**: ~85 files + directories
**Safe to Remove**: ~35 files + 3 directories
**Core Files to Keep**: ~50 files + directories

**Disk Space Savings**: Approximately 40-50% reduction in project size

---

## 🚀 **Cleanup Commands**

To remove all non-essential files, you can run:

```bash
# Remove debug files
rm debug_*.py diagnostic_test.py

# Remove test files  
rm test_*.py

# Remove log files
rm *.log *.txt log.md log hybrid_test_*.json

# Remove backup files
rm *_backup.py contextualized_chunks_backup.json

# Remove analysis files
rm aggregation_optimizer.py analyze_chunk_enhancements.py check_chunks.py optimize_chunks_storage.py

# Remove cache management
rm clear_all_caches.py optimized_cache.py

# Remove experimental files
rm keyword_aggregation_retriever.py hierarchical_processor.py pipeline_orchestrator.py database_migration.py

# Remove external directories
rm -rf amber-ai-search/ docling/ extraction_logs/ old/

# Remove optional documentation
rm COMPREHENSIVE_PROJECT_DOCUMENTATION.md IMPLEMENTATION_LOG.md MAINTAINERS.md PROMPTS.md
```

**⚠️ Warning**: Always backup your project before running cleanup commands!
