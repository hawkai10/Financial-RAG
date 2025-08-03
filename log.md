venv) PS C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG> python test_new_system.py
C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\venv\lib\site-packages\threadpoolctl.py:1226: RuntimeWarning:
Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
the same time. Both libraries are known to be incompatible and this
can cause random crashes or deadlocks on Linux when loaded in the
same Python program.
Using threadpoolctl may cause crashes or deadlocks. For more
information and possible workarounds, please see
    https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md

  warnings.warn(msg, RuntimeWarning)
2025-08-02 01:58:02,039 - utils - INFO - [RAG] Initializing optimizations...
2025-08-02 01:58:02,039 - utils - INFO -    - Chunk Cache: 500 items max
2025-08-02 01:58:02,040 - utils - INFO -    - Embedding Cache: 1000 items max in memory
2025-08-02 01:58:02,040 - utils - INFO -    - Connection Pool: 10 connections
2025-08-02 01:58:02,040 - utils - INFO - Enhanced RAG Backend with Optimizations loaded successfully
2025-08-02 01:58:02,040 - utils - INFO - Features enabled: Progressive=False, Sampling=False, Hybrid=True
2025-08-02 01:58:02,040 - utils - INFO - Chunk limits optimized: {'Standard': (2, 4), 'Analyse': (5, 8), 'Aggregation': (8, 15)}
2025-08-02 01:58:02,040 - utils - INFO - Optimization features: Chunk Caching [SUCCESS], Embedding Caching [SUCCESS], Connection Pooling [SUCCESS]
🚀 Testing New System: Full Retrieval → Score Filtering → Token-Based Processing
================================================================================
🧪 Testing New Full Retrieval + Score Filtering System

🔄 Initializing embeddings...
✅ Embeddings loaded from business-docs-index

================================================================================
🔍 Query 1: List all the parties who have been issued the credit note by Bhartiya Enterprises
⏳ Processing with new system...
2025-08-02 01:58:05,580 - utils - INFO - Cache miss, computing result for: List all the parties who have been issued the cred...
2025-08-02 01:58:05,581 - utils - INFO - Starting enhanced RAG query: List all the parties who have been issued the credit note by Bhartiya Enterprises...
2025-08-02 01:58:06,706 - utils - INFO - Unified processing: Aggregation (confidence: 0.95)
2025-08-02 01:58:06,706 - utils - INFO - Unified processing: Aggregation -> Aggregation (confidence: 0.950)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f50d' in position 41: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1285, in rag_query_enhanced
    logger.info("🔍 Performing full retrieval - getting ALL matching chunks")
Message: '🔍 Performing full retrieval - getting ALL matching chunks'
Arguments: ()
2025-08-02 01:58:06,706 - utils - INFO - 🔍 Performing full retrieval - getting ALL matching chunks
2025-08-02 01:58:06,729 - utils - INFO - Loaded 57 chunk text for BM25
2025-08-02 01:58:06,729 - utils - INFO - Initializing BM25 sparse retriever...
2025-08-02 01:58:06,734 - utils - INFO - BM25 initialized with 57 documents
2025-08-02 01:58:06,735 - utils - INFO - Hybrid search for query 1: List all parties who have received credit notes from Bhartiya Enterprises
2025-08-02 01:58:06,735 - utils - INFO - Performing dense retrieval for: List all parties who have received credit notes fr...
C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\venv\lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.
  return forward_call(*args, **kwargs)
2025-08-02 01:58:07,354 - utils - INFO - Dense retrieval found 57 results
2025-08-02 01:58:07,354 - utils - INFO - Performing sparse retrieval (BM25)...
2025-08-02 01:58:07,355 - utils - INFO - Sparse retrieval found 34 results
2025-08-02 01:58:07,355 - utils - INFO - Hybrid retrieval returned 91 combined results
2025-08-02 01:58:07,356 - utils - INFO - Hybrid search for query 2: Who were the recipients of credit notes from Bhartiya Enterprises?
2025-08-02 01:58:07,356 - utils - INFO - Performing dense retrieval for: Who were the recipients of credit notes from Bhart...
2025-08-02 01:58:07,426 - utils - INFO - Dense retrieval found 57 results
2025-08-02 01:58:07,426 - utils - INFO - Performing sparse retrieval (BM25)...
2025-08-02 01:58:07,427 - utils - INFO - Sparse retrieval found 47 results
2025-08-02 01:58:07,427 - utils - INFO - Hybrid retrieval returned 100 combined results
2025-08-02 01:58:07,427 - utils - INFO - Hybrid search for query 3: Generate a list of parties issued credit notes by Bhartiya Enterprises.
2025-08-02 01:58:07,428 - utils - INFO - Performing dense retrieval for: Generate a list of parties issued credit notes by ...
2025-08-02 01:58:07,496 - utils - INFO - Dense retrieval found 57 results
2025-08-02 01:58:07,496 - utils - INFO - Performing sparse retrieval (BM25)...
2025-08-02 01:58:07,497 - utils - INFO - Sparse retrieval found 47 results
2025-08-02 01:58:07,498 - utils - INFO - Hybrid retrieval returned 100 combined results
2025-08-02 01:58:07,498 - utils - INFO - Hybrid retrieval returned 50 results
2025-08-02 01:58:07,500 - utils - INFO - [CACHE] Source file changed, clearing chunk cache
2025-08-02 01:58:07,503 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,503 - utils - INFO - [CACHE] Source file changed, clearing chunk cache
2025-08-02 01:58:07,503 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 19-20.pdf_text_1, loading from disk
2025-08-02 01:58:07,505 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,506 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn22-23.pdf_text_1, loading from disk
2025-08-02 01:58:07,507 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,508 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn23.-24.pdf_text_1, loading from disk
2025-08-02 01:58:07,509 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,509 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 22.23.pdf_text_1, loading from disk
2025-08-02 01:58:07,511 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,511 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 20-21.pdf_text_1, loading from disk
2025-08-02 01:58:07,514 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,515 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 19.20.pdf_text_1, loading from disk
2025-08-02 01:58:07,518 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,518 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn23.24.pdf_text_1, loading from disk
2025-08-02 01:58:07,521 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,521 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn21.22.pdf_text_1, loading from disk
2025-08-02 01:58:07,523 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,523 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 20.21.pdf_text_1, loading from disk
2025-08-02 01:58:07,525 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,525 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 21-22.pdf_text_1, loading from disk
2025-08-02 01:58:07,527 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,527 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn21.22.pdf_text_2, loading from disk
2025-08-02 01:58:07,529 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,529 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_table_3, loading from disk
2025-08-02 01:58:07,531 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,531 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_table_2, loading from disk
2025-08-02 01:58:07,533 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,534 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_3, loading from disk
2025-08-02 01:58:07,536 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,536 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_text_7, loading from disk
2025-08-02 01:58:07,538 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,538 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_text_3, loading from disk
2025-08-02 01:58:07,540 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,540 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_table_1, loading from disk
2025-08-02 01:58:07,542 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,542 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_4, loading from disk
2025-08-02 01:58:07,543 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,543 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_1, loading from disk
2025-08-02 01:58:07,545 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,545 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\rent_agreement.pdf_text_1, loading from disk
2025-08-02 01:58:07,548 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,548 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_9, loading from disk
2025-08-02 01:58:07,550 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,550 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_7, loading from disk
2025-08-02 01:58:07,552 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,553 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_2, loading from disk
2025-08-02 01:58:07,554 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,555 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 19.20.pdf_table_1, loading from disk
2025-08-02 01:58:07,556 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,556 - utils - INFO - [CACHE] Cache MISS for chunk 46, loading from disk
2025-08-02 01:58:07,558 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,558 - utils - INFO - [CACHE] Cache MISS for chunk 45, loading from disk
2025-08-02 01:58:07,560 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,560 - utils - INFO - [CACHE] Cache MISS for chunk 8, loading from disk
2025-08-02 01:58:07,561 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,561 - utils - INFO - [CACHE] Cache MISS for chunk 2, loading from disk
2025-08-02 01:58:07,563 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,564 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_table_1, loading from disk
2025-08-02 01:58:07,566 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,567 - utils - INFO - [CACHE] Cache MISS for chunk 6, loading from disk
2025-08-02 01:58:07,569 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,569 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn23.24.pdf_table_1, loading from disk
2025-08-02 01:58:07,570 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,570 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_5, loading from disk
2025-08-02 01:58:07,572 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,572 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 20-21.pdf_table_1, loading from disk
2025-08-02 01:58:07,575 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,575 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn21.22.pdf_table_1, loading from disk
2025-08-02 01:58:07,577 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,577 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_6, loading from disk
2025-08-02 01:58:07,580 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,580 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn23.-24.pdf_table_1, loading from disk
2025-08-02 01:58:07,585 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,585 - utils - INFO - [CACHE] Cache MISS for chunk 42, loading from disk
2025-08-02 01:58:07,587 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,588 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 22.23.pdf_table_1, loading from disk
2025-08-02 01:58:07,589 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,589 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_17, loading from disk
2025-08-02 01:58:07,590 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,591 - utils - INFO - [CACHE] Cache MISS for chunk 4, loading from disk
2025-08-02 01:58:07,592 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,593 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn22-23.pdf_table_1, loading from disk
2025-08-02 01:58:07,594 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,594 - utils - INFO - [CACHE] Cache MISS for chunk 5, loading from disk
2025-08-02 01:58:07,598 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,598 - utils - INFO - [CACHE] Cache MISS for chunk 29, loading from disk
2025-08-02 01:58:07,601 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,601 - utils - INFO - [CACHE] Cache MISS for chunk 35, loading from disk
2025-08-02 01:58:07,603 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,604 - utils - INFO - [CACHE] Cache MISS for chunk 51, loading from disk
2025-08-02 01:58:07,605 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,606 - utils - INFO - [CACHE] Cache MISS for chunk 38, loading from disk
2025-08-02 01:58:07,607 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,607 - utils - INFO - [CACHE] Cache MISS for chunk 32, loading from disk
2025-08-02 01:58:07,609 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,609 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 20.21.pdf_table_1, loading from disk
2025-08-02 01:58:07,610 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,610 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\cn 21-22.pdf_table_1, loading from disk
2025-08-02 01:58:07,612 - utils - ERROR - Database lookup with pooling failed: no such table: chunks
2025-08-02 01:58:07,612 - utils - INFO - [CACHE] Cache MISS for chunk C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\Source_Documents\Performance Analysis of a Deletions-Based Investment Strategy in the Indian Equity Market.pdf_text_15, loading from disk
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 41: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1296, in rag_query_enhanced
    logger.info(f"📊 Retrieved {len(chunks)} chunks before reranking and filtering")
Message: '📊 Retrieved 37 chunks before reranking and filtering'
Arguments: ()
2025-08-02 01:58:07,614 - utils - INFO - 📊 Retrieved 37 chunks before reranking and filtering
2025-08-02 01:58:10,545 - sentence_transformers.cross_encoder.CrossEncoder - INFO - Use pytorch device: cpu
2025-08-02 01:58:11,103 - utils - INFO - Cross-encoder reranker loaded successfully
2025-08-02 01:58:11,103 - utils - INFO - Skipping reranking for aggregation query to preserve all results
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3af' in position 41: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1312, in rag_query_enhanced
    logger.info(f"🎯 Score filtering: {len(chunks)} → {len(filtered_chunks)} chunks (threshold: {score_threshold})")
Message: '🎯 Score filtering: 37 → 24 chunks (threshold: 0.4)'
Arguments: ()
2025-08-02 01:58:11,103 - utils - INFO - 🎯 Score filtering: 37 → 24 chunks (threshold: 0.4)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3af' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 120, in process_large_query
    self.set_query_strategy(query_type)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 72, in set_query_strategy
    logger.info(f"🎯 Hierarchical processor set for strategy: {strategy}")
Message: '🎯 Hierarchical processor set for strategy: Aggregation'
Arguments: ()
2025-08-02 01:58:11,109 - hierarchical_processor - INFO - 🎯 Hierarchical processor set for strategy: Aggregation
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f504' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 123, in process_large_query
    should_process, decision_info = self.should_use_hierarchical_processing(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 96, in should_use_hierarchical_processing
    logger.info(f"🔄 Hierarchical processing required: {decision_info['reason']}")
Message: '🔄 Hierarchical processing required: Token count (8081) exceeds limit (3500)'
Arguments: ()
2025-08-02 01:58:11,118 - hierarchical_processor - INFO - 🔄 Hierarchical processing required: Token count (8081) exceeds limit (3500)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 130, in process_large_query
    logger.info(f"🚀 Starting universal hierarchical processing:")
Message: '🚀 Starting universal hierarchical processing:'
Arguments: ()
2025-08-02 01:58:11,120 - hierarchical_processor - INFO - 🚀 Starting universal hierarchical processing:
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 131, in process_large_query
    logger.info(f"   📊 Chunks: {len(chunks)}")
Message: '   📊 Chunks: 24'
Arguments: ()
2025-08-02 01:58:11,122 - hierarchical_processor - INFO -    📊 Chunks: 24
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3af' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 132, in process_large_query
    logger.info(f"   🎯 Strategy: {query_type}")
Message: '   🎯 Strategy: Aggregation'
Arguments: ()
2025-08-02 01:58:11,124 - hierarchical_processor - INFO -    🎯 Strategy: Aggregation
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f522' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 133, in process_large_query
    logger.info(f"   🔢 Estimated tokens: {decision_info['total_tokens']}")
Message: '   🔢 Estimated tokens: 8081'
Arguments: ()
2025-08-02 01:58:11,126 - hierarchical_processor - INFO -    🔢 Estimated tokens: 8081
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f504' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 218, in _create_smart_batches
    logger.info(f"🔄 Creating token-based batches for {len(chunks)} chunks (strategy: {self.query_strategy})")
Message: '🔄 Creating token-based batches for 24 chunks (strategy: Aggregation)'
Arguments: ()
2025-08-02 01:58:11,128 - hierarchical_processor - INFO - 🔄 Creating token-based batches for 24 chunks (strategy: Aggregation)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 225, in _create_smart_batches
    logger.info(f"📊 Token budget: {available_tokens} per batch (limit: {effective_limit}, prompt: {prompt_base_tokens}, reserved: {self.config.reserved_tokens})")
Message: '📊 Token budget: 2607 per batch (limit: 2700, prompt: 93, reserved: 800)'
Arguments: ()
2025-08-02 01:58:11,131 - hierarchical_processor - INFO - 📊 Token budget: 2607 per batch (limit: 2700, prompt: 93, reserved: 800)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 243, in _create_smart_batches
    logger.info(f"✅ Batch {len(batches)} created: {len(current_batch)} chunks, {current_tokens} tokens")
Message: '✅ Batch 1 created: 9 chunks, 2424 tokens'
Arguments: ()
2025-08-02 01:58:11,135 - hierarchical_processor - INFO - ✅ Batch 1 created: 9 chunks, 2424 tokens
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 243, in _create_smart_batches
    logger.info(f"✅ Batch {len(batches)} created: {len(current_batch)} chunks, {current_tokens} tokens")
Message: '✅ Batch 2 created: 9 chunks, 2571 tokens'
Arguments: ()
2025-08-02 01:58:11,138 - hierarchical_processor - INFO - ✅ Batch 2 created: 9 chunks, 2571 tokens
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 261, in _create_smart_batches
    logger.info(f"✅ Final batch {len(batches)} created: {len(current_batch)} chunks, {current_tokens} tokens")
Message: '✅ Final batch 3 created: 6 chunks, 2193 tokens'
Arguments: ()
2025-08-02 01:58:11,141 - hierarchical_processor - INFO - ✅ Final batch 3 created: 6 chunks, 2193 tokens
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3af' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 270, in _create_smart_batches
    logger.info(f"🎯 Batch creation complete:")
Message: '🎯 Batch creation complete:'
Arguments: ()
2025-08-02 01:58:11,150 - hierarchical_processor - INFO - 🎯 Batch creation complete:
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4e6' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 271, in _create_smart_batches
    logger.info(f"   📦 Total batches: {len(batches)}")
Message: '   📦 Total batches: 3'
Arguments: ()
2025-08-02 01:58:11,152 - hierarchical_processor - INFO -    📦 Total batches: 3
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4c4' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 272, in _create_smart_batches
    logger.info(f"   📄 Total chunks: {total_chunks}")
Message: '   📄 Total chunks: 24'
Arguments: ()
2025-08-02 01:58:11,154 - hierarchical_processor - INFO -    📄 Total chunks: 24
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f522' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 273, in _create_smart_batches
    logger.info(f"   🔢 Estimated tokens: {total_estimated_tokens}")
Message: '   🔢 Estimated tokens: 7188'
Arguments: ()
2025-08-02 01:58:11,156 - hierarchical_processor - INFO -    🔢 Estimated tokens: 7188
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 274, in _create_smart_batches
    logger.info(f"   📊 Avg chunks per batch: {total_chunks / len(batches):.1f}")
Message: '   📊 Avg chunks per batch: 8.0'
Arguments: ()
2025-08-02 01:58:11,156 - hierarchical_processor - INFO -    📊 Avg chunks per batch: 8.0
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3af' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 136, in process_large_query
    batches = self._create_smart_batches(chunks, question)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 275, in _create_smart_batches
    logger.info(f"   🎯 Avg tokens per batch: {total_estimated_tokens / len(batches):.1f}")
Message: '   🎯 Avg tokens per batch: 2396.0'
Arguments: ()
2025-08-02 01:58:11,157 - hierarchical_processor - INFO -    🎯 Avg tokens per batch: 2396.0
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f504' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 148, in process_large_query
    logger.info(f"🔄 Processing {len(batches)} batches in parallel")
Message: '🔄 Processing 3 batches in parallel'
Arguments: ()
2025-08-02 01:58:11,158 - hierarchical_processor - INFO - 🔄 Processing 3 batches in parallel
2025-08-02 01:58:11,158 - hierarchical_processor - INFO - Processing wave 1/2 (2 batches)
2025-08-02 01:58:11,904 - hierarchical_processor - INFO - Batch 0 completed: 9 chunks
2025-08-02 01:58:11,974 - hierarchical_processor - INFO - Batch 1 completed: 9 chunks
2025-08-02 01:58:11,975 - hierarchical_processor - INFO - Processing wave 2/2 (1 batches)
2025-08-02 01:58:12,738 - hierarchical_processor - INFO - Batch 2 completed: 6 chunks
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 58: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 175, in process_large_query
    logger.info(f"✅ Hierarchical processing complete:")
Message: '✅ Hierarchical processing complete:'
Arguments: ()
2025-08-02 01:58:12,739 - hierarchical_processor - INFO - ✅ Hierarchical processing complete:
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode characters in position 61-62: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 176, in process_large_query
    logger.info(f"   ⏱️  Total time: {processing_time:.2f}s")
Message: '   ⏱️  Total time: 1.63s'
Arguments: ()
2025-08-02 01:58:12,742 - hierarchical_processor - INFO -    ⏱️  Total time: 1.63s
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 177, in process_large_query
    logger.info(f"   ✅ Success rate: {len(successful_results)}/{len(batch_results)} batches")
Message: '   ✅ Success rate: 3/3 batches'
Arguments: ()
2025-08-02 01:58:12,757 - hierarchical_processor - INFO -    ✅ Success rate: 3/3 batches
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4ca' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 178, in process_large_query
    logger.info(f"   📊 Completeness: {completeness['success_rate']:.1%}")
Message: '   📊 Completeness: 100.0%'
Arguments: ()
2025-08-02 01:58:12,759 - hierarchical_processor - INFO -    📊 Completeness: 100.0%
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f50d' in position 61: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1358, in rag_query_enhanced
    hierarchical_result = processor.process_large_query(corrected_query, chunks, query_strategy)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\hierarchical_processor.py", line 179, in process_large_query
    logger.info(f"   🔍 Conflicts detected: {'Yes' if conflicts['has_conflicts'] else 'No'}")
Message: '   🔍 Conflicts detected: No'
Arguments: ()
2025-08-02 01:58:12,763 - hierarchical_processor - INFO -    🔍 Conflicts detected: No
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\arvin\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f504' in position 41: character maps to <undefined>
Call stack:
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 95, in <module>
    success = test_new_retrieval_system()
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\test_new_system.py", line 44, in test_new_retrieval_system
    result = rag_query_enhanced(test_query, embeddings)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 332, in wrapper
    result = func(question, *args, **kwargs)
  File "C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG\rag_backend.py", line 1364, in rag_query_enhanced
    logger.info(f"🔄 Universal token-based hierarchical processing ({query_strategy}): "
Message: '🔄 Universal token-based hierarchical processing (Aggregation): 3 batches, 1.63s, avg 2694 tokens/batch, est. tokens: 7663, success rate: 100.0%, conflicts: none'
Arguments: ()
2025-08-02 01:58:12,765 - utils - INFO - 🔄 Universal token-based hierarchical processing (Aggregation): 3 batches, 1.63s, avg 2694 tokens/batch, est. tokens: 7663, success rate: 100.0%, conflicts: none
2025-08-02 01:58:12,768 - utils - INFO - Enhanced RAG completed in 7.19s - Strategy: Aggregation, Chunks: 24, Savings: 0.0%

✅ Query 1 processed successfully!
📄 Processing method: hierarchical-universal-aggregation
📊 Retrieval stats:
   Initial chunks: unknown
   After filtering: unknown
   Score threshold: 0.4
📄 Used standard processing (token count within limit)

📝 Answer preview: No relevant information found in the documents....

🎉 All tests passed!
✅ Progressive retrieval disabled
✅ Full retrieval with score filtering (≥0.4) working
✅ Token-based hierarchical processing working
✅ Sparse score weight updated to 40%
2025-08-02 01:58:12,840 - utils - INFO - [DB] All database connections closed
2025-08-02 01:58:12,842 - utils - INFO - [CLEANUP] Optimization cleanup completed
(venv) PS C:\Users\arvin\OneDrive\Desktop\trial\Final FInancial RAG> python test_new_system.py