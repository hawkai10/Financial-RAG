# Financial RAG System Prompts

This document outlines the key prompts used in the Financial RAG system. These prompts are the primary way we interact with the language model (LLM) to perform tasks like query understanding, answer synthesis, and data analysis. All prompts are managed and constructed by the `PromptBuilder` class in `prompt_templates.py`.

## 1. Query Pre-processing Prompts

These prompts are used at the beginning of the RAG pipeline to understand and enhance the user's initial query.

### 1.1. `query_correction`

-   **Purpose**: To correct spelling and grammar in the user's query, ensuring the model receives a clean and understandable question.
-   **When it's used**: This is the very first step in the `rag_query_enhanced` function in `rag_backend.py`.
-   **Template**:
    ```
    Correct the spelling and grammar of this search query for business documents. Keep the corrected query concise and focused. Only fix obvious errors:
    Query to correct: {question}
    Return only the corrected query without additional text.
    ```

### 1.2. `multiquery_generation`

-   **Purpose**: To generate multiple, diverse versions of the user's query. This helps to retrieve a broader and more comprehensive set of relevant documents from the vector database.
-   **When it's used**: Immediately after query correction in `rag_query_enhanced`.
-   **Template**:
    ```
    Generate {num_queries} diverse, rephrased queries that could retrieve relevant information from a database containing invoices, contracts, and financial documents.
    Original query: {question}
    Return each query on a new line without numbering.
    ```

## 2. Main RAG Prompts (Strategy-Based)

The core of the RAG system relies on a "strategy" determined by the `unified_query_processor.py`. This strategy dictates which of the following prompts is used to generate the final answer.

### 2.1. `Standard` Strategy

-   **Purpose**: For direct, fact-based questions (e.g., "What was the total revenue in 2023?"). This prompt instructs the model to find and present specific information from the provided context.
-   **Key Instructions**:
    -   Be direct and comprehensive.
    -   Pay close attention to dates and numbers.
    -   Provide a clear, well-structured response.
-   **Template**:
    ```
    Answer this business question using the provided context. Be direct and comprehensive while staying focused on the query. 

    IMPORTANT: Pay careful attention to temporal references (first year, second year, next months, etc.) and numerical values. When multiple time periods or amounts are mentioned, ensure you answer for the specific period requested in the question.

    Provide a clear, well-structured response.
    Context:
    {context}
    Question: {question}
    Structure your answer clearly and concisely. If multiple time periods are mentioned in the context, be specific about which period you're referencing.
    ```

### 2.2. `Analyse` Strategy

-   **Purpose**: For more complex questions that require analysis, comparison, or identification of trends (e.g., "Analyze the spending trends over the last two years.").
-   **Key Instructions**:
    -   Perform a comprehensive analysis.
    -   Look for patterns, trends, and relationships.
    -   Provide detailed insights.
-   **Template**:
    ```
    Perform a comprehensive analysis of the provided business documents to answer this analytical question. Look for patterns, trends, relationships, and provide detailed insights.
    Context:
    {context}
    Question: {question}
    Provide a well-structured analysis with clear sections for summary, detailed findings, and key insights.
    ```

### 2.3. `Aggregation` Strategy

-   **Purpose**: For questions that require counting or listing information from multiple sources (e.g., "List all invoices from Company X.").
-   **Key Instructions**:
    -   Count, list, or aggregate information precisely.
    -   Provide complete listings.
    -   Start with a summary of totals.
-   **Template**:
    ```
    Count, list, or aggregate information from the provided business documents. Be precise with numbers and provide complete listings. Provide a clear, well-organized response without excessive formatting.
    Context:
    {context}
    Question: {question}
    Provide your response in a clear format:
    - Start with a summary of totals or key findings
    - Follow with detailed breakdown if needed
    - List relevant items clearly and concisely
    ```

## 3. The `PromptBuilder` Class

The `PromptBuilder` class in `prompt_templates.py` is the engine that constructs these prompts. It:
-   Selects the correct template based on the `strategy`.
-   Injects the user's `question` and the retrieved `context_chunks`.
-   Handles smart truncation of the context to ensure the final prompt does not exceed the LLM's token limit.

Understanding this class is key to modifying how the system communicates with the LLM.
