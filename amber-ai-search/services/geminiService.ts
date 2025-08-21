import type { DocumentResult, AiResponse, Filters } from '../types';

// Configuration for the backend API
const API_BASE_URL = 'http://localhost:5000';
const USE_REAL_API = true; // Always use real API

/**
 * Parses a DD.MM.YYYY date string into a Date object.
 * @param dateString The date string to parse.
 * @returns A Date object or null if parsing fails.
 */
const parseDate = (dateString: string): Date | null => {
    try {
        const [day, month, year] = dateString.split('.').map(Number);
        if (!day || !month || !year) return null;
        const date = new Date(year, month - 1, day);
        if (isNaN(date.getTime())) return null;
        return date;
    } catch (e) {
        return null;
    }
};

// Example queries are removed per requirements; no frontend fallback content.

/**
 * Calls the real Python RAG backend API
 * @param query The user's search query
 * @param filters The selected filters to apply to the search
 * @returns A promise that resolves to the search results and AI response
 */
const callRealAPI = async (query: string, filters: Filters): Promise<{ documents: DocumentResult[]; aiResponse: AiResponse }> => {
    try {
        console.log(`🔍 Calling real API: ${API_BASE_URL}/search`);
        
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                filters: {
                    fileType: filters.fileType,
                    timeRange: filters.timeRange,
                    dataSource: filters.dataSource
                }
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API request failed: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        
        // Transform API response to match UI expectations
        const documents: DocumentResult[] = data.documents.map((doc: any) => ({
            id: doc.id,
            sourceType: doc.sourceType,
            sourcePath: doc.sourcePath,
            sourceIcon: null, // Will be set by UI based on sourceType
            docTypeIcon: null, // Will be set by UI based on fileType
            fileType: doc.fileType,
            title: doc.title,
            date: doc.date,
            snippet: doc.snippet,
            author: doc.author,
            missingInfo: doc.missingInfo,
            mustInclude: doc.mustInclude
        }));

        return {
            documents,
            aiResponse: data.aiResponse
        };

    } catch (error) {
        console.error('❌ Real API call failed:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Failed to connect to backend: ${errorMessage}`);
    }
};

/**
 * Main function that handles both real API and mock data
 * In a real application, this function makes an HTTP request to your backend.
 * The backend then:
 * 1.  Use the user's query and filters to search a vector database or other document index (Retrieval).
 * 2.  Collect the most relevant document chunks.
 * 3.  Construct a detailed prompt for the Gemini API, including the user's query and the retrieved document context (Augmentation).
 * 4.  Call the Gemini API to get a generated answer based on the provided context.
 * 5.  Return both the retrieved documents and the AI-generated response to the frontend.
 *
 * @param query The user's search query.
 * @param filters The selected filters to apply to the search.
 * @returns A promise that resolves to the search results and the AI response.
 */
export const fetchSearchResultsAndAiResponse = async (query: string, filters: Filters): Promise<{ documents: DocumentResult[]; aiResponse: AiResponse }> => {
  console.log(`🔍 Search request for query: "${query}" with filters:`, filters);

    // Always use real API, no mock fallback
    return await callRealAPI(query, filters);
};

/**
 * Fetches recent documents from the backend
 * @returns A promise that resolves to an array of recent document objects
 */
const fetchRecentDocuments = async (): Promise<any[]> => {
    try {
        const response = await fetch(`${API_BASE_URL}/recent-documents`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.status === 'success' && Array.isArray(data.documents)) {
            return data.documents;
        } else {
            throw new Error('Invalid response format');
        }
    } catch (error) {
        console.error('Error fetching recent documents:', error);
        // Fallback to empty array
        return [];
    }
};

export { fetchRecentDocuments };
