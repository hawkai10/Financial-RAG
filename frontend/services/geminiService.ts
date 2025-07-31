import { MOCK_SEARCH_RESULTS, MOCK_AI_RESPONSE } from '../constants';
import type { DocumentResult, AiResponse, Filters } from '../types';

// Configuration for the backend API
const API_BASE_URL = 'http://localhost:5000';
const USE_REAL_API = true; // Set to false to use mock data

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

/**
 * Fetches example queries from the backend
 * @returns A promise that resolves to an array of example query strings
 */
const fetchExampleQueries = async (): Promise<string[]> => {
    if (!USE_REAL_API) {
        // Return mock example queries
        return [
            "What standards and certificates are required for exporting the CNC milling machine to the USA?",
            "What are the safety protocols mentioned in the documents?",
            "What technical specifications are detailed in the engineering documents?"
        ];
    }

    try {
        const response = await fetch(`${API_BASE_URL}/example-queries`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.status === 'success' && Array.isArray(data.queries)) {
            return data.queries;
        } else {
            throw new Error('Invalid response format');
        }
    } catch (error) {
        console.error('Error fetching example queries:', error);
        // Fallback to default queries
        return [
            "What are the main topics covered in the documents?",
            "Can you summarize the key information available?",
            "What important details should I know from these documents?"
        ];
    }
};

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

  // Use real API if enabled and available
  if (USE_REAL_API) {
    try {
      return await callRealAPI(query, filters);
    } catch (error) {
      console.warn('⚠️ Real API failed, falling back to mock data:', error);
      // Fall through to mock data
    }
  }

  // Fallback to mock data (original implementation)
  console.log(`📋 Using mock data for query: "${query}"`);
  
  // Simulate network delay to mimic a real API call
  await new Promise(resolve => setTimeout(resolve, 800));

  let filteredDocs = MOCK_SEARCH_RESULTS;

  // Apply fileType filter (multi-select)
  if (filters.fileType.length > 0) {
    filteredDocs = filteredDocs.filter(doc => filters.fileType.includes(doc.fileType));
  }
  
  // Apply dataSource filter (multi-select)
  if (filters.dataSource.length > 0) {
    filteredDocs = filteredDocs.filter(doc => filters.dataSource.includes(doc.sourceType));
  }

  // Apply timeRange filter
  if (filters.timeRange.type !== 'all') {
    const now = new Date();
    now.setHours(23, 59, 59, 999); // End of today

    let startDate: Date | null = null;

    switch (filters.timeRange.type) {
        case '3days':
            startDate = new Date(now);
            startDate.setDate(now.getDate() - 3);
            break;
        case 'week':
            startDate = new Date(now);
            startDate.setDate(now.getDate() - 7);
            break;
        case 'month':
            startDate = new Date(now);
            startDate.setMonth(now.getMonth() - 1);
            break;
        case '3months':
            startDate = new Date(now);
            startDate.setMonth(now.getMonth() - 3);
            break;
        case 'year':
            startDate = new Date(now);
            startDate.setFullYear(now.getFullYear() - 1);
            break;
        case '5years':
            startDate = new Date(now);
            startDate.setFullYear(now.getFullYear() - 5);
            break;
        case 'custom':
            if (filters.timeRange.startDate) {
                startDate = filters.timeRange.startDate;
            }
            break;
    }
    
    if (startDate) {
        startDate.setHours(0, 0, 0, 0); // Start of the day
    }

    const endDate = filters.timeRange.type === 'custom' && filters.timeRange.endDate ? filters.timeRange.endDate : now;
    if (endDate) {
        endDate.setHours(23, 59, 59, 999); // End of the day
    }


    filteredDocs = filteredDocs.filter(doc => {
      const docDate = parseDate(doc.date);
      if (!docDate) return false;

      const isAfterStart = startDate ? docDate >= startDate : true;
      const isBeforeEnd = endDate ? docDate <= endDate : true;
      
      return isAfterStart && isBeforeEnd;
    });
  }

  // Filter AI response based on filtered documents
  const filteredDocIds = new Set(filteredDocs.map(d => d.id));
  const filteredAiItems = MOCK_AI_RESPONSE.items
    .map(item => ({
      ...item,
      references: item.references.filter(ref => filteredDocIds.has(ref.docId)),
    }))
    .filter(item => item.references.length > 0);
  
  const filteredAiResponse: AiResponse = {
    summary: filteredAiItems.length > 0 ? MOCK_AI_RESPONSE.summary : "No relevant information found for the selected filters.",
    items: filteredAiItems,
  };

  return {
    documents: filteredDocs,
    aiResponse: filteredAiResponse,
  };
};

/**
 * Fetches recent documents from the backend
 * @returns A promise that resolves to an array of recent document objects
 */
const fetchRecentDocuments = async (): Promise<any[]> => {
    if (!USE_REAL_API) {
        // Return mock recent documents
        return [
            {
                id: 'doc_1',
                title: 'Export_CNC_Machine_US.docx',
                fileType: 'word',
                sourcePath: 'C:\\Documents\\Export_CNC_Machine_US.docx',
                lastAccessed: '2025-07-28T10:30:00Z',
                sourceType: 'Windows Shares'
            },
            {
                id: 'doc_2',
                title: 'Installation_Handbook_Manual.pdf',
                fileType: 'pdf',
                sourcePath: 'C:\\Documents\\Installation_Handbook_Manual.pdf',
                lastAccessed: '2025-07-28T09:15:00Z',
                sourceType: 'Windows Shares'
            }
        ];
    }

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

export { fetchExampleQueries, fetchRecentDocuments };
