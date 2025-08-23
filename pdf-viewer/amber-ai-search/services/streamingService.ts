import type { DocumentResult, AiResponse, Filters } from '../types';

// Configuration for the backend API
const API_BASE_URL = 'http://localhost:5000';

/**
 * Interface for streaming events
 */
export interface StreamEvent {
  type: 'chunks' | 'answer' | 'complete' | 'error';
  data: {
    documents?: DocumentResult[];
    aiResponse?: AiResponse;
    status?: string;
    method?: string;
    error?: string;
  };
}

/**
 * Streaming search service that handles Server-Sent Events
 */
export class StreamingSearchService {
  private eventSource: EventSource | null = null;
  
  /**
   * Start a streaming search
   */
  async startStreamingSearch(
    query: string,
    filters: Filters,
    onChunks: (documents: DocumentResult[]) => void,
    onAnswer: (aiResponse: AiResponse) => void,
    onComplete: (status: string, method: string) => void,
    onError: (error: string) => void
  ): Promise<void> {
    // Close any existing connection
    this.closeConnection();
    
    try {
      // Use fetch to send the POST request for SSE
      const response = await fetch(`${API_BASE_URL}/search-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
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
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body available for streaming');
      }

      // Read the stream manually since EventSource doesn't support POST
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData: StreamEvent = JSON.parse(line.slice(6)); // Remove "data: " prefix
              this.handleStreamEvent(eventData, onChunks, onAnswer, onComplete, onError);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', parseError, 'Line:', line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming search failed:', error);
      onError(error instanceof Error ? error.message : 'Unknown streaming error');
    }
  }

  /**
   * Handle individual stream events
   */
  private handleStreamEvent(
    event: StreamEvent,
    onChunks: (documents: DocumentResult[]) => void,
    onAnswer: (aiResponse: AiResponse) => void,
    onComplete: (status: string, method: string) => void,
    onError: (error: string) => void
  ): void {
    console.log('📡 Received stream event:', event.type, event.data);
    
    switch (event.type) {
      case 'chunks':
        if (event.data.documents) {
          console.log(`📄 Received ${event.data.documents.length} document chunks`);
          onChunks(event.data.documents);
        }
        break;
        
      case 'answer':
        if (event.data.aiResponse) {
          console.log('🤖 Received AI response');
          onAnswer(event.data.aiResponse);
        }
        break;
        
      case 'complete':
        console.log('✅ Search completed successfully');
        onComplete(
          event.data.status || 'success',
          event.data.method || 'unknown'
        );
        break;
        
      case 'error':
        console.error('❌ Stream error:', event.data.error);
        onError(event.data.error || 'Unknown error');
        break;
        
      default:
        console.warn('Unknown stream event type:', event.type);
    }
  }

  /**
   * Close the streaming connection
   */
  closeConnection(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
}

/**
 * Fallback to original non-streaming service
 */
export const fetchSearchResultsAndAiResponse = async (
  query: string, 
  filters: Filters
): Promise<{ documents: DocumentResult[]; aiResponse: AiResponse }> => {
  console.log(`🔍 Fallback search request for query: "${query}" with filters:`, filters);
  
  try {
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
    console.error('❌ Fallback API call failed:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Failed to connect to backend: ${errorMessage}`);
  }
};
