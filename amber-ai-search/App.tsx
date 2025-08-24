import React, { useState, useRef, createRef, useCallback, useEffect } from 'react';
import Header from './components/Header';
import LeftPane from './components/LeftPane';
import RightPane from './components/RightPane';
import HomeScreen from './components/HomeScreen';
import PdfViewer from './components/PdfViewer';
import { fetchSearchResultsAndAiResponse } from './services/geminiService';
import { StreamingSearchService } from './services/streamingService';
import type { DocumentResult, AiResponse, Filters } from './types';

const App: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [lastExecutedQuery, setLastExecutedQuery] = useState<string>(""); // Track last executed query
  const [documents, setDocuments] = useState<DocumentResult[]>([]);
  const [aiResponse, setAiResponse] = useState<AiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false); // Document loading state
  const [isAnswerLoading, setIsAnswerLoading] = useState<boolean>(false); // Answer loading state
  const [highlightedDocId, setHighlightedDocId] = useState<string | null>(null);
  const [viewingPdfPath, setViewingPdfPath] = useState<string | null>(null); // State for PDF viewer
  // Staged filters (UI selections)
  const [filters, setFilters] = useState<Filters>({
    fileType: [],
    timeRange: { type: 'all', label: 'All Time' },
    dataSource: [],
  });
  // Applied filters (used for actual searches)
  const [appliedFilters, setAppliedFilters] = useState<Filters>({
    fileType: [],
    timeRange: { type: 'all', label: 'All Time' },
    dataSource: [],
  });

  const docRefs = useRef<Map<string, React.RefObject<HTMLDivElement | null>>>(new Map());
  const streamingService = useRef(new StreamingSearchService());

  const getDocRefs = (docs: DocumentResult[]) => {
    const newDocRefs = new Map<string, React.RefObject<HTMLDivElement | null>>();
    docs.forEach(doc => {
      const existingRef = docRefs.current.get(doc.id);
      newDocRefs.set(doc.id, existingRef || createRef<HTMLDivElement>());
    });
    return newDocRefs;
  };

  const executeStreamingSearch = useCallback(async (query: string, filtersOverride?: Filters) => {
    if (!query.trim()) {
      console.log("Empty query, skipping search");
      return;
    }

    setIsLoading(true);
    setIsAnswerLoading(true);
    setLastExecutedQuery(query);
    setDocuments([]);
    setAiResponse(null);
    const filtersToUse = filtersOverride ?? appliedFilters;
    
    try {
      console.log(`🔍 Executing streaming search for: "${query}"`);
      
      await streamingService.current.startStreamingSearch(
        query,
        filtersToUse,
        // onChunks - called when document chunks are received
        (receivedDocuments: DocumentResult[]) => {
          console.log(`📄 Displaying ${receivedDocuments.length} document chunks`);
          docRefs.current = getDocRefs(receivedDocuments);
          setDocuments(receivedDocuments);
          setIsLoading(false); // Stop loading for left pane, show documents
        },
        // onAnswer - called when AI response is received
        (receivedAiResponse: AiResponse) => {
          console.log('🤖 Displaying AI response with typewriter effect');
          setAiResponse(receivedAiResponse);
          setIsAnswerLoading(false); // Stop loading for right pane, show answer
        },
        // onComplete - called when everything is done
        (status: string, method: string) => {
          console.log(`✅ Streaming search completed: ${status} via ${method}`);
        },
        // onError - called if something goes wrong
        (error: string) => {
          console.error('❌ Streaming search error:', error);
          setIsLoading(false);
          setIsAnswerLoading(false);
          // Fallback to regular search
          executeRegularSearch(query, filtersToUse);
        }
      );
    } catch (error) {
      console.error("Streaming search failed, falling back to regular search:", error);
      executeRegularSearch(query, filtersToUse);
    }
  }, [appliedFilters]);

  const executeRegularSearch = useCallback(async (query: string, filtersOverride?: Filters) => {
    console.log("🔄 Starting regular search for:", query);
    setIsLoading(true);
    setIsAnswerLoading(true);
    setLastExecutedQuery(query);
    const filtersToUse = filtersOverride ?? appliedFilters;
    
    try {
      console.log("📡 Calling fetchSearchResultsAndAiResponse...");
      const { documents: fetchedDocs, aiResponse: fetchedAiResponse } = await fetchSearchResultsAndAiResponse(
        query,
        filtersToUse
      );
      console.log("✅ Received response:", { docs: fetchedDocs.length, hasResponse: !!fetchedAiResponse });
      docRefs.current = getDocRefs(fetchedDocs);
      setDocuments(fetchedDocs);
      setAiResponse(fetchedAiResponse);
    } catch (error) {
      console.error("❌ Regular search failed:", error);
      // Show error message to user
      setDocuments([]);
      setAiResponse({
        summary: `Error: ${error instanceof Error ? error.message : 'Search failed'}`,
        items: []
      });
    } finally {
      console.log("🏁 Regular search completed");
      setIsLoading(false);
      setIsAnswerLoading(false);
    }
  }, [appliedFilters]);

    // Handle search submission (Enter key or button click)
  const handleSearchSubmit = useCallback(() => {
    console.log("🔍 Search submitted, using regular search for debugging");
    executeRegularSearch(searchQuery);
  }, [searchQuery, executeRegularSearch]);

  // Handle filter UI changes (staged only, do not execute search automatically)
  const handleFilterChange = useCallback((newFilters: Partial<Filters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  // Apply filters and trigger search only when Apply is pressed
  const handleApplyFilters = useCallback(() => {
    // Persist applied filters
    setAppliedFilters(filters);
    if (lastExecutedQuery) {
      // Re-run the last search immediately with the freshly applied filters
      executeStreamingSearch(lastExecutedQuery, filters);
    }
  }, [filters, lastExecutedQuery, executeStreamingSearch]);

  const handleSearchQueryChange = useCallback((query: string) => {
    setSearchQuery(query);
    // Note: We don't automatically search when query changes
  }, []);

  const handleReferenceClick = (docId: string) => {
    const ref = docRefs.current.get(docId);
    if (ref && ref.current) {
      ref.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
      setHighlightedDocId(docId);
      setTimeout(() => {
        setHighlightedDocId(null);
      }, 2500);
    }
  };

  const handleViewPdf = (sourcePath: string) => {
    console.log('App: Viewing PDF with source path:', sourcePath);
    const pdfUrl = getPdfUrl(sourcePath);
    console.log('App: Generated PDF URL:', pdfUrl);
    setViewingPdfPath(sourcePath);
  };

  const handleClosePdf = () => {
    setViewingPdfPath(null);
  };

  const getPdfUrl = (sourcePath: string | null): string | null => {
    if (!sourcePath) return null;
    const url = new URL('http://localhost:5000/pdf');
    url.searchParams.set('path', sourcePath);
    console.log('App: getPdfUrl - Generated URL:', url.toString());
    return url.toString();
  };

  // Cleanup streaming connection on unmount
  useEffect(() => {
    return () => {
      streamingService.current.closeConnection();
    };
  }, []);

  return (
    <div className="flex flex-col h-screen font-sans bg-slate-50 text-slate-800">
      {/* Show HomeScreen if no search has been executed */}
      {!lastExecutedQuery ? (
        <HomeScreen 
          searchQuery={searchQuery}
          onSearchQueryChange={handleSearchQueryChange}
          onSearchSubmit={handleSearchSubmit}
        />
      ) : (
        <>
          <Header 
            searchQuery={searchQuery}
            onSearchQueryChange={handleSearchQueryChange}
            onSearchSubmit={handleSearchSubmit}
            filters={filters}
            onFilterChange={handleFilterChange}
            onApplyFilters={handleApplyFilters}
            resultsCount={documents.length}
            isLoading={isLoading}
            hasExecutedSearch={!!lastExecutedQuery}
          />
          <main className="flex-grow flex overflow-hidden border-t border-slate-200">
            <div className="w-full lg:w-7/12 xl:w-1/2 h-full overflow-y-auto p-4 lg:p-6 space-y-4">
              <LeftPane 
                documents={documents} 
                docRefs={docRefs.current} 
                highlightedDocId={highlightedDocId} 
                isLoading={isLoading} 
                hasExecutedSearch={!!lastExecutedQuery}
                onViewPdf={handleViewPdf}
              />
            </div>
            <div className="hidden lg:block w-5/12 xl:w-1/2 h-full overflow-y-auto bg-white border-l border-slate-200 p-6">
              <RightPane 
                aiResponse={aiResponse} 
                onReferenceClick={handleReferenceClick} 
                isLoading={isAnswerLoading}
                currentQuery={lastExecutedQuery}
                useTypewriter={true}
              />
            </div>
          </main>
        </>
      )}
      {viewingPdfPath && (
        <PdfViewer
          pdfUrl={getPdfUrl(viewingPdfPath)!}
          fileName={viewingPdfPath.split(/[\\/]/).pop() || 'PDF Document'}
          onClose={handleClosePdf}
        />
      )}
    </div>
  );
};

export default App;