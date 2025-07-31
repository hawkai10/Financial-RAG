import React, { useState, useEffect } from 'react';
import { SearchIcon } from './icons/SearchIcon';
import { fetchExampleQueries, fetchRecentDocuments } from '../services/geminiService';

interface HomeScreenProps {
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSearchSubmit: () => void;
}

interface RecentDocument {
  id: string;
  title: string;
  fileType: string;
  sourcePath: string;
  lastAccessed: string;
  sourceType: string;
}

// File type icons
const FileTypeIcon: React.FC<{ fileType: string; className?: string }> = ({ fileType, className = "w-5 h-5" }) => {
  switch (fileType) {
    case 'word':
      return (
        <svg className={`${className} text-blue-600`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
          <path d="M14 2v6h6"/>
          <path d="M10.5 14.5L8 12l2.5-2.5M13.5 9.5L16 12l-2.5 2.5"/>
        </svg>
      );
    case 'pdf':
      return (
        <svg className={`${className} text-red-600`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
          <path d="M14 2v6h6"/>
          <path d="M9 13h6M9 17h6M9 9h1"/>
        </svg>
      );
    case 'excel':
      return (
        <svg className={`${className} text-green-600`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
          <path d="M14 2v6h6"/>
          <path d="M8 13h8M8 17h8M8 9h8"/>
        </svg>
      );
    default:
      return (
        <svg className={`${className} text-gray-600`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
          <path d="M14 2v6h6"/>
        </svg>
      );
  }
};

const HomeScreen: React.FC<HomeScreenProps> = ({ 
  searchQuery, 
  onSearchQueryChange, 
  onSearchSubmit 
}) => {
  const [exampleQueries, setExampleQueries] = useState<string[]>([]);
  const [recentDocuments, setRecentDocuments] = useState<RecentDocument[]>([]);
  const [isLoadingQueries, setIsLoadingQueries] = useState(true);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);

  // Load example queries and recent documents on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load example queries
        const queries = await fetchExampleQueries();
        setExampleQueries(queries);
      } catch (error) {
        console.error('Failed to load example queries:', error);
        setExampleQueries([
          "What are the main topics covered in the documents?",
          "Can you summarize the key information available?",
          "What important details should I know from these documents?"
        ]);
      } finally {
        setIsLoadingQueries(false);
      }

      try {
        // Load recent documents
        const docs = await fetchRecentDocuments();
        setRecentDocuments(docs);
      } catch (error) {
        console.error('Failed to load recent documents:', error);
        setRecentDocuments([]);
      } finally {
        setIsLoadingDocs(false);
      }
    };

    loadData();
  }, []);

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      onSearchSubmit();
    }
  };

  const handleExampleQueryClick = (query: string) => {
    onSearchQueryChange(query);
    // Auto-submit when clicking example query
    setTimeout(() => onSearchSubmit(), 100);
  };

  const handleDocumentClick = (doc: RecentDocument) => {
    // Set a query about this specific document
    const query = `What information is available about ${doc.title}?`;
    onSearchQueryChange(query);
    setTimeout(() => onSearchSubmit(), 100);
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else {
      return 'Recently';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col">
      {/* Header with user menu */}
      <div className="flex justify-end p-4">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">1</span>
            <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5-5-5 5h5zM15 17v7" />
            </svg>
          </div>
          <div className="w-6 h-6 bg-gray-300 rounded"></div>
          <div className="w-8 h-8 bg-gray-400 rounded-full"></div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center px-8 -mt-16">
        {/* Logo and title */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <div className="w-16 h-16 bg-orange-500 rounded-2xl flex items-center justify-center mr-4">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                <circle cx="9.5" cy="9.5" r="2.5" fill="white" opacity="0.3"/>
              </svg>
            </div>
            <h1 className="text-5xl font-bold text-gray-800">
              amber<span className="text-gray-600">Search</span>
            </h1>
          </div>
        </div>

        {/* Search box */}
        <div className="w-full max-w-2xl mb-8">
          <div className="relative">
            <SearchIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="What are you looking for?"
              className="w-full pl-12 pr-24 py-4 border border-gray-200 rounded-2xl text-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent shadow-lg"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <span className="text-sm text-gray-400 bg-gray-100 px-2 py-1 rounded">Ctrl + K</span>
            </div>
          </div>
        </div>

        {/* Search button */}
        <button
          onClick={onSearchSubmit}
          className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-xl text-lg font-medium transition-colors mb-12 shadow-lg"
        >
          amberSearch
        </button>

        {/* Example queries */}
        {!isLoadingQueries && exampleQueries.length > 0 && (
          <div className="w-full max-w-4xl mb-12">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {exampleQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleQueryClick(query)}
                  className="p-4 text-left bg-white border border-gray-200 rounded-lg hover:border-orange-300 hover:shadow-md transition-all text-sm text-gray-700 hover:text-gray-900"
                >
                  <SearchIcon className="w-4 h-4 text-orange-500 mb-2" />
                  {query}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Recent documents section */}
        <div className="w-full max-w-4xl">
          <div className="flex items-center mb-6">
            <h2 className="text-xl font-semibold text-blue-600 flex items-center">
              Lately interacted documents
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </h2>
          </div>

          {isLoadingDocs ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-white rounded-lg p-4 animate-pulse">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gray-200 rounded"></div>
                    <div className="flex-1">
                      <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {recentDocuments.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => handleDocumentClick(doc)}
                  className="w-full bg-white rounded-lg p-4 hover:shadow-md transition-shadow text-left border border-gray-100 hover:border-orange-200"
                >
                  <div className="flex items-center space-x-3">
                    <FileTypeIcon fileType={doc.fileType} className="w-8 h-8" />
                    <div className="flex-1 min-w-0">
                      <h3 className="text-blue-600 font-medium truncate">{doc.title}</h3>
                      <p className="text-sm text-gray-500">{formatTimeAgo(doc.lastAccessed)}</p>
                    </div>
                    <div className="text-gray-400">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
                      </svg>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HomeScreen;
