import React, { createRef } from 'react';
import DocumentCard from './DocumentCard';
import type { DocumentResult } from '../types';

interface LeftPaneProps {
  documents: DocumentResult[];
  docRefs: Map<string, React.RefObject<HTMLDivElement | null>>;
  highlightedDocId: string | null;
  isLoading: boolean;
  hasExecutedSearch?: boolean;
  onViewPdf: (sourcePath: string) => void;
}

const LoadingSkeleton: React.FC = () => (
    <div className="bg-white border border-slate-200 rounded-lg p-4 animate-pulse">
        <div className="flex items-center space-x-2 mb-3">
            <div className="w-6 h-6 rounded bg-slate-200"></div>
            <div className="h-4 bg-slate-200 rounded w-1/3"></div>
        </div>
        <div className="h-5 bg-slate-200 rounded w-3/4 mb-2"></div>
        <div className="h-3 bg-slate-200 rounded w-1/4 mb-4"></div>
        <div className="space-y-2">
            <div className="h-4 bg-slate-200 rounded"></div>
            <div className="h-4 bg-slate-200 rounded w-5/6"></div>
            <div className="h-4 bg-slate-200 rounded w-1/2"></div>
        </div>
    </div>
);

const LeftPane: React.FC<LeftPaneProps> = ({ documents, docRefs, highlightedDocId, isLoading, hasExecutedSearch = false, onViewPdf }) => {
  if (isLoading) {
    return (
        <>
            <LoadingSkeleton />
            <LoadingSkeleton />
            <LoadingSkeleton />
        </>
    );
  }

  // Show welcome state if no search has been executed
  if (!hasExecutedSearch) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-8 text-center">
        <div className="text-slate-400 mb-4">
          <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-slate-700 mb-2">Ready to search</h3>
        <p className="text-slate-500 mb-4">Type your question in the search box above and press Enter to find relevant documents.</p>
  <p className="text-sm text-slate-400">Use the search box above to start.</p>
      </div>
    );
  }

  // Show no results message if search was executed but no documents found
  if (documents.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-8 text-center">
        <div className="text-slate-400 mb-4">
          <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.44-.896-6.01-2.364M12 21l7-7M5 14L12 7l7 7" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-slate-700 mb-2">No results found</h3>
        <p className="text-slate-500 mb-4">Try rephrasing your question or using different keywords.</p>
        <p className="text-sm text-slate-400">Make sure the documents you're looking for have been indexed.</p>
      </div>
    );
  }
  
  return (
    <>
      {documents.map(doc => {
        const docRef = docRefs.get(doc.id);
        return (
          <DocumentCard 
            key={doc.id} 
            ref={docRef || createRef<HTMLDivElement>()}
            document={doc}
            isHighlighted={highlightedDocId === doc.id} 
            onViewPdf={onViewPdf}
          />
        );
      })}
    </>
  );
};

export default LeftPane;