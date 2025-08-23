import React, { useEffect, useRef, useState } from 'react';
import { renderAsync } from 'docx-preview';
import * as XLSX from 'xlsx';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { MoreVerticalIcon } from './icons/MoreVerticalIcon';
import { WordIcon } from './icons/WordIcon';
import { ExcelIcon } from './icons/ExcelIcon';
import { PdfIcon } from './icons/PdfIcon';

interface DocViewerProps {
  pdfUrl: string; // This prop is used for all document types now
  fileName: string;
  onClose: () => void;
}

const PrintIcon: React.FC = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm7-8V3a1 1 0 00-1-1H9a1 1 0 00-1 1v6m3-3h6" />
  </svg>
);

const DownloadIcon: React.FC = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

const FileTypeIcon = ({ extension, className }: { extension?: string, className?: string }) => {
    const defaultClassName = className || "w-6 h-6";
    switch (extension) {
        case 'pdf':
            return <PdfIcon className={`${defaultClassName} text-red-500`} />;
        case 'docx':
        case 'doc':
            return <WordIcon className={`${defaultClassName} text-blue-600`} />;
        case 'xlsx':
        case 'xls':
            return <ExcelIcon className={`${defaultClassName} text-green-700`} />;
        default:
            return (
                <svg className={`${defaultClassName} text-gray-300`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
            );
    }
};

const LoadingSpinner: React.FC = () => (
    <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-white"></div>
    </div>
);

const DocumentViewer: React.FC<DocViewerProps> = ({ pdfUrl, fileName, onClose }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const fileExtension = fileName.split('.').pop()?.toLowerCase();

  useEffect(() => {
    document.body.style.overflow = 'hidden';
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.body.style.overflow = 'auto';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  useEffect(() => {
    if (!pdfUrl || !fileExtension) return;

    const renderNonPdfDocument = async () => {
      setIsLoading(true);
      setError(null);
      if (viewerRef.current) viewerRef.current.innerHTML = ''; // Clear previous content

      try {
        const response = await fetch(pdfUrl);
        if (!response.ok) {
          throw new Error(`Failed to load document: ${response.statusText}`);
        }

        if (viewerRef.current) {
            if (['docx', 'doc'].includes(fileExtension)) {
                const blob = await response.blob();
                // Correctly call renderAsync with 4 arguments for modern docx-preview versions
                await renderAsync(blob, viewerRef.current, null, {
                    className: "docx-preview",
                    ignoreWidth: false,
                    ignoreHeight: false,
                });
            } else if (['xlsx', 'xls'].includes(fileExtension)) {
                const arrayBuffer = await response.arrayBuffer();
                const workbook = XLSX.read(arrayBuffer, { type: 'buffer' });
                const sheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[sheetName];
                const html = XLSX.utils.sheet_to_html(worksheet);
                viewerRef.current.innerHTML = `<div class="xlsx-preview">${html}</div>`;
            }
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Could not render document.";
        console.error("Error rendering document:", err);
        setError(errorMessage);
        if(viewerRef.current) viewerRef.current.innerHTML = `<div class="p-4 text-red-500">Error: ${errorMessage}</div>`;
      } finally {
        setIsLoading(false);
      }
    };

    if (fileExtension !== 'pdf') {
      renderNonPdfDocument();
    } else {
      setIsLoading(false); // iframe handles its own loading state
    }
  }, [pdfUrl, fileExtension]);

  const renderContent = () => {
    switch (fileExtension) {
      case 'pdf':
        return (
          <iframe
            src={`${pdfUrl}`}
            title={fileName}
            className="w-full h-full border-none"
          />
        );
      case 'docx':
      case 'doc':
      case 'xlsx':
      case 'xls':
        return (
          <div className="w-full h-full bg-white overflow-y-auto">
             <style>{`
                .docx-preview { padding: 2rem; background: white; }
                .docx-preview table { border-collapse: collapse; width: 100%; }
                .docx-preview th, .docx-preview td { border: 1px solid #ccc; padding: 8px; }

                .xlsx-preview table { border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 14px; }
                .xlsx-preview th, .xlsx-preview td { border: 1px solid #ccc; padding: 8px; text-align: left; word-wrap: break-word; }
                .xlsx-preview th { background-color: #f2f2f2; font-weight: bold; }
                .xlsx-preview tr:nth-child(even) { background-color: #f9f9f9; }
            `}</style>
            {isLoading ? <div className="w-full h-full bg-white flex items-center justify-center"><LoadingSpinner/></div> : <div ref={viewerRef} />}
          </div>
        );
      default:
        return (
          <div className="p-8 text-center text-slate-600 bg-white rounded-lg h-full flex flex-col items-center justify-center">
            <h3 className="text-xl font-semibold mb-2">Preview Not Available</h3>
            <p>Preview for ".{fileExtension}" files is not supported.</p>
          </div>
        );
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 z-50 flex flex-col items-center justify-center animate-fade-in" role="dialog" aria-modal="true" aria-labelledby="doc-viewer-title">
        <style>{`
            @keyframes fade-in {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            .animate-fade-in {
              animation: fade-in 0.2s ease-out forwards;
            }
        `}</style>
      
        <header className="bg-[#202124] text-white px-4 py-2 flex items-center justify-between w-full max-w-[95vw] sm:max-w-6xl flex-shrink-0">
          <div className="flex items-center gap-4">
            <button
              onClick={onClose}
              className="p-2 rounded-full text-gray-300 hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Close document viewer"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="flex items-center gap-3">
              <FileTypeIcon extension={fileExtension} className="w-5 h-5"/>
              <h2 id="doc-viewer-title" className="text-base font-medium truncate">{fileName}</h2>
            </div>
          </div>

          <div className="hidden sm:flex flex-1 justify-center px-4">
             <button className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-md text-sm transition-colors">
                <span>Open with</span>
                <ChevronDownIcon className="w-4 h-4 text-white" />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden md:flex items-center gap-1.5 text-sm text-green-400 p-2 rounded-md">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span>Active</span>
            </div>
            <button className="p-2 rounded-full text-gray-300 hover:bg-gray-700 transition-colors" aria-label="Print document"><PrintIcon /></button>
            <a href={pdfUrl} download={fileName} className="p-2 rounded-full text-gray-300 hover:bg-gray-700 transition-colors" aria-label="Download document"><DownloadIcon /></a>
            <button className="p-2 rounded-full text-gray-300 hover:bg-gray-700 transition-colors" aria-label="More options"><MoreVerticalIcon /></button>
          </div>
        </header>

        <main className="flex-grow bg-[#35363a] w-full max-w-[95vw] sm:max-w-6xl h-full overflow-auto p-2 sm:p-4">
           {renderContent()}
        </main>
    </div>
  );
};

export default DocumentViewer;
