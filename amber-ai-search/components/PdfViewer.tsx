// PdfViewer.tsx
'use client';
import React, { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker - Try different approaches
if (typeof window !== 'undefined') {
  // Try local worker first, fallback to CDN
  try {
    pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
      'pdfjs-dist/build/pdf.worker.min.js',
      import.meta.url,
    ).toString();
  } catch {
    // Fallback to CDN
    pdfjsLib.GlobalWorkerOptions.workerSrc = 
      `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.js`;
  }
}

type PDFDocumentProxy = pdfjsLib.PDFDocumentProxy;

interface PdfViewerProps {
  pdfUrl: string;
  fileName: string;
  onClose: () => void;
}

/* ---------------- Icons (20–22px, stroke ~1.8) ---------------- */

const IconBtn: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({ className = '', ...props }) => (
  <button
    className={`h-10 w-10 grid place-items-center rounded hover:bg-white/10 
      focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30 
      disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
    {...props}
  />
);

const XIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" d="M6 6l12 12M18 6L6 18" />
  </svg>
);
const PdfBadge = () => (
  <div className="h-6 px-2 rounded bg-red-600 text-white text-[11px] font-semibold grid place-items-center">PDF</div>
);
const ChevronDown = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
  </svg>
);
const UserIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M12 12a4 4 0 100-8 4 4 0 000 8z" />
    <path d="M6 20a6 6 0 1112 0" strokeLinecap="round" />
  </svg>
);
const PrintIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M6 9V4h12v5" />
    <rect x="6" y="13" width="12" height="8" rx="1.5" />
    <path d="M6 17h12" />
  </svg>
);
const DownloadIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" d="M12 3v12" />
    <path strokeLinecap="round" d="M7 10l5 5 5-5" />
    <path d="M5 21h14" strokeLinecap="round" />
  </svg>
);
const MoreIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
    <circle cx="5" cy="12" r="1.6" />
    <circle cx="12" cy="12" r="1.6" />
    <circle cx="19" cy="12" r="1.6" />
  </svg>
);
const ChevronLeft = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 18l-6-6 6-6" />
  </svg>
);
const ChevronRight = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 6l6 6-6 6" />
  </svg>
);
const MinusIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" d="M5 12h14" />
  </svg>
);
const PlusIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path strokeLinecap="round" d="M12 5v14M5 12h14" />
  </svg>
);

/* ---------------- Component ---------------- */

export default function PdfViewer({ pdfUrl, fileName, onClose }: PdfViewerProps) {
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(100); // 50–200
  const [loading, setLoading] = useState(true);
  const [useFallback, setUseFallback] = useState(false);

  const canvasesRef = useRef<(HTMLCanvasElement | null)[]>([]);
  const mainRef = useRef<HTMLDivElement>(null);

  /* ---------------- Lifecycle & Keyboard ---------------- */

  useEffect(() => {
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') setCurrentPage(p => Math.max(1, p - 1));
      if (e.key === 'ArrowRight') setCurrentPage(p => Math.min(totalPages, p + 1));

      const meta = e.ctrlKey || e.metaKey;
      if (meta && (e.key === '+' || e.key === '=')) {
        e.preventDefault();
        setZoom(z => Math.min(200, z + 10));
      }
      if (meta && e.key === '-') {
        e.preventDefault();
        setZoom(z => Math.max(50, z - 10));
      }
      if (meta && e.key === '0') {
        e.preventDefault();
        setZoom(100);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = 'auto';
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose, totalPages]);

  /* ---------------- Load PDF ---------------- */

  useEffect(() => {
    let cancelled = false;
    console.log('PDF Viewer: Starting to load PDF from URL:', pdfUrl);
    setLoading(true);

    // First test if the URL is accessible
    fetch(pdfUrl)
      .then(response => {
        console.log('PDF Viewer: URL fetch response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.blob();
      })
      .then(blob => {
        console.log('PDF Viewer: Successfully fetched PDF blob, size:', blob.size);
        // Now try to load with PDF.js
        return pdfjsLib.getDocument(pdfUrl).promise;
      })
      .then(d => {
        if (cancelled) return;
        console.log('PDF Viewer: Successfully loaded PDF, pages:', d.numPages);
        setDoc(d);
        setTotalPages(d.numPages);
        setCurrentPage(1);
      })
      .catch(error => {
        console.error('PDF Viewer: Error loading PDF:', error);
        console.error('PDF Viewer: Error details:', {
          message: error.message,
          name: error.name,
          stack: error.stack
        });
        console.log('PDF Viewer: Falling back to iframe method');
        setUseFallback(true);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  /* ---------------- Render current page ---------------- */

  useEffect(() => {
    if (!doc || totalPages === 0) return;

    const renderPage = async (p: number) => {
      const page = await doc.getPage(p);
      const scale = zoom / 100;
      const viewport = page.getViewport({ scale });
      const canvas = canvasesRef.current[p - 1];
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { alpha: false })!;
      // Set sizes (device pixel ratio aware)
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      await page.render({ canvas: canvas, canvasContext: ctx, viewport }).promise;
    };

    // Render current page and optionally neighbors for smooth nav
    renderPage(currentPage);
    if (currentPage > 1) renderPage(currentPage - 1);
    if (currentPage < totalPages) renderPage(currentPage + 1);
  }, [doc, currentPage, totalPages, zoom]);

  /* ---------------- Handlers ---------------- */

  const goPrev = () => setCurrentPage(p => Math.max(1, p - 1));
  const goNext = () => setCurrentPage(p => Math.min(totalPages, p + 1));
  const zoomOut = () => setZoom(z => Math.max(50, z - 10));
  const zoomIn = () => setZoom(z => Math.min(200, z + 10));

  const download = () => {
    const a = document.createElement('a');
    a.href = pdfUrl;
    a.download = fileName || 'document.pdf';
    a.click();
  };

  /* ---------------- UI ---------------- */

  return (
    <div className="fixed inset-0 z-[1000]">
      {/* Scrim */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-[2px]" />

      {/* Shell */}
      <div className="absolute inset-0 flex flex-col">
        {/* Top App Bar */}
        <header
          role="toolbar"
          className="h-14 bg-[#202124] text-[#E8EAED] border-b border-white/10 flex items-center px-2"
        >
          {/* Left: close + file info */}
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <IconBtn aria-label="Close" onClick={onClose}><XIcon /></IconBtn>
            <PdfBadge />
            <div className="min-w-0">
              <div className="truncate text-sm">{fileName || 'document.pdf'}</div>
              <div className="text-[12px] text-[#9AA0A6]">PDF • View only</div>
            </div>
          </div>

          {/* Center: Open with */}
          <div className="flex-1 hidden md:flex justify-center">
            <button className="h-9 px-3 rounded-md bg-white/10 hover:bg-white/15 text-sm flex items-center gap-1">
              Open with <ChevronDown />
            </button>
          </div>

          {/* Right: actions */}
          <div className="flex-1 flex justify-end items-center gap-1">
            <IconBtn aria-label="Account"><UserIcon /></IconBtn>
            <IconBtn aria-label="Print" onClick={() => window.print()}><PrintIcon /></IconBtn>
            <IconBtn aria-label="Download" onClick={download}><DownloadIcon /></IconBtn>
            <IconBtn aria-label="More options"><MoreIcon /></IconBtn>
          </div>
        </header>

        {/* Document Area */}
        <main ref={mainRef} className="flex-1 overflow-auto bg-[#121212] antialiased">
          <div className="mx-auto max-w-[calc(100vw-160px)] py-6 flex flex-col items-center gap-6">
            {loading && (
              <div className="flex flex-col items-center gap-4 text-white/80 text-sm">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                <div>Loading PDF...</div>
                <div className="text-xs text-white/60">URL: {pdfUrl}</div>
              </div>
            )}
            {!loading && useFallback && (
              <div className="w-full h-full bg-white rounded-lg shadow-lg">
                <iframe
                  src={pdfUrl}
                  title={fileName}
                  className="w-full h-full border-none rounded-lg"
                  style={{ minHeight: '80vh' }}
                />
              </div>
            )}
            {!loading && !useFallback && totalPages === 0 && (
              <div className="flex flex-col items-center gap-2 text-white/80 text-sm">
                <div className="text-red-400">❌ Could not load PDF</div>
                <div className="text-xs text-white/60">URL: {pdfUrl}</div>
                <button 
                  onClick={() => setUseFallback(true)} 
                  className="mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                >
                  Try Simple Viewer
                </button>
                <button 
                  onClick={() => window.open(pdfUrl, '_blank')} 
                  className="mt-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white text-xs"
                >
                  Open in New Tab
                </button>
              </div>
            )}
            {!useFallback && totalPages > 0 && Array.from({ length: totalPages }, (_, i) => (
              <div
                key={i}
                className="bg-white rounded-sm shadow-[0_1px_3px_rgba(0,0,0,0.3),0_1px_2px_rgba(0,0,0,0.24)] select-none"
                onClick={() => setCurrentPage(i + 1)}
              >
                <canvas ref={(el) => { canvasesRef.current[i] = el; }} />
              </div>
            ))}
          </div>
        </main>

        {/* Floating Bottom Controls */}
        <div className="pointer-events-none fixed left-1/2 -translate-x-1/2 bottom-6">
          <div className="pointer-events-auto px-3 py-2 bg-[#2A2A2A]/90 backdrop-blur-md border border-white/15 rounded-full text-white shadow-[0_4px_20px_rgba(0,0,0,0.35)] flex items-center gap-1">
            <span className="text-xs text-white/80 px-2">Page</span>
            <IconBtn aria-label="Previous page" onClick={goPrev} disabled={currentPage <= 1}><ChevronLeft /></IconBtn>
            <div className="px-2 tabular-nums text-sm w-[76px] text-center">
              {Math.min(currentPage, totalPages || 0)} / {totalPages || 0}
            </div>
            <IconBtn aria-label="Next page" onClick={goNext} disabled={currentPage >= totalPages}><ChevronRight /></IconBtn>

            <div className="w-px h-6 bg-white/15 mx-1" />

            <IconBtn aria-label="Zoom out" onClick={zoomOut} disabled={zoom <= 50}><MinusIcon /></IconBtn>
            <div className="w-12 text-center tabular-nums text-sm">{zoom}%</div>
            <IconBtn aria-label="Zoom in" onClick={zoomIn} disabled={zoom >= 200}><PlusIcon /></IconBtn>
          </div>
        </div>
      </div>
    </div>
  );
}
