import React, { useState } from 'react';

interface FilePathProps {
  fullPath: string;
  maxLength?: number;
}

const CopyIcon: React.FC<{ className?: string }> = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

const FilePath: React.FC<FilePathProps> = ({ fullPath, maxLength = 50 }) => {
  const [showCopied, setShowCopied] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  const truncatePath = (path: string, maxLen: number): string => {
    if (path.length <= maxLen) return path;
    
    // Find the last part (filename) and some parent directories
    const parts = path.split('\\');
    const filename = parts[parts.length - 1];
    
    if (filename.length >= maxLen - 3) {
      return '...' + filename.slice(-(maxLen - 3));
    }
    
    let result = filename;
    let i = parts.length - 2;
    
    while (i >= 0 && result.length + parts[i].length + 4 <= maxLen) {
      result = parts[i] + '\\' + result;
      i--;
    }
    
    if (i >= 0) {
      result = '...' + result;
    }
    
    return result;
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(fullPath);
      setShowCopied(true);
      setTimeout(() => setShowCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy path:', err);
    }
  };

  const displayPath = truncatePath(fullPath, maxLength);

  return (
    <div className="flex items-center space-x-1 relative">
      <span 
        className="cursor-help"
        title={fullPath}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {displayPath}
      </span>
      
      <div className="relative">
        <button
          onClick={handleCopy}
          className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded hover:bg-slate-100"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          <CopyIcon className="w-3 h-3" />
        </button>
        
        {/* Tooltip */}
        {showTooltip && !showCopied && (
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10">
            Copy path. Paste it in the windows explorer to go to the folder.
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
          </div>
        )}
        
        {/* Copied confirmation */}
        {showCopied && (
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-green-600 text-white text-xs rounded whitespace-nowrap z-10">
            Path copied
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-green-600"></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FilePath;
