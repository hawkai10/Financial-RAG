import { forwardRef } from 'react';
import type { DocumentResult } from '../types';
import { MoreVerticalIcon } from './icons/MoreVerticalIcon';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import FilePath from './FilePath';

interface DocumentCardProps {
  document: DocumentResult;
  isHighlighted: boolean;
}

const DocumentCard = forwardRef<HTMLDivElement, DocumentCardProps>(({ document, isHighlighted }, ref) => {
  const { 
    sourceIcon, 
    sourceType, 
    sourcePath, 
    docTypeIcon, 
    title, 
    date, 
    snippet, 
    author,
    missingInfo,
    mustInclude
  } = document;

  const highlightClass = isHighlighted ? 'border-orange-400 ring-2 ring-orange-400 ring-offset-2' : 'border-slate-200';

  return (
    <div ref={ref} className={`bg-white border rounded-lg p-4 transition-all duration-300 ${highlightClass}`}>
      <div className="flex items-center justify-between text-sm text-slate-500">
        <div className="flex items-center space-x-2">
          {sourceIcon}
          <span>{sourceType}</span>
          <span className="text-slate-400">|</span>
          <FilePath fullPath={sourcePath} maxLength={40} />
        </div>
        <button className="text-slate-400 hover:text-slate-600">
          <MoreVerticalIcon />
        </button>
      </div>

      <div className="mt-3 flex items-start space-x-3">
        <div className="flex-shrink-0 text-blue-600 pt-1">{docTypeIcon}</div>
        <div className="flex-grow">
          <h3 className="text-lg text-blue-700 hover:underline cursor-pointer">{title}</h3>
          <p className="text-sm text-slate-500 mt-1">
            {author && <span>{author} &middot; </span>}
            {date}
          </p>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-slate-600 pl-9 space-y-3">
        <p dangerouslySetInnerHTML={{ __html: snippet }} />
        <div>
            <a href="#" className="text-blue-600 hover:underline font-medium">Page 1 Preview</a>
            <a href="#" className="ml-4 text-blue-600 hover:underline font-medium inline-flex items-center">
                More highlights <ChevronDownIcon className="w-4 h-4 ml-1" />
            </a>
        </div>
        {(missingInfo || mustInclude) && (
            <div className="text-xs text-slate-500">
                {missingInfo && <span>Missing: <span className="text-slate-700">{missingInfo}</span></span>}
                {missingInfo && mustInclude && <span className="mx-2">|</span>}
                {mustInclude && <span>Must include: <span className="text-slate-700">{mustInclude}</span></span>}
            </div>
        )}
      </div>
    </div>
  );
});

DocumentCard.displayName = 'DocumentCard';

export default DocumentCard;
