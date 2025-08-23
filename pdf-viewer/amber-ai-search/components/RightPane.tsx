
import React from 'react';
import AiAnswer from './AiAnswer';
import LoadingScreen from './LoadingScreen';
import type { AiResponse } from '../types';
import { ChatBubbleIcon } from './icons/ChatBubbleIcon';

interface RightPaneProps {
  aiResponse: AiResponse | null;
  onReferenceClick: (docId: string) => void;
  isLoading: boolean;
  currentQuery?: string;
  useTypewriter?: boolean;
}

const RightPane: React.FC<RightPaneProps> = ({ 
  aiResponse, 
  onReferenceClick, 
  isLoading, 
  currentQuery,
  useTypewriter = true 
}) => {
  return (
    <div>
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-orange-400 rounded-full flex items-center justify-center">
          <ChatBubbleIcon className="w-6 h-6 text-white" />
        </div>
              <div className="border-b border-slate-300 pb-4 mb-6">
        <h2 className="text-2xl font-bold text-slate-800">Rag<span className="text-orange-400">AI</span></h2>
        <p className="text-sm text-slate-600 mt-1">Financial Document Intelligence</p>
      </div>
      </div>
      
      {isLoading && <LoadingScreen query={currentQuery} />}

      {!isLoading && aiResponse && (
        <AiAnswer 
          response={aiResponse} 
          onReferenceClick={onReferenceClick} 
          useTypewriter={useTypewriter}
        />
      )}

      {!isLoading && !aiResponse && (
        <div className="text-slate-500">The AI assistant is ready. Results will appear here.</div>
      )}
    </div>
  );
};

export default RightPane;
