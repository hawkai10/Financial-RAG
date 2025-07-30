
import React, { useState } from 'react';
import TypewriterText from './TypewriterText';
import type { AiResponse } from '../types';

interface AiAnswerProps {
  response: AiResponse;
  onReferenceClick: (docId: string) => void;
  useTypewriter?: boolean;
}

const AiAnswer: React.FC<AiAnswerProps> = ({ response, onReferenceClick, useTypewriter = true }) => {
  const [showItems, setShowItems] = useState(!useTypewriter);
  
  return (
    <div className="space-y-6 text-slate-700">
      {useTypewriter ? (
        <TypewriterText
          text={response.summary}
          speed={80} // Characters per second
          className="text-base"
          onComplete={() => setShowItems(true)}
        />
      ) : (
        <p className="text-base">{response.summary}</p>
      )}
      
      {showItems && (
        <div className="space-y-5 animate-fade-in">
          {response.items.map((item, index) => (
            <div key={index}>
              <h4 className="font-bold text-slate-800 text-base mb-1">
                {index + 1}. {item.title}
              </h4>
              <div className="text-base leading-relaxed">
                {useTypewriter ? (
                  <TypewriterText
                    text={item.text}
                    speed={100} // Slightly faster for detailed items
                    className="inline"
                  />
                ) : (
                  <span>{item.text}</span>
                )}
                {item.references.map(ref => (
                  <button
                    key={ref.id}
                    onClick={() => onReferenceClick(ref.docId)}
                    className="inline-flex items-center justify-center w-5 h-5 ml-1.5 -mb-1 bg-orange-100 text-orange-600 text-xs font-bold rounded-full hover:bg-orange-200 transition-all focus:outline-none focus:ring-2 focus:ring-orange-400"
                    aria-label={`Reference ${ref.id} from document ${ref.docId}`}
                  >
                    {ref.id}
                  </button>
                ))}
                .
              </div>
            </div>
          ))}
        </div>
      )}
      
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </div>
  );
};

export default AiAnswer;
