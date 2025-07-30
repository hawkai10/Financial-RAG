import React from 'react';

interface LoadingScreenProps {
  query?: string;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ query }) => {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      {/* AmberAI Logo/Icon */}
      <div className="w-16 h-16 bg-orange-400 rounded-full flex items-center justify-center mb-8">
        <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
          <div className="w-4 h-4 bg-orange-400 rounded-full animate-pulse"></div>
        </div>
      </div>
      
      {/* Loading Text */}
      <div className="text-center mb-8">
        <h3 className="text-xl font-semibold text-slate-700 mb-2">
          Analyzing your query...
        </h3>
        {query && (
          <p className="text-slate-500 mb-4 max-w-md">
            "{query}" is being processed...
          </p>
        )}
        <p className="text-sm text-slate-400">
          Request being processed...
        </p>
      </div>
      
      {/* Loading Animation */}
      <div className="flex space-x-2 mb-8">
        <div className="w-3 h-3 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-3 h-3 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-3 h-3 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
      
      {/* Progress Bar */}
      <div className="w-64 h-2 bg-slate-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-orange-400 to-orange-500 rounded-full"
          style={{
            width: '70%',
            animation: 'loading-progress 3s ease-in-out infinite'
          }}
        ></div>
      </div>
      
      {/* Status Message */}
      <div className="mt-6 text-center">
        <p className="text-sm text-slate-500">
          Searching through documents and generating response...
        </p>
      </div>
      
      <style>{`
        @keyframes loading-progress {
          0% { width: 20%; }
          50% { width: 70%; }
          100% { width: 20%; }
        }
      `}</style>
    </div>
  );
};

export default LoadingScreen;
