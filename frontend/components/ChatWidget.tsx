
import React from 'react';
import { ChatBubbleIcon } from './icons/ChatBubbleIcon';

const ChatWidget: React.FC = () => {
  return (
    <button
      className="fixed bottom-6 right-6 w-16 h-16 bg-orange-400 text-white rounded-full shadow-lg flex items-center justify-center hover:bg-orange-500 transition-all transform hover:scale-110 focus:outline-none focus:ring-4 focus:ring-orange-300"
      aria-label="Open chat"
    >
      <ChatBubbleIcon className="w-8 h-8" />
    </button>
  );
};

export default ChatWidget;
