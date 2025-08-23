import React, { useState, useEffect } from 'react';

interface TypewriterTextProps {
  text: string;
  speed?: number; // Characters per second
  onComplete?: () => void;
  className?: string;
}

const TypewriterText: React.FC<TypewriterTextProps> = ({ 
  text, 
  speed = 50, // Default 50 characters per second (faster than typical typewriter)
  onComplete,
  className = ""
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 1000 / speed); // Convert characters per second to milliseconds per character

      return () => clearTimeout(timeout);
    } else if (onComplete && currentIndex === text.length) {
      onComplete();
    }
  }, [currentIndex, text, speed, onComplete]);

  // Reset when text changes
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
  }, [text]);

  return (
    <span className={className}>
      {displayedText}
      {currentIndex < text.length && (
        <span className="animate-pulse">|</span>
      )}
    </span>
  );
};

export default TypewriterText;
