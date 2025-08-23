import React from 'react';

export const CompressedIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4m16 0l-4-4m4 4l-4 4M4 12l4-4m-4 4l4 4" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16" />
    </svg>
);
