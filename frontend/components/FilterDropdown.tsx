import React, { useState, useRef, useEffect } from 'react';
import { ChevronDownIcon } from './icons/ChevronDownIcon';

interface FilterOption {
    value: string;
    label: string;
    icon?: React.ReactNode;
}

interface FilterDropdownProps {
  icon: React.ReactNode;
  label: string;
  options: FilterOption[];
  selectedValues: string[];
  onChange: (values: string[]) => void;
  disabled?: boolean;
}

const FilterDropdown: React.FC<FilterDropdownProps> = ({ icon, label, options, selectedValues, onChange, disabled }) => {
  const [isOpen, setIsOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [wrapperRef]);
  
  const handleSelect = (value: string) => {
    const newSelected = selectedValues.includes(value)
      ? selectedValues.filter(v => v !== value)
      : [...selectedValues, value];
    onChange(newSelected);
  };

  const displayLabel = selectedValues.length > 0 ? `${label} (${selectedValues.length})` : label;

  return (
    <div className="relative" ref={wrapperRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className="flex items-center space-x-1 px-3 py-1.5 border border-slate-300 rounded-md bg-white hover:bg-slate-50 shadow-sm disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        {icon}
        <span className="truncate max-w-[100px]">{displayLabel}</span>
        <ChevronDownIcon />
      </button>
      {isOpen && (
        <div className="absolute top-full mt-1 w-64 bg-white border border-slate-200 rounded-md shadow-lg z-20" role="listbox">
          <ul className="py-1 max-h-72 overflow-y-auto">
            {options.map(option => (
              <li key={option.value} role="option" aria-selected={selectedValues.includes(option.value)}>
                <label
                  className={`flex w-full items-center space-x-3 cursor-pointer px-4 py-2 text-sm text-slate-700 hover:bg-slate-100 ${selectedValues.includes(option.value) ? 'font-semibold bg-slate-50' : ''}`}
                >
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-slate-300 text-orange-500 focus:ring-orange-400"
                    checked={selectedValues.includes(option.value)}
                    onChange={() => handleSelect(option.value)}
                  />
                  {option.icon && <span className="flex-shrink-0">{option.icon}</span>}
                  <span className="flex-grow">{option.label}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default FilterDropdown;
