import React, { useState, useRef, useEffect } from 'react';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { CalendarIcon } from './icons/CalendarIcon';
import type { TimeRangeFilterValue, TimeRangeType } from '../types';

interface TimeFilterDropdownProps {
  icon: React.ReactNode;
  options: { value: TimeRangeType, label: string }[];
  selectedValue: TimeRangeFilterValue;
  onChange: (value: TimeRangeFilterValue) => void;
  disabled?: boolean;
}

const formatDate = (date: Date) => {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${year}-${month}-${day}`;
}

const TimeFilterDropdown: React.FC<TimeFilterDropdownProps> = ({ icon, options, selectedValue, onChange, disabled }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [customStart, setCustomStart] = useState<Date | null>(selectedValue.startDate || null);
  const [customEnd, setCustomEnd] = useState<Date | null>(selectedValue.endDate || null);
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

  const handleSelect = (option: { value: TimeRangeType; label: string }) => {
    if (option.value !== 'custom') {
      onChange({ type: option.value, label: option.label });
      setIsOpen(false);
    } else {
        onChange({ ...selectedValue, type: 'custom', label: 'Custom Period' });
    }
  };

  const handleApplyCustomDate = () => {
      const label = `${customStart ? formatDate(customStart) : '...'} - ${customEnd ? formatDate(customEnd) : '...'}`;
      onChange({
          type: 'custom',
          label: label,
          startDate: customStart,
          endDate: customEnd
      });
      setIsOpen(false);
  }

  const displayLabel = selectedValue.type === 'custom' && selectedValue.startDate
    ? selectedValue.label
    : options.find(o => o.value === selectedValue.type)?.label || "All Time";

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
        <span className="truncate max-w-[120px]">{displayLabel}</span>
        <ChevronDownIcon />
      </button>
      {isOpen && (
        <div className="absolute top-full mt-1 w-64 bg-white border border-slate-200 rounded-md shadow-lg z-20" role="listbox">
          <ul className="py-1">
            {options.map(option => (
              <li key={option.value} role="option" aria-selected={selectedValue.type === option.value}>
                <label className={`flex w-full items-center space-x-3 cursor-pointer px-4 py-2 text-sm text-slate-700 hover:bg-slate-100 ${selectedValue.type === option.value ? 'font-semibold bg-slate-50' : ''}`}>
                    <input
                        type="radio"
                        name="time-filter"
                        className="h-4 w-4 border-slate-300 text-orange-500 focus:ring-orange-400"
                        checked={selectedValue.type === option.value}
                        onChange={() => handleSelect(option)}
                    />
                    <span>{option.label}</span>
                </label>
              </li>
            ))}
          </ul>
          {selectedValue.type === 'custom' && (
            <div className="border-t border-slate-200 p-3 space-y-3">
                <div className="flex items-center space-x-2 text-sm">
                    <CalendarIcon className="w-5 h-5 text-slate-500" />
                    <span className="font-medium text-slate-600">Set custom date range</span>
                </div>
                <div className="space-y-2">
                    <input type="date" value={customStart ? formatDate(customStart) : ''} onChange={e => setCustomStart(e.target.valueAsDate)} className="w-full text-sm border-slate-300 rounded-md" aria-label="Start date" />
                    <input type="date" value={customEnd ? formatDate(customEnd) : ''} onChange={e => setCustomEnd(e.target.valueAsDate)} className="w-full text-sm border-slate-300 rounded-md" aria-label="End date" />
                </div>
                <button 
                    onClick={handleApplyCustomDate}
                    className="w-full bg-orange-500 text-white text-sm font-semibold py-2 rounded-md hover:bg-orange-600 transition-colors"
                >
                    Apply
                </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TimeFilterDropdown;
