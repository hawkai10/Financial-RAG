import React from 'react';
import FilterDropdown from './FilterDropdown';
import TimeFilterDropdown from './TimeFilterDropdown';
import { SearchIcon } from './icons/SearchIcon';
import { BellIcon } from './icons/BellIcon';
import { PlusIcon } from './icons/PlusIcon';
import { UserIcon } from './icons/UserIcon';
import { FolderIcon } from './icons/FolderIcon';
import { ClockIcon } from './icons/ClockIcon';
import { DataSourceIcon } from './icons/DataSourceIcon';
import type { Filters, FileType, DataSourceType, TimeRangeFilterValue } from '../types';
import { FILE_TYPE_OPTIONS, TIME_RANGE_OPTIONS, DATA_SOURCE_OPTIONS } from '../constants';


interface HeaderProps {
    searchQuery: string;
    onSearchQueryChange: (query: string) => void;
    onSearchSubmit: () => void;
    filters: Filters;
    onFilterChange: (newFilters: Partial<Filters>) => void;
    resultsCount: number;
    isLoading: boolean;
    hasExecutedSearch?: boolean;
}

const Header: React.FC<HeaderProps> = ({ 
    searchQuery, 
    onSearchQueryChange, 
    onSearchSubmit,
    filters, 
    onFilterChange, 
    resultsCount, 
    isLoading,
    hasExecutedSearch = false
}) => {
    


  return (
    <header className="bg-white/80 backdrop-blur-sm px-4 pt-3 pb-2 border-b border-slate-200 sticky top-0 z-10">
      {/* Top row with search and actions */}
      <div className="flex items-center justify-between">
        <div className="flex-grow max-w-3xl">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <SearchIcon />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && onSearchSubmit()}
              placeholder="Type your question and press Enter to search..."
              className="w-full pl-10 pr-24 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-400 focus:border-orange-400 outline-none shadow-sm transition"
            />
            <div className="absolute inset-y-0 right-0 flex items-center">
              <button 
                  onClick={onSearchSubmit}
                  disabled={!searchQuery.trim()}
                  className="mr-1 px-3 py-1 bg-orange-500 text-white text-sm rounded hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
              >
                  Search
              </button>
              <button 
                  onClick={() => onSearchQueryChange("")}
                  className="pr-3 flex items-center text-slate-400 hover:text-slate-600"
                  aria-label="Clear search"
              >
                  &times;
              </button>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4 ml-4">
          <button className="relative text-slate-500 hover:text-slate-800">
            <BellIcon />
            <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">1</span>
          </button>
          <button className="text-slate-500 hover:text-slate-800">
            <PlusIcon />
          </button>
          <button className="text-slate-500 hover:text-slate-800">
            <UserIcon />
          </button>
        </div>
      </div>

      

      {/* Third row with filters */}
      <div className="flex items-center mt-3 space-x-2 text-sm">
        <FilterDropdown
            icon={<FolderIcon />}
            label="File Types"
            disabled={isLoading}
            selectedValues={filters.fileType}
            options={FILE_TYPE_OPTIONS}
            onChange={(values) => onFilterChange({ fileType: values as FileType[] })}
        />
        <TimeFilterDropdown
            icon={<ClockIcon />}
            disabled={isLoading}
            selectedValue={filters.timeRange}
            options={TIME_RANGE_OPTIONS}
            onChange={(value) => onFilterChange({ timeRange: value as TimeRangeFilterValue })}
        />
        <FilterDropdown
            icon={<DataSourceIcon />}
            label="Data Sources"
            disabled={isLoading}
            selectedValues={filters.dataSource}
            options={DATA_SOURCE_OPTIONS}
            onChange={(values) => onFilterChange({ dataSource: values as DataSourceType[] })}
        />
        <div className="flex-grow text-slate-500">
            {isLoading ? 'Searching...' : 
             hasExecutedSearch ? `About ${resultsCount} results` : 
             'Type a question and press Enter to search'}
        </div>
      </div>
    </header>
  );
};

export default Header;