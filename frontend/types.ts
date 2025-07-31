import type React from 'react';

export enum DataSourceType {
  WindowsShare = 'Windows Shares',
  Confluence = 'Confluence',
  SharePoint = 'SharePoint',
  OneDrive = 'OneDrive',
  Website = 'Website',
  Outlook = 'Outlook',
  SharedMailboxes = 'Shared mailboxes',
  Teams = 'Teams',
  OneNote = 'OneNote',
}

export type FileType = 'word' | 'page' | 'pdf' | 'excel' | 'ppt' | 'email' | 'html' | 'txt' | 'compressed';

export type TimeRangeType = 'all' | '3days' | 'week' | 'month' | '3months' | 'year' | '5years' | 'custom';

export interface TimeRangeFilterValue {
  type: TimeRangeType;
  label: string;
  startDate?: Date | null;
  endDate?: Date | null;
}

export interface Filters {
  fileType: FileType[];
  timeRange: TimeRangeFilterValue;
  dataSource: DataSourceType[];
}

export interface DocumentResult {
  id: string;
  sourceType: DataSourceType;
  sourcePath: string;
  sourceIcon: React.ReactNode;
  docTypeIcon: React.ReactNode;
  fileType: FileType;
  title: string;
  date: string; // Keep as DD.MM.YYYY string for parsing
  snippet: string;
  author?: string;
  missingInfo?: string;
  mustInclude?: string;
}

export interface AiReference {
  id: number;
  docId: string;
}

export interface AiAnswerItem {
  title: string;
  text: string;
  references: AiReference[];
}

export interface AiResponse {
  summary: string;
  items: AiAnswerItem[];
}
