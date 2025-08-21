import React from 'react';
import type { DocumentResult, AiResponse, FileType } from './types';
import { DataSourceType, TimeRangeType } from './types';
import { WindowsIcon } from './components/icons/WindowsIcon';
import { ConfluenceIcon } from './components/icons/ConfluenceIcon';
import { WordIcon } from './components/icons/WordIcon';
import { PageIcon } from './components/icons/PageIcon';
import { PdfIcon } from './components/icons/PdfIcon';
import { ExcelIcon } from './components/icons/ExcelIcon';
import { PptIcon } from './components/icons/PptIcon';
import { EmailIcon } from './components/icons/EmailIcon';
import { HtmlIcon } from './components/icons/HtmlIcon';
import { TxtIcon } from './components/icons/TxtIcon';
import { CompressedIcon } from './components/icons/CompressedIcon';
import { SharePointIcon } from './components/icons/SharePointIcon';
import { OneDriveIcon } from './components/icons/OneDriveIcon';
import { WebsiteIcon } from './components/icons/WebsiteIcon';
import { OutlookIcon } from './components/icons/OutlookIcon';
import { SharedMailboxIcon } from './components/icons/SharedMailboxIcon';
import { TeamsIcon } from './components/icons/TeamsIcon';
import { OneNoteIcon } from './components/icons/OneNoteIcon';


export const FILE_TYPE_OPTIONS: { value: FileType; label: string; icon: React.ReactNode }[] = [
    { value: 'pdf', label: 'PDF', icon: React.createElement(PdfIcon) },
    { value: 'excel', label: 'Excel', icon: React.createElement(ExcelIcon) },
    { value: 'word', label: 'Word', icon: React.createElement(WordIcon, { className: "w-5 h-5" }) },
    { value: 'ppt', label: 'PowerPoint', icon: React.createElement(PptIcon) },
    { value: 'email', label: 'E-mail', icon: React.createElement(EmailIcon) },
    { value: 'html', label: 'HTML', icon: React.createElement(HtmlIcon) },
    { value: 'txt', label: 'Text', icon: React.createElement(TxtIcon) },
    { value: 'compressed', label: 'Compressed', icon: React.createElement(CompressedIcon) },
    { value: 'page', label: 'Confluence Page', icon: React.createElement(PageIcon, { className: "w-5 h-5 text-blue-500" }) },
];

export const TIME_RANGE_OPTIONS: { value: TimeRangeType, label: string }[] = [
    { value: 'all', label: 'All Time' },
    { value: '3days', label: 'Last three days' },
    { value: 'week', label: 'Last week' },
    { value: 'month', label: 'Last month' },
    { value: '3months', label: 'Last three months' },
    { value: 'year', label: 'Last year' },
    { value: '5years', label: 'Last five years' },
    { value: 'custom', label: 'Set period...' },
];

export const DATA_SOURCE_OPTIONS: { value: DataSourceType; label: string; icon: React.ReactNode }[] = [
    { value: DataSourceType.WindowsShare, label: 'Windows Shares', icon: React.createElement(WindowsIcon) },
    { value: DataSourceType.SharePoint, label: 'SharePoint', icon: React.createElement(SharePointIcon) },
    { value: DataSourceType.OneDrive, label: 'OneDrive', icon: React.createElement(OneDriveIcon) },
    { value: DataSourceType.Website, label: 'Website', icon: React.createElement(WebsiteIcon) },
    { value: DataSourceType.Outlook, label: 'Outlook', icon: React.createElement(OutlookIcon) },
    { value: DataSourceType.SharedMailboxes, label: 'Shared mailboxes', icon: React.createElement(SharedMailboxIcon) },
    { value: DataSourceType.Teams, label: 'Teams', icon: React.createElement(TeamsIcon) },
    { value: DataSourceType.OneNote, label: 'OneNote', icon: React.createElement(OneNoteIcon) },
    { value: DataSourceType.Confluence, label: 'Confluence', icon: React.createElement(ConfluenceIcon) },
];