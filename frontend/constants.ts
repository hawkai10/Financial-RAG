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


export const MOCK_SEARCH_RESULTS: DocumentResult[] = [
  {
    id: 'doc1',
    sourceType: DataSourceType.WindowsShare,
    sourcePath: 'Z: <> Export_US_Legal_Documents',
    sourceIcon: React.createElement(WindowsIcon),
    docTypeIcon: React.createElement(WordIcon),
    fileType: 'word',
    title: 'Export_CNC_Machine_US.docx',
    date: '18.10.2024',
    snippet: 'It is crucial to comply with the applicable regulations to ensure a smooth <strong>export</strong> process and to align with the <strong>standards</strong> set by the relevant authorities. This document aims to outline the essential steps and <strong>requirements</strong> involved in <strong>exporting CNC machines</strong> to the United States. 2.',
    missingInfo: 'USA milling',
    mustInclude: 'USA milling',
  },
  {
    id: 'doc2',
    sourceType: DataSourceType.Confluence,
    sourcePath: 'Bauteile-Beschaffung <>',
    sourceIcon: React.createElement(ConfluenceIcon),
    docTypeIcon: React.createElement(PageIcon),
    fileType: 'page',
    title: 'Recherche zu Standards für CNC-Exporte in die USA',
    author: 'Michael Jackson',
    date: '20.07.2023',
    snippet: '...der <strong>CNC-Fertigung</strong>. Anforderungen und Bedingungen für den Export in die <strong>USA</strong>. Beim Export von <strong>CNC-Maschinen</strong> in die <strong>USA</strong> müssen folgende Anforderungen und Bedingungen erfüllt werden: Elektrische <strong>Standards</strong> - Die Maschinen müssen den geltenden elektrischen Normen wie den NEMA <strong>Standards</strong> entsprechen.',
  },
    {
    id: 'doc3',
    sourceType: DataSourceType.SharePoint,
    sourcePath: 'Safety_Protocols <>',
    sourceIcon: React.createElement(SharePointIcon),
    docTypeIcon: React.createElement(PdfIcon, {className: "w-6 h-6 text-red-600"}),
    fileType: 'pdf',
    title: 'Safety Certifications for Industrial Machinery',
    author: 'Jane Doe',
    date: '05.01.2024',
    snippet: 'Certifications from organizations like Underwriters Laboratories (<strong>UL</strong>) for product safety and the National Voluntary Laboratory Accreditation Program (<strong>NVLAP</strong>) for testing compliance with <strong>NIST</strong> standards are essential for market access in the United States. These ensure that the machinery meets rigorous safety and quality benchmarks.',
  },
  {
    id: 'doc4',
    sourceType: DataSourceType.Outlook,
    sourcePath: 'Inbox <> Inquiry from US distributor',
    sourceIcon: React.createElement(OutlookIcon),
    docTypeIcon: React.createElement(EmailIcon, {className: "w-6 h-6 text-blue-600"}),
    fileType: 'email',
    title: 'RE: Inquiry about CNC machine specs',
    author: 'Sales Team',
    date: '15.06.2024',
    snippet: 'Following up on your request, please find attached the technical specifications and required <strong>certifications</strong> for our new CNC model. Note the NEMA compliance for all electrical components.',
  },
  {
    id: 'doc5',
    sourceType: DataSourceType.Teams,
    sourcePath: 'Engineering Channel <> Design Files',
    sourceIcon: React.createElement(TeamsIcon),
    docTypeIcon: React.createElement(CompressedIcon, {className: "w-6 h-6 text-yellow-500"}),
    fileType: 'compressed',
    title: 'Final_CNC_Blueprints_US.zip',
    author: 'R&D Dept',
    date: '01.03.2023',
    snippet: 'Contains all CAD files and the final BoM for the US export model. See `README.txt` for details on UL certification requirements.',
  },
   {
    id: 'doc6',
    sourceType: DataSourceType.OneDrive,
    sourcePath: 'Personal <> Marketing Docs',
    sourceIcon: React.createElement(OneDriveIcon),
    docTypeIcon: React.createElement(PptIcon, {className: "w-6 h-6 text-orange-600"}),
    fileType: 'ppt',
    title: 'US_Export_Strategy_Q3.pptx',
    author: 'Marketing',
    date: '10.08.2024',
    snippet: 'Presentation outlining the marketing strategy for penetrating the US market. Slide 5 covers the required <strong>OSHA</strong> and <strong>ANSI B11</strong> compliance.',
  },
];

export const MOCK_AI_RESPONSE: AiResponse = {
  summary: 'To export CNC milling machines to the USA, compliance with several standards and certifications is required:',
  items: [
    {
      title: 'Standards',
      text: 'Machines must adhere to US-specific standards such as NEMA (electrical wiring and safety), ANSI B11 (safety requirements for CNC machines), OSHA (workplace safety), and NIST (precision and quality in CNC manufacturing).',
      references: [
        { id: 1, docId: 'doc1' },
        { id: 2, docId: 'doc2' },
        { id: 3, docId: 'doc6' },
      ],
    },
    {
      title: 'Certifications',
      text: 'Certifications from organizations like Underwriters Laboratories (UL) for product safety and the National Voluntary Laboratory Accreditation Program (NVLAP) for testing compliance with NIST standards are essential.',
      references: [
        { id: 4, docId: 'doc3' },
        { id: 5, docId: 'doc4' },
        { id: 6, docId: 'doc5' },
      ],
    },
    {
      title: 'Export Documentation',
      text: 'Key documents include a Bill of Lading, Commercial Invoice, Packing List, Shipping Marks, Export License, and VAT/Customs documentation to ensure smooth customs clearance.',
      references: [{ id: 7, docId: 'doc1' }],
    },
    {
      title: 'Environmental and Safety Compliance',
      text: 'Machines must also comply with EPA regulations for environmental impact and OSHA standards for workplace safety. Proper labeling and documentation are critical.',
      references: [{ id: 8, docId: 'doc3' }],
    },
  ],
};