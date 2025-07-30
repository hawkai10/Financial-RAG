import { render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { html } from 'htm/preact';
import { GoogleGenAI, Chat } from '@google/genai';

// --- MOCK DATA & CONFIG ---

const API_KEY = process.env.API_KEY;
const ai = new GoogleGenAI({ apiKey: API_KEY });

const mockDocuments = [
  {
    id: 'doc1',
    name: 'Q3_2024_Invoice_ACME.pdf',
    type: 'pdf',
    content: `ACME Corp - Invoice #12345
Date: July 15, 2024
To: John Doe

Item      | Qty | Price
------------------------
Widget A  | 2   | $50.00
Widget B  | 1   | $75.00
------------------------
Subtotal: $175.00
Tax (8%): $14.00
Total:    $189.00

Payment Terms: Payment is due within 30 days of the invoice date. Late payments will incur a 5% fee. Please send payment to the address above. For questions, contact billing@acme.com.
`,
  },
  {
    id: 'doc2',
    name: 'MSA_Global_Inc.docx',
    type: 'docx',
    content: `MASTER SERVICES AGREEMENT
This Master Services Agreement ("MSA") is entered into on August 1, 2024, between Global Inc. ("Client") and ServiceProvider LLC ("Provider").

1. SCOPE OF SERVICES
Provider agrees to perform services as described in individual Statements of Work ("SOW") which shall be incorporated herein by reference.

2. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and continue until terminated by either party with 90 days written notice. Either party may terminate this agreement for cause if the other party is in material breach and fails to cure such breach within 30 days of notice.

3. CONFIDENTIALITY
Both parties agree to maintain the confidentiality of all proprietary information disclosed during the term of this agreement.
`,
  },
];

// This simulates a RAG backend finding the most relevant chunk.
const findRelevantChunk = (query: string) => {
    // A real implementation would involve a vector search backend.
    // Here, we'll just do simple keyword matching for the demo.
    query = query.toLowerCase();
    if (query.includes('invoice') || query.includes('payment') || query.includes('late fee') || query.includes('total')) {
        return {
            document: mockDocuments[0],
            chunk: 'Payment Terms: Payment is due within 30 days of the invoice date. Late payments will incur a 5% fee.',
            answer: "According to 'Q3_2024_Invoice_ACME.pdf', payment is due within 30 days. A 5% fee is applied for late payments."
        };
    }
    if (query.includes('agreement') || query.includes('terminate') || query.includes('msa')) {
        return {
            document: mockDocuments[1],
            chunk: 'This Agreement shall commence on the Effective Date and continue until terminated by either party with 90 days written notice.',
            answer: "The Master Services Agreement with Global Inc. can be terminated by either party with 90 days written notice."
        };
    }
    return null;
};

// --- ICONS ---
const SearchIcon = () => html`<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" /></svg>`;
const SendIcon = () => html`<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" /></svg>`;
const BackIcon = () => html`<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" /></svg>`;


// --- COMPONENTS ---

const Header = ({ onBack, showBackButton }) => html`
  <header class="app-header">
    ${showBackButton && html`
      <button onClick=${onBack} class="back-button" aria-label="Back to search">
        <${BackIcon} />
      </button>
    `}
    <div class="logo">DocuChat AI</div>
  </header>
`;

const SearchView = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handlePillClick = (text) => {
      setQuery(text);
      onSearch(text);
  }

  const recentQueries = [
    'What are the payment terms?',
    'How to terminate the MSA?',
    'What was the invoice total?'
  ];

  return html`
    <div class="search-view">
      <h1>Ask Your Documents</h1>
      <form class="search-box" onSubmit=${handleSubmit}>
        <input
          type="search"
          class="search-input"
          value=${query}
          onInput=${(e) => setQuery(e.currentTarget.value)}
          placeholder="Ask about your invoices, tax documents, receipts…"
          aria-label="Search documents"
        />
        <button type="submit" class="search-button" aria-label="Search">
          <${SearchIcon} />
        </button>
      </form>
      <div class="recent-queries">
        ${recentQueries.map(q => html`
          <button class="query-pill" onClick=${() => handlePillClick(q)}>${q}</button>
        `)}
      </div>
    </div>
  `;
};

const DocumentViewer = ({ result }) => {
    const { document, chunk } = result;

    const getHighlightedContent = () => {
        if (!document.content || !chunk) {
            return html`<p>No content to display.</p>`;
        }
        const parts = document.content.split(chunk);
        return html`
            ${parts[0]}<mark>${chunk}</mark>${parts[1]}
        `;
    };

    return html`
        <div class="document-viewer">
            <div class="document-header">
                <h2>${document.name}</h2>
                <p>Type: ${document.type}</p>
            </div>
            <div class="document-content" aria-live="polite">
                ${getHighlightedContent()}
            </div>
        </div>
    `;
};


const ChatPanel = ({ result, chatHistory, onSendMessage, isLoading }) => {
  const [newMessage, setNewMessage] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (newMessage.trim() && !isLoading) {
      onSendMessage(newMessage.trim());
      setNewMessage('');
    }
  };

  return html`
    <div class="chat-panel">
        <div class="chat-history">
            ${chatHistory.map(msg => html`
                <div class="chat-message ${msg.role}">
                    ${msg.role === 'ai' && msg.isFirst ? html`
                        <strong>Answer from ${result.document.name}</strong>
                    ` : ''}
                    <p>${msg.text}</p>
                </div>
            `)}
            ${isLoading && html`
                <div class="chat-message ai">
                    <div class="loader"><div class="spinner"></div></div>
                </div>
            `}
            <div ref=${chatEndRef}></div>
        </div>
        <form class="chat-input-form" onSubmit=${handleSubmit}>
            <input
                class="chat-input"
                value=${newMessage}
                onInput=${(e) => setNewMessage(e.currentTarget.value)}
                placeholder="Ask a follow-up question..."
                aria-label="Follow-up question"
                disabled=${isLoading}
            />
            <button type="submit" class="chat-send-button" aria-label="Send message" disabled=${isLoading}>
                <${SendIcon} />
            </button>
        </form>
    </div>
  `;
};

const App = () => {
  const [view, setView] = useState('search'); // 'search', 'loading', 'results'
  const [selectedResult, setSelectedResult] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [chat, setChat] = useState(null);

  const handleSearch = (query) => {
    setView('loading');
    setTimeout(() => { // Simulate network latency for RAG search
      const result = findRelevantChunk(query);
      if (result) {
        setSelectedResult(result);
        const initialAiMessage = { role: 'ai', text: result.answer, isFirst: true };
        setChatHistory([initialAiMessage]);
        
        // Initialize a new chat session for follow-ups
        const newChat = ai.chats.create({
            model: 'gemini-2.5-flash',
            config: {
                systemInstruction: `You are a helpful assistant answering questions based on a provided document excerpt. Be concise and stick to the information given in the excerpt. The document is named '${result.document.name}'. The key information is in this excerpt: "${result.chunk}"`
            }
        });
        setChat(newChat);
        
        setView('results');
      } else {
          alert('Sorry, no relevant documents found for that query.');
          setView('search');
      }
    }, 1000);
  };

  const handleSendMessage = async (message) => {
    const userMessage = { role: 'user', text: message };
    setChatHistory(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
        if (!chat) throw new Error("Chat not initialized.");
        
        const response = await chat.sendMessage({ message });

        const aiMessage = { role: 'ai', text: response.text };
        setChatHistory(prev => [...prev, aiMessage]);
    } catch (error) {
        console.error('Error sending message:', error);
        const errorMessage = { role: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
        setChatHistory(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
    }
  };
  
  const handleBackToSearch = () => {
      setView('search');
      setSelectedResult(null);
      setChatHistory([]);
      setChat(null);
  }

  const renderView = () => {
    switch(view) {
        case 'loading':
            return html`<div class="loader"><div class="spinner"></div></div>`;
        case 'results':
            return html`
                <div class="results-view">
                    <${DocumentViewer} result=${selectedResult} />
                    <${ChatPanel} 
                        result=${selectedResult}
                        chatHistory=${chatHistory}
                        onSendMessage=${handleSendMessage}
                        isLoading=${isLoading}
                    />
                </div>
            `;
        case 'search':
        default:
            return html`<${SearchView} onSearch=${handleSearch} />`;
    }
  };

  return html`
    <${Header} onBack=${handleBackToSearch} showBackButton=${view === 'results'} />
    <main>
      ${renderView()}
    </main>
  `;
};

render(html`<${App} />`, document.getElementById('app'));