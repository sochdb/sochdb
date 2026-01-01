import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { open } from '@tauri-apps/plugin-dialog';
import './App.css';

// SVG Icons component with new 'clean' style
const Icon = ({ name, size = 16, className = "" }: { name: string, size?: number, className?: string }) => {
  const icons: Record<string, React.ReactNode> = {
    // UI Icons
    'x': <path d="M18 6L6 18M6 6l12 12" />,
    'maximize': <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />,
    'minimize': <path d="M5 12h14" />,
    'folder': <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />,
    'refresh': <path d="M23 4v6h-6M1 20v-6h6" />,
    'settings': <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />,
    'search': <><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></>,
    'zap': <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />,
    'database': <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></>,
    'play': <polygon points="5 3 19 12 5 21 5 3" />,
    'copy': <><rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></>,
    'check': <polyline points="20 6 9 17 4 12" />,
    'arrow-right': <><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></>,
    'chevron-down': <polyline points="6 9 12 15 18 9" />,
    'terminal': <><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></>,
    'layout': <><rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><line x1="3" y1="9" x2="21" y2="9" /><line x1="9" y1="21" x2="9" y2="9" /></>,
    'moon': <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />,
    'sun': <><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></>,
    'message': <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />,
    'bug': <><rect x="8" y="6" width="8" height="12" rx="4" /><path d="M12 2v4" /><line x1="12" y1="2" x2="12" y2="6" /><line x1="20" y1="10" x2="16" y2="10" /><line x1="4" y1="10" x2="8" y2="10" /><line x1="20" y1="14" x2="16" y2="14" /><line x1="4" y1="14" x2="8" y2="14" /><line x1="21" y1="7" x2="17.6" y2="8.9" /><line x1="3" y1="7" x2="6.4" y2="8.9" /></>,
  };
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      {icons[name] || <circle cx="12" cy="12" r="10" />}
    </svg>
  );
};

const ToonIcon = ({ size = 24 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="2" y="2" width="20" height="20" rx="6" fill="url(#grad1)" />
    <path d="M7 8H17" stroke="white" strokeWidth="2" strokeLinecap="round" />
    <path d="M7 12H17" stroke="white" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
    <path d="M7 16H12" stroke="white" strokeWidth="2" strokeLinecap="round" opacity="0.3" />
    <defs>
      <linearGradient id="grad1" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#13bbaf" />
        <stop offset="1" stopColor="#5c98f9" />
      </linearGradient>
    </defs>
  </svg>
);

// --- Components ---

const TitleBar = () => {
  return (
    <div className="h-10 flex items-center justify-between select-none z-50 px-3 pt-3 app-region-drag" data-tauri-drag-region>
      <div className="flex items-center gap-3 pointer-events-none">
        <ToonIcon size={20} />
        <span className="text-sm font-medium text-text-muted tracking-wide">ToonDB Studio</span>
        <span className="bg-background-muted text-text-muted text-[10px] px-2 py-0.5 rounded-full border border-border-default">BETA</span>
      </div>

      <div className="flex items-center gap-2">
        <div className="flex bg-background-muted rounded-lg p-1 border border-border-default backdrop-blur-sm">
          <button
            onClick={() => getCurrentWindow().minimize()}
            className="p-1 hover:bg-background-medium rounded text-text-muted hover:text-text-default transition-colors"
          >
            <Icon name="minimize" size={14} />
          </button>
          <button
            onClick={() => getCurrentWindow().toggleMaximize()}
            className="p-1 hover:bg-background-medium rounded text-text-muted hover:text-text-default transition-colors"
          >
            <Icon name="maximize" size={14} />
          </button>
          <button
            onClick={() => getCurrentWindow().close()}
            className="p-1 hover:bg-red-500/20 rounded text-text-muted hover:text-red-400 transition-colors"
          >
            <Icon name="x" size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Sidebar Component ---
const Sidebar = ({ activeTab, setActiveTab, connected }: { activeTab: string, setActiveTab: (tab: string) => void, connected: boolean }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Overview', icon: 'layout' },
    { id: 'query', label: 'Query Engine', icon: 'play' },
    { id: 'explorer', label: 'KV Explorer', icon: 'database' },
    { id: 'assistant', label: 'AI Assistant', icon: 'zap' },
    { id: 'console', label: 'Console', icon: 'terminal' },
    { id: 'settings', label: 'Settings', icon: 'settings' },
  ];

  return (
    <div className="w-64 flex flex-col h-full z-20 p-3 pt-6">
      {/* Status Card */}
      <div className={`p-4 rounded-xl border mb-6 transition-all duration-300 ${connected
        ? 'bg-background-muted border-teal/30'
        : 'bg-background-muted border-red-500/30'
        }`}>
        <div className="flex items-center gap-3 mb-2">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-teal' : 'bg-red-500'}`} />
            {connected && <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-teal animate-ping opacity-75" />}
          </div>
          <span className={`text-xs font-semibold uppercase tracking-wider ${connected ? 'text-teal' : 'text-red-400'}`}>
            {connected ? 'System Online' : 'Disconnected'}
          </span>
        </div>
        <div className="text-[10px] text-text-muted font-mono truncate">
          0.1.0
        </div>
      </div>

      <nav className="flex-1 space-y-1">
        {menuItems.map((item, index) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`sidebar-item w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group relative ${activeTab === item.id
              ? 'text-text-default bg-background-muted shadow-sm'
              : 'text-text-muted hover:text-text-default hover:bg-background-muted/50'
              }`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {activeTab === item.id && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-teal rounded-r-sm" />
            )}
            <Icon
              name={item.icon}
              size={18}
              className={`transition-colors ${activeTab === item.id ? 'text-teal' : 'text-text-muted group-hover:text-text-default'}`}
            />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="mt-auto px-2 py-4 border-t border-border-default">
        <p className="text-[10px] text-center text-text-muted opacity-50">ToonDB Studio © 2024</p>
      </div>
    </div>
  );
};

// ... Stat Card ...
const StatCard = ({ icon, label, value, sub, color }: any) => {
  const colorMap: Record<string, string> = {
    teal: 'text-teal bg-teal/10 border-teal/20',
    orange: 'text-orange bg-orange/10 border-orange/20',
    blue: 'text-text-info bg-background-info/10 border-border-info',
    red: 'text-text-danger bg-background-danger/10 border-border-danger',
  };

  const theme = colorMap[color] || colorMap['teal'];

  return (
    <div className="p-5 rounded-2xl glass-panel glass-panel-hover transition-all duration-300 group">
      <div className="flex justify-between items-start mb-4">
        <div className={`p-2.5 rounded-xl border ${theme}`}>
          <Icon name={icon} size={20} />
        </div>
        <span className="text-xs font-mono text-text-muted bg-background-muted px-2 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
          2s ago
        </span>
      </div>
      <div>
        <h3 className="text-2xl font-bold text-text-default mb-1 tracking-tight">{value}</h3>
        <p className="text-sm text-text-muted font-medium">{label}</p>
        <p className="text-[10px] text-text-muted mt-2 opacity-60">{sub}</p>
      </div>
    </div>
  );
};


// --- LLM Settings Tab Component ---
const LlmSettingsTab = () => {
  const [provider, setProvider] = useState('openai');
  const [apiKey, setApiKey] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [model, setModel] = useState('gpt-4o-mini');
  const [azureApiVersion, setAzureApiVersion] = useState('2024-02-01');
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  // Load config on mount
  useEffect(() => {
    invoke('get_llm_config').then((config: any) => {
      if (config) {
        setProvider(config.provider || 'openai');
        setApiKey(config.api_key || '');
        setEndpoint(config.endpoint || '');
        setModel(config.model || 'gpt-4o-mini');
        setAzureApiVersion(config.azure_api_version || '2024-02-01');
      }
    }).catch(() => { });
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await invoke('save_llm_config', {
        config: {
          provider,
          api_key: apiKey,
          endpoint: endpoint || null,
          model,
          azure_api_version: provider === 'azure' ? azureApiVersion : null,
        }
      });
      setTestResult('✓ Configuration saved');
    } catch (e: any) {
      setTestResult(`Error: ${e?.message || e}`);
    }
    setSaving(false);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      // Save first, then test
      await invoke('save_llm_config', {
        config: {
          provider,
          api_key: apiKey,
          endpoint: endpoint || null,
          model,
          azure_api_version: provider === 'azure' ? azureApiVersion : null,
        }
      });
      const result: string = await invoke('test_llm_connection');
      setTestResult(result);
    } catch (e: any) {
      setTestResult(`Error: ${e?.message || e}`);
    }
    setTesting(false);
  };

  const handleClear = async () => {
    try {
      await invoke('clear_llm_config');
      setProvider('openai');
      setApiKey('');
      setEndpoint('');
      setModel('gpt-4o-mini');
      setTestResult('Configuration cleared');
    } catch (e: any) {
      setTestResult(`Error: ${e?.message || e}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="p-6 rounded-xl border border-border-default bg-background-muted/30">
        <h3 className="text-lg font-semibold text-text-default mb-2">LLM Configuration</h3>
        <p className="text-sm text-text-muted mb-6">Configure AI assistant with OpenAI or compatible APIs.</p>

        {/* Provider Selection */}
        <div className="mb-4">
          <label className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 block">Provider</label>
          <div className="flex gap-2">
            {['openai', 'azure', 'custom'].map((p) => (
              <button
                key={p}
                onClick={() => setProvider(p)}
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${provider === p
                  ? 'bg-teal/10 border-teal text-teal'
                  : 'bg-background-muted border-border-default text-text-muted hover:text-text-default'
                  }`}
              >
                {p === 'openai' ? 'OpenAI' : p === 'azure' ? 'Azure OpenAI' : 'Custom'}
              </button>
            ))}
          </div>
        </div>

        {/* API Key */}
        <div className="mb-4">
          <label className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 block">API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={provider === 'azure' ? 'Azure API Key' : 'sk-...'}
            className="w-full bg-background-app border border-border-default rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none transition-all placeholder:text-text-muted/50"
          />
        </div>

        {/* API Endpoint (optional - for proxies, compatible APIs, etc) */}
        <div className="mb-4">
          <label className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 block">
            {provider === 'azure' ? 'Azure Endpoint' : 'API Endpoint'}
            {provider === 'openai' && <span className="text-text-muted/60 font-normal ml-1">(optional)</span>}
          </label>
          <input
            type="text"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            placeholder={
              provider === 'azure'
                ? 'https://your-resource.openai.azure.com'
                : provider === 'openai'
                  ? 'Leave blank for api.openai.com, or enter custom URL'
                  : 'http://localhost:11434/v1'
            }
            className="w-full bg-background-app border border-border-default rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none transition-all placeholder:text-text-muted/50"
          />
          {provider === 'openai' && (
            <p className="text-xs text-text-muted mt-1">Leave empty for default OpenAI, or enter a proxy/compatible API URL</p>
          )}
        </div>

        {/* Azure API Version */}
        {provider === 'azure' && (
          <div className="mb-4">
            <label className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 block">API Version</label>
            <input
              type="text"
              value={azureApiVersion}
              onChange={(e) => setAzureApiVersion(e.target.value)}
              placeholder="2024-02-01"
              className="w-full bg-background-app border border-border-default rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none transition-all placeholder:text-text-muted/50"
            />
          </div>
        )}

        {/* Model */}
        <div className="mb-6">
          <label className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 block">Model</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="e.g., gpt-4o-mini, llama3.2, claude-3-sonnet..."
            className="w-full bg-background-app border border-border-default rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none placeholder:text-text-muted/50"
          />
          <p className="text-xs text-text-muted mt-1">Enter the model name as expected by your API endpoint</p>
        </div>

        {/* Test Result */}
        {testResult && (
          <div className={`mb-4 p-3 rounded-lg text-sm font-mono ${testResult.startsWith('✓') ? 'bg-teal/10 text-teal' : 'bg-red-500/10 text-red-400'
            }`}>
            {testResult}
          </div>
        )}

        {/* Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleTest}
            disabled={!apiKey || testing}
            className="px-4 py-2 bg-background-muted border border-border-default hover:bg-background-medium rounded-lg text-sm font-medium text-text-default transition-colors disabled:opacity-50"
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button
            onClick={handleSave}
            disabled={!apiKey || saving}
            className="px-4 py-2 bg-teal hover:bg-teal/90 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
          <button
            onClick={handleClear}
            className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg text-sm font-medium transition-colors"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
};


// --- AI Assistant View Component ---
const AssistantView = () => {
  const [messages, setMessages] = useState<{ role: string, content: string, toolResults?: { tool_name: string, result: string }[] }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isConfigured, setIsConfigured] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check if LLM is configured on mount
  useEffect(() => {
    invoke('get_llm_config').then((config: any) => {
      setIsConfigured(config && config.api_key && config.api_key.length > 0);
    }).catch(() => {
      setIsConfigured(false);
    });
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setError(null);
    setLoading(true);

    // Add user message
    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);

    try {
      // Build messages for API (include system prompt)
      const apiMessages = [
        {
          role: 'system',
          content: `You are a helpful AI assistant with access to ToonDB - an AI-native embedded database.
You can use the available tools to query the database, get/put/delete data, and help users manage their data.
When the user asks about data, use the appropriate tools. Always be helpful and explain what you're doing.`,
        },
        ...newMessages.map(m => ({ role: m.role, content: m.content || '' })),
      ];

      const response: any = await invoke('chat_completion', { messages: apiMessages });

      // Add assistant response
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message.content || '',
        toolResults: response.tool_results,
      }]);

    } catch (e: any) {
      console.error('Chat error:', e);
      const errorMsg = typeof e === 'string' ? e : (e?.message || e?.toString() || JSON.stringify(e) || 'Failed to send message');
      setError(errorMsg);
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full animate-in p-8">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-text-default mb-2">AI Assistant</h1>
        <p className="text-text-muted">Chat with AI to query and manage your ToonDB data</p>
      </header>

      <div className="flex-1 glass-panel rounded-2xl flex flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 p-4 overflow-auto space-y-4">
          {messages.length === 0 ? (
            isConfigured === false ? (
              <div className="h-full flex flex-col items-center justify-center text-text-muted">
                <Icon name="settings" size={48} className="mb-4 opacity-50" />
                <p className="text-lg font-semibold text-text-default mb-2">LLM Not Configured</p>
                <p className="text-sm mb-4 text-center max-w-md">
                  To use the AI Assistant, please configure your LLM API key in Settings.
                </p>
                <p className="text-xs text-text-muted mb-4">Settings → LLM → Enter API Key → Save</p>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-text-muted opacity-50">
                <Icon name="zap" size={48} className="mb-4" />
                <p>Ask me anything about your data!</p>
                <p className="text-sm mt-2">Try: "List all tables" or "Show me data in users/"</p>
              </div>
            )
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-xl px-4 py-3 ${msg.role === 'user'
                  ? 'bg-teal text-white'
                  : 'bg-background-muted border border-border-default text-text-default'
                  }`}>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.toolResults && msg.toolResults.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/20">
                      <p className="text-xs opacity-70 mb-2">🔧 Tools Used:</p>
                      {msg.toolResults.map((tr, j) => (
                        <div key={j} className="text-xs bg-black/20 rounded px-2 py-1 mt-1 font-mono">
                          <span className="text-teal">{tr.tool_name}</span>
                          <pre className="mt-1 whitespace-pre-wrap opacity-80 max-h-32 overflow-auto">{tr.result.slice(0, 500)}{tr.result.length > 500 ? '...' : ''}</pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-background-muted border border-border-default rounded-xl px-4 py-3">
                <div className="flex items-center gap-2 text-text-muted text-sm">
                  <div className="w-2 h-2 bg-teal rounded-full animate-pulse" />
                  Thinking...
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Error */}
        {error && (
          <div className="px-4 py-2 bg-red-500/10 border-t border-red-500/20">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Input */}
        <div className="border-t border-border-default bg-background-muted/30 p-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask about your data..."
              className="flex-1 bg-background-app border border-border-default rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none transition-all placeholder:text-text-muted/50"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              className="px-6 py-2.5 bg-teal hover:bg-teal/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};


// --- Settings View with Working Tabs and Theme Toggle ---
const SettingsView = ({ theme, setTheme, connected }: { theme: string, setTheme: (t: string) => void, connected: boolean }) => {
  const [copied, setCopied] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState('mcp');

  const mcpConfig = {
    mcpServers: {
      toondb: {
        command: "/path/to/toondb-mcp",
        args: ["--db", "/path/to/data"]
      }
    }
  };

  const copyConfig = async () => {
    await invoke('get_mcp_config').then((c: any) => {
      navigator.clipboard.writeText(JSON.stringify(c, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      navigator.clipboard.writeText(JSON.stringify(mcpConfig, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const tabs = ['General', 'Connection', 'LLM', 'MCP', 'About'];

  return (
    <div className="flex flex-col h-full animate-in p-8 overflow-y-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-text-default mb-2">Settings</h1>
        <p className="text-text-muted">Application configuration and integrations</p>
      </header>

      {/* Tabs */}
      <div className="flex gap-6 border-b border-border-default mb-8">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveSettingsTab(tab.toLowerCase())}
            className={`pb-3 text-sm font-medium transition-colors border-b-2 ${activeSettingsTab === tab.toLowerCase()
              ? 'text-teal border-teal'
              : 'text-text-muted border-transparent hover:text-text-default'
              }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="space-y-6 max-w-4xl flex-1">
        {/* General Tab */}
        {activeSettingsTab === 'general' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl border border-border-default bg-background-muted/30">
              <h3 className="text-lg font-semibold text-text-default mb-2">Theme</h3>
              <p className="text-sm text-text-muted mb-4">Customize the look and feel of ToonDB Studio</p>
              <div className="flex gap-2">
                <button
                  onClick={() => setTheme('light')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${theme === 'light'
                    ? 'bg-background-app text-text-default border-teal'
                    : 'bg-background-muted text-text-muted border-border-default hover:border-text-muted'
                    }`}
                >
                  <Icon name="sun" size={16} /> Light
                </button>
                <button
                  onClick={() => setTheme('dark')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${theme === 'dark'
                    ? 'bg-background-app text-text-default border-teal'
                    : 'bg-background-muted text-text-muted border-border-default hover:border-text-muted'
                    }`}
                >
                  <Icon name="moon" size={16} /> Dark
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Connection Tab */}
        {activeSettingsTab === 'connection' && (
          <div className="p-6 rounded-xl border border-border-default bg-background-muted/30">
            <h3 className="text-lg font-semibold text-text-default mb-2">Connection Status</h3>
            <div className="flex items-center gap-3 mt-4">
              <div className={`w-3 h-3 rounded-full ${connected ? 'bg-teal' : 'bg-red-500'}`} />
              <span className="text-text-default">{connected ? 'Connected to database' : 'Not connected'}</span>
            </div>
            <p className="text-sm text-text-muted mt-4">
              {connected
                ? 'Your database is active and ready to receive queries.'
                : 'Connect to a database using the connection modal.'}
            </p>
          </div>
        )}

        {/* LLM Tab */}
        {activeSettingsTab === 'llm' && (
          <LlmSettingsTab />
        )}

        {/* MCP Tab */}
        {activeSettingsTab === 'mcp' && (
          <div className="rounded-xl border border-border-default bg-background-muted/30 overflow-hidden">
            <div className="p-6 border-b border-border-default flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-text-default mb-1">MCP Configuration</h3>
                <p className="text-sm text-text-muted">Connect ToonDB to Claude, Cursor, or other AI agents.</p>
              </div>
              <button
                onClick={copyConfig}
                className="flex items-center gap-2 px-3 py-1.5 bg-teal hover:bg-teal/90 text-white text-xs font-medium rounded-md transition-all"
              >
                {copied ? <Icon name="check" size={14} /> : <Icon name="copy" size={14} />}
                {copied ? 'Copied!' : 'Copy Config'}
              </button>
            </div>
            <div className="p-0 bg-background-app">
              <pre className="p-6 text-xs font-mono text-text-muted overflow-x-auto">
                {JSON.stringify(mcpConfig, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* About Tab */}
        {activeSettingsTab === 'about' && (
          <div className="p-6 rounded-xl border border-border-default bg-background-muted/30">
            <h3 className="text-lg font-semibold text-text-default mb-4">About ToonDB Studio</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between"><span className="text-text-muted">Version</span><span className="text-text-default font-mono">0.1.0-alpha</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Engine</span><span className="text-text-default font-mono">ToonDB Core</span></div>
              <div className="flex justify-between"><span className="text-text-muted">Build</span><span className="text-text-default font-mono">2024.12.08</span></div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="pt-8 border-t border-border-default flex justify-between text-xs text-text-muted mt-auto">
        <span>Version 0.1.0</span>
        <span className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-teal' : 'bg-red-500'}`} />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </div>
  );
};


// ... Connection Modal ...
const ConnectionModal = ({ onConnect }: any) => {
  const [dbPath, setDbPath] = useState('');

  async function pickDir() {
    try {
      const selected = await open({ directory: true });
      if (selected) setDbPath(selected as string);
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-in">
      <div className="bg-background-default border border-border-default rounded-2xl w-full max-w-4xl flex overflow-hidden shadow-2xl h-[500px]">
        {/* Sidebar */}
        <div className="w-72 bg-background-muted/50 border-r border-border-default p-6 flex flex-col gap-2">
          <h2 className="text-lg font-bold text-text-default mb-6 flex items-center gap-2">
            <ToonIcon size={24} /> ToonDB
          </h2>
          <button className="flex items-center gap-3 px-4 py-3 rounded-xl bg-background-app border border-border-accent text-text-default shadow-sm">
            <Icon name="database" size={18} className="text-teal" />
            <div className="text-left">
              <div className="text-sm font-semibold">Local Storage</div>
              <div className="text-[10px] text-text-muted">Embedded LSM Engine</div>
            </div>
          </button>
          <button className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-background-muted text-text-muted transition-colors opacity-50 cursor-not-allowed">
            <Icon name="zap" size={18} />
            <div className="text-left">
              <div className="text-sm font-semibold">Remote Server</div>
              <div className="text-[10px] text-text-muted">Connect via TCP/HTTP</div>
            </div>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 p-8 flex flex-col bg-mesh relative">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-text-default mb-2">Connect to Database</h1>
            <p className="text-text-muted">Select a directory to initialize or load a ToonDB instance.</p>
          </div>

          <div className="space-y-4 max-w-lg">
            <label className="text-xs font-semibold text-text-muted uppercase tracking-wider">Data Directory</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={dbPath}
                onChange={(e) => setDbPath(e.target.value)}
                placeholder="/path/to/data"
                className="flex-1 bg-background-input border border-border-input rounded-lg px-4 py-2.5 text-text-default text-sm focus:ring-2 focus:ring-teal/50 outline-none transition-all placeholder:text-text-muted/50"
              />
              <button onClick={pickDir} className="px-4 py-2 bg-background-muted border border-border-default hover:bg-background-medium rounded-lg text-text-muted hover:text-text-default transition-colors">
                <Icon name="folder" size={18} />
              </button>
            </div>
          </div>

          <div className="mt-auto flex justify-end">
            <button
              onClick={() => onConnect(dbPath)}
              disabled={!dbPath}
              className="flex items-center gap-2 px-8 py-3 bg-teal hover:bg-teal/90 text-white rounded-lg font-semibold shadow-lg shadow-teal/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Connect <Icon name="arrow-right" size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};


function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<{ columns: string[], rows: any[][] } | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  const [showConnectionModal, setShowConnectionModal] = useState(true);
  const [theme, setTheme] = useState('dark');

  // Console state
  const [consoleInput, setConsoleInput] = useState('');
  const [consoleHistory, setConsoleHistory] = useState<{ type: 'input' | 'output' | 'error', text: string }[]>([
    { type: 'output', text: 'Welcome to ToonDB Console. Type \'help\' for available commands.' }
  ]);

  // KV Explorer state
  const [kvKeys, setKvKeys] = useState<string[]>([]);
  const [kvLoading, setKvLoading] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [selectedValue, setSelectedValue] = useState<string | null>(null);
  const [kvSearch, setKvSearch] = useState('');
  const [selectedTable, setSelectedTable] = useState<string>('');

  // Schema state
  const [tables, setTables] = useState<string[]>([]);

  // Auto-connect effect
  useEffect(() => {
    const lastPath = localStorage.getItem("last_db_path");
    if (lastPath) {
      handleConnect(lastPath);
    } else {
      setShowConnectionModal(true);
    }
  }, []);

  // Poll stats
  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(async () => {
      try {
        const s = await invoke('get_stats');
        setStatus(s);
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(interval);
  }, [connected]);

  const handleConnect = async (path: string) => {
    try {
      await invoke('connect', { path });
      setConnected(true);
      setShowConnectionModal(false);
      localStorage.setItem("last_db_path", path);
      setActiveTab('dashboard');
      // Load initial data
      loadTables();
    } catch (e) {
      console.error("Connection failed", e);
      setShowConnectionModal(true);
    }
  };

  // Load tables from database
  const loadTables = async () => {
    try {
      const res: any = await invoke('mcp_call_tool', {
        toolName: 'toondb_list_tables',
        arguments: {}
      });
      if (res && res.content) {
        // Parse table list from MCP response (toon format)
        // Format: "results[N]{header}:\nrow1\nrow2\n..."
        const content = res.content[0]?.text || '';
        const lines = content.split('\n').filter((t: string) => t.trim());

        // Skip the header line (starts with "results[")
        const dataLines = lines.filter((line: string) => !line.startsWith('results['));

        // Extract table names (first field before comma)
        const tableNames = dataLines.map((line: string) => {
          const parts = line.split(',');
          return parts[0]?.trim() || '';
        }).filter((name: string) => name && name !== 'null');

        setTables(tableNames);
      }
    } catch (e) {
      console.error('Failed to load tables', e);
    }
  };

  // Create sample data
  const createSampleData = async () => {
    try {
      // Create users table with sample data
      await invoke('mcp_call_tool', {
        toolName: 'toondb_put',
        arguments: { path: 'users/1', value: { id: 1, name: 'Alice', email: 'alice@example.com', role: 'admin' } }
      });
      await invoke('mcp_call_tool', {
        toolName: 'toondb_put',
        arguments: { path: 'users/2', value: { id: 2, name: 'Bob', email: 'bob@example.com', role: 'user' } }
      });
      await invoke('mcp_call_tool', {
        toolName: 'toondb_put',
        arguments: { path: 'users/3', value: { id: 3, name: 'Carol', email: 'carol@example.com', role: 'user' } }
      });
      // Create products table
      await invoke('mcp_call_tool', {
        toolName: 'toondb_put',
        arguments: { path: 'products/1', value: { id: 1, name: 'Widget', price: 29.99, stock: 100 } }
      });
      await invoke('mcp_call_tool', {
        toolName: 'toondb_put',
        arguments: { path: 'products/2', value: { id: 2, name: 'Gadget', price: 49.99, stock: 50 } }
      });
      setConsoleHistory(prev => [...prev,
      { type: 'output', text: '✓ Created sample data: users/1, users/2, users/3, products/1, products/2' }
      ]);
      loadTables();
      refreshKvExplorer();
    } catch (e: any) {
      setConsoleHistory(prev => [...prev,
      { type: 'error', text: `Failed to create sample data: ${e?.message || e}` }
      ]);
    }
  };

  // Refresh KV Explorer - scan keys for the selected table
  const refreshKvExplorer = async (tableToScan?: string) => {
    const table = tableToScan || selectedTable;
    if (!table) {
      setKvKeys([]);
      return;
    }

    setKvLoading(true);
    setKvKeys([]);
    setSelectedKey(null);
    setSelectedValue(null);

    try {
      const scanRes: any = await invoke('mcp_call_tool', {
        toolName: 'toondb_query',
        arguments: { query: `SELECT * FROM ${table}`, limit: 50, format: 'json' }
      });

      let keys: string[] = [];

      if (scanRes && scanRes.content) {
        const scanContent = scanRes.content[0]?.text || '';

        // Try parsing as JSON first
        try {
          const jsonData = JSON.parse(scanContent);
          if (Array.isArray(jsonData)) {
            for (const item of jsonData) {
              if (item._path) {
                keys.push(item._path);
              }
            }
          }
        } catch {
          // Fallback: parse toon format
          const scanLines = scanContent.split('\n').filter((t: string) => t.trim());
          for (const line of scanLines) {
            if (line.startsWith('results[')) continue;
            const firstField = line.split(',')[0]?.trim();
            if (firstField && firstField.startsWith('/')) {
              keys.push(firstField);
            } else if (firstField && /^[a-zA-Z0-9_-]+$/.test(firstField)) {
              keys.push(`/${table}/${firstField}`);
            }
          }
        }
      }

      setKvKeys(keys.length > 0 ? keys : [`No records in ${table}`]);
    } catch (e) {
      setKvKeys([`Error scanning ${table}`]);
    }
    setKvLoading(false);
  };

  // Get value for a key
  const getKeyValue = async (key: string) => {
    setSelectedKey(key);
    try {
      const res: any = await invoke('mcp_call_tool', {
        toolName: 'toondb_get',
        arguments: { path: key }
      });
      if (res && res.content) {
        setSelectedValue(res.content[0]?.text || JSON.stringify(res, null, 2));
      }
    } catch (e: any) {
      setSelectedValue(`Error: ${e?.message || e}`);
    }
  };

  // Console command execution
  const executeConsoleCommand = async () => {
    if (!consoleInput.trim()) return;

    const cmd = consoleInput.trim();
    setConsoleHistory(prev => [...prev, { type: 'input', text: `toondb> ${cmd}` }]);
    setConsoleInput('');

    // Parse and execute command
    const parts = cmd.split(/\s+/);
    const command = parts[0].toLowerCase();

    try {
      let result = '';

      switch (command) {
        case 'help':
          result = `Available commands:
  help              - Show this help
  list              - List all tables
  get <path>        - Get value at path
  put <path> <json> - Store value at path
  delete <path>     - Delete value at path
  describe <table>  - Show table schema
  query <sql>       - Execute a query
  create-sample     - Create sample data
  clear             - Clear console`;
          break;

        case 'clear':
          setConsoleHistory([{ type: 'output', text: 'Console cleared.' }]);
          return;

        case 'list':
          const listRes: any = await invoke('mcp_call_tool', {
            toolName: 'toondb_list_tables',
            arguments: {}
          });
          const rawListOutput = listRes?.content?.[0]?.text || '';
          // Parse toon format and format nicely
          const listLines = rawListOutput.split('\n').filter((t: string) => t.trim());
          const listDataLines = listLines.filter((line: string) => !line.startsWith('results['));
          if (listDataLines.length > 0) {
            result = 'Tables:\n' + listDataLines.map((line: string) => {
              const tableName = line.split(',')[0]?.trim();
              return `  • ${tableName}`;
            }).join('\n');
          } else {
            result = 'No tables found. Try: create-sample';
          }
          break;

        case 'get':
          if (parts.length < 2) {
            result = 'Usage: get <path>';
          } else {
            const getRes: any = await invoke('mcp_call_tool', {
              toolName: 'toondb_get',
              arguments: { path: parts[1] }
            });
            result = getRes?.content?.[0]?.text || 'Not found';
          }
          break;

        case 'put':
          if (parts.length < 3) {
            result = 'Usage: put <path> <json>';
          } else {
            const path = parts[1];
            const valueStr = parts.slice(2).join(' ');
            const value = JSON.parse(valueStr);
            await invoke('mcp_call_tool', {
              toolName: 'toondb_put',
              arguments: { path, value }
            });
            result = `✓ Stored at ${path}`;
          }
          break;

        case 'delete':
          if (parts.length < 2) {
            result = 'Usage: delete <path>';
          } else {
            await invoke('mcp_call_tool', {
              toolName: 'toondb_delete',
              arguments: { path: parts[1] }
            });
            result = `✓ Deleted ${parts[1]}`;
          }
          break;

        case 'describe':
          if (parts.length < 2) {
            result = 'Usage: describe <table>';
          } else {
            const descRes: any = await invoke('mcp_call_tool', {
              toolName: 'toondb_describe',
              arguments: { table: parts[1] }
            });
            result = descRes?.content?.[0]?.text || 'Table not found';
          }
          break;

        case 'query':
          if (parts.length < 2) {
            result = 'Usage: query <sql>';
          } else {
            const queryStr = parts.slice(1).join(' ');
            const queryRes: any = await invoke('mcp_call_tool', {
              toolName: 'toondb_query',
              arguments: { query: queryStr }
            });
            result = queryRes?.content?.[0]?.text || 'No results';
          }
          break;

        case 'create-sample':
          await createSampleData();
          return;

        default:
          // Try as a query
          const defaultRes: any = await invoke('mcp_call_tool', {
            toolName: 'toondb_query',
            arguments: { query: cmd }
          });
          result = defaultRes?.content?.[0]?.text || 'Unknown command. Type "help" for available commands.';
      }

      setConsoleHistory(prev => [...prev, { type: 'output', text: result }]);

    } catch (e: any) {
      setConsoleHistory(prev => [...prev, { type: 'error', text: `Error: ${e?.message || e}` }]);
    }
  };

  const runQuery = async () => {
    if (!query.trim()) return;
    setQueryError(null);
    try {
      const res: any = await invoke('execute_query', { query });
      // Handle both array format (columns, rows) and object format
      if (res.columns && res.rows) {
        setResults({ columns: res.columns, rows: res.rows });
      } else if (Array.isArray(res)) {
        // Legacy format: array of objects
        const cols = res.length > 0 ? Object.keys(res[0]) : [];
        const rows = res.map(r => cols.map(c => r[c]));
        setResults({ columns: cols, rows: rows });
      } else {
        setResults({ columns: ['Result'], rows: [[JSON.stringify(res)]] });
      }
    } catch (e: any) {
      console.error(e);
      setQueryError(e?.message || 'Error executing query');
      setResults(null);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="p-8 h-full overflow-y-auto animate-in">
            <header className="mb-8">
              <h1 className="text-3xl font-bold text-text-default mb-2 font-display">Overview</h1>
              <p className="text-text-muted">Real-time database performance metrics</p>
            </header>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              <StatCard icon="zap" label="MemTable Usage" value={status && typeof status.memtable_size_bytes === 'number' ? `${(status.memtable_size_bytes / 1024).toFixed(2)} KB` : '0 KB'} sub="Active Buffer" color="teal" />
              <StatCard icon="database" label="Tables" value={status?.total_tables ?? '0'} sub="Registered" color="orange" />
              <StatCard icon="check" label="Uptime" value={status?.uptime_seconds ? `${Math.floor(status.uptime_seconds / 60)}m ${status.uptime_seconds % 60}s` : '0s'} sub="Since Restart" color="blue" />
              <StatCard icon="bug" label="Active Txns" value={status?.active_transactions ?? '0'} sub="In Progress" color="red" />
            </div>

            {/* Tables Section */}
            <div className="mt-8">
              <h2 className="text-xl font-bold text-text-default mb-4">Tables</h2>
              <div className="glass-panel rounded-xl p-4">
                {tables.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {tables.map((table, i) => (
                      <div key={i} className="flex items-center gap-2 px-3 py-2 bg-background-muted/50 rounded-lg">
                        <Icon name="database" size={14} className="text-teal" />
                        <span className="text-sm font-mono text-text-default">{table}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-text-muted">
                    <p className="text-sm mb-2">No tables found</p>
                    <button
                      onClick={() => setActiveTab('console')}
                      className="text-teal text-sm hover:underline"
                    >
                      Go to Console to create sample data
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      case 'query':
        return (
          <div className="flex flex-col h-full animate-in pr-3 pb-3 pt-3">
            <div className="glass-panel flex-1 rounded-2xl flex flex-col overflow-hidden shadow-2xl">
              <div className="h-12 border-b border-border-default bg-background-muted/50 flex items-center justify-between px-4">
                <span className="text-xs font-mono text-text-muted">QUERY EDITOR</span>
                <button onClick={runQuery} className="flex items-center gap-2 px-3 py-1.5 bg-teal hover:bg-teal/90 text-white text-xs font-semibold rounded transition-colors">
                  <Icon name="play" size={12} /> Run
                </button>
              </div>
              <div className="h-1/3 bg-background-app/50 relative">
                <textarea
                  className="w-full h-full bg-transparent text-text-default font-mono text-sm p-4 outline-none resize-none code-editor placeholder:text-text-muted/30"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="SELECT * FROM users WHERE id = 1..."
                />
              </div>
              <div className="flex-1 border-t border-border-default bg-background-default flex flex-col">
                <div className="px-4 py-2 border-b border-border-default bg-background-muted/30 text-xs font-semibold text-text-muted">RESULTS</div>
                <div className="flex-1 p-4 overflow-auto">
                  {queryError ? (
                    <div className="h-full flex flex-col items-center justify-center text-red-400">
                      <Icon name="bug" size={48} className="mb-4" />
                      <p className="text-sm font-mono">{queryError}</p>
                    </div>
                  ) : results ? (
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr>
                          {results.columns.map((col, i) => (
                            <th key={i} className="p-2 text-xs font-mono text-text-muted border-b border-border-default">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.rows.map((row, i) => (
                          <tr key={i} className="hover:bg-white/5">
                            {row.map((v: any, j: number) => (
                              <td key={j} className="p-2 text-sm text-text-default border-b border-border-default/50 font-mono">{String(v)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-text-muted opacity-50">
                      <Icon name="terminal" size={48} className="mb-4" />
                      <p>Execute a query to see results</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      case 'assistant':
        return <AssistantView />;
      case 'settings':
        return <SettingsView theme={theme} setTheme={setTheme} connected={connected} />;
      case 'console':
        return (
          <div className="flex flex-col h-full animate-in p-8">
            <header className="mb-6 flex justify-between items-start">
              <div>
                <h1 className="text-3xl font-bold text-text-default mb-2">Console</h1>
                <p className="text-text-muted">Interactive ToonQL REPL</p>
              </div>
              <button
                onClick={createSampleData}
                className="px-4 py-2 bg-teal hover:bg-teal/90 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Create Sample Data
              </button>
            </header>
            <div className="flex-1 glass-panel rounded-2xl flex flex-col overflow-hidden">
              {/* Output area */}
              <div className="flex-1 p-4 font-mono text-sm overflow-auto">
                {consoleHistory.map((entry, i) => (
                  <div key={i} className={`mb-1 ${entry.type === 'input' ? 'text-teal' :
                    entry.type === 'error' ? 'text-red-400' :
                      'text-text-muted'
                    }`}>
                    <pre className="whitespace-pre-wrap">{entry.text}</pre>
                  </div>
                ))}
              </div>
              {/* Input area */}
              <div className="border-t border-border-default bg-background-muted/30 p-4">
                <div className="flex items-center gap-2">
                  <span className="text-teal font-mono">toondb&gt;</span>
                  <input
                    type="text"
                    value={consoleInput}
                    onChange={(e) => setConsoleInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        executeConsoleCommand();
                      }
                    }}
                    placeholder="Type a command... (try 'help')"
                    className="flex-1 bg-transparent outline-none text-text-default font-mono placeholder:text-text-muted/50"
                    autoFocus
                  />
                </div>
              </div>
            </div>
          </div>
        );
      case 'explorer':
        const filteredKeys = kvKeys.filter(k => k.toLowerCase().includes(kvSearch.toLowerCase()));
        return (
          <div className="flex flex-col h-full animate-in p-8">
            <header className="mb-6">
              <h1 className="text-3xl font-bold text-text-default mb-2">KV Explorer</h1>
              <p className="text-text-muted">Browse key-value pairs stored in ToonDB</p>
            </header>
            <div className="flex-1 glass-panel rounded-2xl overflow-hidden flex">
              {/* Keys panel */}
              <div className="w-1/3 border-r border-border-default flex flex-col">
                {/* Table Selector */}
                <div className="h-12 border-b border-border-default bg-background-muted/50 flex items-center px-4 gap-2">
                  <select
                    value={selectedTable}
                    onChange={(e) => {
                      setSelectedTable(e.target.value);
                      if (e.target.value) {
                        refreshKvExplorer(e.target.value);
                      } else {
                        setKvKeys([]);
                      }
                    }}
                    className="flex-1 bg-background-app border border-border-default rounded-lg px-3 py-1.5 text-sm text-text-default outline-none focus:ring-1 focus:ring-teal/50"
                  >
                    <option value="">Select a table...</option>
                    {tables.map((t, i) => (
                      <option key={i} value={t}>{t}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => refreshKvExplorer()}
                    disabled={!selectedTable}
                    className="px-3 py-1.5 bg-teal hover:bg-teal/90 text-white text-xs font-medium rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {kvLoading ? '...' : 'Refresh'}
                  </button>
                </div>
                {/* Filter */}
                {selectedTable && (
                  <div className="h-10 border-b border-border-default bg-background-muted/30 flex items-center px-4">
                    <input
                      type="text"
                      value={kvSearch}
                      onChange={(e) => setKvSearch(e.target.value)}
                      placeholder="Filter keys..."
                      className="flex-1 bg-transparent text-sm text-text-default outline-none placeholder:text-text-muted/50"
                    />
                  </div>
                )}
                {/* Keys list */}
                <div className="flex-1 overflow-auto">
                  {!selectedTable ? (
                    <div className="p-4 text-center text-text-muted opacity-60 h-full flex flex-col items-center justify-center">
                      <Icon name="database" size={32} className="mx-auto mb-2 opacity-50" />
                      <p className="text-sm">Select a table to browse</p>
                      {tables.length === 0 && (
                        <p className="text-xs mt-2 text-text-muted/60">No tables found. Create sample data first.</p>
                      )}
                    </div>
                  ) : filteredKeys.length > 0 ? (
                    <div className="divide-y divide-border-default">
                      {filteredKeys.map((key, i) => (
                        <button
                          key={i}
                          onClick={() => getKeyValue(key)}
                          className={`w-full text-left px-4 py-3 text-sm font-mono hover:bg-background-muted/50 transition-colors ${selectedKey === key ? 'bg-teal/10 text-teal' : 'text-text-default'
                            }`}
                        >
                          {key}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 text-center text-text-muted opacity-60">
                      <p className="text-sm">{kvLoading ? 'Loading...' : `No records in ${selectedTable}`}</p>
                    </div>
                  )}
                </div>
              </div>
              {/* Value panel */}
              <div className="flex-1 flex flex-col">
                <div className="h-12 border-b border-border-default bg-background-muted/50 flex items-center px-4">
                  <span className="text-xs font-mono text-text-muted">
                    {selectedKey ? `Key: ${selectedKey}` : 'Select a key to view its value'}
                  </span>
                </div>
                <div className="flex-1 p-4 overflow-auto">
                  {selectedValue ? (
                    <pre className="text-sm font-mono text-text-default whitespace-pre-wrap bg-background-muted/30 rounded-lg p-4">
                      {selectedValue}
                    </pre>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-text-muted opacity-50">
                      <Icon name="search" size={48} className="mb-4" />
                      <p>Select a key from the list</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      default:
        return (
          <div className="h-full flex flex-col items-center justify-center text-text-muted animate-in">
            <div className="p-6 bg-background-muted rounded-full mb-6">
              <Icon name="layout" size={48} className="opacity-50" />
            </div>
            <h2 className="text-xl font-semibold text-text-default mb-2">Coming Soon</h2>
            <p className="max-w-md text-center opacity-70">The {activeTab} module is under development.</p>
          </div>
        );
    }
  };

  return (
    <div className="h-screen w-screen bg-mesh text-text-default flex flex-col overflow-hidden font-sans selection:bg-teal/30">
      <TitleBar />
      <div className="flex-1 flex overflow-hidden">
        {connected && <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} connected={connected} />}
        <main className="flex-1 overflow-hidden relative">
          {renderContent()}
        </main>
      </div>
      {showConnectionModal && <ConnectionModal onConnect={handleConnect} />}
    </div>
  );
}

export default App;
