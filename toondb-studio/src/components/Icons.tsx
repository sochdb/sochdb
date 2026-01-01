// Simple Icon component - uses custom TD cartoon for branding, emoji for everything else
export const ToonIcon = ({ size = 24, className = "" }: { size?: number; className?: string }) => (
    <div
        className={`inline-block ${className}`}
        style={{ width: size, height: size }}
    >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width={size} height={size}>
            <defs>
                <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#ffb347" />
                    <stop offset="50%" stopColor="#ff6f61" />
                    <stop offset="100%" stopColor="#ffcc33" />
                </linearGradient>
                <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="0" dy="10" stdDeviation="18" floodColor="#000000" floodOpacity="0.3" />
                </filter>
            </defs>
            <rect x="32" y="32" width="448" height="448" rx="100" ry="100" fill="url(#bgGrad)" />
            <g filter="url(#softShadow)">
                <path d="M150 150 h 220 a 40 40 0 0 1 40 40 v 120 a 40 40 0 0 1 -40 40 h -120 l -35 45 a 10 10 0 0 1 -18 -6 v -39 h -47 a 40 40 0 0 1 -40 -40 v -120 a 40 40 0 0 1 40 -40 z"
                    fill="#ffffff" stroke="#f25c54" strokeWidth="10" strokeLinejoin="round" />
            </g>
            <g>
                <circle cx="210" cy="220" r="28" fill="#ffffff" />
                <circle cx="218" cy="226" r="14" fill="#333333" />
                <circle cx="290" cy="220" r="28" fill="#ffffff" />
                <circle cx="298" cy="226" r="14" fill="#333333" />
                <path d="M220 260 Q 250 285 280 260" fill="none" stroke="#333333" strokeWidth="7" strokeLinecap="round" />
            </g>
            <text x="50%" y="340" textAnchor="middle" fontFamily="system-ui, -apple-system" fontSize="72" fontWeight="700" fill="#333333">TD</text>
        </svg>
    </div>
);

// Simple emoji-based icons for UI elements
export const Icon = ({ name, size = 18, className = "" }: { name: string; size?: number; className?: string }) => {
    const icons: Record<string, string> = {
        // Navigation
        'dashboard': '📊',
        'table': '📋',
        'database': '💾',
        'terminal': '💻',
        'settings': '⚙️',

        // Actions
        'play': '▶️',
        'refresh': '🔄',
        'save': '💾',
        'x': '✕',
        'minus': '−',
        'square': '⬜',

        // System
        'cpu': '🖥️',
        'network': '🌐',
        'bot': '🤖',
        'zap': '⚡',
        'server': '🖥️',
        'activity': '📈',
        'hard-drive': '💿',
    };

    return (
        <span
            className={`inline-flex items-center justify-center ${className}`}
            style={{ fontSize: size }}
        >
            {icons[name] || '•'}
        </span>
    );
};
