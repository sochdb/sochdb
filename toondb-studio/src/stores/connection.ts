/**
 * Connection Store - Global connection state
 */

export interface Connection {
    id: string;
    name: string;
    path: string;
    status: 'connected' | 'disconnected' | 'connecting';
}

// For now, use a simple React-based store without zustand
// We can add zustand later if needed

export const defaultConnections: Connection[] = [
    {
        id: 'local-dev',
        name: 'Local Development',
        path: './dev.toondb',
        status: 'connected',
    },
    {
        id: 'production',
        name: 'Production',
        path: '/var/toondb/prod',
        status: 'disconnected',
    },
    {
        id: 'remote',
        name: 'Test Server',
        path: '192.168.1.10:7654',
        status: 'disconnected',
    },
];

// Simple state management without external dependencies
class ConnectionStore {
    private connections: Connection[] = defaultConnections;
    private activeConnectionId: string | null = 'local-dev';
    private listeners: (() => void)[] = [];

    getConnections(): Connection[] {
        return this.connections;
    }

    getActiveConnection(): Connection | null {
        return this.connections.find(c => c.id === this.activeConnectionId) || null;
    }

    getActiveConnectionId(): string | null {
        return this.activeConnectionId;
    }

    setActiveConnection(id: string): void {
        this.activeConnectionId = id;
        this.notifyListeners();
    }

    addConnection(connection: Connection): void {
        this.connections.push(connection);
        this.notifyListeners();
    }

    removeConnection(id: string): void {
        this.connections = this.connections.filter(c => c.id !== id);
        if (this.activeConnectionId === id) {
            this.activeConnectionId = this.connections[0]?.id || null;
        }
        this.notifyListeners();
    }

    updateConnectionStatus(id: string, status: Connection['status']): void {
        const conn = this.connections.find(c => c.id === id);
        if (conn) {
            conn.status = status;
            this.notifyListeners();
        }
    }

    subscribe(listener: () => void): () => void {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    private notifyListeners(): void {
        this.listeners.forEach(l => l());
    }
}

export const connectionStore = new ConnectionStore();
