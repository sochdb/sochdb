/**
 * Query Store - Global query state
 */

import { QueryResult } from '../hooks/useToonDB';
import { QueryHistoryEntry } from '../hooks/useQuery';

class QueryStore {
    private query: string = 'SELECT * FROM users LIMIT 100';
    private result: QueryResult | null = null;
    private isExecuting: boolean = false;
    private error: string | null = null;
    private history: QueryHistoryEntry[] = [];
    private listeners: (() => void)[] = [];

    getQuery(): string {
        return this.query;
    }

    setQuery(query: string): void {
        this.query = query;
        this.notifyListeners();
    }

    getResult(): QueryResult | null {
        return this.result;
    }

    setResult(result: QueryResult | null): void {
        this.result = result;
        this.notifyListeners();
    }

    isQueryExecuting(): boolean {
        return this.isExecuting;
    }

    setIsExecuting(executing: boolean): void {
        this.isExecuting = executing;
        this.notifyListeners();
    }

    getError(): string | null {
        return this.error;
    }

    setError(error: string | null): void {
        this.error = error;
        this.notifyListeners();
    }

    getHistory(): QueryHistoryEntry[] {
        return this.history;
    }

    addToHistory(entry: QueryHistoryEntry): void {
        this.history = [entry, ...this.history.slice(0, 49)];
        this.notifyListeners();
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

export const queryStore = new QueryStore();
