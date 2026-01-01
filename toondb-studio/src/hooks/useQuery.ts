/**
 * Query Hook - Manage query state and execution
 */

import { useState, useCallback } from 'react';
import { useToonDB, QueryResult } from './useToonDB';

export interface QueryState {
    query: string;
    result: QueryResult | null;
    isExecuting: boolean;
    error: string | null;
    history: QueryHistoryEntry[];
}

export interface QueryHistoryEntry {
    query: string;
    timestamp: Date;
    rowCount: number;
    executionTimeMs: number;
    success: boolean;
}

export function useQuery() {
    const { executeQuery: executeQueryCmd } = useToonDB();

    const [query, setQuery] = useState('SELECT * FROM users LIMIT 100');
    const [result, setResult] = useState<QueryResult | null>(null);
    const [isExecuting, setIsExecuting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [history, setHistory] = useState<QueryHistoryEntry[]>([]);

    const executeQuery = useCallback(async () => {
        setIsExecuting(true);
        setError(null);

        try {
            const res = await executeQueryCmd(query);
            setResult(res);

            // Add to history
            setHistory(prev => [
                {
                    query,
                    timestamp: new Date(),
                    rowCount: res.stats.row_count,
                    executionTimeMs: res.stats.execution_time_ms,
                    success: true,
                },
                ...prev.slice(0, 49), // Keep last 50 entries
            ]);
        } catch (e) {
            const errorMessage = e instanceof Error ? e.message : String(e);
            setError(errorMessage);

            // Add failed query to history
            setHistory(prev => [
                {
                    query,
                    timestamp: new Date(),
                    rowCount: 0,
                    executionTimeMs: 0,
                    success: false,
                },
                ...prev.slice(0, 49),
            ]);
        } finally {
            setIsExecuting(false);
        }
    }, [query, executeQueryCmd]);

    const clearResult = useCallback(() => {
        setResult(null);
        setError(null);
    }, []);

    const loadFromHistory = useCallback((entry: QueryHistoryEntry) => {
        setQuery(entry.query);
    }, []);

    return {
        query,
        setQuery,
        result,
        isExecuting,
        error,
        history,
        executeQuery,
        clearResult,
        loadFromHistory,
    };
}

export default useQuery;
