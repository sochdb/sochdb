/**
 * ToonDB Hook - Interface to ToonDB backend
 */

import { invoke } from '@tauri-apps/api/core';

// Types
export interface QueryResult {
    columns: string[];
    rows: any[][];
    stats: {
        row_count: number;
        execution_time_ms: number;
        scanned_rows: number;
    };
}

export interface ColumnInfo {
    name: string;
    data_type: string;
    nullable: boolean;
    is_primary_key: boolean;
}

export interface TableInfo {
    name: string;
    row_count: number;
    columns: ColumnInfo[];
}

export interface IndexInfo {
    name: string;
    table_name: string;
    column_name: string;
    index_type: string;
    config?: string;
}

export interface PathInfo {
    path: string;
    count: number;
}

export interface DatabaseSchema {
    tables: TableInfo[];
    indexes: IndexInfo[];
    paths: PathInfo[];
}

export interface VectorResult {
    id: string;
    score: number;
    content: string;
    metadata: Record<string, any>;
}

export interface DatabaseStats {
    memtable_size_bytes: number;
    wal_size_bytes: number;
    total_tables: number;
    total_rows: number;
    active_transactions: number;
    last_checkpoint_lsn: number;
    uptime_seconds: number;
    version: string;
    active_snapshots: number;
    min_active_timestamp: number;
    garbage_versions: number;
}

export interface ConnectionInfo {
    path: string;
    connected: boolean;
    version: string;
    uptime_seconds: number;
}

// Hook implementation
export function useToonDB() {
    // Query operations
    const executeQuery = async (query: string): Promise<QueryResult> => {
        return invoke<QueryResult>('execute_query', { query });
    };

    const explainQuery = async (query: string): Promise<string> => {
        return invoke<string>('explain_query', { query });
    };

    // Schema operations
    const getSchema = async (): Promise<DatabaseSchema> => {
        return invoke<DatabaseSchema>('get_schema');
    };

    const getTableInfo = async (tableName: string): Promise<TableInfo> => {
        return invoke<TableInfo>('get_table_info', { tableName });
    };

    // Vector operations
    const vectorSearch = async (
        table: string,
        column: string,
        queryText: string,
        limit: number,
        metric: string
    ): Promise<VectorResult[]> => {
        return invoke<VectorResult[]>('vector_search', {
            table,
            column,
            queryText,
            limit,
            metric,
        });
    };

    const getVectorColumns = async (table: string): Promise<string[]> => {
        return invoke<string[]>('get_vector_columns', { table });
    };

    // Admin operations
    const getStats = async (): Promise<DatabaseStats> => {
        return invoke<DatabaseStats>('get_stats');
    };

    const connect = async (path: string): Promise<ConnectionInfo> => {
        return invoke<ConnectionInfo>('connect', { path });
    };

    const disconnect = async (): Promise<void> => {
        return invoke<void>('disconnect');
    };

    const checkpoint = async (): Promise<number> => {
        return invoke<number>('checkpoint');
    };

    const gc = async (): Promise<number> => {
        return invoke<number>('gc');
    };

    const analyze = async (): Promise<void> => {
        return invoke<void>('analyze');
    };

    const compact = async (): Promise<void> => {
        return invoke<void>('compact');
    };

    return {
        // Query
        executeQuery,
        explainQuery,
        // Schema
        getSchema,
        getTableInfo,
        // Vector
        vectorSearch,
        getVectorColumns,
        // Admin
        getStats,
        connect,
        disconnect,
        checkpoint,
        gc,
        analyze,
        compact,
    };
}

export default useToonDB;
