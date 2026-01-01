/**
 * Schema Hook - Manage database schema state
 */

import { useState, useEffect, useCallback } from 'react';
import { useToonDB, DatabaseSchema, TableInfo } from './useToonDB';

export function useSchema() {
    const { getSchema, getTableInfo } = useToonDB();

    const [schema, setSchema] = useState<DatabaseSchema | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [tableDetails, setTableDetails] = useState<TableInfo | null>(null);

    // Load schema on mount
    useEffect(() => {
        loadSchema();
    }, []);

    // Load table details when selected table changes
    useEffect(() => {
        if (selectedTable) {
            loadTableInfo(selectedTable);
        }
    }, [selectedTable]);

    const loadSchema = useCallback(async () => {
        setIsLoading(true);
        setError(null);

        try {
            const s = await getSchema();
            setSchema(s);

            // Select first table by default
            if (s.tables.length > 0 && !selectedTable) {
                setSelectedTable(s.tables[0].name);
            }
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setIsLoading(false);
        }
    }, [getSchema, selectedTable]);

    const loadTableInfo = useCallback(async (tableName: string) => {
        try {
            const info = await getTableInfo(tableName);
            setTableDetails(info);
        } catch (e) {
            console.error('Failed to load table info:', e);
        }
    }, [getTableInfo]);

    const refreshSchema = useCallback(() => {
        loadSchema();
    }, [loadSchema]);

    return {
        schema,
        isLoading,
        error,
        selectedTable,
        setSelectedTable,
        tableDetails,
        refreshSchema,
    };
}

export default useSchema;
