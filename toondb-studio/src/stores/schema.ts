/**
 * Schema Store - Global database schema state
 */

import { DatabaseSchema } from '../hooks/useToonDB';

class SchemaStore {
    private schema: DatabaseSchema | null = null;
    private selectedTable: string | null = 'users';
    private expandedNodes: string[] = ['tables', 'indexes'];
    private listeners: (() => void)[] = [];

    getSchema(): DatabaseSchema | null {
        return this.schema;
    }

    setSchema(schema: DatabaseSchema): void {
        this.schema = schema;
        this.notifyListeners();
    }

    getSelectedTable(): string | null {
        return this.selectedTable;
    }

    setSelectedTable(table: string): void {
        this.selectedTable = table;
        this.notifyListeners();
    }

    getExpandedNodes(): string[] {
        return this.expandedNodes;
    }

    toggleNode(node: string): void {
        if (this.expandedNodes.includes(node)) {
            this.expandedNodes = this.expandedNodes.filter(n => n !== node);
        } else {
            this.expandedNodes.push(node);
        }
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

export const schemaStore = new SchemaStore();
