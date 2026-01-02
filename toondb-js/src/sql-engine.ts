/**
 * SQL Engine for ToonDB JavaScript SDK
 *
 * Provides SQL support on top of the KV storage backend.
 * Tables are stored as:
 *   - Schema: _sql/tables/{table_name}/schema -> JSON schema definition
 *   - Rows: _sql/tables/{table_name}/rows/{row_id} -> JSON row data
 *
 * @packageDocumentation
 */

// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import { v4 as uuidv4 } from 'uuid';

/**
 * Result of a SQL query execution.
 */
export interface SQLQueryResult {
  /** Result rows */
  rows: Array<Record<string, any>>;
  /** Column names */
  columns: string[];
  /** Number of rows affected (for INSERT/UPDATE/DELETE) */
  rowsAffected: number;
}

/**
 * Column definition for a table.
 */
interface Column {
  name: string;
  type: string; // INT, TEXT, FLOAT, BOOL, BLOB
  nullable: boolean;
  primaryKey: boolean;
  default?: any;
}

/**
 * Table schema definition.
 */
interface TableSchema {
  name: string;
  columns: Column[];
  primaryKey?: string;
}

/**
 * Parsed SQL operation result.
 */
type ParsedSQL = {
  operation: string;
  data: Record<string, any>;
};

/**
 * Simple SQL parser for DDL and DML operations.
 */
export class SQLParser {
  /**
   * Parse a SQL statement.
   */
  static parse(sql: string): ParsedSQL {
    sql = sql.trim();
    const upper = sql.toUpperCase();

    if (upper.startsWith('CREATE TABLE')) {
      return SQLParser.parseCreateTable(sql);
    } else if (upper.startsWith('DROP TABLE')) {
      return SQLParser.parseDropTable(sql);
    } else if (upper.startsWith('INSERT')) {
      return SQLParser.parseInsert(sql);
    } else if (upper.startsWith('SELECT')) {
      return SQLParser.parseSelect(sql);
    } else if (upper.startsWith('UPDATE')) {
      return SQLParser.parseUpdate(sql);
    } else if (upper.startsWith('DELETE')) {
      return SQLParser.parseDelete(sql);
    } else {
      throw new Error(`Unsupported SQL statement: ${sql.substring(0, 50)}`);
    }
  }

  private static parseCreateTable(sql: string): ParsedSQL {
    const match = sql.match(
      /CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*)\)/is
    );
    if (!match) {
      throw new Error(`Invalid CREATE TABLE: ${sql}`);
    }

    const tableName = match[1];
    const colsStr = match[2];
    const columns: Column[] = [];
    let primaryKey: string | undefined;

    const colDefs = SQLParser.splitColumns(colsStr);

    for (const colDef of colDefs) {
      const trimmed = colDef.trim();
      if (!trimmed) continue;

      // Check for PRIMARY KEY constraint
      if (trimmed.toUpperCase().startsWith('PRIMARY KEY')) {
        const pkMatch = trimmed.match(/PRIMARY\s+KEY\s*\((\w+)\)/i);
        if (pkMatch) {
          primaryKey = pkMatch[1];
        }
        continue;
      }

      // Parse column: name TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT value]
      const parts = trimmed.split(/\s+/);
      if (parts.length < 2) continue;

      const colName = parts[0];
      let colType = parts[1].toUpperCase();

      // Normalize types
      if (['INTEGER', 'INT', 'BIGINT', 'SMALLINT'].includes(colType)) {
        colType = 'INT';
      } else if (['VARCHAR', 'CHAR', 'STRING', 'TEXT'].includes(colType)) {
        colType = 'TEXT';
      } else if (['REAL', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC'].includes(colType)) {
        colType = 'FLOAT';
      } else if (['BOOLEAN', 'BOOL'].includes(colType)) {
        colType = 'BOOL';
      } else if (['BLOB', 'BYTES', 'BINARY'].includes(colType)) {
        colType = 'BLOB';
      }

      const colUpper = trimmed.toUpperCase();
      const isPk = colUpper.includes('PRIMARY KEY');
      const nullable = !colUpper.includes('NOT NULL');

      if (isPk) {
        primaryKey = colName;
      }

      columns.push({
        name: colName,
        type: colType,
        nullable,
        primaryKey: isPk,
      });
    }

    return {
      operation: 'CREATE_TABLE',
      data: { table: tableName, columns, primaryKey },
    };
  }

  private static splitColumns(colsStr: string): string[] {
    const result: string[] = [];
    let current = '';
    let depth = 0;

    for (const char of colsStr) {
      if (char === '(') {
        depth++;
        current += char;
      } else if (char === ')') {
        depth--;
        current += char;
      } else if (char === ',' && depth === 0) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }

    if (current.trim()) {
      result.push(current);
    }

    return result;
  }

  private static parseDropTable(sql: string): ParsedSQL {
    const match = sql.match(/DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)/i);
    if (!match) {
      throw new Error(`Invalid DROP TABLE: ${sql}`);
    }
    return { operation: 'DROP_TABLE', data: { table: match[1] } };
  }

  private static parseInsert(sql: string): ParsedSQL {
    // INSERT INTO table (col1, col2) VALUES (val1, val2)
    let match = sql.match(
      /INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)/is
    );

    if (match) {
      const table = match[1];
      const columns = match[2].split(',').map((c) => c.trim());
      const values = SQLParser.parseValues(match[3]);
      return { operation: 'INSERT', data: { table, columns, values } };
    }

    // INSERT INTO table VALUES (val1, val2)
    match = sql.match(/INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)/is);
    if (match) {
      const table = match[1];
      const values = SQLParser.parseValues(match[2]);
      return { operation: 'INSERT', data: { table, columns: null, values } };
    }

    throw new Error(`Invalid INSERT: ${sql}`);
  }

  private static parseValues(valuesStr: string): any[] {
    const values: any[] = [];
    let current = '';
    let inString = false;
    let stringChar: string | null = null;

    for (const char of valuesStr) {
      if ((char === '"' || char === "'") && !inString) {
        inString = true;
        stringChar = char;
        current += char;
      } else if (char === stringChar && inString) {
        inString = false;
        stringChar = null;
        current += char;
      } else if (char === ',' && !inString) {
        values.push(SQLParser.parseValue(current.trim()));
        current = '';
      } else {
        current += char;
      }
    }

    if (current.trim()) {
      values.push(SQLParser.parseValue(current.trim()));
    }

    return values;
  }

  private static parseValue(valStr: string): any {
    if (!valStr || valStr.toUpperCase() === 'NULL') {
      return null;
    }

    // String literals
    if (
      (valStr.startsWith("'") && valStr.endsWith("'")) ||
      (valStr.startsWith('"') && valStr.endsWith('"'))
    ) {
      return valStr.slice(1, -1);
    }

    // Boolean
    if (valStr.toUpperCase() === 'TRUE') return true;
    if (valStr.toUpperCase() === 'FALSE') return false;

    // Numbers
    if (valStr.includes('.')) {
      const num = parseFloat(valStr);
      if (!isNaN(num)) return num;
    }
    const intVal = parseInt(valStr, 10);
    if (!isNaN(intVal)) return intVal;

    return valStr;
  }

  private static parseSelect(sql: string): ParsedSQL {
    // Extract table name first
    const tableMatch = sql.match(/FROM\s+(\w+)/i);
    if (!tableMatch) {
      throw new Error(`Invalid SELECT: ${sql}`);
    }
    const table = tableMatch[1];

    // Extract columns
    const colsMatch = sql.match(/SELECT\s+(.+?)\s+FROM/is);
    let columns: string[] = ['*'];
    if (colsMatch) {
      const colsStr = colsMatch[1].trim();
      columns = colsStr === '*' ? ['*'] : colsStr.split(',').map((c) => c.trim());
    }

    // Extract WHERE clause
    let conditions: Array<[string, string, any]> = [];
    const whereMatch = sql.match(/WHERE\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s+OFFSET|$)/is);
    if (whereMatch) {
      conditions = SQLParser.parseWhere(whereMatch[1]);
    }

    // Extract ORDER BY
    let orderBy: Array<[string, string]> = [];
    const orderMatch = sql.match(/ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)/i);
    if (orderMatch) {
      for (const part of orderMatch[1].split(',')) {
        const trimmed = part.trim();
        if (trimmed.toUpperCase().endsWith(' DESC')) {
          orderBy.push([trimmed.slice(0, -5).trim(), 'DESC']);
        } else if (trimmed.toUpperCase().endsWith(' ASC')) {
          orderBy.push([trimmed.slice(0, -4).trim(), 'ASC']);
        } else {
          orderBy.push([trimmed, 'ASC']);
        }
      }
    }

    // Extract LIMIT
    let limit: number | undefined;
    const limitMatch = sql.match(/LIMIT\s+(\d+)/i);
    if (limitMatch) {
      limit = parseInt(limitMatch[1], 10);
    }

    // Extract OFFSET
    let offset: number | undefined;
    const offsetMatch = sql.match(/OFFSET\s+(\d+)/i);
    if (offsetMatch) {
      offset = parseInt(offsetMatch[1], 10);
    }

    return {
      operation: 'SELECT',
      data: { table, columns, where: conditions, orderBy, limit, offset },
    };
  }

  private static parseWhere(whereClause: string): Array<[string, string, any]> {
    const conditions: Array<[string, string, any]> = [];
    const parts = whereClause.split(/\s+AND\s+/i);

    for (const part of parts) {
      const match = part.match(/(\w+)\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT\s+LIKE)\s*(.+)/i);
      if (match) {
        const col = match[1];
        let op = match[2].toUpperCase().replace(/\s+/g, '_');
        if (op === '<>') op = '!=';
        const val = SQLParser.parseValue(match[3].trim());
        conditions.push([col, op, val]);
      }
    }

    return conditions;
  }

  private static parseUpdate(sql: string): ParsedSQL {
    const match = sql.match(/UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$/is);
    if (!match) {
      throw new Error(`Invalid UPDATE: ${sql}`);
    }

    const table = match[1];
    const setClause = match[2];
    const whereClause = match[3];

    // Parse SET clause
    const updates: Record<string, any> = {};
    for (const part of setClause.split(',')) {
      const eqMatch = part.match(/\s*(\w+)\s*=\s*(.+)\s*/);
      if (eqMatch) {
        updates[eqMatch[1]] = SQLParser.parseValue(eqMatch[2].trim());
      }
    }

    let conditions: Array<[string, string, any]> = [];
    if (whereClause) {
      conditions = SQLParser.parseWhere(whereClause);
    }

    return { operation: 'UPDATE', data: { table, updates, where: conditions } };
  }

  private static parseDelete(sql: string): ParsedSQL {
    const match = sql.match(/DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$/is);
    if (!match) {
      throw new Error(`Invalid DELETE: ${sql}`);
    }

    const table = match[1];
    let conditions: Array<[string, string, any]> = [];
    if (match[2]) {
      conditions = SQLParser.parseWhere(match[2]);
    }

    return { operation: 'DELETE', data: { table, where: conditions } };
  }
}

/**
 * Interface for database operations required by SQLExecutor.
 */
interface DatabaseInterface {
  get(key: Buffer | string): Promise<Buffer | null>;
  put(key: Buffer | string, value: Buffer | string): Promise<void>;
  delete(key: Buffer | string): Promise<void>;
  scan(prefix: string): Promise<Array<{ key: Buffer; value: Buffer }>>;
}

/**
 * SQL Executor that operates on a KV database.
 */
export class SQLExecutor {
  private db: DatabaseInterface;

  // Key prefixes for SQL data
  private readonly TABLE_PREFIX = '_sql/tables/';
  private readonly SCHEMA_SUFFIX = '/schema';
  private readonly ROWS_PREFIX = '/rows/';

  constructor(db: DatabaseInterface) {
    this.db = db;
  }

  /**
   * Execute a SQL statement.
   */
  async execute(sql: string): Promise<SQLQueryResult> {
    const { operation, data } = SQLParser.parse(sql);

    switch (operation) {
      case 'CREATE_TABLE':
        return this.createTable(data);
      case 'DROP_TABLE':
        return this.dropTable(data);
      case 'INSERT':
        return this.insert(data);
      case 'SELECT':
        return this.select(data);
      case 'UPDATE':
        return this.update(data);
      case 'DELETE':
        return this.deleteRows(data);
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
  }

  private schemaKey(table: string): string {
    return this.TABLE_PREFIX + table + this.SCHEMA_SUFFIX;
  }

  private rowKey(table: string, rowId: string): string {
    return this.TABLE_PREFIX + table + this.ROWS_PREFIX + rowId;
  }

  private rowPrefix(table: string): string {
    return this.TABLE_PREFIX + table + this.ROWS_PREFIX;
  }

  private async getSchema(table: string): Promise<TableSchema | null> {
    const data = await this.db.get(this.schemaKey(table));
    if (!data) return null;
    return JSON.parse(data.toString());
  }

  private async createTable(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;
    const columns = data.columns as Column[];
    const primaryKey = data.primaryKey;

    // Check if table exists
    if (await this.getSchema(table)) {
      throw new Error(`Table '${table}' already exists`);
    }

    const schema: TableSchema = { name: table, columns, primaryKey };
    await this.db.put(this.schemaKey(table), JSON.stringify(schema));

    return { rows: [], columns: [], rowsAffected: 0 };
  }

  private async dropTable(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;

    // Delete all rows
    const prefix = this.rowPrefix(table);
    const rows = await this.db.scan(prefix);
    let rowsDeleted = 0;

    for (const { key } of rows) {
      await this.db.delete(key);
      rowsDeleted++;
    }

    // Delete schema
    await this.db.delete(this.schemaKey(table));

    return { rows: [], columns: [], rowsAffected: rowsDeleted };
  }

  private async insert(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;
    let columns = data.columns as string[] | null;
    const values = data.values as any[];

    const schema = await this.getSchema(table);
    if (!schema) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // If no columns specified, use schema order
    if (!columns) {
      columns = schema.columns.map((c) => c.name);
    }

    if (columns.length !== values.length) {
      throw new Error(
        `Column count (${columns.length}) doesn't match value count (${values.length})`
      );
    }

    // Create row object
    const row: Record<string, any> = {};
    for (let i = 0; i < columns.length; i++) {
      row[columns[i]] = values[i];
    }

    // Generate row ID
    let rowId: string;
    if (schema.primaryKey && schema.primaryKey in row) {
      rowId = String(row[schema.primaryKey]);
    } else {
      rowId = uuidv4();
    }
    row['_id'] = rowId;

    await this.db.put(this.rowKey(table, rowId), JSON.stringify(row));

    return { rows: [], columns: [], rowsAffected: 1 };
  }

  private async select(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;
    let columns = data.columns as string[];
    const conditions = data.where as Array<[string, string, any]>;
    const orderBy = data.orderBy as Array<[string, string]>;
    const limit = data.limit as number | undefined;
    const offset = data.offset as number | undefined;

    const schema = await this.getSchema(table);
    if (!schema) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // Get column names
    if (columns.length === 1 && columns[0] === '*') {
      columns = schema.columns.map((c) => c.name);
    }

    // Scan all rows
    const prefix = this.rowPrefix(table);
    const scanResults = await this.db.scan(prefix);
    let rows: Array<Record<string, any>> = [];

    for (const { value } of scanResults) {
      const row = JSON.parse(value.toString());

      // Apply WHERE conditions
      if (this.matchesConditions(row, conditions)) {
        // Project columns
        const projected: Record<string, any> = {};
        for (const col of columns) {
          if (col in row) {
            projected[col] = row[col];
          }
        }
        rows.push(projected);
      }
    }

    // Apply ORDER BY
    if (orderBy && orderBy.length > 0) {
      rows.sort((a, b) => {
        for (const [col, direction] of orderBy) {
          const aVal = a[col];
          const bVal = b[col];

          // Handle nulls
          if (aVal === null || aVal === undefined) return 1;
          if (bVal === null || bVal === undefined) return -1;

          let cmp = 0;
          if (aVal < bVal) cmp = -1;
          else if (aVal > bVal) cmp = 1;

          if (direction === 'DESC') cmp = -cmp;
          if (cmp !== 0) return cmp;
        }
        return 0;
      });
    }

    // Apply OFFSET and LIMIT
    if (offset && offset > 0) {
      rows = rows.slice(offset);
    }
    if (limit !== undefined && limit > 0) {
      rows = rows.slice(0, limit);
    }

    return { rows, columns, rowsAffected: 0 };
  }

  private matchesConditions(
    row: Record<string, any>,
    conditions: Array<[string, string, any]>
  ): boolean {
    for (const [col, op, val] of conditions) {
      const rowVal = row[col];

      switch (op) {
        case '=':
          if (rowVal !== val) return false;
          break;
        case '!=':
          if (rowVal === val) return false;
          break;
        case '>':
          if (rowVal === null || rowVal === undefined || rowVal <= val) return false;
          break;
        case '>=':
          if (rowVal === null || rowVal === undefined || rowVal < val) return false;
          break;
        case '<':
          if (rowVal === null || rowVal === undefined || rowVal >= val) return false;
          break;
        case '<=':
          if (rowVal === null || rowVal === undefined || rowVal > val) return false;
          break;
        case 'LIKE': {
          if (rowVal === null || rowVal === undefined) return false;
          const pattern = String(val).replace(/%/g, '.*').replace(/_/g, '.');
          if (!new RegExp(`^${pattern}$`, 'i').test(String(rowVal))) return false;
          break;
        }
        case 'NOT_LIKE': {
          if (rowVal === null || rowVal === undefined) return true;
          const pattern = String(val).replace(/%/g, '.*').replace(/_/g, '.');
          if (new RegExp(`^${pattern}$`, 'i').test(String(rowVal))) return false;
          break;
        }
      }
    }
    return true;
  }

  private async update(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;
    const updates = data.updates as Record<string, any>;
    const conditions = data.where as Array<[string, string, any]>;

    const schema = await this.getSchema(table);
    if (!schema) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // Scan all rows
    const prefix = this.rowPrefix(table);
    const scanResults = await this.db.scan(prefix);
    let rowsAffected = 0;

    for (const { key, value } of scanResults) {
      const row = JSON.parse(value.toString());

      // Apply WHERE conditions
      if (this.matchesConditions(row, conditions)) {
        // Apply updates
        for (const [col, val] of Object.entries(updates)) {
          row[col] = val;
        }

        await this.db.put(key, JSON.stringify(row));
        rowsAffected++;
      }
    }

    return { rows: [], columns: [], rowsAffected };
  }

  private async deleteRows(data: Record<string, any>): Promise<SQLQueryResult> {
    const table = data.table;
    const conditions = data.where as Array<[string, string, any]>;

    const schema = await this.getSchema(table);
    if (!schema) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // Scan all rows and collect keys to delete
    const prefix = this.rowPrefix(table);
    const scanResults = await this.db.scan(prefix);
    const keysToDelete: Buffer[] = [];

    for (const { key, value } of scanResults) {
      const row = JSON.parse(value.toString());

      // Apply WHERE conditions
      if (this.matchesConditions(row, conditions)) {
        keysToDelete.push(key);
      }
    }

    // Delete collected keys
    for (const key of keysToDelete) {
      await this.db.delete(key);
    }

    return { rows: [], columns: [], rowsAffected: keysToDelete.length };
  }
}
