/**
 * ToonDB Query Builder
 *
 * Fluent query interface for ToonDB.
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

import { IpcClient } from './ipc-client';

/**
 * Query result row.
 */
export interface QueryResult {
  [key: string]: unknown;
}

/**
 * Fluent query builder for ToonDB.
 *
 * @example
 * ```typescript
 * const results = await db.query('users/')
 *   .limit(10)
 *   .select(['name', 'email'])
 *   .toList();
 * ```
 */
export class Query {
  private _client: IpcClient;
  private _pathPrefix: string;
  private _limit?: number;
  private _offset?: number;
  private _columns?: string[];

  constructor(client: IpcClient, pathPrefix: string) {
    this._client = client;
    this._pathPrefix = pathPrefix;
  }

  /**
   * Limit the number of results.
   *
   * @param n - Maximum number of results to return
   * @returns This query builder for chaining
   */
  limit(n: number): Query {
    this._limit = n;
    return this;
  }

  /**
   * Skip the first n results.
   *
   * @param n - Number of results to skip
   * @returns This query builder for chaining
   */
  offset(n: number): Query {
    this._offset = n;
    return this;
  }

  /**
   * Select specific columns to return.
   *
   * @param columns - Array of column names to select
   * @returns This query builder for chaining
   */
  select(columns: string[]): Query {
    this._columns = columns;
    return this;
  }

  /**
   * Execute the query and return results as TOON string.
   *
   * @returns TOON formatted string (e.g., "result[N]{cols}: row1; row2")
   */
  async execute(): Promise<string> {
    return this._client.query(this._pathPrefix, {
      limit: this._limit,
      offset: this._offset,
      columns: this._columns,
    });
  }

  /**
   * Execute and parse results into a list of objects.
   *
   * @returns Array of result objects
   */
  async toList(): Promise<QueryResult[]> {
    const toonStr = await this.execute();
    return this._parseToon(toonStr);
  }

  /**
   * Execute and return the first result, or null if none.
   *
   * @returns First result or null
   */
  async first(): Promise<QueryResult | null> {
    const originalLimit = this._limit;
    this._limit = 1;
    const results = await this.toList();
    this._limit = originalLimit;
    return results.length > 0 ? results[0] : null;
  }

  /**
   * Execute and return the count of results.
   *
   * @returns Number of matching results
   */
  async count(): Promise<number> {
    const results = await this.toList();
    return results.length;
  }

  /**
   * Simple TOON parser.
   *
   * Parses TOON format: "result[N]{col1,col2}: val1,val2; val3,val4"
   */
  private _parseToon(toonStr: string): QueryResult[] {
    if (!toonStr || toonStr === 'result[0]{}:') {
      return [];
    }

    // Parse header: result[N]{cols}:
    const headerMatch = toonStr.match(/^result\[(\d+)\]\{([^}]*)\}:\s*/);
    if (!headerMatch) {
      // Try to parse as JSON if not TOON format
      try {
        return JSON.parse(toonStr);
      } catch {
        return [];
      }
    }

    const count = parseInt(headerMatch[1], 10);
    if (count === 0) {
      return [];
    }

    const columns = headerMatch[2].split(',').map((c) => c.trim());
    const body = toonStr.substring(headerMatch[0].length);

    // Split rows by semicolon
    const rows = body.split(';').map((r) => r.trim()).filter((r) => r.length > 0);

    return rows.map((row) => {
      const values = row.split(',').map((v) => v.trim());
      const result: QueryResult = {};

      columns.forEach((col, idx) => {
        if (col && idx < values.length) {
          // Try to parse as JSON/number
          let value: unknown = values[idx];
          if (value === 'null') {
            value = null;
          } else if (value === 'true') {
            value = true;
          } else if (value === 'false') {
            value = false;
          } else if (!isNaN(Number(value))) {
            value = Number(value);
          }
          result[col] = value;
        }
      });

      return result;
    });
  }
}
