/**
 * ToonDB IPC Client
 *
 * Connects to a ToonDB IPC server via Unix domain socket.
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

import * as net from 'net';
import { ConnectionError, ProtocolError, TransactionError } from './errors';
import { Query } from './query';

/**
 * Wire protocol opcodes.
 */
export const OpCode = {
  // Client → Server (must match toondb-storage/src/ipc_server.rs)
  Put: 0x01,
  Get: 0x02,
  Delete: 0x03,
  BeginTxn: 0x04,
  CommitTxn: 0x05,
  AbortTxn: 0x06,
  Query: 0x07,
  CreateTable: 0x08,
  PutPath: 0x09,
  GetPath: 0x0A,
  Scan: 0x0B,
  Checkpoint: 0x0C,
  Stats: 0x0D,
  Ping: 0x0E,

  // Server → Client
  OK: 0x80,
  Error: 0x81,
  Value: 0x82,
  TxnId: 0x83,
  Row: 0x84,
  EndStream: 0x85,
  StatsResp: 0x86,
  Pong: 0x87,
} as const;

// Internal OpCode map for backwards compatibility
const InternalOpCode = {
  PUT: OpCode.Put,
  GET: OpCode.Get,
  DELETE: OpCode.Delete,
  BEGIN_TXN: OpCode.BeginTxn,
  COMMIT_TXN: OpCode.CommitTxn,
  ABORT_TXN: OpCode.AbortTxn,
  QUERY: OpCode.Query,
  PUT_PATH: OpCode.PutPath,
  GET_PATH: OpCode.GetPath,
  SCAN: OpCode.Scan,
  CHECKPOINT: OpCode.Checkpoint,
  STATS: OpCode.Stats,
  PING: OpCode.Ping,
  OK: OpCode.OK,
  ERROR: OpCode.Error,
  VALUE: OpCode.Value,
  TXN_ID: OpCode.TxnId,
  ROW: OpCode.Row,
  END_STREAM: OpCode.EndStream,
  STATS_RESP: OpCode.StatsResp,
  PONG: OpCode.Pong,
} as const;

const MAX_MESSAGE_SIZE = 16 * 1024 * 1024; // 16 MB

/**
 * Configuration options for IpcClient.
 */
export interface IpcClientConfig {
  /** Path to Unix domain socket */
  socketPath: string;
  /** Connection timeout in milliseconds (default: 5000) */
  connectTimeout?: number;
  /** Read timeout in milliseconds (default: 30000) */
  readTimeout?: number;
}

/**
 * IPC Client for ToonDB.
 *
 * Connects to a ToonDB server via Unix domain socket.
 *
 * @example
 * ```typescript
 * import { IpcClient } from '@sushanth/toondb';
 *
 * const client = await IpcClient.connect('/tmp/toondb.sock');
 *
 * await client.put(Buffer.from('key'), Buffer.from('value'));
 * const value = await client.get(Buffer.from('key'));
 *
 * await client.close();
 * ```
 */
export class IpcClient {
  private _socket: net.Socket | null = null;
  private _config: Required<IpcClientConfig>;
  private _pendingReads: Array<{
    resolve: (buf: Buffer) => void;
    reject: (err: Error) => void;
  }> = [];
  private _readBuffer: Buffer = Buffer.alloc(0);
  private _closed = false;

  private constructor(config: IpcClientConfig) {
    this._config = {
      connectTimeout: 5000,
      readTimeout: 30000,
      ...config,
    };
  }

  /**
   * Connect to a ToonDB IPC server.
   *
   * @param socketPath - Path to the Unix domain socket
   * @returns A connected IpcClient instance
   *
   * @example
   * ```typescript
   * const client = await IpcClient.connect('/tmp/toondb.sock');
   * ```
   */
  static async connect(socketPath: string): Promise<IpcClient> {
    const client = new IpcClient({ socketPath });
    await client._connect();
    return client;
  }

  private async _connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = net.createConnection(
        { path: this._config.socketPath },
        () => {
          this._socket = socket;
          resolve();
        }
      );

      socket.setTimeout(this._config.connectTimeout);

      socket.on('timeout', () => {
        socket.destroy();
        reject(new ConnectionError('Connection timeout'));
      });

      socket.on('error', (err) => {
        reject(new ConnectionError(`Connection failed: ${err.message}`));
      });

      socket.on('data', (data) => {
        this._readBuffer = Buffer.concat([this._readBuffer, data]);
        this._processBuffer();
      });

      socket.on('close', () => {
        this._closed = true;
        for (const pending of this._pendingReads) {
          pending.reject(new ConnectionError('Connection closed'));
        }
        this._pendingReads = [];
      });
    });
  }

  private _processBuffer(): void {
    // Process complete messages from the buffer
    while (this._readBuffer.length >= 5 && this._pendingReads.length > 0) {
      const length = this._readBuffer.readUInt32LE(1);
      const totalLength = 5 + length;

      if (this._readBuffer.length >= totalLength) {
        const message = this._readBuffer.subarray(0, totalLength);
        this._readBuffer = this._readBuffer.subarray(totalLength);

        const pending = this._pendingReads.shift();
        if (pending) {
          pending.resolve(message);
        }
      } else {
        break;
      }
    }
  }

  private async _send(opcode: number, payload: Buffer = Buffer.alloc(0)): Promise<Buffer> {
    if (this._closed || !this._socket) {
      throw new ConnectionError('Not connected');
    }

    // Encode message: opcode (1) + length (4 LE) + payload
    const message = Buffer.alloc(5 + payload.length);
    message.writeUInt8(opcode, 0);
    message.writeUInt32LE(payload.length, 1);
    payload.copy(message, 5);

    return new Promise((resolve, reject) => {
      this._pendingReads.push({ resolve, reject });

      this._socket!.write(message, (err) => {
        if (err) {
          this._pendingReads.pop();
          reject(new ConnectionError(`Write failed: ${err.message}`));
        }
      });

      // Set timeout for response
      setTimeout(() => {
        const idx = this._pendingReads.findIndex((p) => p.resolve === resolve);
        if (idx !== -1) {
          this._pendingReads.splice(idx, 1);
          reject(new ConnectionError('Read timeout'));
        }
      }, this._config.readTimeout);
    });
  }

  private _parseResponse(response: Buffer): { opcode: number; payload: Buffer } {
    const opcode = response.readUInt8(0);
    const length = response.readUInt32LE(1);
    const payload = response.subarray(5, 5 + length);

    if (opcode === InternalOpCode.ERROR) {
      throw new ProtocolError(payload.toString('utf8'));
    }

    return { opcode, payload };
  }

  /**
   * Encode a key for the wire protocol.
   * @internal
   */
  static encodeKey(key: Buffer): Buffer {
    // Format: [length:4][op:1][key_len:4][key:...]
    const msgLen = 1 + 4 + key.length;
    const msg = Buffer.alloc(4 + msgLen);
    msg.writeUInt32BE(msgLen, 0);
    msg.writeUInt8(InternalOpCode.GET, 4);
    msg.writeUInt32BE(key.length, 5);
    key.copy(msg, 9);
    return msg;
  }

  /**
   * Encode a key-value pair for the wire protocol.
   * @internal
   */
  static encodeKeyValue(key: Buffer, value: Buffer): Buffer {
    // Format: [length:4][op:1][key_len:4][key:...][value_len:4][value:...]
    const msgLen = 1 + 4 + key.length + 4 + value.length;
    const msg = Buffer.alloc(4 + msgLen);
    msg.writeUInt32BE(msgLen, 0);
    msg.writeUInt8(InternalOpCode.PUT, 4);
    msg.writeUInt32BE(key.length, 5);
    key.copy(msg, 9);
    msg.writeUInt32BE(value.length, 9 + key.length);
    value.copy(msg, 13 + key.length);
    return msg;
  }

  /**
   * Get a value by key.
   */
  async get(key: Buffer): Promise<Buffer | null> {
    const response = await this._send(InternalOpCode.GET, key);
    const { opcode, payload } = this._parseResponse(response);

    if (opcode === InternalOpCode.VALUE) {
      // If payload is empty, the key doesn't exist
      if (payload.length === 0) {
        return null;
      }
      return payload;
    }
    return null;
  }

  /**
   * Put a key-value pair.
   */
  async put(key: Buffer, value: Buffer): Promise<void> {
    // Encode: key_len (4 LE) + key + value
    const payload = Buffer.alloc(4 + key.length + value.length);
    payload.writeUInt32LE(key.length, 0);
    key.copy(payload, 4);
    value.copy(payload, 4 + key.length);

    const response = await this._send(InternalOpCode.PUT, payload);
    this._parseResponse(response);
  }

  /**
   * Delete a key.
   */
  async delete(key: Buffer): Promise<void> {
    const response = await this._send(InternalOpCode.DELETE, key);
    this._parseResponse(response);
  }

  /**
   * Get a value by path.
   * Wire format: path_count(2 LE) + [path_len(2 LE) + path_segment]...
   */
  async getPath(path: string): Promise<Buffer | null> {
    // Encode path as single segment for now (could split on '/' for multi-segment)
    const pathBuf = Buffer.from(path, 'utf8');
    const payload = Buffer.alloc(2 + 2 + pathBuf.length);
    payload.writeUInt16LE(1, 0); // path_count = 1
    payload.writeUInt16LE(pathBuf.length, 2); // path_len
    pathBuf.copy(payload, 4); // path
    
    const response = await this._send(InternalOpCode.GET_PATH, payload);
    const { opcode, payload: responsePayload } = this._parseResponse(response);

    if (opcode === InternalOpCode.VALUE) {
      // If payload is empty, the key doesn't exist
      if (responsePayload.length === 0) {
        return null;
      }
      return responsePayload;
    }
    return null;
  }

  /**
   * Put a value at a path.
   * Wire format: path_count(2 LE) + [path_len(2 LE) + path_segment]... + value
   */
  async putPath(path: string, value: Buffer): Promise<void> {
    const pathBuf = Buffer.from(path, 'utf8');
    const payload = Buffer.alloc(2 + 2 + pathBuf.length + value.length);
    payload.writeUInt16LE(1, 0); // path_count = 1
    payload.writeUInt16LE(pathBuf.length, 2); // path_len
    pathBuf.copy(payload, 4); // path
    value.copy(payload, 4 + pathBuf.length); // value

    const response = await this._send(InternalOpCode.PUT_PATH, payload);
    this._parseResponse(response);
  }

  /**
   * Execute a query and return TOON-formatted results.
   * 
   * Wire format: path_len(2) + path + limit(4) + offset(4) + cols_count(2) + [col_len(2) + col]...
   */
  async query(
    pathPrefix: string,
    options?: { limit?: number; offset?: number; columns?: string[] }
  ): Promise<string> {
    const opts = options || {};
    const pathBuf = Buffer.from(pathPrefix, 'utf8');
    const columns = opts.columns || [];
    
    // Calculate payload size
    let size = 2 + pathBuf.length + 4 + 4 + 2;
    for (const col of columns) {
      size += 2 + Buffer.byteLength(col, 'utf8');
    }
    
    const payload = Buffer.alloc(size);
    let offset = 0;
    
    // Path: path_len(2 LE) + path
    payload.writeUInt16LE(pathBuf.length, offset);
    offset += 2;
    pathBuf.copy(payload, offset);
    offset += pathBuf.length;
    
    // Limit (4 LE) - 0 means no limit
    payload.writeUInt32LE(opts.limit || 0, offset);
    offset += 4;
    
    // Offset (4 LE)
    payload.writeUInt32LE(opts.offset || 0, offset);
    offset += 4;
    
    // Columns: count(2 LE) + [col_len(2 LE) + col]...
    payload.writeUInt16LE(columns.length, offset);
    offset += 2;
    for (const col of columns) {
      const colBuf = Buffer.from(col, 'utf8');
      payload.writeUInt16LE(colBuf.length, offset);
      offset += 2;
      colBuf.copy(payload, offset);
      offset += colBuf.length;
    }

    const response = await this._send(InternalOpCode.QUERY, payload);
    const { payload: resultPayload } = this._parseResponse(response);

    return resultPayload.toString('utf8');
  }

  /**
   * Scan for keys with a prefix, returning key-value pairs.
   * This is the preferred method for simple prefix-based iteration.
   * 
   * Wire format: prefix string
   * Response format: count(4 LE) + [key_len(2 LE) + key + val_len(4 LE) + val]...
   */
  async scan(prefix: string): Promise<Array<{ key: Buffer; value: Buffer }>> {
    const prefixBuf = Buffer.from(prefix, 'utf8');
    const response = await this._send(InternalOpCode.SCAN, prefixBuf);
    const { opcode, payload } = this._parseResponse(response);

    if (opcode !== InternalOpCode.VALUE && opcode !== InternalOpCode.OK) {
      throw new ProtocolError(`Unexpected scan response opcode: 0x${opcode.toString(16)}`);
    }

    if (payload.length < 4) {
      return []; // Empty result
    }

    // Parse response: count(4 LE) + [key_len(2 LE) + key + val_len(4 LE) + val]...
    const count = payload.readUInt32LE(0);
    const results: Array<{ key: Buffer; value: Buffer }> = [];
    let offset = 4;

    for (let i = 0; i < count; i++) {
      if (offset + 2 > payload.length) {
        throw new ProtocolError('Truncated scan response (key_len)');
      }
      const keyLen = payload.readUInt16LE(offset);
      offset += 2;

      if (offset + keyLen + 4 > payload.length) {
        throw new ProtocolError('Truncated scan response (key+val_len)');
      }
      const key = payload.subarray(offset, offset + keyLen);
      offset += keyLen;

      const valLen = payload.readUInt32LE(offset);
      offset += 4;

      if (offset + valLen > payload.length) {
        throw new ProtocolError('Truncated scan response (value)');
      }
      const value = payload.subarray(offset, offset + valLen);
      offset += valLen;

      results.push({ key, value });
    }

    return results;
  }

  /**
   * Create a query builder.
   */
  queryBuilder(pathPrefix: string): Query {
    return new Query(this, pathPrefix);
  }

  /**
   * Begin a new transaction.
   */
  async beginTransaction(): Promise<bigint> {
    const response = await this._send(InternalOpCode.BEGIN_TXN);
    const { opcode, payload } = this._parseResponse(response);

    if (opcode === InternalOpCode.TXN_ID) {
      return payload.readBigUInt64LE(0);
    }
    throw new TransactionError('Failed to begin transaction');
  }

  /**
   * Commit a transaction.
   */
  async commitTransaction(txnId: bigint): Promise<void> {
    const payload = Buffer.alloc(8);
    payload.writeBigUInt64LE(txnId, 0);
    const response = await this._send(InternalOpCode.COMMIT_TXN, payload);
    this._parseResponse(response);
  }

  /**
   * Abort a transaction.
   */
  async abortTransaction(txnId: bigint): Promise<void> {
    const payload = Buffer.alloc(8);
    payload.writeBigUInt64LE(txnId, 0);
    const response = await this._send(InternalOpCode.ABORT_TXN, payload);
    this._parseResponse(response);
  }

  /**
   * Force a checkpoint.
   */
  async checkpoint(): Promise<void> {
    const response = await this._send(InternalOpCode.CHECKPOINT);
    this._parseResponse(response);
  }

  /**
   * Get storage statistics.
   */
  async stats(): Promise<{
    memtableSizeBytes: number;
    walSizeBytes: number;
    activeTransactions: number;
  }> {
    const response = await this._send(InternalOpCode.STATS);
    const { payload } = this._parseResponse(response);
    const json = JSON.parse(payload.toString('utf8'));
    return {
      memtableSizeBytes: json.memtable_size_bytes || 0,
      walSizeBytes: json.wal_size_bytes || 0,
      activeTransactions: json.active_transactions || 0,
    };
  }

  /**
   * Ping the server.
   */
  async ping(): Promise<boolean> {
    try {
      const response = await this._send(InternalOpCode.PING);
      const { opcode } = this._parseResponse(response);
      return opcode === InternalOpCode.PONG;
    } catch {
      return false;
    }
  }

  /**
   * Close the connection.
   */
  async close(): Promise<void> {
    if (this._closed) return;
    this._closed = true;

    if (this._socket) {
      return new Promise((resolve) => {
        this._socket!.end(() => {
          this._socket = null;
          resolve();
        });
      });
    }
  }
}
