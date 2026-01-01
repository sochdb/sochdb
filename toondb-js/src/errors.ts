/**
 * ToonDB Error Classes
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

/**
 * Base error class for all ToonDB errors.
 */
export class ToonDBError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ToonDBError';
    Object.setPrototypeOf(this, ToonDBError.prototype);
  }
}

/**
 * Error thrown when connection to the database fails.
 */
export class ConnectionError extends ToonDBError {
  constructor(message: string) {
    super(message);
    this.name = 'ConnectionError';
    Object.setPrototypeOf(this, ConnectionError.prototype);
  }
}

/**
 * Error thrown when a transaction operation fails.
 */
export class TransactionError extends ToonDBError {
  constructor(message: string) {
    super(message);
    this.name = 'TransactionError';
    Object.setPrototypeOf(this, TransactionError.prototype);
  }
}

/**
 * Error thrown when there's a protocol error in IPC communication.
 */
export class ProtocolError extends ToonDBError {
  constructor(message: string) {
    super(message);
    this.name = 'ProtocolError';
    Object.setPrototypeOf(this, ProtocolError.prototype);
  }
}

/**
 * Error thrown when a database operation fails.
 */
export class DatabaseError extends ToonDBError {
  constructor(message: string) {
    super(message);
    this.name = 'DatabaseError';
    Object.setPrototypeOf(this, DatabaseError.prototype);
  }
}
