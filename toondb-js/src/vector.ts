/**
 * ToonDB Vector Index
 *
 * HNSW vector search support for ToonDB.
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

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { DatabaseError } from './errors';

/**
 * Vector search result.
 */
export interface VectorSearchResult {
  /** ID of the vector */
  id: number;
  /** Distance from query vector */
  distance: number;
}

/**
 * Configuration for vector index.
 */
export interface VectorIndexConfig {
  /** Number of dimensions */
  dimension: number;
  /** Maximum connections per node (default: 16) */
  m?: number;
  /** Construction beam width (default: 100) */
  efConstruction?: number;
  /** Distance metric: 'cosine' | 'euclidean' | 'dot' (default: 'cosine') */
  metric?: 'cosine' | 'euclidean' | 'dot';
}

/**
 * Bulk build statistics.
 */
export interface BulkBuildStats {
  /** Number of vectors indexed */
  vectors: number;
  /** Build time in seconds */
  buildTimeSeconds: number;
  /** Throughput in vectors/second */
  rate: number;
}

/**
 * Get the path to the toondb-bulk binary.
 */
function findBulkBinary(): string {
  const platform = process.platform;
  const arch = process.arch;

  let target: string;
  if (platform === 'darwin') {
    target = arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
  } else if (platform === 'win32') {
    target = 'x86_64-pc-windows-msvc';
  } else {
    target = arch === 'arm64' ? 'aarch64-unknown-linux-gnu' : 'x86_64-unknown-linux-gnu';
  }

  const binaryName = platform === 'win32' ? 'toondb-bulk.exe' : 'toondb-bulk';

  // Search paths
  const searchPaths = [
    // Bundled in package
    path.join(__dirname, '..', '_bin', target, binaryName),
    path.join(__dirname, '_bin', target, binaryName),
    // Development
    path.join(__dirname, '..', '..', '..', 'target', 'release', binaryName),
    path.join(__dirname, '..', '..', '..', 'target', 'debug', binaryName),
  ];

  for (const p of searchPaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  // Try PATH
  const pathDirs = (process.env.PATH || '').split(path.delimiter);
  for (const dir of pathDirs) {
    const p = path.join(dir, binaryName);
    if (fs.existsSync(p)) {
      return p;
    }
  }

  throw new DatabaseError(
    `Could not find ${binaryName}. Install via: cargo build --release -p toondb-tools`
  );
}

/**
 * HNSW Vector Index for high-performance similarity search.
 *
 * @example
 * ```typescript
 * import { VectorIndex } from '@sushanth/toondb';
 *
 * // Build an index
 * const stats = await VectorIndex.bulkBuild(embeddings, {
 *   output: 'my_index.hnsw',
 *   dimension: 768,
 *   m: 16,
 *   efConstruction: 100,
 * });
 *
 * // Query the index
 * const results = await VectorIndex.query('my_index.hnsw', queryVector, {
 *   k: 10,
 *   efSearch: 64,
 * });
 * ```
 */
export class VectorIndex {
  /**
   * Build an HNSW index from vectors.
   *
   * @param vectors - Float32Array of vectors (dimension × count)
   * @param options - Build options
   * @returns Build statistics
   */
  static async bulkBuild(
    vectors: Float32Array,
    options: {
      output: string;
      dimension: number;
      m?: number;
      efConstruction?: number;
      metric?: 'cosine' | 'euclidean' | 'dot';
    }
  ): Promise<BulkBuildStats> {
    const bulkPath = findBulkBinary();
    const vectorCount = vectors.length / options.dimension;

    // Write vectors to temp file
    const tempFile = path.join(
      require('os').tmpdir(),
      `toondb_vectors_${Date.now()}.bin`
    );
    fs.writeFileSync(tempFile, Buffer.from(vectors.buffer));

    try {
      const args = [
        'build',
        '--input', tempFile,
        '--output', options.output,
        '--dimension', options.dimension.toString(),
        '--count', vectorCount.toString(),
        '--m', (options.m || 16).toString(),
        '--ef-construction', (options.efConstruction || 100).toString(),
        '--metric', options.metric || 'cosine',
      ];

      const startTime = Date.now();

      return new Promise((resolve, reject) => {
        const child = spawn(bulkPath, args);
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        child.on('close', (code) => {
          const buildTimeSeconds = (Date.now() - startTime) / 1000;

          if (code !== 0) {
            reject(new DatabaseError(`Bulk build failed: ${stderr}`));
            return;
          }

          resolve({
            vectors: vectorCount,
            buildTimeSeconds,
            rate: vectorCount / buildTimeSeconds,
          });
        });

        child.on('error', (err) => {
          reject(new DatabaseError(`Failed to spawn bulk process: ${err.message}`));
        });
      });
    } finally {
      // Clean up temp file
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    }
  }

  /**
   * Query an HNSW index.
   *
   * @param indexPath - Path to the index file
   * @param query - Query vector (Float32Array)
   * @param options - Query options
   * @returns Array of search results
   */
  static async query(
    indexPath: string,
    query: Float32Array,
    options?: {
      k?: number;
      efSearch?: number;
    }
  ): Promise<VectorSearchResult[]> {
    const bulkPath = findBulkBinary();
    const k = options?.k || 10;
    const efSearch = options?.efSearch || 64;

    // Write query to temp file
    const tempFile = path.join(
      require('os').tmpdir(),
      `toondb_query_${Date.now()}.bin`
    );
    fs.writeFileSync(tempFile, Buffer.from(query.buffer));

    try {
      const args = [
        'query',
        '--index', indexPath,
        '--query', tempFile,
        '--k', k.toString(),
        '--ef-search', efSearch.toString(),
      ];

      return new Promise((resolve, reject) => {
        const child = spawn(bulkPath, args);
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        child.on('close', (code) => {
          if (code !== 0) {
            reject(new DatabaseError(`Query failed: ${stderr}`));
            return;
          }

          try {
            const results: VectorSearchResult[] = JSON.parse(stdout);
            resolve(results);
          } catch {
            // Parse line format: id,distance
            const results: VectorSearchResult[] = stdout
              .trim()
              .split('\n')
              .filter((line) => line.length > 0)
              .map((line) => {
                const [id, distance] = line.split(',');
                return {
                  id: parseInt(id, 10),
                  distance: parseFloat(distance),
                };
              });
            resolve(results);
          }
        });

        child.on('error', (err) => {
          reject(new DatabaseError(`Failed to spawn query process: ${err.message}`));
        });
      });
    } finally {
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    }
  }

  /**
   * Get index metadata.
   *
   * @param indexPath - Path to the index file
   * @returns Index metadata
   */
  static async info(indexPath: string): Promise<{
    vectors: number;
    dimension: number;
    metric: string;
  }> {
    const bulkPath = findBulkBinary();

    return new Promise((resolve, reject) => {
      const child = spawn(bulkPath, ['info', '--index', indexPath]);
      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        if (code !== 0) {
          reject(new DatabaseError(`Info failed: ${stderr}`));
          return;
        }

        try {
          const info = JSON.parse(stdout);
          resolve({
            vectors: info.vectors || info.count || 0,
            dimension: info.dimension || 0,
            metric: info.metric || 'cosine',
          });
        } catch {
          reject(new DatabaseError(`Failed to parse info: ${stdout}`));
        }
      });

      child.on('error', (err) => {
        reject(new DatabaseError(`Failed to spawn info process: ${err.message}`));
      });
    });
  }

  // Instance properties for non-static usage
  private _path: string;
  private _config: VectorIndexConfig;

  /**
   * Create a VectorIndex instance.
   * 
   * @param indexPath - Path where the index will be stored
   * @param config - Optional configuration
   */
  constructor(indexPath: string, config?: Partial<VectorIndexConfig>) {
    this._path = indexPath;
    this._config = {
      dimension: config?.dimension || 0,
      m: config?.m || 16,
      efConstruction: config?.efConstruction || 100,
      metric: config?.metric || 'cosine',
    };
  }

  /**
   * Build an index from vectors (instance method).
   * 
   * @param vectors - Array of number arrays
   * @param labels - Optional labels for each vector
   */
  async bulkBuild(vectors: number[][], labels?: string[]): Promise<BulkBuildStats> {
    if (vectors.length === 0) {
      return { vectors: 0, buildTimeSeconds: 0, rate: 0 };
    }

    // Validate dimensions
    const dim = vectors[0].length;
    for (let i = 1; i < vectors.length; i++) {
      if (vectors[i].length !== dim) {
        throw new DatabaseError(
          `Vector ${i} has dimension ${vectors[i].length}, expected ${dim}`
        );
      }
    }

    // Flatten to Float32Array
    const flat = new Float32Array(vectors.length * dim);
    for (let i = 0; i < vectors.length; i++) {
      for (let j = 0; j < dim; j++) {
        flat[i * dim + j] = vectors[i][j];
      }
    }

    return VectorIndex.bulkBuild(flat, {
      output: this._path,
      dimension: dim,
      m: this._config.m,
      efConstruction: this._config.efConstruction,
      metric: this._config.metric,
    });
  }

  /**
   * Compute cosine distance between two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Cosine distance (0 = identical, 2 = opposite)
   */
  static computeCosineDistance(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) {
      return 1;
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return 1 - similarity;
  }

  /**
   * Compute Euclidean (L2) distance between two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Euclidean distance
   */
  static computeEuclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Compute dot product of two vectors.
   * 
   * @param a - First vector
   * @param b - Second vector
   * @returns Dot product
   */
  static computeDotProduct(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * Normalize a vector to unit length.
   * 
   * @param v - Input vector
   * @returns Normalized vector (new array)
   */
  static normalizeVector(v: number[]): number[] {
    let norm = 0;
    for (const x of v) {
      norm += x * x;
    }
    norm = Math.sqrt(norm);

    if (norm === 0) {
      return [...v];
    }

    return v.map(x => x / norm);
  }
}
