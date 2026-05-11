// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB Node.js/TypeScript SDK — Thin gRPC client wrapper

import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import * as path from "path";

/** SochDB client options. */
export interface SochDBOptions {
  /** Server address (host:port) */
  address?: string;
  /** API key for authentication */
  apiKey?: string;
}

/** SochDB client — provides access to all gRPC services. */
export class SochDB {
  private client: grpc.Client;
  private metadata: grpc.Metadata;
  private address: string;
  private packageDef: any;

  constructor(options: SochDBOptions = {}) {
    this.address = options.address || "localhost:50051";
    this.metadata = new grpc.Metadata();

    if (options.apiKey) {
      this.metadata.set("x-api-key", options.apiKey);
    }

    // Load proto definition dynamically
    const PROTO_PATH = path.resolve(
      __dirname,
      "../../sochdb-grpc/proto/sochdb.proto"
    );

    this.packageDef = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    });

    const protoDescriptor = grpc.loadPackageDefinition(this.packageDef);
    this.client = new grpc.Client(
      this.address,
      grpc.credentials.createInsecure()
    );
  }

  /** Get a gRPC service stub by name. */
  getService(serviceName: string): any {
    const protoDescriptor = grpc.loadPackageDefinition(this.packageDef);
    const sochdb = (protoDescriptor as any).sochdb?.v1;
    if (!sochdb || !sochdb[serviceName]) {
      throw new Error(`Service ${serviceName} not found in proto`);
    }
    return new sochdb[serviceName](
      this.address,
      grpc.credentials.createInsecure()
    );
  }

  /** Close the client connection. */
  close(): void {
    this.client.close();
  }
}

/** Helper: wrap a gRPC call in a Promise. */
export function promisify<TReq, TRes>(
  stub: any,
  method: string,
  request: TReq,
  metadata?: grpc.Metadata
): Promise<TRes> {
  return new Promise((resolve, reject) => {
    stub[method](request, metadata || new grpc.Metadata(), (err: any, response: TRes) => {
      if (err) reject(err);
      else resolve(response);
    });
  });
}
